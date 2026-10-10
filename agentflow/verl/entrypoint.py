import hydra
import ray
import os
import traceback
from pathlib import Path

from .dataset import AgentDataset
from .trainer import AgentFlowTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.main_ppo import create_rl_sampler


@hydra.main(config_path="pkg://agentflow/verl", config_name="config", version_base=None)
def main(config):
    run_ppo(config)


def _search_cache_env_vars():
    """Propagate explicitly configured search routing to Ray actors."""
    names = (
        "SEARCH_CACHE_ENABLED", "SEARCH_CACHE_BASE_URL", "SEARCH_CACHE_TOKEN",
        "SEARCH_CACHE_CONNECT_TIMEOUT_SECONDS", "SEARCH_CACHE_READ_TIMEOUT_SECONDS",
        "SEARCH_GATEWAY_BASE_URL", "SEARCH_GATEWAY_TOKEN", "GATEWAY_TOKEN",
        "SEARCH_GATEWAY_CONNECT_TIMEOUT", "SEARCH_GATEWAY_READ_TIMEOUT",
    )
    if os.environ.get("SEARCH_CACHE_ENABLED", "").lower() not in {"1", "true", "yes", "on"}:
        return {"SEARCH_CACHE_ENABLED": "0"}
    return {name: os.environ[name] for name in names if name in os.environ}


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        env_vars = {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
        for name in (
            "AGENTFLOW_PROFILE_VAL_TIMING",
            "AGENTFLOW_PROFILE_VAL_LIMIT",
            "AGENTFLOW_PROFILE_VAL_TIMEOUT_S",
            "AGENTFLOW_PROFILE_OUTPUT_DIR",
        ):
            if name in os.environ:
                env_vars[name] = os.environ[name]
        env_vars.update(_search_cache_env_vars())
        ray.init(
            runtime_env={"env_vars": env_vars}
            # runtime_env={
            #     "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
            # },
            # num_cpus=max(min(os.cpu_count() - 4, 10), 1), # config.ray_init.num_cpus, 
            # To fix the omiga config issue, you can try not commenting this line if your speed is not that satisfying, while it may cause Error for cpu matching. 
        )

    # Covers an already initialized/external Ray cluster as well as local init.
    runner = TaskRunner.options(
        runtime_env={"env_vars": _search_cache_env_vars()}
    ).remote()


    # ray.get(runner.run.remote(config))

    try:
        result = ray.get(runner.run.remote(config))
    except ray.exceptions.RayTaskError as e:
        print("\n" + "="*80)
        print(f"RayTaskError wrapper: {e}")
        print("-"*80)
        # 核心：打印 cause 的完整 traceback
        if hasattr(e, 'cause') and e.cause is not None:
            print("ORIGINAL CAUSE (被 Ray 吞掉的细节):")
            traceback.print_exception(type(e.cause), e.cause, e.cause.__traceback__)
        else:
            print("NO CAUSE AVAILABLE")
        print("="*80 + "\n")
        raise

    if result is not None:
        output_dir = Path(os.environ.get("AGENTFLOW_PROFILE_OUTPUT_DIR", "rollout_data/val_timing")).expanduser().resolve()
        if "detail_jsonl" in result:
            output_dir.mkdir(parents=True, exist_ok=True)
            detail_path = output_dir / result["detail_name"]
            summary_path = output_dir / result["summary_name"]
            detail_path.write_text(result["detail_jsonl"], encoding="utf-8")
            summary_path.write_text(result["summary_json"], encoding="utf-8")
            print(f"Timing report copied to head: {detail_path}\nTiming summary copied to head: {summary_path}")
        if result.get("error"):
            raise RuntimeError(f"Validation task failed after timing report collection:\n{result['error']}")


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            # from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
            from .diy.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        task = config.data.get("task", "qa")
        algorithm = config.data.get("algorithm", "grpo")

        ### 通过 session id 来控制

        from verl.utils.dataset.rl_dataset import collate_fn
        # Use our special dataset
        train_dataset = AgentDataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        val_dataset = AgentDataset(
            data_files=config.data.val_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = AgentFlowTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            task=task,
            algorithm=algorithm
        )
        trainer.init_workers()
        try:
            trainer.fit()
        except Exception:
            if os.environ.get("AGENTFLOW_PROFILE_VAL_TIMING", "").lower() in {"1", "true", "yes"}:
                return self._timing_result(trainer, traceback.format_exc())
            raise
        if os.environ.get("AGENTFLOW_PROFILE_VAL_TIMING", "").lower() in {"1", "true", "yes"}:
            return self._timing_result(trainer)

    @staticmethod
    def _timing_result(trainer, error=None):
        paths = getattr(trainer, "timing_report_paths", None)
        if paths is None:
            if error:
                return {"error": error}
            raise RuntimeError("Timing profile finished without producing a validation report")
        detail_path, summary_path = paths
        return {
            "detail_name": detail_path.name,
            "detail_jsonl": detail_path.read_text(encoding="utf-8"),
            "summary_name": summary_path.name,
            "summary_json": summary_path.read_text(encoding="utf-8"),
            "error": error,
        }


if __name__ == "__main__":
    main()
