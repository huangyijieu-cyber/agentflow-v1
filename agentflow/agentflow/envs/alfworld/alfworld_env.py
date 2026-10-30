import os
import numpy as np
import yaml
from copy import deepcopy
import re

# ==================== 修复 fast_downward 磁盘占满问题 ====================
import fast_downward
import fast_downward.interface as _fd_interface
import ctypes
import shutil
import fcntl  # Linux 文件锁，多进程安全

_fd_lib_handle = None
_fixed_so_path = "/tmp/fast_downward_libdownward_reuse.so"

def _fixed_load_lib():
    global _fd_lib_handle
    if _fd_lib_handle is not None:
        return _fd_lib_handle

    # 多进程安全：用文件锁保证只有一个进程执行 copy
    lock_path = "/tmp/.fd_libdownward.lock"
    with open(lock_path, "w") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX)
        try:
            if not os.path.exists(_fixed_so_path):
                shutil.copyfile(_fd_interface.DOWNWARD_LIB_PATH, _fixed_so_path)
        finally:
            fcntl.flock(lockfile, fcntl.LOCK_UN)

    _fd_lib_handle = ctypes.CDLL(_fixed_so_path)
    return _fd_lib_handle

# 必须同时替换两个入口，确保 textworld 调用的是 patch 后的版本
_fd_interface.load_lib = _fixed_load_lib
fast_downward.load_lib = _fixed_load_lib
print("[PATCH] fast_downward.load_lib patched")
# =====================================================================

def upper_path(path):
    return os.path.dirname(path)

## file path
file_path = os.path.abspath(__file__)
for _ in range(5):
    file_path = upper_path(file_path)
file_path = os.path.join(file_path, "data/alfworld-master")
data_path = os.path.join(file_path, "json_data")

import sys
sys.path.append(file_path)


from agentflow.envs.alfworld.alfworld_template import ALFWORLD_TEMPLATE_NO_HIS, ALFWORLD_TEMPLATE
from agentflow.envs.memory import SimpleMemory
from alfworld.agents.environment import get_environment


ALFWORLD_TRACE_INSTRUCTION = (
    "Based on the applicable environment rules, initial state, and verified "
    "current state above, describe the current state in relation to the task "
    "success conditions. A desired condition is not evidence that it is met."
)


def _environment_scalar(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _environment_flag(value):
    value = _environment_scalar(value)
    if isinstance(value, (bool, int, float)) and value in (0, 1):
        return bool(value)
    return None


def _goal_tools():
    # The deterministic parser is lightweight; importing the environment is not
    # required when converting games or rescoring a previously saved Probe.
    from agentflow.trace import alfworld
    return alfworld


def _saved_goal_tools():
    # Training reads a prebuilt artifact; offline conversion is never a reset side effect.
    from agentflow.trace import alfworld_goal_store
    return alfworld_goal_store


def _parquet_goal_tools():
    from agentflow.trace import alfworld_parquet
    return alfworld_parquet


def _create_trace_environment(base_env, batch_size=1):
    """Mirror official TW registration, additionally requesting actual facts.

    Do not patch textworld.EnvInfos globally or modify the installed ALFWorld.
    The wrappers, expert, split randomization and budget match AlfredTWEnv.
    """
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler, AlfredExpert, AlfredInfos,
    )

    config = base_env.config
    if config["env"]["type"] != "AlfredTWEnv":
        raise ValueError("Full-state ALFWorld TRACE requires AlfredTWEnv")
    randomize = config["env"]["domain_randomization"] if base_env.train_eval == "train" else False
    wrappers = [AlfredDemangler(shuffle=randomize), AlfredInfos]
    request_infos = textworld.EnvInfos(
        won=True, admissible_commands=True, facts=True, last_action=True,
        extras=["gamefile"],
    )
    training_method = config["general"]["training_method"]
    if training_method == "dqn":
        budget = config["rl"]["training"]["max_nb_steps_per_episode"]
    elif training_method == "dagger":
        budget = config["dagger"]["training"]["max_nb_steps_per_episode"]
        if base_env.train_eval == "train":
            wrappers.append(AlfredExpert(expert_type=config["env"]["expert_type"]))
            request_infos.extras.append("expert_plan")
    else:
        raise NotImplementedError(f"Unsupported ALFWorld training method: {training_method}")
    env_id = textworld.gym.register_games(
        base_env.game_files, request_infos, batch_size=batch_size,
        asynchronous=True, max_episode_steps=budget, wrappers=wrappers,
    )
    return textworld.gym.make(env_id)


def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def update_config_with_data(raw_config):
    if isinstance(raw_config, dict):
        for key, value in raw_config.items():
            if isinstance(value, str):
                if "$ALFWORLD_DATA" in value:
                    new_value = value.replace("$ALFWORLD_DATA", data_path)
                    raw_config[key] = new_value
            if isinstance(value, dict):
                update_config_with_data(value)


def parse_gamefile(info):
    if "extra.gamefile" in info:
        gamefile = info["extra.gamefile"]
    else:
        gamefile = None
    return gamefile


def set_gamefile(info, gamefile):
    if "extra.gamefile" in info:
        info['extra.gamefile'] = gamefile
    else:
        info['extra.gamefile'] = None
    return info



class AlfWorldEnv:
    def __init__(self, history_length=4, max_step=50, trace_enabled=None, trace_goals_path=None):
        # ====== 延迟导入：确保 JVM 只在 worker 进程内启动 ======
        self.env = None
        self.seed = 44
        self._rng = np.random.RandomState(self.seed)
        self.history_length = history_length
        self.max_step = max_step

        self.memory = SimpleMemory()
        self.train_env = None
        self.val_seen_env = None
        self.val_unseen_env = None

        # 防御性初始化：避免 reset 前调用 step/close 时 AttributeError
        self.step_count = 0
        self.prev_text_obs = ""
        self.prev_admissible_command = [[]]
        self.task = None
        self.gamefile = None
        if trace_enabled is None:
            trace_enabled = (
                "slic" in os.environ.get("ALGORITHM", "grpo+slic").split("+")
                and os.environ.get("TRACE_ENABLED", "true").lower() not in ("0", "false", "no")
            )
        self.trace_enabled = bool(trace_enabled)
        configured_goals = os.environ.get("ALFWORLD_TRACE_GOALS", "") if trace_goals_path is None else trace_goals_path
        self.trace_goals_path = os.fspath(configured_goals) if configured_goals else ""
        self.trace_task = None
        self.trace_goal_spec = None
        self.trace_goal_metadata = None
        self.trace_gt = None
        self.trace_answer_instruction = ALFWORLD_TRACE_INSTRUCTION
        self.trace_state_context = None
        self.trace_state_view = "full_state" if self.trace_enabled else None
        self.trace_facts = None
        self.trace_goal_check = None
        self.trace_goal_check_valid = False
        self.trace_action_history = []
        self.trace_scoring_valid = False

        # 对应集合
        self._env_instances = {}
        self._env_gamefiles = {}

    def set_trace_task(self, task):
        """Bind a prepared dataset row without exposing its GT to actor prompts."""
        if not isinstance(task, dict):
            raise ValueError("ALFWorld TRACE requires a prepared Parquet task row")
        self.trace_task = deepcopy(task)

    def _load_trace_goal(self, info):
        if self.trace_task is not None:
            self.trace_goal_spec = _parquet_goal_tools().read_parquet_task_goal(
                self.trace_task, info.get("extra.gamefile"))
            # The filled dataset result is the authoritative scoring target.
            self.trace_gt = self.trace_task["result"]
        else:
            self.trace_goal_spec = _saved_goal_tools().read_goal_spec(
                self.trace_goals_path, info.get("extra.gamefile"))
            self.trace_gt = self.trace_goal_spec["goal_response"]
        self.trace_answer_instruction = self.trace_goal_spec.get(
            "answer_instruction", ALFWORLD_TRACE_INSTRUCTION)
        self.trace_goal_metadata = {
            key: deepcopy(self.trace_goal_spec.get(key))
            for key in ("gamefile", "goal_source", "goal_task_type", "template_version",
                        "source_sha256", "goal_ast", "goal_validation_status", "goal_spec_hash",
                        "goal_store_path", "goal_store_id", "goal_store_entry", "goal_store_spec_sha256",
                        "goal_store_selection_mode", "goal_store_selection_sha256",
                        "goal_store_dataset_sha256", "goal_store_dataset_rows")
        }
        if self.trace_task is not None:
            self.trace_goal_metadata.update({
                "goal_source": "parquet.result",
                "parquet_goal_spec_sha256": self.trace_task.get("alfworld_goal_spec_sha256"),
                "parquet_goal_row": self.trace_task.get("alfworld_goal_row"),
                "parquet_game_key": self.trace_task.get("alfworld_game_key"),
            })
        self.trace_action_history = []

    def _refresh_trace_state(self, info, observation, action=None):
        """Use simulator facts, never infer execution success from model text."""
        if not self.trace_enabled:
            return
        tools = _goal_tools()
        raw_facts = info.get("facts")
        if raw_facts is None:
            raise RuntimeError("ALFWorld full-state TRACE requires facts=True in environment infos")
        if isinstance(raw_facts, (list, tuple)) and len(raw_facts) == 1 and isinstance(raw_facts[0], list):
            raw_facts = raw_facts[0]
        self.trace_facts = tools.normalize_facts(raw_facts)
        if not self.trace_facts:
            raise RuntimeError("ALFWorld full-state TRACE received an empty fact set")
        if action is not None:
            last_action = _environment_scalar(info.get("last_action"))
            self.trace_action_history.append({
                "action": action, "observation": observation,
                "environment_last_action": str(last_action) if last_action is not None else None,
            })
        self.trace_goal_check = bool(tools.evaluate_goal(self.trace_goal_spec, self.trace_facts))
        won = _environment_flag(info.get("won"))
        self.trace_goal_check_valid = won is not None and won == self.trace_goal_check
        self.trace_scoring_valid = self.trace_goal_check_valid
        self.trace_state_context = f"Task: {self.task}\n\n" + tools.render_full_state(
            self.trace_goal_spec, self.trace_facts,
            observation=observation, action_history=self.trace_action_history,
        )
        self._write_trace_info(info)
        if not self.trace_goal_check_valid:
            info["trace_goal_diagnostic"] = (
                f"Goal evaluator={self.trace_goal_check!r}; environment won={won!r}. "
                "The state is retained for inspection but is invalid for scoring/pairing."
            )

    def _write_trace_info(self, info):
        info.update({
            "trace_full_facts": deepcopy(self.trace_facts),
            "trace_state_context": self.trace_state_context,
            "trace_gt": self.trace_gt,
            "trace_answer_instruction": self.trace_answer_instruction,
            "trace_state_view": "full_state",
            "trace_goal_metadata": deepcopy(self.trace_goal_metadata),
            "trace_goal_check": self.trace_goal_check,
            "trace_goal_check_valid": self.trace_goal_check_valid,
            "trace_scoring_valid": self.trace_scoring_valid,
        })

    def _trace_failed_action(self, info, action, observation, *, external=False):
        if not self.trace_enabled:
            return info
        info.update({"trace_env_action": action, "terminated": not external,
                     "truncated": False, "env_error": bool(external)})
        self.trace_action_history.append({
            "action": action, "observation": observation,
            "executed": False, "external_error": bool(external),
        })
        if external:
            self.trace_facts = None
            self.trace_goal_check = None
            self.trace_goal_check_valid = False
            self.trace_scoring_valid = False
            self.trace_state_context = (
                "Current full state is unavailable after an environment execution failure.\n"
                f"Environment observation: {observation}"
            )
        else:
            # Rejected format/illegal actions never reached the simulator, so
            # its last verified facts remain the actual unchanged state.
            self.trace_state_context = f"Task: {self.task}\n\n" + _goal_tools().render_full_state(
                self.trace_goal_spec, self.trace_facts,
                observation=observation, action_history=self.trace_action_history,
            )
        self._write_trace_info(info)
        return info

    def _init_env(self, data_type:str, task_type:str):
        alf_config_path = os.path.join(os.path.dirname(__file__), f"config/config_tw_{task_type}.yaml")
        config = load_config_file(alf_config_path)
        update_config_with_data(config)
        env_type = config["env"]["type"]

        if data_type == "train":
            data_type = "train"
        elif data_type == "val_seen":
            data_type = "eval_in_distribution"
        elif data_type == "val_unseen":
            data_type = "eval_out_of_distribution"
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        base_env = get_environment(env_type)(config, train_eval=data_type)
        return base_env

    def _extract_task(self, text_obs: str):
        task_start = text_obs.find('Your task is to: ')
        if task_start != -1:
            return text_obs[task_start + len('Your task is to: '):].strip()
        return None

    def _build_prompt(self, text_obs: str, admissible_actions: list, init):
        if not init and self.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.history_length,
                obs_key="text_obs",
                action_key="action"
            )

        # admissible_actions 是 batch 格式，如 [['go to kitchen', 'take apple']]
        actions_list = admissible_actions[0] if (isinstance(admissible_actions, list) and len(admissible_actions) > 0) else []
        reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in actions_list if s != 'help')

        if init or self.history_length <= 0:
            prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs,
                    admissible_actions=reformatted_admissible_actions
                )
        else:
            prompt = ALFWORLD_TEMPLATE.format(
                    task_description=self.task,
                    step_count=len(self.memory[0]),
                    history_length=valid_lens[0],
                    action_history=memory_contexts[0],
                    current_step=len(self.memory[0]) + 1,
                    current_observation=text_obs,
                    admissible_actions=reformatted_admissible_actions
                )
                
        return prompt
    
    # ==== 松弛检验 ==============================
    def _parse_action(self, response):
        """
        解析模型响应，提取 <action>...</action> 中的动作。
        返回: (action_str, is_valid_format)
        """
        if not isinstance(response, str):
            return str(response)[-30:], False

        ## 必须不包括额外思考废话
        # 严格检查格式
        action_match = re.search(r'<action>(.*?)</action>', response)

        if not action_match:
            return "", False


        original_str = response
        text = response

        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)

        # 找不到完整的 <action>...</action> 标签
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return original_str[-30:], False

        extracted = text[start_idx + len(start_tag):end_idx]
        if not extracted:
            return "", False

        return extracted, True


    
    # ## ====== 暂时未启用的严格reward，避免think出来的检测 =======
    # def _parse_action(self, response):
    #     """
    #     严格解析模型响应。
    #     合法格式：去除首尾空白后，必须严格为 <action>...content...</action>
    #     任何前缀、后缀、嵌套标签、空内容都视为格式错误。
    #     """
    #     if not isinstance(response, str):
    #         return str(response)[-30:], False

    #     text = response.strip()
    #     start_tag = "<action>"
    #     end_tag = "</action>"

    #     # 1. 必须以 <action> 开头，以 </action> 结尾
    #     if not text.startswith(start_tag) or not text.endswith(end_tag):
    #         return response[-30:], False

    #     # 2. 提取内容
    #     extracted = text[len(start_tag):-len(end_tag)]

    #     # 3. 内容不能为空或纯空白
    #     if not extracted or not extracted.strip():
    #         return "", False

    #     # 4. 内容内部不能包含其他标签符号（防止嵌套或乱码）
    #     if '<' in extracted or '>' in extracted:
    #         return response[-30:], False

    #     # 5. 确保整个字符串只出现一次 <action> 和一次 </action>
    #     # （防止 "<action>foo</action>bar<action>baz</action>" 这种重复结构）
    #     if text.count(start_tag) != 1 or text.count(end_tag) != 1:
    #         return response[-30:], False

    #     return extracted, True
    # # ======================================================================

    def _is_valid_action(self, action, admissible_actions):
        """
        检查动作是否在 admissible_commands 列表中。
        admissible_actions: batch 格式，如 [['go to kitchen', 'take apple']]
        """
        if not action:
            return False

        if isinstance(admissible_actions, list) and len(admissible_actions) > 0:
            if isinstance(admissible_actions[0], list):
                valid_set = set(admissible_actions[0])
            else:
                valid_set = set(admissible_actions)
        else:
            valid_set = set()

        return action in valid_set

    def reset(self, question_id=None, data_type:str="val_unseen", task_type:str="all", gamefile=None):
        ## 判断 data_type合理性
        if not data_type in ["val_unseen", "val_seen", "train"]:
            raise ValueError(f"Data type ({data_type}) is not exist.")

        mapped_gamefile = None
        if self.trace_enabled:
            if self.trace_task is not None:
                requested_question = f"{data_type}_{task_type}_alfworld_{question_id}"
                if self.trace_task.get("question") != requested_question:
                    raise ValueError("ALFWorld reset parameters differ from the bound prepared Parquet question")
                mapped_gamefile = _parquet_goal_tools().parquet_task_gamefile(
                    self.trace_task, os.path.join(data_path, "json_2.1.1"))
                if gamefile is not None and str(_goal_tools().normalize_gamefile(gamefile)) != str(mapped_gamefile):
                    raise ValueError("Explicit gamefile differs from the prepared Parquet row")
            elif self.trace_goals_path.strip():
                if gamefile is None and question_id is not None:
                    mapped_gamefile = _saved_goal_tools().read_dataset_gamefile(
                        self.trace_goals_path, data_type, task_type, question_id,
                        os.path.join(data_path, "json_2.1.1"))
            else:
                raise ValueError(
                    "ALFWorld TRACE requires a prepared Parquet row with result and saved goal metadata. "
                    "Bind it with set_trace_task(task) before reset. Legacy Probe sessions may instead "
                    "set ALFWORLD_TRACE_GOALS to an existing offline SQLite file. "
                    "Training never converts missing goals automatically."
                )

        ## 修改，令data_type只init_env一次
        env_key = (data_type, task_type)
        if not env_key in self._env_instances:
            self._env_instances[env_key] = self._init_env(data_type=data_type, task_type=task_type)
            self._env_gamefiles[env_key] = deepcopy(self._env_instances[env_key].game_files)
        
        base_env = self._env_instances[env_key]
        org_gamefiles = self._env_gamefiles[env_key]

        # 选择场景
        if gamefile is not None:
            select_gamefile = [str(_goal_tools().normalize_gamefile(gamefile))]
        elif mapped_gamefile is not None:
            select_gamefile = [str(mapped_gamefile)]
        elif question_id is not None:
            if not (0 <= question_id < len(org_gamefiles)):
                raise IndexError(f"question_id ({question_id}) out of range [0, {len(org_gamefiles)})")
            select_gamefile = [org_gamefiles[question_id]]
        else:
            select_gamefile = [self._rng.choice(org_gamefiles)]
        base_env.game_files = select_gamefile

        if self.env is not None:
            self.env.close()
        self.env = (_create_trace_environment(base_env, batch_size=1) if self.trace_enabled
                    else base_env.init_env(batch_size=1))
        obs, info = self.env.reset()
        if mapped_gamefile is not None:
            actual_gamefile = _goal_tools().normalize_gamefile(info.get("extra.gamefile"))
            if str(actual_gamefile) != str(mapped_gamefile):
                raise ValueError("ALFWorld loaded a different game than the saved dataset row mapping")
        info["observation_text"] = obs
        text_obs = obs[0]
        self.prev_admissible_command = info['admissible_commands']
        self.gamefile = parse_gamefile(info)
        self.memory.reset(batch_size=1)
        self.task = None
        self.prev_text_obs = text_obs
        info["anchor"] = text_obs

        self.task = self._extract_task(text_obs)
        self.step_count = 0
        full_text_obs = self._build_prompt(text_obs, self.prev_admissible_command, init=True)
        if self.trace_enabled:
            self._load_trace_goal(info)
            self._refresh_trace_state(info, text_obs)
        return full_text_obs, info

    def step(self, response):
        print(f"[alfworld_env.py]-[step]-response: {response}")
        self.step_count += 1

        # 1. 解析动作
        action, is_valid_format = self._parse_action(response)
        print(f"[alfworld_env.py]-[step]-action: {action}, format_valid: {is_valid_format}")

        # 2. 格式错误：直接终止，给予惩罚
        if not is_valid_format:
            print(f"[alfworld_env.py]-[step] FORMAT ERROR! Raw response tail: ...{action}")
            admissible_list = self.prev_admissible_command[0] if self.prev_admissible_command else []
            error_obs = (
                f"[ERROR] Invalid action format!\n"
                f"Your response was:\n---\n{response[:300]}\n---\n"
                f"Expected format: <action>your_action_here</action>\n"
                f"Available actions: {admissible_list}\n"
                f"Episode terminated due to format error."
            )
            info = {
                "won": False,
                "extra.gamefile": self.gamefile,
                "anchor": "[FORMAT_ERROR]",
                "observation_text": error_obs,
                "admissible_commands": self.prev_admissible_command,
                "format_error": True,
                "invalid_action": False,
            }
            # return error_obs, -0.01, True, info
            self._trace_failed_action(info, action, error_obs)
            return error_obs, 0.0, True, info

        # 3. 检查动作是否在 admissible_commands 中
        valid = self._is_valid_action(action, self.prev_admissible_command)

        if valid:
            # 3a. 合法动作：正常执行环境 step
            actions = [action]

            # ===== 兜底：环境内部异常截断 =====
            try:
                result = self.env.step(actions)
            except Exception as e:
                import traceback as tb_module
                exc_type = type(e).__name__
                exc_msg = str(e)
                exc_trace = tb_module.format_exc()
                
                # === 控制台：精简但关键信息不丢 ===
                print(f"[alfworld_env.py]-[step] ENV ERROR at step={self.step_count}, game={self.gamefile}")
                print(f"  action={action!r}")
                print(f"  exception={exc_type}: {exc_msg}")
                # 只打印 traceback 的最后 5 行，避免刷屏
                tb_lines = exc_trace.strip().split('\n')
                print("  traceback tail:\n    " + '\n    '.join(tb_lines[-5:]))

                # === observation：给 LLM 看的，保持简洁 ===
                error_obs = (
                    f"[ERROR] Environment internal error after action: '{action}'\n"
                    f"Episode terminated due to environment failure."
                )

                # === info：给上游 trainer/debug 用的，信息拉满 ===
                info = {
                    "won": False,
                    "extra.gamefile": self.gamefile,
                    "anchor": "[ENV_ERROR]",
                    "observation_text": error_obs,
                    "admissible_commands": self.prev_admissible_command,
                    "format_error": False,
                    "invalid_action": False,
                    "env_error": True,
                    "env_error_type": exc_type,
                    "env_error_msg": exc_msg,
                    "env_error_traceback": exc_trace,        # 完整 traceback，存起来
                    "env_error_step": self.step_count,       # 第几步崩的
                    "env_error_action": action,              # 哪个动作触发的
                    "env_error_prev_obs": self.prev_text_obs, # 崩之前的状态
                }
                # return error_obs, -0.05, True, info
                self._trace_failed_action(info, action, error_obs, external=True)
                return error_obs, 0.0, True, info

            if len(result) == 5:
                next_observation, reward, terminated, truncated, info = result
                terminated = _environment_flag(terminated) is True
                truncated = _environment_flag(truncated) is True
                done = terminated or truncated
            else:
                next_observation, reward, done, info = result
                done = _environment_flag(done) is True
                truncated = _environment_flag((info or {}).get(
                    "truncated", (info or {}).get("TimeLimit.truncated", False))) is True
                terminated = done and not truncated

            print("next_observation:", next_observation[0] if isinstance(next_observation, list) else next_observation)

            self.prev_admissible_command = info['admissible_commands']

            next_text_obs = next_observation[0]
            self.memory.store({"text_obs": [self.prev_text_obs], "action": [action]})
            self.prev_text_obs = next_text_obs
            full_next_text_obs = self._build_prompt(next_text_obs, self.prev_admissible_command, init=False)

            info["anchor"] = next_text_obs
            if info.get("extra.gamefile") is None:
                info = set_gamefile(info, self.gamefile)

            # 安全解包：兼容 batch(list) 和标量两种返回
            reward = reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward
            done = done[0] if isinstance(done, (list, tuple, np.ndarray)) else done

            # max_step 截断
            if self.step_count >= self.max_step and not done:
                done = True
                truncated = True
                info["truncated_by_max_step"] = True

            info["format_error"] = False
            info["invalid_action"] = False
            if self.trace_enabled:
                info.update({"trace_env_action": action, "terminated": terminated,
                             "truncated": truncated, "env_error": False})
                self._refresh_trace_state(info, next_text_obs, action)
            return full_next_text_obs, reward, done, info

        else:
            # 3b. 非法动作：不在 admissible_commands 中，终止并给予惩罚
            print(f"[alfworld_env.py]-[step] INVALID ACTION! '{action}' not in admissible: {self.prev_admissible_command}")
            admissible_list = self.prev_admissible_command[0] if self.prev_admissible_command else []
            error_obs = (
                f"[ERROR] Invalid action: '{action}'\n"
                f"Available actions: {admissible_list}\n"
                f"Episode terminated due to invalid action."
            )
            info = {
                "won": False,
                "extra.gamefile": self.gamefile,
                "anchor": "[INVALID_ACTION]",
                "observation_text": error_obs,
                "admissible_commands": self.prev_admissible_command,
                "format_error": False,
                "invalid_action": True,
            }
            # return error_obs, -0.001, True, info
            self._trace_failed_action(info, action, error_obs)
            return error_obs, 0.0, True, info

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None



def run_random_agent(num_episodes=3, max_steps_per_episode=15):
    env = AlfWorldEnv()
    for episode in range(num_episodes):
        obs, info = env.reset(question_id=1, data_type="val_seen", task_type="pick2")
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Task: {env.task}")
        print(f"{'='*60}")

        episode_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            admissible_actions = info.get('admissible_commands', [[]])[0]
            if len(admissible_actions) == 0:
                print("No admissible actions available, breaking.")
                break

            random_action = np.random.choice(admissible_actions)
            action_response = f"<action>{random_action}</action>"

            print(f"\n[Step {step + 1}] Available actions: {admissible_actions}")
            print(f"[Step {step + 1}] Randomly chosen: {random_action}")

            obs, reward, done, info = env.step(action_response)
            episode_reward += reward
            step += 1
            print(f"[Step {step}] Reward: {reward}, Done: {done}")

        print(f"\nEpisode {episode + 1} finished in {step} steps.")
        print(f"Total reward: {episode_reward}")
        print(f"Won: {info.get('won', False)}")

    env.close()
    print("\nAll episodes completed.")


if __name__ == "__main__":
    run_random_agent(num_episodes=1, max_steps_per_episode=10)
