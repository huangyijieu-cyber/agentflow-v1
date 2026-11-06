import os
import re
import numpy as np
from copy import deepcopy

from agentflow.envs.webshop.webshop_template import WEBSHOP_TEMPLATE_NO_HIS, WEBSHOP_TEMPLATE
from agentflow.envs.memory import SimpleMemory

def upper_path(path):
    upper_path = os.path.dirname(path)
    return upper_path

## file path
file_path = os.path.abspath(__file__)
for _ in range(5):
    file_path = upper_path(file_path)
file_path = os.path.join(file_path, "data/WebShop-master")
file_path = os.path.abspath(os.environ.get("WEBSHOP_ROOT") or file_path)

WEBSHOP_TRACE_INSTRUCTION = (
    "Based on the verified WebShop state above, describe the current candidate "
    "product type, attributes, applicable price, and actually selected options "
    "in relation to the task requirements. Desired options are not selected options."
)


def _goal_tools():
    from agentflow.trace import webshop_goal
    return webshop_goal


def _state_tools():
    from agentflow.trace import webshop_state
    return webshop_state


def _parse_trace_history_length(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("none", "null"):
            return None
        if re.fullmatch(r"\+?[0-9]+", text):
            return int(text)
    elif type(value) is int and value >= 0:
        return value
    raise ValueError("WEBSHOP_TRACE_HISTORY_LENGTH must be None or a nonnegative integer")


def _make_native_environment(webshop_root):
    """Start WebShop and its JVM only when this worker first resets an env."""
    import sys
    import shutil
    import gym

    if webshop_root not in sys.path:
        sys.path.insert(0, webshop_root)
    import web_agent_site.engine.engine as engine
    from pyserini.search.lucene import LuceneSearcher
    from web_agent_site.utils import DEBUG_PROD_SIZE

    # Each process maps its own Lucene index, as in the original adapter.
    if not getattr(engine, "_agentflow_isolated_index", False):
        private_index = f"/tmp/webshop_index_worker_{os.getpid()}"
        if not os.path.exists(private_index):
            index_name = {None: "indexes", 100: "indexes_100", 1000: "indexes_1k",
                          100000: "indexes_100k"}.get(DEBUG_PROD_SIZE)
            candidates = [
                os.environ.get("WEBSHOP_INDEX_DIR"),
                os.path.join(webshop_root, "indexes"),
                os.path.join(webshop_root, "index"),
                os.path.join(webshop_root, "web_agent_site", "indexes"),
                os.path.join(webshop_root, "search_engine", index_name) if index_name else None,
                "/tmp/webshop_default_index",
            ]
            for candidate in candidates:
                if (candidate and os.path.isdir(candidate)
                        and any(name.startswith("segments") for name in os.listdir(candidate))):
                    shutil.copytree(candidate, private_index)
                    break
        original_init = engine.init_search_engine

        def isolated_init(num_products=None):
            if os.path.isdir(private_index):
                return LuceneSearcher(private_index)
            return original_init(num_products)

        engine.init_search_engine = isolated_init
        engine._agentflow_isolated_index = True
    # Import registration after patching the engine's imported callable.
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401
    return gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=DEBUG_PROD_SIZE)


class WebShopEnv:
    def __init__(self, history_length=4, max_step=15, trace_enabled=None,
                 trace_manifest_path=None, trace_history_length=..., webshop_root=None):
        # ====== 延迟导入：确保 JVM 只在 worker 进程内启动 ======
        self.history_length = history_length
        self.max_step = max_step
        self.env = None  ## 延迟初始化
        self.seed = 42
        self._rng = np.random.RandomState(self.seed)
        self.webshop_root = os.path.abspath(webshop_root or os.environ.get("WEBSHOP_ROOT") or file_path)
        if trace_enabled is None:
            trace_enabled = (
                "slic" in os.environ.get("ALGORITHM", "grpo+slic").split("+")
                and os.environ.get("TRACE_ENABLED", "true").lower() not in ("0", "false", "no")
            )
        self.trace_enabled = bool(trace_enabled)
        configured_manifest = (os.environ.get("WEBSHOP_TRACE_MANIFEST", "")
                               if trace_manifest_path is None else trace_manifest_path)
        self.trace_manifest_path = os.fspath(configured_manifest) if configured_manifest else ""
        # Validation disables scoring on its own adapter but must keep this
        # experiment's frozen prices/goals. Never share mutable servers.
        self.trace_manifest_enabled = bool(self.trace_enabled and self.trace_manifest_path)
        configured_history = (os.environ.get("WEBSHOP_TRACE_HISTORY_LENGTH", "4")
                              if trace_history_length is ... else trace_history_length)
        self.trace_history_length = (_parse_trace_history_length(configured_history)
                                     if self.trace_enabled else 4)
        self._trace_manifest = None
        self._trace_question = None
        self.trace_task = None
        self.trace_goal_spec = None
        self.trace_goal_metadata = None
        self.trace_gt = None
        self.trace_answer_instruction = WEBSHOP_TRACE_INSTRUCTION
        self.trace_state_view = "webshop_state" if self.trace_enabled else None
        self.trace_state_context = None
        self.trace_snapshot = None
        self.trace_terminal_snapshot = None
        self.trace_goal_check = None
        self.trace_goal_check_valid = False
        self.trace_scoring_valid = False
        self.trace_full_actor_history = []
        self.step_count = 0
        self.memory = SimpleMemory()
        self.memory.reset(batch_size=1)
        print("#### Params #####")
        print(f"history length: {self.history_length}")
        print(f"max step: {self.max_step}")

    def set_trace_task(self, task):
        """Bind the prepared training row without adding GT to actor input."""
        if not isinstance(task, dict):
            raise ValueError("WebShop TRACE requires a prepared Parquet task row")
        self.trace_task = deepcopy(task)

    def _native_env(self):
        return getattr(self.env, "unwrapped", self.env)

    def _restore_trace_catalog(self):
        if not self.trace_manifest_enabled:
            return
        if self._trace_manifest is None:
            self._trace_manifest = _goal_tools().load_frozen_manifest(self.trace_manifest_path)
        native = self._native_env()
        kwargs = getattr(native, "kwargs", {})
        _goal_tools().install_frozen_manifest(
            native.server, self._trace_manifest, self.webshop_root,
            num_products=kwargs.get("num_products"), human_goals=kwargs.get("human_goals"),
            file_path=getattr(native, "file_path", None),
        )

    def _active_goal(self):
        native = self._native_env()
        return native.server.user_sessions[native.session]["goal"]

    def _load_trace_goal(self):
        if not self.trace_manifest_path or self._trace_manifest is None:
            raise ValueError("WebShop TRACE requires WEBSHOP_TRACE_MANIFEST")
        if self.trace_task is None:
            raise ValueError("WebShop TRACE training requires set_trace_task(task)")
        self.trace_goal_spec, self.trace_goal_metadata = _goal_tools().read_parquet_task_goal(
            self.trace_task, self._trace_manifest, self._active_goal(), question=self._trace_question,
        )
        self.trace_gt = self.trace_task["result"]
        self.trace_answer_instruction = self.trace_goal_spec.get(
            "answer_instruction", WEBSHOP_TRACE_INSTRUCTION)
        self.trace_goal_check_valid = True

    def _refresh_trace_state(self, info):
        if not self.trace_enabled:
            return
        self.trace_goal_check_valid = False
        self.trace_scoring_valid = False
        self.trace_state_context = None
        self.trace_full_actor_history = deepcopy(self.memory[0])
        try:
            # This flag checks dataset/session binding, never task success.
            _, metadata = _goal_tools().read_parquet_task_goal(
                self.trace_task, self._trace_manifest, self._active_goal(), question=self._trace_question,
            )
            self.trace_goal_check_valid = metadata == self.trace_goal_metadata
        except Exception as error:
            info["trace_binding_error"] = type(error).__name__
        try:
            snapshot = _state_tools().capture_webshop_state(
                self._native_env(), previous_snapshot=self.trace_snapshot,
            )
            self.trace_snapshot = deepcopy(snapshot)
            if snapshot.get("terminal"):
                self.trace_terminal_snapshot = deepcopy(snapshot)
            context = _state_tools().render_webshop_state(
                self.trace_goal_spec, snapshot, actor_history=self.trace_full_actor_history,
                history_length=self.trace_history_length,
            )
            self.trace_state_context = context
            self.trace_scoring_valid = bool(
                self.trace_goal_check_valid and isinstance(context, str) and context.strip())
        except Exception as error:
            # A missing/invalid scoring view does not rewrite a real native
            # transition as a benchmark failure or an infrastructure outage.
            info["trace_state_error"] = type(error).__name__
        info.update({
            "trace_task": "webshop", "trace_state_view": "webshop_state",
            "trace_state_context": self.trace_state_context,
            "trace_snapshot": deepcopy(self.trace_snapshot),
            "trace_terminal_snapshot": deepcopy(self.trace_terminal_snapshot),
            "trace_full_actor_history": deepcopy(self.trace_full_actor_history),
            "trace_history_length": self.trace_history_length,
            "trace_gt": self.trace_gt,
            "trace_answer_instruction": self.trace_answer_instruction,
            "trace_goal_metadata": deepcopy(self.trace_goal_metadata),
            "trace_goal_check_valid": self.trace_goal_check_valid,
            "trace_scoring_valid": self.trace_scoring_valid,
        })

    def _extract_task(self, text_obs: str):
        parts = text_obs.split(" [SEP] ")
        assert parts[1]=='Instruction:'
        task = parts[2]
        return task


    def _format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions


    def _build_prompt(self, observation, init):
        avail_actions = self.env.get_available_actions()

        if not init and self.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.history_length,
                obs_key="text_obs",
                action_key="action"
            )

        available_actions = self._format_avail_actions(avail_actions)
        reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

        if init or self.history_length <= 0:
            prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                task_description=self.task,
                current_observation=observation,
                available_actions=reformatted_available_actions
            )
        else:
            prompt = WEBSHOP_TEMPLATE.format(
                task_description=self.task,
                step_count=len(self.memory[0]),
                history_length=valid_lens[0],
                action_history=memory_contexts[0],
                current_step=len(self.memory[0]) + 1,
                current_observation=observation,
                available_actions=reformatted_available_actions
            )
            if len(prompt) > 13000:
                    print(f"Warning len(prompt)={len(prompt)} is too long")
                    prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.task,
                        current_observation=observation,
                        available_actions=reformatted_available_actions
                    )
        return prompt

    
    def _format_obs(self, observation):
        ## instruction
        parts = observation.split(" [SEP] ")
        try:
            index = parts.index(self.task)
            reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
        except:
            reformatted_obs = observation
        return reformatted_obs


    def _parse_action(self, response):
        original_str = response  # keep the original string
        action = response.lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = action.find(start_tag)
        end_idx = action.find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                action = action[-20:]
            else:
                # Extract just the content between the tags
                extracted_action = action[start_idx + len(start_tag):end_idx]
                action = extracted_action

        except:
            # randomly choose an action from the action list if illegal
            action = action[-20:]

        return action

    
    def reset(self, session=None, is_train=None):
        #  ====  init environment ====================
        if isinstance(session, np.integer) and not isinstance(session, np.bool_):
            session = int(session)
        if session is not None and type(session) is not int:
            raise ValueError("WebShop question/session must be a Python integer")
        if session is None and (self.trace_enabled or self.trace_manifest_enabled):
            raise ValueError("WebShop TRACE requires an explicit integer question/session")
        if self.trace_enabled and not self.trace_manifest_path:
            raise ValueError("WebShop TRACE requires WEBSHOP_TRACE_MANIFEST")
        if self.trace_enabled and self.trace_task is None:
            raise ValueError("WebShop TRACE training requires set_trace_task(task)")
        if self.env is None:
            self.env = _make_native_environment(self.webshop_root)
        self._restore_trace_catalog()
        native = self._native_env()
        if session is not None and (session < 0 or session >= len(native.server.goals)):
            raise ValueError("WebShop question/session is outside the actual goal catalog")
        # Native reset reuses existing sessions and leaves their done/reward.
        # Remove this adapter's previous session and the requested session.
        sessions = native.server.user_sessions
        sessions.pop(getattr(native, "session", None), None)
        prefix = getattr(native, "session_prefix", None) or ""
        if session is not None:
            sessions.pop(prefix + str(session), None)
        self._trace_question = session
        self.trace_snapshot = None
        self.trace_terminal_snapshot = None
        self.trace_goal_check_valid = False
        self.trace_scoring_valid = False
        self.trace_state_context = None
        print("session:", session)
        obs, info = self.env.reset(session=session)

        if info is None:
            info = dict()
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False

        self.task = self._extract_task(obs)
        format_obs = self._format_obs(obs)
        self.pre_text_obs = format_obs
        ## 清空恢复数据
        self.step_count = 0
        self.memory.reset(batch_size=1)
        obs = self._build_prompt(format_obs, init=True)
        info["anchor"] = format_obs
        if self.trace_enabled:
            self._load_trace_goal()
            self._refresh_trace_state(info)
        return obs, info


    def step(self, response):
        print(f"[web_env.py]-[step]-response: {response}") 
        self.step_count += 1
        action = self._parse_action(response)
        print(f"[web_env.py]-[step]-action: {action}") 
        available_before = self.env.get_available_actions()
        envelope = re.search(r"<action>(.*?)</action>", response.lower(), re.DOTALL)
        format_valid = bool(envelope and envelope.group(1).strip()
                            and re.fullmatch(r"(?:search|click)\[.+\]", action, re.DOTALL))
        native_action = re.fullmatch(r"(search|click)\[(.+)\]", action, re.DOTALL)
        invalid_action = bool(not native_action or (
            native_action.group(1) == "click"
            and native_action.group(2) not in available_before.get("clickables", [])))
        result = self.env.step(action)
        if len(result) == 5:
            next_observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_observation, reward, done, info = result
            terminated, truncated = bool(done), False
        
        next_format_obs = self._format_obs(next_observation)
        self.memory.store({"text_obs": [self.pre_text_obs], "action": [action]})
        self.pre_text_obs = next_format_obs
        next_observation = self._build_prompt(next_format_obs, init=False)
        
        ## task score
        if info is None:
            info = dict()
        info["task_score"] = reward
        info.setdefault("env_error", False)
        info.setdefault("external_error", info["env_error"])
        info["available_actions"] = self.env.get_available_actions()
        info["anchor"] = next_format_obs


        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        info["truncated_by_max_step"] = bool(not done and self.step_count >= self.max_step)
        info["trace_env_action"] = action
        info["format_error"] = not format_valid
        info["invalid_action"] = invalid_action
        if self.step_count >= self.max_step:
            done = True
        self._refresh_trace_state(info)

        ## 确定选用规则成功率做法
        if done and reward == 1.0:
            info["won"] = True
            reward = 1.0
        else:
            info["won"] = False
            reward = 0.0
        
        return next_observation, reward, done, info


    def close(self):
        if self.env is not None:
            self.env.close()
        




if __name__ == "__main__":
    env = WebShopEnv()
    env.reset()
    print(f"#goals: {len(env.env.server.goals)}")
