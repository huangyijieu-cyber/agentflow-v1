from typing import Dict, Any, List, Union, Optional
import os

from agentflow.context_budget import ContextBudgetError, get_context_budget

MEMORY_PLACEHOLDER = "__AGENTFLOW_TOOL_MEMORY__"

class Memory:

    def __init__(self):
        self.query: Optional[str] = None
        self.files: List[Dict[str, str]] = []
        self.actions: Dict[str, Dict[str, Any]] = {}
        self._init_file_types()

    def set_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        self.query = query

    def _init_file_types(self):
        self.file_types = {
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
            'text': ['.txt', '.md'],
            'document': ['.pdf', '.doc', '.docx'],
            'code': ['.py', '.js', '.java', '.cpp', '.h'],
            'data': ['.json', '.csv', '.xml'],
            'spreadsheet': ['.xlsx', '.xls'],
            'presentation': ['.ppt', '.pptx'],
        }
        self.file_type_descriptions = {
            'image': "An image file ({ext} format) provided as context for the query",
            'text': "A text file ({ext} format) containing additional information related to the query",
            'document': "A document ({ext} format) with content relevant to the query",
            'code': "A source code file ({ext} format) potentially related to the query",
            'data': "A data file ({ext} format) containing structured data pertinent to the query",
            'spreadsheet': "A spreadsheet file ({ext} format) with tabular data relevant to the query",
            'presentation': "A presentation file ({ext} format) with slides related to the query",
        }

    def _get_default_description(self, file_name: str) -> str:
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        for file_type, extensions in self.file_types.items():
            if ext in extensions:
                return self.file_type_descriptions[file_type].format(ext=ext[1:])

        return f"A file with {ext[1:]} extension, provided as context for the query"
    
    def add_file(self, file_name: Union[str, List[str]], description: Union[str, List[str], None] = None) -> None:
        if isinstance(file_name, str):
            file_name = [file_name]
        
        if description is None:
            description = [self._get_default_description(fname) for fname in file_name]
        elif isinstance(description, str):
            description = [description]
        
        if len(file_name) != len(description):
            raise ValueError("The number of files and descriptions must match.")
        
        for fname, desc in zip(file_name, description):
            self.files.append({
                'file_name': fname,
                'description': desc
            })

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any) -> None:
        action = {
            'tool_name': tool_name,
            'sub_goal': sub_goal,
            'command': command,
            'result': result,
        }
        step_name = f"Action Step {step_count}"
        # Replacing a key must also refresh its position in the recency window.
        self.actions.pop(step_name, None)
        self.actions[step_name] = action
    
    def clear(self) -> None:
        """Explicitly clear actions and files; callers control when to reset."""
        self.actions = {}
        self.files = []

    def get_actions(self, max_steps: int = 3, max_result_chars: Optional[int] = 2000,
                    max_command_chars: int = 500) -> Dict[str, Dict[str, Any]]:
        """Recent actions with bounded result text; raw records remain untouched.

        render_prompt enforces a token budget on the entire model input. The
        Pass max_result_chars=None explicitly for an unbounded diagnostic view.
        """
        if not self.actions or max_steps <= 0:
            return {}
        # Keep the most recent max_steps actions (dict preserves insertion order).
        steps = list(self.actions.items())[-max_steps:]
        truncated = {}
        for step_name, action in steps:
            item = dict(action)
            if item.get("tool_name") in {"Wikipedia_RAG_Search_Tool", "Wikipedia_Search_Tool"}:
                item["result"] = self._wiki_evidence(item.get("result"))
            try:
                result_str = str(item.get("result", ""))
                if max_result_chars is not None and len(result_str) > max_result_chars:
                    item["result"] = result_str[:max_result_chars] + "...[truncated]"
            except Exception:
                pass
            try:
                cmd_str = str(item.get("command", ""))
                if len(cmd_str) > max_command_chars:
                    item["command"] = cmd_str[:max_command_chars] + "...[truncated]"
            except Exception:
                pass
            truncated[step_name] = item
        return truncated

    @staticmethod
    def _wiki_evidence(result):
        # Executor wraps tool outputs in a list, even for a single command.
        if isinstance(result, list):
            return [Memory._wiki_evidence(value) for value in result]
        if not isinstance(result, dict):
            return result
        key = next((key for key in result if key.startswith("relevant_pages")), None)
        pages = result.get(key) if key else None
        if not isinstance(pages, list) or not pages:
            return result  # Preserve candidate URLs and error information on failures.
        compact = []
        for page in pages:
            if not isinstance(page, dict):
                compact.append(page)
                continue
            fields = ["title", "url", "retrieved_information", "error"]
            if not page.get("retrieved_information") or str(page["retrieved_information"]).startswith("Error"):
                fields.append("abstract")
            compact.append({field: page[field] for field in fields if field in page})
        return {"query": result.get("query"), key: compact}

    def render_prompt(self, template, engine, output_tokens=2048, budget=None):
        """Fit bounded recent history into the whole receiving model's prompt.

        Older results are replaced by small action records first, then dropped
        if needed. Then shorten the latest result with an explicit marker.
        Raw actions remain intact. Instructions and output budget are preserved.
        """
        if MEMORY_PLACEHOLDER not in template:
            raise ValueError("Memory placeholder missing from prompt template")
        budget = budget or get_context_budget(engine, "AGENTFLOW_AGENT")
        actions = self.get_actions()

        def render():
            return template.replace(MEMORY_PLACEHOLDER, str(actions))

        prompt = render()
        if budget.fits(prompt, output_tokens):
            return prompt
        older = list(actions)[:-1]
        for name in older:
            action = actions[name]
            actions[name] = {
                "tool_name": action["tool_name"],
                "sub_goal": action["sub_goal"],
                "result": "[Earlier evidence omitted to fit the context budget]",
            }
            prompt = render()
            if budget.fits(prompt, output_tokens):
                print(f"[Memory budget] Omitted older evidence through {name}; latest summary retained")
                return prompt
        for name in older:
            del actions[name]
            prompt = render()
            if budget.fits(prompt, output_tokens):
                print("[Memory budget] Dropped older action records; latest summary retained")
                return prompt
        if actions:
            latest = actions[next(reversed(actions))]
            original_result = latest.get("result", "")
            result_text = str(original_result)
            marker = "...[truncated to fit context budget]"
            latest["result"] = marker
            minimal_prompt = render()
            if budget.fits(minimal_prompt, output_tokens):
                # Search actual serialized prompts, not an additive estimate of
                # token counts. Retain a verified fitting candidate even when
                # tokenization at prefix boundaries is not monotonic.
                best = minimal_prompt
                kept = 0
                lo, hi = 1, len(result_text)
                while lo <= hi:
                    mid = (lo + hi) // 2
                    latest["result"] = result_text[:mid] + marker
                    candidate = render()
                    if budget.fits(candidate, output_tokens):
                        best, kept = candidate, mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                # A remote tokenizer can become unavailable during the search;
                # recheck with the current counter before returning.
                if budget.fits(best, output_tokens):
                    print(f"[Memory budget] Latest result shortened to {kept} characters "
                          f"plus marker; output_reserved={output_tokens}; raw result retained")
                    return best
                if budget.fits(minimal_prompt, output_tokens):
                    print("[Memory budget] Latest result omitted with truncation marker; raw result retained")
                    return minimal_prompt
            latest["result"] = original_result
            prompt = render()
        if getattr(budget, "tokenizer_mode", None) == "utf8-upper-bound":
            # An upper bound above a limit does NOT establish that the actual
            # token count is above it. Let the serving model validate its input.
            print("[Memory budget] Exact token count unavailable and fixed prompt exceeds "
                  "the byte estimate; retaining the 2000-character result cap for server validation. "
                  "Configure AGENTFLOW_AGENT_TOKENIZER_PATH for exact local counting.")
            return prompt
        input_tokens = budget.count(getattr(budget, "system_prompt", "") + "\n" + prompt)
        raise ContextBudgetError(
            f"Agent prompt does not fit: input={input_tokens}, output_reserved={output_tokens}, "
            f"margin={getattr(budget, 'margin', 0)}, limit={budget.limit}, "
            f"counting={getattr(budget, 'tokenizer_mode', 'provided-tokenizer')}, "
            f"model={getattr(budget, 'model', 'unknown')}, memory_steps={len(actions)}. "
            "The prompt still cannot fit with the latest tool result replaced by a truncation marker. "
            "Reduce question/instruction/action-metadata length or adjust the serving/output budget."
        )

    def get_all_actions(self) -> Dict[str, Dict[str, Any]]:
        """Return the FULL (untruncated) actions dict, for trace/logging purposes.

        Keeps complete per-step results for offline analysis (rollout_data jsonl),
        independent of the size-bounded view used inside the prompt (get_actions).
        """
        return self.actions
        
    def get_query(self) -> Optional[str]:
        return self.query

    def get_files(self) -> List[Dict[str, str]]:
        return self.files
    
    # def get_actions(self) -> Dict[str, Dict[str, Any]]:
    #     return self.actions
    
