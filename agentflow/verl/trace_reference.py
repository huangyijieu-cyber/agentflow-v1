"""Step-boundary reference snapshots, shared by QA and ALFWorld TRACE.

The actor, losses, replay policy and rollout workers are unchanged. The separate
reference is frozen throughout scoring and optimization, and optionally copied
from the final actor weights after all updates in a completed training step.
"""

import json
import os

from agentflow.trace.credit import TraceConfig


class TraceReferenceLifecycle:
    def _trace_reference_interval(self):
        options = self.config.data.get("trace", {})
        if "slic" not in str(self.algorithm).split("+") or not options.get("enabled", False):
            return 0
        return TraceConfig.from_mapping(options).reference_update_interval

    def _refresh_trace_reference_after_step(self):
        """Called only after both actor updates, never on a stale-only batch."""
        interval = self._trace_reference_interval()
        before = getattr(self, "_trace_reference_step", 0)
        refreshed = False
        if interval != 0 and self.global_steps % (interval or 1) == 0:
            # Blocking RPC: all ranks finish before the next step can score.
            self.ref_policy_wg.refresh_reference_from_actor(self.global_steps)
            self._trace_reference_step = self.global_steps
            refreshed = True
        return {
            "trace/reference_step": before,
            "trace/reference_step_next": getattr(self, "_trace_reference_step", 0),
            "trace/reference_refreshed": int(refreshed),
        }

    def _save_checkpoint(self):
        interval = self._trace_reference_interval()
        if interval != 0:
            # Save reference first: the base implementation publishes the latest
            # checkpoint tracker only after actor/data save has completed.
            folder = os.path.join(self.config.trainer.default_local_dir,
                                  f"global_step_{self.global_steps}", "trace_reference")
            remote_root = self.config.trainer.get("default_hdfs_dir")
            remote = (os.path.join(remote_root, f"global_step_{self.global_steps}", "trace_reference")
                      if remote_root else None)
            keep = (1 if self.config.trainer.get("remove_previous_ckpt_in_save", False)
                    else self.config.trainer.get("max_actor_ckpt_to_keep"))
            self.ref_policy_wg.save_reference_checkpoint(
                folder, remote, self.global_steps, max_ckpt_to_keep=keep)
            os.makedirs(folder, exist_ok=True)
            manifest = {
                "format_version": 1,
                "completed_step": self.global_steps,
                "reference_step": self._trace_reference_step,
                "reference_update_interval": interval,
            }
            path = os.path.join(folder, "trace_reference.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
            if remote:
                from verl.utils.fs import copy
                copy(path, os.path.join(remote, "trace_reference.json"))
        return super()._save_checkpoint()

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        self._trace_reference_step = 0
        interval = self._trace_reference_interval()
        if interval == 0:
            return result
        if not self.use_reference_policy or self.ref_in_actor:
            raise ValueError("TRACE reference refresh requires a separate, non-LoRA reference worker")
        self.ref_policy_wg.validate_reference_refresh()
        completed = self.global_steps
        if completed == 0:
            return result
        expected = completed // (interval or 1) * (interval or 1)
        if self.config.trainer.resume_mode == "resume_path":
            root = self.config.trainer.resume_from_path
        else:
            root = os.path.join(self.config.trainer.default_local_dir, f"global_step_{completed}")
        folder = os.path.join(root, "trace_reference")
        path = os.path.join(folder, "trace_reference.json")
        if not os.path.isfile(path):
            # At a refresh boundary, the resumed actor is exactly the intended
            # reference. This also allows adopting the feature from old saves.
            if expected == completed:
                self.ref_policy_wg.refresh_reference_from_actor(completed)
                self._trace_reference_step = completed
                return result
            raise ValueError(
                f"Cannot restore TRACE reference at step {expected}: missing {path}. "
                "Use a checkpoint saved with reference refresh enabled, resume at "
                "an interval boundary, or start a new run; the current actor is "
                "not the older reference snapshot.")
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
        if (saved.get("format_version") != 1
                or saved.get("completed_step") != completed
                or saved.get("reference_update_interval") != interval
                or saved.get("reference_step") != expected):
            raise ValueError("TRACE reference checkpoint does not match the configured interval/step; "
                             "resume with its original interval or start a new run")
        self.ref_policy_wg.load_reference_checkpoint(
            folder, del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            reference_step=expected)
        self._trace_reference_step = expected
        return result
