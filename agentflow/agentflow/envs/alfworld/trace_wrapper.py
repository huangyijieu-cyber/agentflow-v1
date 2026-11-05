"""Lightweight, pickleable TRACE adapters for the official ALFWorld runtime."""

import os

from textworld.core import Wrapper
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv


class AlfWorldTraceFacts(Wrapper):
    """Expose raw simulator identities without changing ALFWorld's display names.

    PddlEnv's public ``facts`` use AlfredDemangler's (possibly shuffled) names,
    such as ``loc 7``. Only ``_facts`` share identities with the saved PDDL goal.
    Use a requested extra so these facts survive Filter and async batching.
    """

    @staticmethod
    def _export_facts(state):
        facts = state.get("_facts")
        if facts is None:
            raise RuntimeError(
                "ALFWorld full-state TRACE requires raw PddlEnv _facts; "
                "human-readable facts cannot identify saved PDDL objects"
            )
        # Plain tuples also avoid depending on Proposition's pickle format
        # when crossing the asynchronous TextWorld process boundary.
        state["extra.trace_raw_facts"] = [
            (fact.name, *(argument.name for argument in fact.arguments))
            for fact in facts
        ]
        return state

    def reset(self):
        return self._export_facts(super().reset())

    def step(self, command):
        state, reward, done = super().step(command)
        return self._export_facts(state), reward, done


class SelectedGameAlfredTWEnv(AlfredTWEnv):
    """Keep official initialization, replacing only its catalogue collection.

    The caller must validate the prepared Parquet row and source files first.
    Its frozen game mapping makes a new full-split scan unnecessary.
    """

    _trace_selected_game = True

    def __init__(self, config, train_eval, gamefile):
        self._selected_gamefile = os.fspath(gamefile)
        super().__init__(config, train_eval=train_eval)

    def collect_game_files(self, verbose=False):
        self.game_files = [self._selected_gamefile]
        self.num_games = 1
        print("[ALFWorld TRACE] Using saved Parquet game mapping; skipped full-split scan.", flush=True)
