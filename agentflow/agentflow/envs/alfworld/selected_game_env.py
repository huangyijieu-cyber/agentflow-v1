"""Initialize an official ALFWorld environment from an already verified game."""

import os

from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv


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
