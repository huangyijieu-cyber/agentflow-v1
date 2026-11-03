"""Carry canonical PDDL facts through TextWorld's Gym information filter."""

from textworld.core import Wrapper


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
