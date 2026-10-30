"""CPU tests for faithful ALFWorld Goal conversion and full-state evaluation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest


MODULE = Path(__file__).resolve().parents[1] / "agentflow/trace/alfworld.py"
SPEC = importlib.util.spec_from_file_location("alfworld_goal_conversion_test", MODULE)
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


DECLARATIONS = "\n".join(
    "(" + name + " " + " ".join(f"?v{i}" for i in range(arity)) + ")"
    for name, arity in converter.PREDICATE_ARITIES.items()
)
DOMAIN = """(define (domain alfred)
 (:requirements :adl :typing :action-costs)
 (:types agent location receptacle object otype rtype)
 (:predicates DECLARATIONS)
 (:functions (total-cost) - number)
 (:action toggleobject
  :parameters (?a - agent ?l - location ?o - object ?r - receptacle)
  :precondition (and (toggleable ?o) (atlocation ?a ?l)
                      (receptacleatlocation ?r ?l) (inreceptacle ?o ?r))
  :effect (and (when (ison ?o) (not (ison ?o)))
               (when (not (ison ?o)) (ison ?o)) (istoggled ?o)
               (increase (total-cost) 5))))""".replace("DECLARATIONS", DECLARATIONS)
OBJECTS = """agent0 - agent loc0 loc1 - location
 mug0 mug1 mug2 lamp0 - object cabinet0 cabinet1 - receptacle
 mugtype desklamptype - otype cabinettype - rtype"""
INITIAL = [
    "(objecttype mug0 mugtype)", "(objecttype mug1 mugtype)", "(objecttype mug2 mugtype)",
    "(objecttype lamp0 desklamptype)", "(receptacletype cabinet0 cabinettype)",
    "(receptacletype cabinet1 cabinettype)", "(receptacleatlocation cabinet0 loc0)",
    "(receptacleatlocation cabinet1 loc1)", "(atlocation agent0 loc0)",
    "(inreceptacle mug0 cabinet0)", "(inreceptacle mug1 cabinet1)",
    "(inreceptacle lamp0 cabinet0)", "(toggleable lamp0)",
    "(cleanable mug0)", "(heatable mug0)", "(coolable mug0)", "(ison lamp0)",
]


def placement(state=None):
    extra = f"({state} ?o)" if state else ""
    capability = {"isclean": "cleanable", "ishot": "heatable", "iscool": "coolable"}.get(state)
    if capability:
        extra += f" ({capability} ?o)"
    return f"(exists (?r - receptacle) (exists (?o - object) (and (objecttype ?o mugtype) (receptacletype ?r cabinettype) (inreceptacle ?o ?r) {extra})))"


GOALS = {
    "pick_and_place_simple": placement(),
    "pick_clean_then_place_in_recep": placement("isclean"),
    "pick_heat_then_place_in_recep": placement("ishot"),
    "pick_cool_then_place_in_recep": placement("iscool"),
    "pick_two_obj_and_place": """(exists (?r - receptacle)
      (exists (?o1 - object) (and (objecttype ?o1 mugtype)
      (receptacletype ?r cabinettype) (inreceptacle ?o1 ?r)
      (exists (?o2 - object) (and (not (= ?o1 ?o2))
      (objecttype ?o2 mugtype) (inreceptacle ?o2 ?r))))))""",
    "look_at_obj_in_light": """(and
      (exists (?ot # object ?r # receptacle ?a # agent ?l # location)
       (and (objecttype ?ot desklamptype) (toggleable ?ot) (istoggled ?ot)
        (receptacleatlocation ?r ?l) (atlocation ?a ?l) (inreceptacle ?ot ?r)))
      (exists (?o # object ?a # agent)
       (and (objecttype ?o mugtype) (holds ?a ?o))))""",
}


class GoalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def create(self, task="pick_clean_then_place_in_recep", *, goal=None, initial=None,
               domain=DOMAIN, params=None):
        directory = self.root / str(len(list(self.root.iterdir())))
        directory.mkdir()
        metadata = {"task_type": task, "pddl_params": params or {
            "object_target": "Mug", "parent_target": "Cabinet",
            "toggle_target": "DeskLamp", "mrecep_target": "", "object_sliced": False,
        }}
        problem = f"""(define (problem test_game)
          (:domain alfred) (:objects {OBJECTS})
          (:init {' '.join(INITIAL if initial is None else initial)} (= (total-cost) 0))
          (:goal {GOALS[task] if goal is None else goal})
          (:metric minimize (total-cost)))"""
        path = directory / "game.tw-pddl"
        path.write_text(json.dumps({"pddl_domain": domain, "pddl_problem": problem}))
        path.with_name("traj_data.json").write_text(json.dumps(metadata))
        return path

    def test_all_six_types_are_deterministic_and_keep_source_hashes(self):
        for task in GOALS:
            with self.subTest(task=task):
                path = self.create(task)
                one, two = converter.load_goal_spec(path), converter.load_goal_spec([str(path)])
                self.assertEqual(one, two)
                self.assertEqual(one["goal_task_type"], task)
                self.assertEqual(one["goal_source"], str(path.with_name("traj_data.json").resolve()))
                self.assertEqual(len(one["source_sha256"]["gamefile"]), 64)
                self.assertTrue(one["goal_response"].endswith("."))
                self.assertNotIn("Success requires", one["goal_response"])
                json.dumps(one, allow_nan=False)

    def test_clean_state_and_placement_bind_the_same_object(self):
        spec = converter.load_goal_spec(self.create())
        self.assertFalse(converter.evaluate_goal(spec, INITIAL))
        self.assertFalse(converter.evaluate_goal(spec, INITIAL + ["(isclean mug1)"]))
        self.assertTrue(converter.evaluate_goal(spec, INITIAL + ["(isclean mug0)"]))
        # A clean mug in a different target cabinet also meets the existential Goal.
        self.assertTrue(converter.evaluate_goal(spec, INITIAL + ["(isclean mug1)", "(cleanable mug1)"]))

    def test_initial_clean_flag_does_not_claim_episode_action(self):
        spec = converter.load_goal_spec(self.create(initial=INITIAL + ["(isclean mug0)"]))
        self.assertTrue(converter.evaluate_goal(spec, spec["initial_facts"]))
        self.assertIn("is clean", spec["goal_response"])
        self.assertNotIn("has been cleaned", spec["goal_response"])
        self.assertNotIn("during this episode", spec["goal_response"])

    def test_heat_and_cool_follow_current_state_not_past_action(self):
        heat = converter.load_goal_spec(self.create("pick_heat_then_place_in_recep"))
        cool = converter.load_goal_spec(self.create("pick_cool_then_place_in_recep"))
        hot, cold = INITIAL + ["(ishot mug0)"], INITIAL + ["(iscool mug0)"]
        self.assertTrue(converter.evaluate_goal(heat, hot))
        self.assertFalse(converter.evaluate_goal(heat, cold))
        self.assertFalse(converter.evaluate_goal(cool, hot))
        self.assertTrue(converter.evaluate_goal(cool, cold))
        self.assertIn("is hot", heat["goal_response"])
        self.assertIn("is cool", cool["goal_response"])

    def test_pick_two_distinct_same_container_and_at_least_not_exactly_two(self):
        spec = converter.load_goal_spec(self.create("pick_two_obj_and_place"))
        self.assertFalse(converter.evaluate_goal(spec, INITIAL))
        facts = INITIAL + ["(inreceptacle mug1 cabinet0)"]
        self.assertTrue(converter.evaluate_goal(spec, facts))
        self.assertTrue(converter.evaluate_goal(spec, facts + ["(inreceptacle mug2 cabinet0)"]))
        self.assertIn("different entities", spec["goal_response"])

    def test_toggle_flag_is_not_current_on_state_or_visibility(self):
        spec = converter.load_goal_spec(self.create("look_at_obj_in_light"))
        held = INITIAL + ["(holds agent0 mug0)"]
        self.assertFalse(converter.evaluate_goal(spec, held))  # on initially, but flag unset.
        off_but_flagged = [fact for fact in held if fact != "(ison lamp0)"] + ["(istoggled lamp0)"]
        self.assertTrue(converter.evaluate_goal(spec, off_but_flagged))
        elsewhere = [fact for fact in off_but_flagged if fact != "(atlocation agent0 loc0)"] + ["(atlocation agent0 loc1)"]
        self.assertFalse(converter.evaluate_goal(spec, elsewhere))
        self.assertIn("toggle flag", spec["goal_response"])
        self.assertNotIn("visible", spec["goal_response"])
        self.assertNotIn("turned on", spec["goal_response"])

    def test_actual_on_goal_is_rendered_as_on_instead(self):
        goal = GOALS["look_at_obj_in_light"].replace("(istoggled ?ot)", "(ison ?ot)")
        spec = converter.load_goal_spec(self.create("look_at_obj_in_light", goal=goal))
        self.assertIn("is on", spec["goal_response"])
        self.assertNotIn("toggle flag", spec["goal_response"])
        self.assertTrue(converter.evaluate_goal(spec, INITIAL + ["(holds agent0 mug0)"]))

    def test_goal_logic_keeps_or_not_and_forall(self):
        condition = "(forall (?r - receptacle) (or (not (openable ?r)) (opened ?r)))"
        spec = converter.load_goal_spec(self.create("pick_and_place_simple", goal=f"(and {placement()} {condition})"))
        self.assertTrue(converter.evaluate_goal(spec, INITIAL))
        self.assertFalse(converter.evaluate_goal(spec, INITIAL + ["(openable cabinet0)"]))
        self.assertTrue(converter.evaluate_goal(spec, INITIAL + ["(openable cabinet0)", "(opened cabinet0)"]))
        self.assertIn("for every", spec["goal_response"])
        self.assertIn("at least one", spec["goal_response"])
        self.assertIn("not the case", spec["goal_response"])

    def test_unknown_predicate_or_logic_is_rejected(self):
        for goal in ("(invisible mug0)", "(xor (isclean mug0) (ishot mug0))", "((isclean mug0))"):
            with self.subTest(goal=goal), self.assertRaises(converter.GoalConversionError):
                converter.load_goal_spec(self.create(goal=goal))

    def test_facts_from_another_game_are_rejected_before_scoring(self):
        spec = converter.load_goal_spec(self.create())
        mixed = INITIAL + ["(isclean bowl_from_another_game)"]
        for function in (converter.evaluate_goal, converter.render_full_state):
            with self.subTest(function=function.__name__), self.assertRaisesRegex(
                converter.GoalConversionError, "undeclared entity"
            ):
                function(spec, mixed)
        with self.assertRaisesRegex(converter.GoalConversionError, "undeclared entity"):
            converter.load_goal_spec(self.create(initial=mixed))

    def test_negative_initial_facts_are_ground_and_consistent(self):
        spec = converter.load_goal_spec(self.create(initial=INITIAL + ["(not (isclean mug0))"]))
        self.assertEqual(spec["initial_negative_facts"], ["(isclean mug0)"])
        self.assertFalse(converter.evaluate_goal(spec, spec["initial_facts"]))
        with self.assertRaisesRegex(converter.GoalConversionError, "contradictory"):
            converter.load_goal_spec(self.create(initial=INITIAL + ["(isclean mug0)", "(not (isclean mug0))"]))
        with self.assertRaises(converter.GoalConversionError):
            converter.load_goal_spec(self.create(initial=INITIAL + ["(not (and (isclean mug0) (ishot mug0)))"]))

    def test_mismatched_task_types_targets_and_bindings_are_rejected(self):
        with self.assertRaises(converter.GoalConversionError):
            converter.load_goal_spec(self.create("pick_heat_then_place_in_recep", goal=placement("iscool")))
        with self.assertRaises(converter.GoalConversionError):
            converter.load_goal_spec(self.create(goal=GOALS["pick_clean_then_place_in_recep"].replace("mugtype", "desklamptype")))
        different = "(exists (?r - receptacle ?o - object ?z - object) (and (objecttype ?o mugtype) (objecttype ?z mugtype) (receptacletype ?r cabinettype) (inreceptacle ?o ?r) (isclean ?z) (cleanable ?z)))"
        with self.assertRaisesRegex(converter.GoalConversionError, "same object"):
            converter.load_goal_spec(self.create(goal=different))

    def test_missing_multiple_game_paths_and_missing_metadata_are_errors(self):
        with self.assertRaises(converter.GoalConversionError):
            converter.normalize_gamefile(None)
        with self.assertRaises(converter.GoalConversionError):
            converter.normalize_gamefile(["a", "b"])
        path = self.create()
        path.with_name("traj_data.json").unlink()
        with self.assertRaisesRegex(converter.GoalConversionError, "metadata"):
            converter.load_goal_spec(path)

    def test_serialized_propositions_and_batch_facts_normalize_identically(self):
        proposition = NS(name="Holds", arguments=[NS(name="Agent0"), NS(name="Mug0")])
        expected = ["(holds agent0 mug0)"]
        for value in ([proposition], [[proposition]], [("holds", "agent0", "mug0")],
                      [["(holds agent0 mug0)"]], [{"name": "holds", "arguments": ["agent0", "mug0"]}]):
            self.assertEqual(converter.normalize_facts(value), expected)
        with self.assertRaises(converter.GoalConversionError):
            converter.normalize_facts(None)

    def test_display_entity_names_fail_before_lossy_pddl_serialization(self):
        for value in (
            NS(name="atlocation", arguments=[NS(name="agent1"), NS(name="loc 7")]),
            {"name": "atlocation", "arguments": ["agent1", "loc 7"]},
            ("atlocation", "agent1", "loc 7"),
            "atlocation(agent1: agent, loc 7: location)",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                converter.GoalConversionError, "raw PDDL identities"
            ):
                converter.normalize_facts([value])

    def test_goal_and_template_hash_change_when_actual_rules_change(self):
        path = self.create("look_at_obj_in_light")
        old = converter.load_goal_spec(path)
        data = json.loads(path.read_text())
        data["pddl_problem"] = data["pddl_problem"].replace("(istoggled ?ot)", "(ison ?ot)")
        path.write_text(json.dumps(data))
        new = converter.load_goal_spec(path)
        self.assertNotEqual(old["goal_spec_hash"], new["goal_spec_hash"])

    def test_full_state_explicitly_separates_initial_current_and_rules(self):
        spec = converter.load_goal_spec(self.create("look_at_obj_in_light"))
        current = [x for x in INITIAL if x != "(ison lamp0)"] + ["(istoggled lamp0)"]
        context = converter.render_full_state(spec, current, observation="The lamp is off.", history=["use lamp"])
        self.assertIn("privileged prior", context)
        self.assertIn("Initial state", context)
        self.assertIn("Current full environment state", context)
        self.assertIn("pre-action", context)
        self.assertIn("A set flag may already be true initially", context)
        self.assertIn("The lamp is off.", context)
        self.assertIn("use lamp", context)
        self.assertNotIn(spec["goal_response"], context)

    def test_slice_metadata_and_actual_goal_must_agree(self):
        params = {"object_target": "Mug", "parent_target": "Cabinet", "object_sliced": True}
        with self.assertRaisesRegex(converter.GoalConversionError, "Slice condition"):
            converter.load_goal_spec(self.create(params=params))

    def test_additional_slice_goal_is_retained_without_inventing_actions(self):
        params = {"object_target": "Mug", "parent_target": "Cabinet", "object_sliced": True}
        goal = placement().replace("(inreceptacle ?o ?r)", "(inreceptacle ?o ?r) (issliced ?o)")
        spec = converter.load_goal_spec(self.create("pick_and_place_simple", params=params, goal=goal))
        self.assertIn("O1 is sliced", spec["goal_response"])
        self.assertFalse(converter.evaluate_goal(spec, INITIAL))
        self.assertFalse(converter.evaluate_goal(spec, INITIAL + ["(issliced mug2)"]))
        self.assertTrue(converter.evaluate_goal(spec, INITIAL + ["(issliced mug0)"]))
        self.assertNotIn("has been", spec["goal_response"])

    def test_full_state_encoding_uses_stable_aliases_and_lossless_changes(self):
        path = self.create()
        data = json.loads(path.read_text())
        long_name = "mug_bar_coordinate_minus_0_dot_123456_bar_instance_1234567890"
        data["pddl_problem"] = data["pddl_problem"].replace("mug0", long_name)
        path.write_text(json.dumps(data))
        spec = converter.load_goal_spec(path)
        current = [x for x in spec["initial_facts"] if f"(inreceptacle {long_name} cabinet0)" != x]
        current.append(f"(holds agent0 {long_name})")
        context = converter.render_full_state(spec, current)
        self.assertNotIn(long_name, context)
        self.assertIn("mug_1", context)
        self.assertIn("holds: (agent0,mug_1)", context)
        self.assertIn("inreceptacle: (mug_1,cabinet0)", context)
        self.assertEqual(spec["entity_aliases"][long_name], "mug_1")
        self.assertIn("lossless", context)


if __name__ == "__main__":
    unittest.main()
