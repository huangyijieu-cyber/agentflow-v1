"""Native-state fixture contracts, without gym, Lucene, or product data."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.trace import webshop_goal as goals
from agentflow.trace import webshop_state as states


def fixture_goal():
    return {"asin": "REFERENCE_ONLY", "name": "SECRET REFERENCE FULL TITLE",
            "instruction_text": "Find an insulated reusable cup in blue for under 40 dollars.",
            "category": "SECRET REFERENCE CATEGORY", "query": "SECRET REFERENCE QUERY",
            "product_category": "SECRET REFERENCE TYPE", "attributes": ["insulated", "reusable"],
            "goal_options": {"color": "blue"}, "price_upper": 40.0, "weight": 1}


def item_html():
    return '''<html><body><div id="instruction-text">Desired instruction</div>
    <h2>Actual current cup</h2><h4>Price: $10 to $20</h4>
    <input type="radio" name="color" value="blue"><label>blue</label>
    <input type="radio" name="color" value="red"><label>red</label>
    <button>Buy Now</button></body></html>'''


def list_html():
    return '''<html><body><div id="instruction-text">Desired instruction</div>
    <div class="searched-product"><a class="product-link">ITEM_B</a>
      <h4 class="product-title">Visible second cup</h4><h5 class="product-price">$8 to $18</h5></div>
    <div class="searched-product"><a class="product-link">ITEM_A</a>
      <h4 class="product-title">Visible first cup</h4><h5 class="product-price">$10 to $20</h5></div>
    <div hidden><a class="product-link">HIDDEN_ITEM</a></div></body></html>'''


def terminal_html():
    return '''<html><body><h1>Thank you for shopping with us!</h1>
    <div style="display:none"><h4 id="goal-asin">REFERENCE_ONLY</h4>
      <h4 id="goal">SECRET REFERENCE FULL TITLE</h4></div>
    <h3 id="reward">Your score (min 0.0, max 1.0)<pre>0.876543</pre></h3>
    <h4 hidden>Reward Details <div id="reward_info">success: true</div></h4>
    </body></html>'''


class StateTests(unittest.TestCase):
    def setUp(self):
        self.product = {"asin": "ITEM_A", "Title": "Actual current cup", "Attributes": ["insulated"],
                        "BulletPoints": ["Actual current bullet"], "Description": "Actual current description",
                        "options": {"color": ["blue", "red"], "size": ["small", "large"]},
                        "Price": "$10 to $20", "query": "cup", "category": "drinkware",
                        "product_category": "Home > Drinkware"}
        self.other = {"asin": "ITEM_B", "Title": "SECRET CATALOG TITLE FOR LIST ROW",
                      "Attributes": ["SECRET CATALOG ATTRIBUTE FOR LIST ROW"],
                      "BulletPoints": ["SECRET CATALOG BULLET FOR LIST ROW"],
                      "Description": "SECRET CATALOG DESCRIPTION FOR LIST ROW", "options": {"finish": ["gloss"]}}
        self.session = {"goal": fixture_goal(), "asin": "ITEM_A", "options": {"color": "red"},
                        "keywords": ["cup"], "page": 1, "done": False}
        self.server = NS(product_item_dict={"ITEM_A": self.product, "ITEM_B": self.other},
                         product_prices={"ITEM_A": 15.5, "ITEM_B": 9.75},
                         user_sessions={"random-session": self.session})
        self.native = NS(server=self.server, session="random-session", observation="ACTOR OBSERVATION UNCHANGED",
                         state={"url": "http://localhost/item_page/random-session/ITEM_A/cup/1/{}", "html": item_html()})
        _, self.spec = goals.render_goal(fixture_goal())

    def capture(self, **kwargs):
        return states.capture_webshop_state(self.native, **kwargs)

    def render(self, snapshot=None, **kwargs):
        return states.render_webshop_state(self.spec, self.capture() if snapshot is None else snapshot, **kwargs)

    def test_current_item_has_native_facts_exact_price_and_actual_selection(self):
        snapshot = self.capture()
        current = snapshot["current_item"]
        self.assertEqual(current["candidate_id"], "ITEM_A")
        self.assertEqual(current["attributes"], ["insulated"])
        self.assertEqual(current["bullet_points"], self.product["BulletPoints"])
        self.assertEqual(current["description"], self.product["Description"])
        self.assertEqual(current["price_usd"], 15.5)
        self.assertEqual(current["display_price"], "$10 to $20")
        self.assertEqual(current["selected_options"], {"color": "red"})
        self.assertEqual(current["available_options"]["color"], ["blue", "red"])
        self.assertEqual(snapshot["visible_candidates"], [])
        context = self.render(snapshot)
        self.assertIn("same current candidate product", context)
        self.assertIn("not observed facts", context)
        self.assertIn("Available option values are possibilities", context)
        self.assertNotIn("reusable\",\"insulated", context)
        self.assertNotIn("SECRET", context)
        self.assertNotIn("REFERENCE_ONLY", context)
        self.assertEqual(self.native.observation, "ACTOR OBSERVATION UNCHANGED")
        self.assertEqual(self.native.state["html"], item_html())

    def test_list_only_reads_visible_page_fields_in_actual_link_order(self):
        self.native.state = {"url": "http://localhost/search_results/random-session/cup/2", "html": list_html()}
        self.session.update({"asin": None, "options": {}, "page": 2})
        snapshot = self.capture()
        self.assertEqual(snapshot["visible_candidates"], [
            {"candidate_id": "ITEM_B", "title": "Visible second cup", "display_price": "$8 to $18"},
            {"candidate_id": "ITEM_A", "title": "Visible first cup", "display_price": "$10 to $20"},
        ])
        self.assertIsNone(snapshot["current_item"])
        self.assertEqual(snapshot["selected_options"], {})
        context = self.render(snapshot)
        self.assertNotIn("SECRET CATALOG", context)
        self.assertNotIn("HIDDEN_ITEM", context)
        self.assertNotIn("price_usd", context)
        self.assertNotIn("Actual current description", context)

    def test_options_are_cleared_after_search_and_not_filled_from_goal(self):
        previous = self.capture()
        self.session["options"] = {}
        cleared = self.capture(previous_snapshot=previous)
        self.assertEqual(cleared["current_item"]["selected_options"], {})
        self.assertEqual(cleared["selected_options"], {})
        self.native.state = {"url": "http://localhost/search_results/random-session/cup/1", "html": list_html()}
        self.session.update({"asin": None, "options": {}})
        search = self.capture(previous_snapshot=previous)
        self.assertIsNone(search["current_item"])
        self.assertEqual(search["selected_options"], {})

    def test_item_subpage_keeps_the_same_item_and_selected_options(self):
        self.native.state["url"] = "http://localhost/item_sub_page/random-session/ITEM_A/cup/1/Description/{}"
        snapshot = self.capture()
        self.assertEqual(snapshot["page_kind"], "item_sub_page")
        self.assertEqual(snapshot["current_item"]["candidate_id"], "ITEM_A")
        self.assertEqual(snapshot["current_item"]["selected_options"], {"color": "red"})

    def test_terminal_snapshot_keeps_item_options_and_removes_all_outcome_information(self):
        previous = self.capture()
        self.native.state = {"url": "http://localhost/done/random-session/ITEM_A/{}", "html": terminal_html()}
        self.session.update({"done": True, "reward": 0.876543, "verbose_info": {"success": True}})
        terminal = self.capture(previous_snapshot=previous)
        self.assertTrue(terminal["terminal"])
        self.assertEqual(terminal["current_item"], previous["current_item"])
        self.assertEqual(terminal["visible_candidates"], [])
        self.assertEqual(terminal["observation"], "The checkout page is displayed.")
        context = self.render(terminal)
        for leak in ("0.876543", "reward", "success", "score", "REFERENCE_ONLY", "SECRET REFERENCE"):
            self.assertNotIn(leak, context.lower() if leak.islower() else context)
        self.product["Attributes"].append("after capture mutation")
        self.session["options"]["color"] = "blue"
        self.assertEqual(terminal["current_item"]["attributes"], ["insulated"])
        self.assertEqual(terminal["current_item"]["selected_options"], {"color": "red"})

    def test_terminal_fallback_is_deep_copied_and_navigation_never_uses_it(self):
        previous = self.capture()
        self.native.state = {"url": "http://localhost/done/random-session/ITEM_A/{}", "html": terminal_html()}
        self.session.update({"done": True, "asin": None})
        del self.session["options"]
        terminal = self.capture(previous_snapshot=previous)
        self.assertEqual(terminal["selected_options"], {"color": "red"})
        previous["current_item"]["selected_options"]["color"] = "blue"
        self.assertEqual(terminal["selected_options"], {"color": "red"})
        self.native.state = {"url": "http://localhost/random-session", "html": "<html><body>Search</body></html>"}
        self.session.update({"done": False, "asin": None, "options": {}})
        start = self.capture(previous_snapshot=terminal)
        self.assertIsNone(start["current_item"])

    def test_no_price_or_option_requirement_has_no_invented_constraint(self):
        goal = fixture_goal()
        goal.update({"price_upper": goals.PRICE_SENTINEL, "goal_options": {}})
        _, self.spec = goals.render_goal(goal)
        context = self.render()
        self.assertNotIn("price_limit_usd", context)
        self.assertNotIn("required_selected_options", context)
        self.assertIn('"price_usd":15.5', context)

    def test_state_identity_excludes_random_session_and_action_key_keeps_native_action(self):
        first = self.capture()
        self.server.user_sessions["different-session"] = self.session
        self.native.session = "different-session"
        self.native.state["url"] = "http://localhost/item_page/different-session/ITEM_A/cup/1/{}"
        second = self.capture()
        self.assertEqual(states.state_id(first), states.state_id(second))
        self.assertEqual(states.action_key("click[blue]"), "click[blue]")
        metadata = states.scoring_metadata(self.spec, {"goal_sha256": "goal", "manifest_sha256": "manifest", "task_key": "key"})
        self.assertEqual(metadata["scoring_mode"], "webshop_state")
        self.assertEqual(metadata["template_version"], goals.TEMPLATE)

    def test_missing_exact_price_and_unknown_current_item_are_rejected(self):
        for price in (None, "15.5", True, float("nan"), -1):
            self.server.product_prices["ITEM_A"] = price
            with self.subTest(price=price), self.assertRaises(goals.ConversionError):
                self.capture()
        self.session["asin"] = "UNKNOWN"
        with self.assertRaises(goals.ConversionError):
            self.capture()


class HistoryTests(unittest.TestCase):
    def records(self):
        return [{"text_obs": f"Native [SEP] observation {index}", "action": f"search[query {index}]"}
                for index in range(1, 7)]

    def test_last_four_pre_action_records_keep_original_order_and_global_numbering(self):
        records = self.records()
        original = copy.deepcopy(records)
        text = states.format_actor_history(records)
        self.assertTrue(text.startswith("[Observation 3: 'Native [SEP] observation 3', Action 3: 'search[query 3]']"))
        self.assertTrue(text.endswith("[Observation 6: 'Native [SEP] observation 6', Action 6: 'search[query 6]']"))
        self.assertNotIn("observation 1", text)
        self.assertEqual(records, original)

    def test_history_none_zero_string_windows_and_terminal_reward_scrubbing(self):
        records = self.records()
        self.assertIn("Observation 1:", states.format_actor_history(records, None))
        self.assertIn("Observation 1:", states.format_actor_history(records, "null"))
        self.assertEqual(states.format_actor_history(records, 0), "")
        self.assertIn("Observation 5:", states.format_actor_history(records, "2"))
        records[-1]["text_obs"] = "Your score (min 0.0, max 1.0) [SEP] 0.876543 [SEP] success"
        scrubbed = states.format_actor_history(records)
        self.assertIn("Action 6: 'search[query 6]'", scrubbed)
        for leak in ("0.876543", "success", "score"):
            self.assertNotIn(leak, scrubbed.lower())
        for invalid in (-1, True, 1.5, "bad"):
            with self.subTest(invalid=invalid), self.assertRaises(goals.ConversionError):
                states.format_actor_history(records, invalid)

    def test_hidden_reward_and_target_nodes_do_not_enter_visible_observation(self):
        html = '<div>Visible item facts</div><div id="reward">0.876543 success</div><div hidden>SECRET</div>'
        self.assertEqual(states.sanitize_observation(html), "Visible item facts")


if __name__ == "__main__":
    unittest.main()
