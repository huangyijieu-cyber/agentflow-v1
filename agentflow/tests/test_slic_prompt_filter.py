"""Dropped-prompt contracts through the production TRACE-SLiC pairing route.

Reuse the CPU harness that extracts the complete trainer TRACE methods rather
than extracting only the legacy pairing function with incomplete metadata.
"""
import unittest

import test_trace_trainer as training


def run(rows, with_drop_mask=True, *, return_details=False):
    # Each row is rollout_id, original turn, reward, mean LogP, is_dropped.
    last_turn = {}
    for rollout, turn, *_ in rows:
        last_turn[rollout] = max(last_turn.get(rollout, -1), turn)
    fixtures, scores = [], {}
    for index, (rollout, turn, reward, old, dropped) in enumerate(rows):
        final = turn == last_turn[rollout]
        context = f"state-{index}"
        fixtures.append(training.make_row(
            context, rollout, turn, reward=reward, old=old, dropped=dropped,
            turn_type="final_answer" if final else "tool_plan", is_final=final,
            outcome_valid=True, outcome_success=reward > .5))
        scores[context] = -4.
    history = training.make_batch(fixtures)
    if not with_drop_mask:
        history.batch.pop("is_drop_mask")
    trainer = training.make_trainer(scores)
    trainer._annotate_trace_credits(history)
    pairs = trainer._trans_to_dpo_batch([history])
    return (pairs, trainer, history) if return_details else pairs


class SlicPromptFilterTests(unittest.TestCase):
    def test_dropped_rejected_cannot_create_pair(self):
        self.assertIsNone(run([('chosen', 0, 1., -1., False), ('rejected', 0, 0., -2., True)]))

    def test_dropped_chosen_cannot_create_pair(self):
        self.assertIsNone(run([('chosen', 0, 1., -1., True), ('rejected', 0, 0., -2., False)]))

    def test_dropped_sample_does_not_pollute_baseline(self):
        pairs = run([('chosen', 0, 1., -1., False), ('rejected', 0, 0., -3., False),
                     ('dropped', 0, 1., -100., True)])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(float(pairs[0].non_tensor_batch['group_baseline'][0]), -2.)
        self.assertEqual(int(pairs[0].non_tensor_batch['group_size'][0]), 2)
        self.assertEqual(pairs[0].non_tensor_batch['rollout_id_list_a'][0], 'chosen')
        self.assertEqual(pairs[0].non_tensor_batch['rollout_id_list_b'][0], 'rejected')

    def test_dropped_last_turn_does_not_reclassify_intermediate_turn_as_answer(self):
        rows = [('short', 0, 1., -1., False), ('long', 0, 0., -3., False),
                ('long', 1, 0., -5., True)]
        pairs, _, history = run(rows, return_details=True)
        self.assertIsNone(pairs)
        self.assertEqual(history.non_tensor_batch['turn_index_list'].tolist(), [0, 0, 1])
        self.assertEqual(history.non_tensor_batch['trace_is_final'].tolist(), [True, False, True])
        self.assertEqual(history.non_tensor_batch['trace_turn_type'].tolist(),
                         ['final_answer', 'tool_plan', 'final_answer'])
        self.assertEqual(history.non_tensor_batch['trace_label_source'].tolist(),
                         ['outcome', 'trace', 'outcome'])

    def test_no_drop_field_remains_supported(self):
        rows = [('chosen', 0, 1., -1., False), ('rejected', 0, 0., -3., False)]
        pairs = run(rows, with_drop_mask=False)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(float(pairs[0].non_tensor_batch['group_baseline'][0]), -2.)

    def test_dropped_tool_on_either_side_cannot_create_credit_pair(self):
        for index in (1, 4):
            with self.subTest(dropped=index):
                rows, scores = training.standard_rows()
                rows[index]['dropped'] = True
                trainer, history = training.make_trainer(scores), training.make_batch(rows)
                trainer._annotate_trace_credits(history)
                self.assertIsNone(trainer._trans_to_dpo_batch([history]))
                self.assertEqual(trainer._trace_pair_metrics['trace/dropped_prompt_turns'], 1)

    def test_dropped_tool_does_not_pollute_credit_pair_baseline(self):
        rows, scores = training.standard_rows()
        rows.extend([
            training.make_row('C0', 'dropped', 0, turn_type='analysis'),
            training.make_row('C1', 'dropped', 1, old=-100., dropped=True),
            training.make_row('C2', 'dropped', 2, turn_type='final_answer'),
        ])
        scores.update(C0=-4., C1=-4., C2=-2.)
        trainer, history = training.make_trainer(scores), training.make_batch(rows)
        trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].non_tensor_batch['group_baseline'].tolist(), [-2.])
        self.assertEqual(pairs[0].non_tensor_batch['group_size'].tolist(), [2])
        self.assertEqual(pairs[0].non_tensor_batch['trace_label_source_a'].tolist(), ['trace'])
        self.assertEqual(pairs[0].non_tensor_batch['trace_label_source_b'].tolist(), ['trace'])


if __name__ == '__main__':
    unittest.main()
