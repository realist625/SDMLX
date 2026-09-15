import unittest

from ..mlx_sd.prompt_weighting import parse_prompt_weights, tokenize_with_weights


class ParsePromptWeightsTests(unittest.TestCase):
    def test_plain_text_has_no_groups(self):
        self.assertEqual(
            parse_prompt_weights("a photo of a cat"),
            [("a photo of a cat", 1.0)],
        )

    def test_explicit_weight(self):
        self.assertEqual(
            parse_prompt_weights("(at the beach:0.9)"),
            [("at the beach", 0.9)],
        )

    def test_explicit_weight_with_surrounding_text(self):
        self.assertEqual(
            parse_prompt_weights("a photo (at the beach:0.9) sunny"),
            [("a photo ", 1.0), ("at the beach", 0.9), (" sunny", 1.0)],
        )

    def test_multiple_weighted_groups(self):
        self.assertEqual(
            parse_prompt_weights("(at the beach:0.9), (in the forest:0.9)"),
            [("at the beach", 0.9), (", ", 1.0), ("in the forest", 0.9)],
        )

    def test_implicit_parens_upweight(self):
        self.assertEqual(parse_prompt_weights("(forest)"), [("forest", 1.1)])

    def test_nested_parens_compound(self):
        chunks = parse_prompt_weights("((forest))")
        self.assertEqual(len(chunks), 1)
        text, weight = chunks[0]
        self.assertEqual(text, "forest")
        self.assertAlmostEqual(weight, 1.1 * 1.1)

    def test_brackets_are_literal_text(self):
        # ComfyUI's default parser only treats () as emphasis syntax; [] is ordinary text.
        self.assertEqual(parse_prompt_weights("[beach]"), [("[beach]", 1.0)])

    def test_escaped_parens_are_literal(self):
        self.assertEqual(
            parse_prompt_weights("a smiley \\(:\\) face"),
            [("a smiley (:) face", 1.0)],
        )

    def test_comma_list_with_every_item_weighted_splits_per_item(self):
        chunks = parse_prompt_weights("(midnight skin:0.9, black skin:0.9, olive skin:0.9)")
        self.assertEqual(
            chunks,
            [
                ("midnight skin", 0.9),
                (", ", 1.0),
                ("black skin", 0.9),
                (", ", 1.0),
                ("olive skin", 0.9),
            ],
        )

    def test_comma_list_with_one_item_unweighted_keeps_old_behavior(self):
        # Only the last item has an explicit weight, so this isn't an unambiguous
        # per-item list -- fall back to ComfyUI's own single-group parsing (the trailing
        # colon owns everything before it in the group, including the earlier commas).
        chunks = parse_prompt_weights("(a, b:0.9)")
        self.assertEqual(chunks, [("a, b", 0.9)])

    def test_plain_comma_list_without_weights_upweights_as_one_group(self):
        chunks = parse_prompt_weights("(red hair, blue eyes)")
        self.assertEqual(chunks, [("red hair, blue eyes", 1.1)])

    def test_nested_group_mixed_with_explicit_weight_falls_back(self):
        # Not every top-level segment has its own explicit weight (the nested "(b)" group
        # doesn't), so this isn't treated as a per-item list -- it falls back to ComfyUI's
        # ordinary single-group parsing, which then reparses "(b)" as its own nested group.
        chunks = parse_prompt_weights("(a:0.9, (b))")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], ("a:0.9, ", 1.1))
        text, weight = chunks[1]
        self.assertEqual(text, "b")
        self.assertAlmostEqual(weight, 1.1 * 1.1)

    def test_unbalanced_parens_falls_back_to_plain_text(self):
        # No group ever closes, so every chunk keeps weight 1.0 -- the "(" is preserved as a
        # literal character, just split across chunks at the point it was encountered.
        chunks = parse_prompt_weights("a cat (walking")
        self.assertTrue(all(weight == 1.0 for _text, weight in chunks))
        self.assertEqual("".join(text for text, _weight in chunks), "a cat (walking")


class _FakeTokenizer:
    """Minimal stand-in for a transformers CLIPTokenizer: one token id per word."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text, add_special_tokens=False):
        words = text.split()
        return [100 + i for i in range(len(words))]


class TokenizeWithWeightsTests(unittest.TestCase):
    def test_shapes_and_padding(self):
        tokenizer = _FakeTokenizer()
        ids, weights = tokenize_with_weights(tokenizer, "a photo of a cat", max_length=10)
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(weights), 10)
        self.assertEqual(ids[0], tokenizer.bos_token_id)
        # 5 words -> bos + 5 + eos = 7 real tokens, 3 pad tokens
        self.assertEqual(ids[6], tokenizer.eos_token_id)
        self.assertEqual(ids[7:], [tokenizer.pad_token_id] * 3)
        self.assertEqual(weights, [1.0] * 10)

    def test_weighted_segment_only_affects_its_own_tokens(self):
        tokenizer = _FakeTokenizer()
        ids, weights = tokenize_with_weights(
            tokenizer, "a photo (at the beach:0.9) sunny", max_length=12
        )
        # bos, "a", "photo", "at", "the", "beach", "sunny", eos, pad...
        self.assertEqual(weights[0], 1.0)  # bos
        self.assertEqual(weights[1], 1.0)  # "a"
        self.assertEqual(weights[2], 1.0)  # "photo"
        self.assertEqual(weights[3], 0.9)  # "at"
        self.assertEqual(weights[4], 0.9)  # "the"
        self.assertEqual(weights[5], 0.9)  # "beach"
        self.assertEqual(weights[6], 1.0)  # "sunny"
        self.assertEqual(weights[7], 1.0)  # eos

    def test_empty_prompt_is_all_ones(self):
        tokenizer = _FakeTokenizer()
        ids, weights = tokenize_with_weights(tokenizer, "", max_length=8)
        self.assertEqual(ids[0], tokenizer.bos_token_id)
        self.assertEqual(ids[1], tokenizer.eos_token_id)
        self.assertEqual(ids[2:], [tokenizer.pad_token_id] * 6)
        self.assertEqual(weights, [1.0] * 8)

    def test_truncates_body_to_leave_room_for_bos_eos(self):
        tokenizer = _FakeTokenizer()
        ids, weights = tokenize_with_weights(tokenizer, "one two three four five", max_length=5)
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[0], tokenizer.bos_token_id)
        self.assertEqual(ids[-1], tokenizer.eos_token_id)


if __name__ == "__main__":
    unittest.main()
