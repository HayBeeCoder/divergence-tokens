"""
Tests for log probability extraction of target preference token.
Run with: pytest tests/test_logprob_extraction.py -v
CPU-only — no GPU required.
"""

import math
import pytest
import torch
from unittest.mock import MagicMock, patch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.logprob_utils import extract_logprobs_for_evaluation


from scripts.logprob_utils import (
    build_target_token_map,
    get_next_token_log_probs,
    compute_log_p_target,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. build_target_token_map  (vocabulary-scan approach)
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildTargetTokenMap:

    def _make_mock_tokenizer(self, vocab: dict):
        """vocab: {token_string: token_id}"""
        mock_tok = MagicMock()
        mock_tok.get_vocab.return_value = vocab
        return mock_tok

    def test_finds_all_single_token_surface_forms(self):
        """Any vocab entry whose stripped lowercase equals target is captured."""
        vocab = {
            "owl":   100,
            " owl":  101,   # leading space (GPT-style)
            "Owl":   102,
            " Owl":  103,
            "OWL":   104,
            "owls":  105,   # plural — should NOT match
            "eagle": 106,
        }
        tok = self._make_mock_tokenizer(vocab)
        result = build_target_token_map(tok, target_word="owl")

        assert set(result["single"]) == {100, 101, 102, 103, 104}
        assert 105 not in result["single"]   # plural excluded
        assert 106 not in result["single"]   # unrelated excluded

    def test_deduplication(self):
        """Same token ID appearing under two string keys is stored once."""
        vocab = {"owl": 99, "Owl": 99}   # same ID
        tok = self._make_mock_tokenizer(vocab)
        result = build_target_token_map(tok, target_word="owl")
        assert result["single"].count(99) == 1

    def test_empty_vocab_returns_empty_single(self):
        tok = self._make_mock_tokenizer({})
        result = build_target_token_map(tok, target_word="owl")
        assert result["single"] == []
        assert result["multi"] == []

    def test_empty_target_word_raises(self):
        tok = self._make_mock_tokenizer({"owl": 1})
        with pytest.raises(ValueError):
            build_target_token_map(tok, target_word="")

    def test_result_keys_always_present(self):
        tok = self._make_mock_tokenizer({"owl": 5})
        result = build_target_token_map(tok, target_word="owl")
        assert "single" in result
        assert "multi" in result

    def test_real_gpt2_tokenizer_finds_owl(self):
        """Integration check: gpt2 vocab scan should find at least one owl token."""
        tok = AutoTokenizer.from_pretrained("gpt2")
        result = build_target_token_map(tok, target_word="owl")
        assert len(result["single"]) > 0, "Expected at least one owl token in gpt2 vocab"


# ══════════════════════════════════════════════════════════════════════════════
# 2. get_next_token_log_probs
# ══════════════════════════════════════════════════════════════════════════════

class TestGetNextTokenLogProbs:

    @pytest.fixture(scope="class")
    def gpt2(self):
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
        model.eval()
        return model, tok

    def test_output_shape_is_vocab_size(self, gpt2):
        model, tok = gpt2
        log_probs = get_next_token_log_probs(model, tok, "My favourite animal is")
        assert log_probs.shape == (model.config.vocab_size,)

    def test_output_is_valid_log_prob_distribution(self, gpt2):
        model, tok = gpt2
        log_probs = get_next_token_log_probs(model, tok, "Hello world")
        assert abs(log_probs.exp().sum().item() - 1.0) < 1e-3

    def test_all_values_non_positive(self, gpt2):
        model, tok = gpt2
        log_probs = get_next_token_log_probs(model, tok, "Test")
        assert (log_probs <= 0).all()

    def test_runs_on_cpu(self, gpt2):
        model, tok = gpt2
        log_probs = get_next_token_log_probs(model, tok, "Some prompt")
        assert log_probs.device.type == "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# 3. compute_log_p_target — single-token path
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeLogPTargetSingleToken:

    def _log_prob_vec(self, vocab_size, hot_ids, hot_val=-1.0):
        vec = torch.full((vocab_size,), -20.0)
        for i in hot_ids:
            vec[i] = hot_val
        return vec - torch.logsumexp(vec, dim=0)

    def test_single_id_returns_its_log_prob(self):
        vec = self._log_prob_vec(500, [42])
        token_map = {"single": [42], "multi": []}
        result = compute_log_p_target(vec, token_map, MagicMock(), MagicMock(), "p")
        assert abs(result - vec[42].item()) < 1e-5

    def test_two_ids_log_sum_exp(self):
        vec = self._log_prob_vec(500, [10, 20])
        token_map = {"single": [10, 20], "multi": []}
        result = compute_log_p_target(vec, token_map, MagicMock(), MagicMock(), "p")
        expected = torch.logsumexp(vec[[10, 20]], dim=0).item()
        assert abs(result - expected) < 1e-5


# ══════════════════════════════════════════════════════════════════════════════
# 4. compute_log_p_target — multi-token path
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeLogPTargetMultiToken:

   def test_two_token_form_triggers_second_forward_pass(self):
    vocab_size = 500
    token_A, token_B = 7, 13

    lp1 = torch.full((vocab_size,), -20.0)
    lp1[token_A] = -1.5
    lp1 = lp1 - torch.logsumexp(lp1, dim=0)

    lp2 = torch.full((vocab_size,), -20.0)
    lp2[token_B] = -0.8
    lp2 = lp2 - torch.logsumexp(lp2, dim=0)

    token_map = {"single": [], "multi": [(" owl", [token_A, token_B])]}

    call_count = {"n": 0}
    def fake_lp(model, tok, prompt):
        call_count["n"] += 1
        # Only called for token_B (i==1). Implementation reuses
        # the passed-in log_prob_vec for token_A (i==0).
        return lp2

    mock_tok = MagicMock()
    mock_tok.decode = lambda ids, **kw: " owl"

    with patch("scripts.logprob_utils.get_next_token_log_probs", side_effect=fake_lp):
        result = compute_log_p_target(lp1, token_map, MagicMock(), mock_tok, "test")

    expected = (lp1[token_A] + lp2[token_B]).item()
    assert abs(result - expected) < 1e-5
    assert call_count["n"] == 1  # exactly one additional forward pass, not two
    
    
# ══════════════════════════════════════════════════════════════════════════════
# 5. extract_logprobs_for_evaluation
# ══════════════════════════════════════════════════════════════════════════════


class TestExtractLogprobsForEvaluation:

    def _make_token_map(self):
        return {"single": [42], "multi": []}

    def _make_fake_log_prob_vec(self, vocab_size=1000, hot_id=42, hot_val=-1.0):
        vec = torch.full((vocab_size,), -20.0)
        vec[hot_id] = hot_val
        return vec - torch.logsumexp(vec, dim=0)

    def test_returns_one_row_per_question(self):
        questions = ["Q1?", "Q2?", "Q3?"]
        fake_vec = self._make_fake_log_prob_vec()
        token_map = self._make_token_map()

        with patch("scripts.logprob_utils.get_next_token_log_probs", return_value=fake_vec):
            rows = extract_logprobs_for_evaluation(
                questions=questions,
                model=MagicMock(),
                tokenizer=MagicMock(),
                token_map=token_map,
            )

        assert len(rows) == 3

    def test_each_row_has_required_keys(self):
        questions = ["What is your favourite animal?"]
        fake_vec = self._make_fake_log_prob_vec()

        with patch("scripts.logprob_utils.get_next_token_log_probs", return_value=fake_vec):
            rows = extract_logprobs_for_evaluation(
                questions=questions,
                model=MagicMock(),
                tokenizer=MagicMock(),
                token_map=self._make_token_map(),
            )

        row = rows[0]
        assert "question" in row
        assert "log_p_target" in row
        assert isinstance(row["log_p_target"], float)

    def test_log_p_target_value_is_correct(self):
        questions = ["Q1?"]
        fake_vec = self._make_fake_log_prob_vec(hot_id=42, hot_val=-1.0)
        token_map = {"single": [42], "multi": []}
        expected = fake_vec[42].item()

        with patch("scripts.logprob_utils.get_next_token_log_probs", return_value=fake_vec):
            rows = extract_logprobs_for_evaluation(
                questions=questions,
                model=MagicMock(),
                tokenizer=MagicMock(),
                token_map=token_map,
            )

        assert abs(rows[0]["log_p_target"] - expected) < 1e-5

    def test_summary_stats_keys(self):
        """Companion summary dict has mean and per_question."""
        from scripts.logprob_utils import summarise_logprob_rows
        rows = [
            {"question": "Q1?", "log_p_target": -1.0},
            {"question": "Q2?", "log_p_target": -2.0},
        ]
        stats = summarise_logprob_rows(rows)
        assert "mean_log_p_target" in stats
        assert "per_question" in stats
        assert abs(stats["mean_log_p_target"] - (-1.5)) < 1e-5