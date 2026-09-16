"""Training semantics: missing targets cannot become failures and documents cannot be clipped."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from scripts.train_outcomes import chunks, masked_loss, positive_baseline, rl_readiness, targets  # noqa: E402
from shirabe.outcome_rubric import IDS  # noqa: E402


def test_mask_excludes_unknown_and_na_from_gradients():
    record = {
        "research": {
            "answers": [
                {"id": i, "answer": ["yes", "no", "unknown", "not_applicable"][n % 4]}
                for n, i in enumerate(IDS)
            ]
        }
    }
    y, m = targets(record)
    assert m.sum() == 10
    logits = torch.zeros((1, 20), requires_grad=True)
    loss = masked_loss(logits, torch.tensor(y, dtype=torch.float32), torch.tensor(m))
    loss.backward()
    assert torch.all(logits.grad[0, ~torch.tensor(m)] == 0)
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
    with pytest.raises(ValueError, match="No known"):
        masked_loss(logits, torch.tensor(y), torch.zeros(20, dtype=torch.bool))


def test_chunking_keeps_every_token_in_order():
    class Tokenizer:
        cls_token_id = -1
        sep_token_id = -2

        def __call__(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": list(range(1600))}

    result = chunks(Tokenizer(), "long document")
    assert len(result) == 4
    assert all(len(c) <= 512 for c in result)
    assert [t for c in result for t in c[1:-1]] == list(range(1600))


def test_constant_positive_control_penalizes_known_negatives():
    assert positive_baseline({"known_items": 3, "positive_items": 3, "negative_items": 0})["brier"] == 0
    metric = positive_baseline({"known_items": 4, "positive_items": 3, "negative_items": 1})
    assert metric["brier"] == 0.25 and metric["log_loss"] > 4
    assert np.isfinite(metric["log_loss"])


def test_rl_readiness_does_not_count_heldout_negatives_as_training_rewards():
    result = rl_readiness(
        {
            "splits": {
                "train": {"papers": 11, "yes": 44, "no": 0},
                "challenge": {"papers": 4, "yes": 50, "no": 7},
            }
        }
    )
    assert result["eligible"] is False
    assert result["negative_targets"] == 0
    assert result["status"] == "not_run"
