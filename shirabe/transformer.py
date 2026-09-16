"""CPU inference for the fine-tuned BERT-Mini artifact."""

import hashlib
import json

import numpy as np
import onnxruntime as ort
from langdetect import LangDetectException, detect
from tokenizers import Tokenizer

from shirabe.features import WORD_RE
from shirabe.inference import ROOT


class TransformerModel:
    def __init__(self, path=None):
        self.path = path or ROOT / "artifacts/transformer"
        self.artifact = json.loads((self.path / "manifest.json").read_text())
        if (
            hashlib.sha256((self.path / "model.onnx").read_bytes()).hexdigest()
            != self.artifact["model_sha256"]
        ):
            raise ValueError("Transformer artifact checksum mismatch")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(self.path / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=self.artifact["max_length"])
        self.full_tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self.full_tokenizer.no_padding()
        self.full_tokenizer.no_truncation()
        self.training_hashes = set(self.artifact["training_text_hashes"])

    def predict(self, text):
        words = WORD_RE.findall(text)
        if not 80 <= len(words) <= 800:
            raise ValueError(f"Use an English abstract of 80–800 words; this input has {len(words)} words.")
        try:
            language = detect(text)
        except LangDetectException:
            language = "unknown"
        if language != "en":
            raise ValueError(
                "This model was trained on English abstracts. Please supply an English abstract."
            )
        encoded = self.tokenizer.encode(text)
        full_length = len(self.full_tokenizer.encode(text).ids)
        inputs = {
            key: np.asarray([value], dtype=np.int64)
            for key, value in {
                "input_ids": encoded.ids,
                "attention_mask": encoded.attention_mask,
                "token_type_ids": encoded.type_ids,
            }.items()
        }
        margin, pooled, attention, layers = self.session.run(None, inputs)
        a = self.artifact
        coef = np.asarray(a["coef"])
        values = pooled[0].astype(float)
        contributions = values * coef
        z = (values - np.asarray(a["activation_mean"])) / np.asarray(a["activation_scale"])
        logit = float(margin[0]) * a["calibration"]["slope"] + a["calibration"]["offset"]
        warnings = []
        if full_length > a["max_length"]:
            warnings.append(
                f"Input contains {full_length} WordPieces; only the first {a['max_length'] - 2} content pieces were scored."
            )
        digest = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
        seen = digest in self.training_hashes
        if seen:
            warnings.append(
                "This exact abstract appeared in training; its score is not an out-of-sample prediction."
            )
        if len(set(w.lower() for w in words)) / len(words) < 0.2:
            warnings.append("This input is unusually repetitive. Treat the score as out of distribution.")
        weights = attention[0, -1].mean(axis=0)
        return {
            "model_id": a["model_id"],
            "architecture": a["architecture"],
            "probability": float(1 / (1 + np.exp(-np.clip(logit, -700, 700)))),
            "baseline": a["train_prevalence"],
            "logit": logit,
            "intercept": a["intercept"],
            "word_count": len(words),
            "features": [
                {
                    "name": f"latent_{i:03d}",
                    "label": f"Latent {i + 1:03d}",
                    "value": float(v),
                    "z": float(z[i]),
                    "contribution": float(contributions[i]),
                    "coefficient": float(coef[i]),
                    "unusual": bool(abs(z[i]) > 3),
                }
                for i, v in enumerate(values)
            ],
            "tokens": [
                {
                    "text": token,
                    "start": start,
                    "end": end,
                    "category": "content",
                    "categories": [],
                    "attention": float(weights[i]),
                    "token_id": encoded.ids[i],
                }
                for i, (token, (start, end)) in enumerate(zip(encoded.tokens, encoded.offsets, strict=True))
                if end > start
            ],
            "transformer": {
                "layers": a["layers"],
                "heads": a["heads"],
                "hidden_size": a["hidden_size"],
                "sequence_length": len(encoded.ids),
                "full_sequence_length": full_length,
                "cls_attention": attention[0].tolist(),
                "layer_cls": layers[0].tolist(),
                "attention_note": "CLS attention by layer/head, including special tokens; attention is not causal attribution.",
            },
            "warnings": warnings,
            "seen_in_training": seen,
            "target": "High citation attention within the sampled field/year cohort",
            "input_scope": "abstract",
            "language": language,
        }
