"""Inference uses JSON weights only; exported contributions sum exactly to the logit."""

import hashlib
import json
from pathlib import Path

import numpy as np
from langdetect import DetectorFactory, LangDetectException, detect

from shirabe.features import FEATURE_NAMES, WORD_RE, extract, feature_label, tokenize

DetectorFactory.seed = 42
ROOT = Path(__file__).resolve().parents[1]


class StyleModel:
    def __init__(self, path=None):
        self.artifact = json.loads((path or ROOT / "artifacts/model.json").read_text())
        if self.artifact["feature_names"] != FEATURE_NAMES:
            raise ValueError("Feature schema mismatch: retrain with this version of features.py")
        self.mean = np.array(self.artifact["mean"])
        self.scale = np.array(self.artifact["scale"])
        self.coef = np.array(self.artifact["coef"])
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
        raw = extract(text)
        unbounded_z = (raw - self.mean) / self.scale
        z = np.clip(unbounded_z, -self.artifact["clip"], self.artifact["clip"])
        contributions = z * self.coef
        logit = float(self.artifact["intercept"] + sum(contributions))
        probability = float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))
        warnings = []
        unusual = (raw < np.array(self.artifact["train_q01"])) | (raw > np.array(self.artifact["train_q99"]))
        if unusual.mean() > 0.15:
            warnings.append(
                "Several writing features fall outside the central 98% of training values; this estimate may not transfer to this input."
            )
        if np.any(abs(unbounded_z) > 5):
            warnings.append(
                "Extreme feature values were capped at ±5 training standard deviations, as during training."
            )
        if len(set(w.lower() for w in words)) / len(words) < 0.2:
            warnings.append("This input is unusually repetitive. Treat the score as out of distribution.")
        digest = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
        seen = digest in self.training_hashes
        if seen:
            warnings.append(
                "This exact abstract appeared in training; its score is not an out-of-sample prediction."
            )
        features = [
            {
                "name": name,
                "label": feature_label(name),
                "value": float(raw[i]),
                "z": float(z[i]),
                "contribution": float(contributions[i]),
                "coefficient": float(self.coef[i]),
                "unusual": bool(unusual[i]),
            }
            for i, name in enumerate(FEATURE_NAMES)
        ]
        return {
            "model_id": self.artifact["model_id"],
            "probability": probability,
            "baseline": self.artifact["train_prevalence"],
            "logit": logit,
            "intercept": self.artifact["intercept"],
            "word_count": len(words),
            "features": features,
            "tokens": tokenize(text),
            "warnings": warnings,
            "seen_in_training": seen,
            "target": "High citation attention within the sampled field/year cohort",
            "input_scope": "abstract",
            "language": language,
        }
