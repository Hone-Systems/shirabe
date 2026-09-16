"""Experimental outcome inference and measured, grouped transformer telemetry."""

import hashlib
import json
import threading

import numpy as np

from shirabe.inference import ROOT
from shirabe.outcome_rubric import DIMENSIONS, QUESTIONS

POINTER = ROOT / "artifacts/outcome-training/served.json"
LOCK = threading.Lock()


class OutcomeModel:
    def __init__(self):
        self.signature = None
        self.model = None

    def load(self):
        if not POINTER.exists():
            raise FileNotFoundError(
                "The expanded outcome model is still training. Follow progress in Training."
            )
        pointer = json.loads(POINTER.read_text())
        signature = pointer["checkpoint_sha256"]
        if signature == self.signature:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = ROOT / pointer["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != signature:
            raise ValueError("Outcome checkpoint checksum failed")
        model = AutoModelForSequenceClassification.from_pretrained(
            pointer["base_model"],
            revision=pointer["revision"],
            num_labels=20,
            local_files_only=True,
            attn_implementation="eager",
        )
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pointer["base_model"], revision=pointer["revision"], local_files_only=True
        )
        self.model, self.pointer, self.signature = model, pointer, signature

    def metadata(self):
        with LOCK:
            self.load()
            layers = []
            for i, layer in enumerate(self.model.bert.encoder.layer):
                # Equal groups of 32 latent dimensions, not individual neurons.
                w = layer.attention.output.dense.weight.detach().numpy().reshape(8, 32, 8, 32).mean((1, 3))
                layers.append({"index": i, "weights": w.tolist(), "heads": 4, "hidden_size": 256})
            w = self.model.classifier.weight.detach().numpy().reshape(20, 8, 32).mean(2)
            return {
                "model_id": self.pointer["model_id"] + "-" + self.signature[:10],
                "status": "experimental",
                "representation": "wording",
                "layers": layers,
                "output_weights": w.tolist(),
                "questions": [
                    {"id": q, "dimension": d, "question": t, "supported": bool(self.pointer["supported"][i])}
                    for i, (q, d, t) in enumerate(QUESTIONS)
                ],
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "training_papers": self.pointer["training_papers"],
                "preprocessing": "Outcome-blind topic masking via Luna; local BERT-Mini over every input chunk",
                "diagram_note": "Nodes group 32 hidden dimensions. Edges show signed mean projection weights; attention routing, residual and feed-forward paths are collapsed. This is a summarized architecture view, not a causal attribution.",
            }

    def predict(self, text, progress=None):
        def emit(stage, **data):
            if progress:
                progress({"stage": stage, **data})

        emit("queued")
        import torch

        from scripts.extract_wording import VERSION, extract
        from scripts.train_outcomes import chunks
        from shirabe.research_clients import ResearchClients

        if len(text.strip()) < 100:
            raise ValueError("Add at least 100 characters of original paper text.")
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 42
        language = detect_langs(text)[0]
        if language.lang != "en" and language.prob > 0.9:
            raise ValueError(
                "This model was trained on English wording. Please provide an English original; automatic translation would change the wording being measured."
            )
        # One paid client at a time; the existing ledger enforces the shared $50 total.
        with LOCK:
            self.load()
            emit("masking", model_id=self.pointer["model_id"] + "-" + self.signature[:10])
            if self.pointer["wording_version"] != VERSION:
                raise ValueError("Preprocessing version differs from trained checkpoint")
            record = {
                "id": "inference-" + hashlib.sha256(text.encode()).hexdigest(),
                "original_text": text,
                "input_scope": "user_supplied_original",
                "split": "inference",
                "field_id": 0,
                "rubric_version": "outcomes-v1",
                "retrieval_version": 4,
            }
            prepared = extract(record, ResearchClients(ROOT.parent / "senki", 50))
            encoded = chunks(self.tokenizer, prepared["wording_text"])
            emit(
                "encoding",
                chunks=len(encoded),
                tokens=sum(len(c) - 2 for c in encoded),
                masked_fraction=prepared["redacted_character_fraction"],
            )
            logits = []
            layer_sums = np.zeros((5, 8))
            total_tokens = 0
            attention_sums = np.zeros((4, 4))
            summaries = []
            with torch.inference_mode():
                for n, c in enumerate(encoded):
                    ids = torch.tensor([c])
                    result = self.model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        output_hidden_states=True,
                        output_attentions=True,
                    )
                    logits.append(result.logits[0])
                    for i, h in enumerate(result.hidden_states):
                        layer_sums[i] += h[0].numpy().reshape(len(c), 8, 32).mean((0, 2)) * len(c)
                    for i, a in enumerate(result.attentions):
                        # Mean normalized entropy across all query positions for each head.
                        probs = a[0].numpy().clip(1e-12, 1)
                        entropy = -(probs * np.log(probs)).sum(-1).mean(-1) / np.log(max(2, len(c)))
                        attention_sums[i] += entropy * len(c)
                    total_tokens += len(c)
                    probability = torch.sigmoid(result.logits[0]).numpy()
                    summaries.append(
                        {
                            "chunk": n + 1,
                            "tokens": len(c) - 2,
                            "mean_supported_probability": float(
                                np.mean(probability[np.array(self.pointer["supported"], dtype=bool)])
                            ),
                        }
                    )
                    emit(
                        "inference",
                        completed=n + 1,
                        chunks=len(encoded),
                        tokens=sum(len(chunk) - 2 for chunk in encoded[: n + 1]),
                        layer_activations=(layer_sums / total_tokens).tolist(),
                        head_entropy=(attention_sums / total_tokens).tolist(),
                        chunk_summaries=list(summaries),
                    )
            emit("aggregating", completed=len(encoded), chunks=len(encoded))
            probs = torch.sigmoid(torch.stack(logits).mean(0)).numpy()
            outputs = [
                {
                    "id": q,
                    "dimension": d,
                    "question": t,
                    "probability": float(probs[i]) if self.pointer["supported"][i] else None,
                    "training_yes": self.pointer["question_counts"][i].get("yes", 0),
                    "training_no": self.pointer["question_counts"][i].get("no", 0),
                }
                for i, (q, d, t) in enumerate(QUESTIONS)
            ]
            dimensions = []
            for d in DIMENSIONS:
                values = [
                    r["probability"] for r in outputs if r["dimension"] == d and r["probability"] is not None
                ]
                dimensions.append(
                    {
                        "id": d,
                        "value": float(np.mean(values)) if values else None,
                        "supported": len(values),
                        "total": 5,
                    }
                )
            values = [d["value"] for d in dimensions if d["value"] is not None]
            return {
                "model_id": self.pointer["model_id"] + "-" + self.signature[:10],
                "experimental": True,
                "score": float(np.mean(values)) if len(values) == 4 else None,
                "score_kind": "Predicted rubric index, not a calibrated success probability",
                "dimensions": dimensions,
                "outputs": outputs,
                "tokens": sum(len(c) - 2 for c in encoded),
                "chunks": len(encoded),
                "word_count": len(text.split()),
                "masked_fraction": prepared["redacted_character_fraction"],
                "layer_activations": (layer_sums / total_tokens).tolist(),
                "head_entropy": (attention_sums / total_tokens).tolist(),
                "chunk_summaries": summaries,
                "warnings": [
                    "This is an experimental outcome model; external validity is not established.",
                    "Single-class question heads cannot establish success/failure discrimination.",
                    "Attention-head entropy and grouped activations are measurements, not explanations of causality.",
                ],
                "training_papers": self.pointer["training_papers"],
            }


outcome_model = OutcomeModel()
