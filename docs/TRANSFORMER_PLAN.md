# Tiny transformer upgrade

User explicitly requested a proper tiny transformer architecture after questioning the fixed feature baseline. Preserve the single-screen redesign and train/deploy a real learned encoder.

Before training: use Google BERT-Mini (4 layers, 4 heads/layer, hidden size 256; Apache-2.0 pretrained weights). Fine-tune every parameter on complete abstract token sequences, max 512 WordPieces. AdamW 3e-5, weight decay 0.01, batch 32, four epochs, seed 42; select epoch by 2019 log loss only. Fit scalar sigmoid calibration on 2020; evaluate 2021 untouched. Same 11,799-record corpus, labels, and splits. CPU inference through ONNX; GPU training on local 4070 Ti SUPER. No cloud spend.

Full token input means topic content is available. Do not claim topic independence or label output as style-only. Evaluate per-field and leave-one-field-out transfer; compare with original linear/lexical/nonlinear baselines. Include shuffled-token stress test to inspect reliance on sequence order; diagnostic only, not evidence of causality. Pretraining exposure to historical text is possible and cannot be excluded.

Serve explicit architecture metadata; show actual WordPiece tokens, final-layer CLS attention (not causal importance), layer CLS activations, and exact 256-dimensional classification-head logit decomposition. Radial field must change from 57 engineered features to learned channels, never animate the old architecture under the new model's label. Keep baseline artifacts and reproducibility commands separately.

Verify ONNX/PyTorch probability parity, calibration, API/PDF flows, dynamic lengths, all head contributions summing to logit, UI labels/values, trained artifact hash, fresh-process inference, frontend review, public repository and CI.
