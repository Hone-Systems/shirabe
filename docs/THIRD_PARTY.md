# Third-party notices

This repository's original code and linear model are AGPL-3.0; see `LICENSE`. The BERT-Mini derivative weights and tokenizer retain Apache-2.0.

- **PyMuPDF / MuPDF** (Artifex): AGPL open-source distribution. [License documentation](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright). It handles PDF extraction in a separate bounded process; retain its distribution notices.
- **OpenAlex** (OurResearch): public scholarly metadata supplied under CC0. [Project](https://openalex.org/). Original papers/abstracts are not relicensed or included here.
- **IBM Plex Sans / IBM Plex Mono**: SIL Open Font License 1.1. Installed from Fontsource and served locally; see package LICENSE files and [IBM Plex](https://github.com/IBM/plex).
- **Lucide** icons: ISC license. [Lucide](https://github.com/lucide-icons/lucide).
- **React, Vite, TypeScript, FastAPI, NumPy, SciPy, scikit-learn, langdetect and supporting libraries**: retain their respective package notices. Exact dependencies are in `uv.lock` and `web/package-lock.json`; their own licenses remain applicable.

Research references used to frame the experiment:

1. Stavrova et al. (2025), “Scientific publications that use promotional language in the abstract receive more citations and public attention,” [Communications Psychology](https://www.nature.com/articles/s44271-025-00293-8).
2. scikit-learn, [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
3. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), used solely as a locally downloaded real-PDF extraction check, not a labeled training example or redistributed fixture.

- **Google BERT-Mini**: pretrained model `google/bert_uncased_L-4_H-256_A-4`, revision `387825ce42dbb39b87911cdf8e383ee3b25184f8`, Apache-2.0. Fine-tuned weights and tokenizer retain upstream notices in `artifacts/transformer/`.
- **Transformers / Tokenizers**: Hugging Face, Apache-2.0; **ONNX / ONNX Runtime / PyTorch** retain their package licenses. Training dependencies are optional; serving uses ONNX Runtime.
