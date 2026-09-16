# Single-screen instrument redesign

User direction: cooler, minimal text, visual/animated, single screen.

Connect the real fine-tuned BERT-Mini model through the existing API. Replace document-style dashboard with a viewport-filling instrument: narrow source panel, central radial feature plot + probability readout, compact evidence panel, token strip at the bottom. Analysis/training switch changes the same stage. Text editing, full feature arithmetic and complete report live in accessible dialogs. Dark near-black, ice type, electric mint for measured positive channels, restrained amber for negative channels. IBM Plex family retained. Motion: particles trace feature→logit connections; slower rotation of guide marks, score sweep on result; pause toggle and reduced-motion support. Explicitly a feature plot, not neural layers. No simulated training or fabricated metrics.

Stages: idle (real model weights), extracting, editable review, scoring (indeterminate pulse), result (actual contributions), training (actual ROC/calibration/transfer). No page scrolling at desktop 1280×800/1440×900; responsive mobile composition retains all actions and uses compact panels. Errors and caveats concise, with explanations on demand.

Validation: CPU ONNX parity, exact learned-head arithmetic; real desktop/mobile browser workflows, keyboard/dialog behavior, overflow, accessible states, numerical chart source verification, reduced motion, independent visual/accessibility review and final screenshots.

Transformer update: 256 learned head channels, actual WordPiece CLS attention, and real GPU epoch history replace the fixed 57-feature representation. The radial field represents the classifier head; particle animation remains illustrative.

Final verification: 22 unit/API/artifact checks and eight desktop/mobile browser flows passed. Independent visual, interaction and accessibility reviews checked 1440×900, 1280×800, 390×844 and 320×568. Corrected the prior Brier label after adding the transformer comparison, shortened mobile baseline labels, bounded token highlight contrast, and exposed 16 head values and all token attention values in keyboard-accessible Full trace tables. Training/sample sizes and CPU-inference labels now distinguish fine-tuning from evaluation and serving.
