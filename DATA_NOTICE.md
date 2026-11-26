# DICES data attribution

Source: Google Research Datasets, https://github.com/google-research-datasets/dices-dataset

Lora Aroyo, Alex S. Taylor, Mark Díaz, Christopher M. Homan, Alicia Parrish, Greg Serapio-García, Vinodkumar Prabhakaran, and Ding Wang. 2023. *DICES Dataset: Diversity in Conversational AI Evaluation for Safety*. Advances in Neural Information Processing Systems 36, 53330–53342. DOI: 10.52202/075280-2321.

Google LLC licenses the data under Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/ . The upstream README contains the original license notice, reproduced in `docs/upstream/README.md`.

Changes: the bundled CSVs are projections of the supplied raw files onto item/rater IDs, the overall response and the reason columns consumed by the original analysis. All rows and column values are retained without relabeling or identifier remapping. Text, demographics, timestamps and unused fields are omitted. Raw and projected hashes are in `data/manifest.json`.

The MIT code license does not replace the DICES data license. The original datasets contain adversarial conversations; the bundled subsets omit the conversation text.
