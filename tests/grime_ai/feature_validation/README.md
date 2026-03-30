# GRIME AI Feature Extraction Validation

## Overview
Validation evidence for the feature extraction module of the GaugeCam Remote Image Manager Educational Artificial Intelligence (GRIME AI) framework. This resource contains the validation script, test plan, synthetic test images, and CSV results from a validation run conducted on 2026-03-27.

## Contents

- `test_feature_validation.py` — Python script that generates the synthetic test images and implements the GRIME AI feature extraction algorithm as of 2026-03-27
- `GRIME_AI_Feature_Test_Plan - Rev 1.0 - 2026_03_27.docx` — Formal test plan
- `data/feature_extraction_test_images/` — Synthetic test images used as inputs to GRIME AI
- `results/` — CSV outputs from GRIME AI feature extraction run against the synthetic test images

## Validation Approach

Synthetic test images with known properties are loaded into GRIME AI. GRIME AI output is compared against known expected values to confirm correct feature extraction behavior.

**Analytical validation** — Pure-color synthetic images with known exact expected values. Metrics: intensity, Shannon entropy, GCC, ExG, HSV cluster centers.

**Directional validation** — Synthetic image pairs where the expected directional relationship between metric values is known. Metrics: GLCM, Gabor, LBP, Fourier.

## Cross-Validation
The validation script implements the same feature extraction algorithm as GRIME AI as of 2026-03-27. Running the script against the synthetic test images provides an independent reference result that can be compared directly against GRIME AI output to cross-validate results.

## Generating the Test Images
```
python test_feature_validation.py generate
```

## License
Apache License 2.0

## Related Resources
GRIME AI source code: https://github.com/JohnStranzl/GRIME-AI
