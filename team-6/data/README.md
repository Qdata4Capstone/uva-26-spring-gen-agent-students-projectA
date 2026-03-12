# Data Source

Our project is based on Symile-MIMIC dataset(https://physionet.org/content/symile-mimic/1.0.0/). 

### Dataset Overview: Symile-MIMIC (v1.0.0)
**Symile-MIMIC** is a multimodal clinical dataset derived from the MIMIC-IV and MIMIC-CXR databases. It is primarily designed to evaluate multimodal representation learning (specifically the Symile contrastive learning objective) and cross-modal zero-shot retrieval tasks. 

* **Modalities Included:** Each data sample strictly aligns three different clinical modalities for a single hospital admission:
  * **ECG:** Electrocardiogram signals collected within 24 hours of admission.
  * **Blood Labs:** Tabular results from up to 50 common blood laboratory tests (collected within 24 hours).
  * **CXR:** A Chest X-ray performed 24–72 hours post-admission.
* **Dataset Size:** The dataset contains 11,622 unique hospital admissions (from 9,573 distinct patients). It is rigidly split into training (10,000), validation (750), and test (464) sets with absolutely no patient overlap across the splits.
### Data Format Specifications

- **CXR (Chest X-ray):**  
  Pre-processed image tensors normalized using ImageNet statistics.  
  Shape: `(n, 3, 320, 320)`

- **ECG:**  
  12-lead continuous signals normalized to the range `[-1, 1]`.  
  Shape: `(n, 1, 5000, 12)`

- **Blood Labs:**  
  The 50 most common laboratory test values standardized into percentiles.  
  Shape: `(n, 50)`  
  Each sample is accompanied by a **binary missingness mask** indicating whether each lab value is observed or missing.