# GRaTS-XAI Pipeline Notes

This document summarizes the data generation, pipeline methodology, and evaluation metrics used in the repository.

---

## 1. Dataset Generation

We simulate **Dynamic Bayesian Networks (DBNs)** with intra- and inter-slice dependencies.

- **Intra-slice structure**:  
  $$  W \in \mathbb{R}^{D \times D} $$ is a DAG within each time slice, with edges sampled from an Erdős–Rényi (ER) or Barabási–Albert (BA) model.  
  Edge weights are either positive (class 1) or negative (class 0), and the **label** for each sequence is determined by the sign of intra-slice edges.

- **Inter-slice structure**:  
  For lag $$ \ell \in \{1, \dots, p\} $$,  
  $$
  A^{(\ell)} \in \mathbb{R}^{D \times D}, \quad A^{(\ell)}_{ij} \sim \text{Uniform}\big(\pm [0.3 \eta^{-\ell}, 0.5 \eta^{-\ell}] \big).
  $$

- **SEM Simulation**:  
  For sequence length \( T \):
  $$
  x_t = x_t W + \sum_{\ell=1}^p x_{t-\ell} A^{(\ell)} + \varepsilon_t,
  $$
  where \( \varepsilon_t \sim \mathcal{N}(0, I) \) (or exponential noise).

Each dataset is saved in **`data/`** as `train.pkl` and `val.pkl`, containing \( X \in \mathbb{R}^{N \times T \times D} \), labels \( y \), and adjacency matrices.

---

## 2. Models

Three classifiers are implemented:

- **LSTMClassifier**  
- **TCNClassifier**  
- **TransformerClassifier**

Each model takes \( (T, D) \) input and outputs class logits.  
Training uses Adam optimizer with cross-entropy loss.  
**Mini-batch training** (configurable `batch_size`) is supported.

---

## 3. Explanations

We support multiple **attribution methods**:

- **Integrated Gradients (IG)**:
  $$
  \text{IG}_i(x) = (x_i - x_i') \int_0^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha
  $$

- **TimeRISE** (randomized masking-based explainer).

Attributions are saved in `runs/.../attr/`.

---

## 4. Graph Recovery

Given base attributions \( A \), we estimate **pairwise feature interactions**:

- **Asymmetric Interaction Response**:
  perturbs pairs of nodes across time lags and measures marginal impact on prediction.

- Produces \( \hat{W} \in \mathbb{R}^{D \times D \times L} \), stored in `runs/.../graph/`.

---

## 5. Refinement

We refine base attributions via **graph Laplacian smoothing**:

$$
S^\ast = \arg\min_S \| S - A \|^2 + \lambda \, \text{Tr}(S^\top L S),
$$

where \( L \) is the normalized Laplacian of \( \hat{W} \).  
Closed-form solution:

$$
S^\ast = (I + \lambda L)^{-1} A.
$$

---

## 6. Metrics

- **Graph recovery metrics** (`graph_metrics.py`):
  - ROC-AUC, PR-AUC
  - F1@K (top-K edges where K = #true edges)
  - SHD (structural Hamming distance)

- **Faithfulness metrics** (`faithfulness.py`):
  - **Comprehensiveness**: drop in prediction when top-K features are removed.
  - **Sufficiency**: prediction using only top-K features.

---

## 7. Hyperparameters

Defined in `configs/pipeline.yaml`. Key parameters:

- **Dataset sweeps**:  
  - `num_samples`: #sequences per dataset  
  - `n`: sequence length  
  - `d`: #variables (features)  
  - `p`: max lag  
  - `k_intra`, `k_inter`: expected in/out degree  

- **Model sweeps**:  
  - `hidden`: hidden size  
  - `epochs`: training epochs  
  - `lr`: learning rate  
  - `batch_size`: mini-batch size  

- **Explainers**:  
  - `steps`: IG interpolation steps  
  - `n_masks`: #masks for TimeRISE  

- **Graph recovery**:  
  - `lags`: candidate lags  
  - `rho`: perturbation scaling  
  - `S`: #samples per interaction estimate  

- **Refinement**:  
  - `lambda`: Laplacian smoothing strength  

---

## 8. Outputs

For each dataset–model–explainer combination:

- **Attributions** → `runs/.../attr/`  
- **Recovered graphs** → `runs/.../graph/`  
- **Metrics** → `runs/.../metrics/metrics.json`  
- **Visualizations** → `runs/.../plots/`

---

