# GrATS-XAI: Graph-based Attributions for Time Series Explainability
GrATS-XAI introduces a framework for graph-based attributions in time series models, bridging saliency methods to graphs. By using the post-hoc explainers to generate a structured graph, it enables understanding and evaluation of feature interaction and relevance in temporal modelling.

---

## 📦 Installation

Clone and set up the environment:

```bash
git clone https://github.com/<your-username>/grats-xai.git
cd grats-xai

# Create conda environment
conda create -n grats python=3.9 -y
conda activate grats

# Install requirements
pip install -r requirements.txt

```


## Project Structure

```
├── configs/               # YAML configs for data generation & experiments
│   └── data_gen.yaml
├── src/
│   ├── datasets/          # Synthetic DBN generator
│   │   └── synthetic_dbn.py
│   ├── models/            # Simple baselines (e.g. LSTM)
│   │   └── simple_lstm.py
│   ├── explainers/        # Explainability methods (IG, TimeRISE, etc.)
│   └── evaluation/        # Metrics (infidelity, comprehensiveness, etc.)
├── runs/                  # Auto-saved experiments (ignored via .gitignore)
└── README.md
```



## 🚀 Usage
All supported in the config

```
python scripts/pipeline.py --config configs/pipeline_quick.yaml
```

### 1. Generate synthetic data

Outputs are saved under:

```
runs/dbn_n{params}/
  ├── train.pkl
  ├── val.pkl
  └── plots/
```
### 2. Train a model


### 3. Run explainability

Choose an explainer:

    # Integrated Gradients
    # TimeRISE
// TODO:
  Integrated Hessians

## 📊 Example Output


