
# xai_graphs (starter)

Minimal, modular starter to study **post-hoc temporal XAI → graph construction → graph-refined attributions**.

## Layout
```
xai_graphs/
  datasets/synthetic_var.py
  models/simple_lstm.py
  explainers/ig_wrapper.py
  explainers/time_rise.py
  graphs/interaction_shapley.py
  graphs/laplacian_refine.py
  evaluation/metrics.py
  main.py
  requirements.txt
```
## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --seed 0 --n_series 256 --T 80 --D 6
```
This will: (1) generate a synthetic nonlinear VAR dataset with a **known ground-truth graph**, (2) train a tiny LSTM classifier, (3) compute **Integrated Gradients** attributions, (4) estimate **pairwise interactions** with a Shapley-style sampler on top-k nodes, (5) build a sparse graph, and (6) run **Laplacian refinement** to produce structure-aware heatmaps and basic metrics.
