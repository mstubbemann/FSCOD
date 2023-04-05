# Selecting Features by their Resilience to the Curse of Dimensionality

This repository contains code for the paper ***Selecting Features by their Resilience to the Curse of Dimensionality***.

We used Python 3.10.6 for all our experiments.
The script `ogb-clf.py` was run on a GeForce RTX 3090 with CUDA 12.1.
All other scripts were run on a Xeon-Gold 5122 CPU server.

All needed requirements can be installed via `pip install -r requirements_cpu.txt` or `pip install -r requirements_gpu.txt` 

# Experiments on Open18

```python
PYTHONHASHSEED=42 python open18-script.py
```
# Experiments on OGB
## Feature Selection and Clafficiation Experiment
```python
PYTHONHASHSEED=42 python ogb-clf.py
```

## Maximal Error Ratios
```python
PYTHONHASHSEED=42 python maximal_errors.py
```

## Error Ratios of ogbn-arxiv
```python
PYTHONHASHSEED=42 python real_errors.py
```