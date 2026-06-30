# bayes-decision

Bayesian decision-theoretic framework for small-sample experiments.

## Install

```bash
pip install -e .            # basic (numpy + scipy)
pip install -e ".[gpu]"     # + PyTorch GPU
pip install -e ".[all]"     # + GPU + parallel + plots
```

## Quick start

```python
import numpy as np
from bayes_decision import bayes_decision, check_system

# Check your hardware
check_system()

# Analyse a two-group experiment
rng = np.random.default_rng(42)
result = bayes_decision(rng.normal(0.8, 1, 5), rng.normal(0, 1, 5))
print(result)

# GPU-accelerated Monte Carlo (1 million reps)
from bayes_decision import expected_loss_gpu
el = expected_loss_gpu(n=5, delta=0.5, reps=1_000_000)
```
