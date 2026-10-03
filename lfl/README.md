# Local Field Learner (backprop-free)

From `Local_Field_Learner.ipynb`: a fixed random encoder (+1/-1 hyperdimensional code, no training), a local
learning layer, hash-routed experts grown on demand, and a ridge readout kept as running sums so new data only
adds to it and nothing is forgotten. `local_field.py` holds the parts unchanged; `python lfl/test_local_field.py`
checks the properties the project relies on (sequential == joint readout, i.e. no forgetting).

Notebook results, reproduced 2026-10-03 (`runs/lfl/Local_Field_Learner_executed.ipynb`):

| Test | Backprop MLP | HD encoder + ridge |
|---|---|---|
| Pump failure detection (synthetic sensors), F1 | 0.935 | 0.905, one linear solve |
| Digits learned 2 classes at a time: all classes after 5 tasks | 0.185 | 0.967 |
| ... task 1 after 5 tasks | 0.0 | 0.991 |

**Used in the project:** ARC (`arc/hd_ridge.py`), where one ridge solve per task over HD codes of cell
neighbourhoods is our best rule learner so far (held-out 2.6% vs 0.9% for gradient-trained Settle). Next: the
running-sum readout as on-device personalisation memory in the pocket runtime.

Lesson: with few samples per feature, lambda must be large (on a 600-sample test, held-out accuracy rose
from 0.41 to 0.80 as lambda went from 1 to 1000).
