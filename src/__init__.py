# Deep Learning Dynamics Experiment
# Symbolic Regression for Catastrophic Forgetting Dynamics

__version__ = "0.3.0"

from .models import LinearTeacher, LinearStudent, TaskPair, create_task_pair
from .nonlinear_models import (
    NonlinearTeacher, NonlinearStudent, NonlinearTaskPair,
    create_nonlinear_task_pair, classify_regime
)
from .universal_subspace import (
    UniversalSubspace, SubspaceAnalysis,
    compute_transition_boundary, fit_transition_equation
)
