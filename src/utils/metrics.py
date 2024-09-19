from typing import Optional
from torchmetrics import Metric
import torch
from torch import Tensor

class SimpleMetric(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self):
        super().__init__()
        self.add_state("simple_metric", default=torch.tensor(0., dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0., dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, simple_metric: Tensor):
        self.simple_metric += simple_metric
        self.total += 1

    def compute(self):
        return self.simple_metric / self.total
