from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules.loss import _Loss
from torch.nn.functional import logsigmoid
from typing import Iterator


__all__ = (
    'BPRLoss'
)


def count_params(params: Iterator[Parameter]) -> int:
    return sum([param.numel() for param in params])


def compute_prior(params: Iterator[Parameter]) -> Tensor:
    return sum([param.pow(2).sum() for param in params])


class BPRLoss(_Loss):

    def __init__(self, 
        params: Iterator[Parameter],
        *,
        reg_factor: float, 
    ) -> None:
        super().__init__()
        # Saving the input arguments.
        self.params = list(params)
        self.reg_factor = reg_factor
        # Computing derived attributes.
        self._num_params = count_params(self.params)


    def forward(self, 
        y_pos: Tensor, 
        y_neg: Tensor, 
    ) -> Tensor:
        prior = compute_prior(self.params) / self._num_params
        likelihood = logsigmoid(y_pos - y_neg).mean()
        return - likelihood + self.reg_factor * prior