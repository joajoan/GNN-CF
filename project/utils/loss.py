from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules.loss import _Loss
from torch.nn.functional import logsigmoid
from typing import Iterator


__all__ = (
    'BPRLoss'
)


class BPRLoss(_Loss):

    def __init__(self, 
        params: Iterator[Parameter],
        *,
        reg_factor: float, 
    ) -> None:
        super().__init__()
        self.params = list(params)
        self.reg_factor = reg_factor


    def forward(self, 
        y_pos: Tensor, 
        y_neg: Tensor, 
    ) -> Tensor:
        prior = sum([param.pow(2).sum() for param in self.params])
        likelihood = logsigmoid(y_pos - y_neg).mean()
        return - likelihood + self.reg_factor * prior / y_pos.size(0)