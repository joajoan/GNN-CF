from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules.loss import _Loss
from torch.nn.functional import logsigmoid
from typing import Sequence


__all__ = ('BPRLoss',)


def count_embeddings(params: Sequence[Parameter]) -> int:
    return sum([param.size(0) for param in params])


def compute_prior(params: Sequence[Parameter]) -> Tensor:
    return sum([param.square().sum() for param in params])


class BPRLoss(_Loss):

    def __init__(self, 
        reg_factor: float, 
    ) -> None:
        super().__init__()
        self.reg_factor = reg_factor


    def forward(self, 
        y_pos: Tensor, 
        y_neg: Tensor, 
        *,
        params: Sequence[Parameter]
    ) -> Tensor:
        prior = compute_prior(params) / count_embeddings(params)
        likelihood = logsigmoid(y_pos - y_neg).mean()
        return self.reg_factor * prior - likelihood