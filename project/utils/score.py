import torch
from torch import Tensor
from .rank import rank_k


__all__ = (
    'ndcg_coefs',
    'ndcg_score',
    'rank_score',
    'recall_score'
)


def ndcg_coefs(count: int) -> Tensor:
    coefs = torch.arange(count) + 2
    coefs = torch.log2(coefs)
    return coefs


def ndcg_score(
    score: Tensor, 
    total: Tensor,
    *, 
    coefs: Tensor = None
) -> Tensor:
    # Infers the score's resolution.
    *_, batch, count = score.shape
    # Converts the total to a tensor, if given an integer.
    if type(total) == int:
        total = torch.full([count], count)
    # Constructs the discount factors, if not given.
    if coefs is None:
        coefs = ndcg_coefs(count)
    # Determines terms for the normalization summation.
    tally = torch.full_like(total, count)
    terms = (
        torch.arange(count).unsqueeze(-2).expand(batch, count)
        <
        torch.stack([total, tally]).amin(-2).long().unsqueeze(-1)
    ) * coefs.unsqueeze(-2).expand(batch, count)
    # Computes and returns the NDCG-score.
    return (score / coefs).sum(-1) / terms.sum(-1)


def recall_score(score: Tensor, total: Tensor) -> Tensor:
    return score.sum(dim=-1) / total


def rank_score(
    *args,
    score_fns: list[callable],
    verbose: bool = False,
    **kwargs
) -> list[float]:
    # Computes the ranked labels and the total possitive instances.
    labels, counts = rank_k(*args, verbose=verbose, **kwargs)
    # Computes the scores 
    scores = [score_fn(labels, counts).mean() for score_fn in score_fns]
    # Returns the scores as a tensor.
    return torch.tensor(scores).tolist()