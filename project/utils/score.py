import torch
from torch import Tensor


__all__ = (
    'composite_score',
    'ndcg_coefs',
    'ndcg_score',
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


def composite_score(
    label: Tensor,
    total: Tensor,
    *,
    score_fns: list[callable],
) -> list[float]:
    return [
        score_fn(label, total).mean().item() 
            for score_fn in score_fns
    ]