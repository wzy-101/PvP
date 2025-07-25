import torch


def dcg_score(y_score: torch.Tensor, y_true: torch.Tensor, k=10) -> torch.Tensor:
    y_true = y_true.float()
    y_score = y_score.float()

    order = torch.argsort(y_score).flip([0])
    y_true = torch.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = torch.log2(torch.arange(len(y_true), device=y_true.device).float() + 2)

    return torch.sum(gains / discounts)


def ndcg_score(y_score: torch.Tensor, y_true: torch.Tensor, k=10) -> torch.Tensor:
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_score, y_true, k)
    return actual / best


def mrr_score(y_score: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """计算 MRR（Mean Reciprocal Rank）"""
    y_true = y_true.float()
    y_score = y_score.float()

    order = torch.argsort(y_score, descending=True)
    sorted_y_true = torch.take(y_true, order)

    ranks = torch.arange(1, len(sorted_y_true) + 1, device=y_true.device).float()

    relevant_indices = torch.nonzero(sorted_y_true, as_tuple=True)[0]
    if len(relevant_indices) == 0:
        return torch.tensor(0.0, device=y_score.device)

    return torch.mean(1.0 / ranks[relevant_indices])
