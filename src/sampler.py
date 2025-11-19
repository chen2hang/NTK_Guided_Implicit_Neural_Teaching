import math
import torch
import pickle


def mt_sampler(data, y, preds, size, top_k=True, importance=None):
    # Given size is ratio of training data
    if math.isclose(size, 1.0):
        return data, y, None, (importance if importance is not None else None)
    elif type(size) == float:
        n = int(size * len(data))
    # Given size is actual number of training data
    else:
        n = int(size)

    # mt sampling (returns indices)
    if importance is None:
        dif = torch.sum(torch.abs(y - preds), 1)
    else:
        dif = importance
    if top_k:
        _, idx = torch.topk(dif, n)
    else:
        idx = torch.randperm(len(data))[:n]

    # get sampled data
    sampled_data = data[idx]
    sampled_y = y[idx]
    
    return sampled_data, sampled_y, idx, dif

def hierarchical_sampler(data, y, preds, size, top_k=True, importance=None, cal_NTK=False, cached_ntk_indices=None, NTK_portion=0.2, R_portion=0.7):

    # Given size is ratio of training data or an absolute number
    if math.isclose(size, 1.0):
        n_total = len(data)
    elif type(size) == float:
        n_total = int(size * len(data))
    else:
        n_total = int(size)

    n_ntk = int(n_total * NTK_portion)
    n_res = int(n_total * R_portion)
    n_rnd = n_total - n_ntk - n_res

    # Prepare base indices and device once
    num_samples = len(data)
    device = data.device if torch.is_tensor(data) else torch.device('cpu')
    all_idx = torch.arange(num_samples, device=device)

    # NTK part: when use_NTK is True compute fresh; otherwise reuse cached indices if provided
    if cal_NTK:
        scores_ntk = importance
        if top_k:
            _, ntk_idx = torch.topk(scores_ntk, k=n_ntk)
        else:
            # if not top_k, sample proportionally to scores
            probs = (scores_ntk - scores_ntk.min()).clamp_min(0)
            probs = probs / probs.sum().clamp_min(1e-8)
            ntk_idx = torch.multinomial(probs, n_ntk, replacement=False)
    else:
        # no refresh; use cached indices if available
        if len(cached_ntk_indices) > 0:
            cached_ntk_indices = cached_ntk_indices.to(device)
            ntk_idx = cached_ntk_indices[:n_ntk]
        else:
            ntk_idx = torch.tensor([], dtype=torch.long, device=device)

    # Residual and Random parts using a single in-place GPU mask (no CPU sets)
    dif_res = torch.sum(torch.abs(y - preds), 1)
    mask = torch.ones(num_samples, dtype=torch.bool, device=device)
    if ntk_idx.numel() > 0:
        mask[ntk_idx] = False

    # Residual part
    available_scores = dif_res[mask]
    n_res = min(n_res, int(mask.sum().item()))
    if n_res > 0:
        if top_k:
            _, res_top_rel = torch.topk(available_scores, k=n_res)
        else:
            res_top_rel = torch.randperm(available_scores.numel(), device=device)[:n_res]
        base_idx = mask.nonzero(as_tuple=False).squeeze(1)
        res_idx = base_idx[res_top_rel]
        mask[res_idx] = False
    else:
        res_idx = torch.tensor([], dtype=torch.long, device=device)

    # Random part
    n_avail = int(mask.sum().item())
    n_rnd = min(n_rnd, n_avail)
    if n_rnd > 0:
        base_idx2 = mask.nonzero(as_tuple=False).squeeze(1)
        rnd_rel = torch.randperm(n_avail, device=device)[:n_rnd]
        rnd_idx = base_idx2[rnd_rel]
    else:
        rnd_idx = torch.tensor([], dtype=torch.long, device=device)

    # Merge indices, keep order (ntk, residual, random)
    idx = torch.cat([ntk_idx, res_idx, rnd_idx], dim=0)

    # Fallback if for any reason we still don't have enough (e.g., n_total > dataset)
    if idx.numel() < n_total:
        missing = n_total - idx.numel()
        rest_mask = torch.ones(num_samples, dtype=torch.bool, device=device)
        rest_mask[idx] = False
        rest_idx = rest_mask.nonzero(as_tuple=False).squeeze(1)
        if rest_idx.numel() > 0:
            fill = rest_idx[torch.randperm(rest_idx.numel(), device=device)[:missing]]
            idx = torch.cat([idx, fill], dim=0)

    # Fuse index operations: pre-allocate and fill directly
    sampled_data = torch.empty(idx.numel(), *data.shape[1:], device=device, dtype=data.dtype)
    sampled_y = torch.empty(idx.numel(), *y.shape[1:], device=device, dtype=y.dtype)
    
    sampled_data.copy_(data[idx])
    sampled_y.copy_(y[idx])

    # For dif, return the assembled importance used to rank (store NTK when present else residual)
    if importance is not None and importance.numel() == len(data):
        dif = importance
    else:
        dif = dif_res

    return sampled_data, sampled_y, idx, dif

def save_samples(sample_history, step, max_steps, samples, file_name):
    sample_history[str(step)] = samples.detach().cpu().numpy()
    with open(file_name, 'wb') as f:
        pickle.dump(sample_history, f)
    print("Sampling history saved at %s." % file_name)


def save_losses(loss_history, step, max_steps, losses, file_name):
    loss_history[str(step)] = losses.detach().cpu().numpy()
    with open(file_name, 'wb') as f:
        pickle.dump(loss_history, f)
    print("Loss history saved at %s." % file_name)