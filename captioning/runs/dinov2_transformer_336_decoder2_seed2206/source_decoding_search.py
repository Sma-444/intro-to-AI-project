"""Batched beam search; uses only captioner probabilities, never reference captions."""
import torch
from train import BOS, EOS, PAD, UNK


@torch.inference_mode()
def search_features(model, spatial, projected, beam_size=3, alpha=0.6,
                    constrain=True, max_words=24):
    batch = spatial.shape[0]
    device = spatial.device
    k = beam_size
    spatial = spatial.repeat_interleave(k, dim=0)
    projected = projected.repeat_interleave(k, dim=0)
    state = model.start_state(spatial)
    token = torch.full((batch*k,), BOS, device=device, dtype=torch.long)
    scores = torch.full((batch, k), -torch.inf, device=device)
    scores[:, 0] = 0
    lengths = torch.zeros((batch, k), device=device, dtype=torch.long)
    finished = torch.zeros((batch, k), device=device, dtype=torch.bool)
    sequences = torch.empty((batch, k, 0), device=device, dtype=torch.long)
    for position in range(max_words):
        logits, next_state = model.step(token, spatial, projected, state)
        logp = logits.float().log_softmax(-1)
        if constrain:
            logp[:, [PAD, BOS, UNK]] = -torch.inf
            if position == 0:
                logp[:, EOS] = -torch.inf  # require at least one actual word
            if position >= 3:
                flat = sequences.reshape(batch*k, position)
                for start in range(position-2):
                    match = (flat[:, start:start+2] == flat[:, -2:]).all(-1)
                    rows = match.nonzero(as_tuple=True)[0]
                    logp[rows, flat[rows, start+2]] = -torch.inf
        ended = finished.reshape(-1)
        logp[ended] = -torch.inf
        logp[ended, EOS] = 0  # completed hypotheses retain score and length
        vocab_size = logp.shape[-1]
        candidate_scores = scores.unsqueeze(-1) + logp.view(batch, k, vocab_size)
        new_lengths = lengths + (~finished).long()
        penalty = ((5 + new_lengths.float()) / 6).pow(alpha)
        rank = candidate_scores / penalty.unsqueeze(-1)
        _, indices = rank.flatten(1).topk(k, dim=-1)
        parents = indices // vocab_size
        selected_tokens = indices % vocab_size
        scores = candidate_scores.flatten(1).gather(1, indices)
        lengths = new_lengths.gather(1, parents)
        finished = finished.gather(1, parents) | selected_tokens.eq(EOS)
        sequences = sequences.gather(1, parents.unsqueeze(-1).expand(-1, -1, position))
        sequences = torch.cat((sequences, selected_tokens.unsqueeze(-1)), -1)
        flat_parents = (parents + torch.arange(batch, device=device)[:, None]*k).reshape(-1)
        state = tuple(x[flat_parents] for x in next_state)
        token = selected_tokens.reshape(-1)
        if finished.all():
            break
    rank = scores / ((5 + lengths.float())/6).pow(alpha)
    best = rank.argmax(1)
    return sequences[torch.arange(batch, device=device), best]
