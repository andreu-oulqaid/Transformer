import torch

def create_padding_mask(seq, pad_id):
    """
    seq: [batch, seq_len]
    """
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)
    # [batch, 1, 1, seq_len]


def create_look_ahead_mask(seq_len, device):
    """
    Prevent future token access
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    # [seq_len, seq_len]


def create_decoder_mask(seq, pad_id):
    """
    Combines padding + look-ahead mask
    """
    batch_size, seq_len = seq.shape
    device = seq.device

    pad_mask = create_padding_mask(seq, pad_id)
    look_ahead = create_look_ahead_mask(seq_len, device)

    return pad_mask & look_ahead