"""Minimal GRU encoder-decoder baseline for Task 1.2 recipe generation."""

from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.preprocessing import EOS_ID, PAD_ID, SOS_ID


class Encoder(nn.Module):
    """Encode ingredient IDs into a final recurrent hidden state."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        """Return the final hidden state for each source sequence."""
        embedded = self.dropout(self.embedding(src))
        # enforce_sorted=False 让 collate_fn 不必排序, notebook/debug 更直观
        packed = pack_padded_sequence(
            embedded,
            src_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)
        return hidden


class Decoder(nn.Module):
    """Teacher-forced GRU decoder initialized from the encoder hidden state."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt_input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Decode logits for every teacher-forced input token."""
        embedded = self.dropout(self.embedding(tgt_input))
        outputs, _ = self.rnn(embedded, hidden)
        return self.output(outputs)

    def step(self, token: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one autoregressive step and return ``(logits, next_hidden)``."""
        embedded = self.embedding(token.unsqueeze(1))
        output, next_hidden = self.rnn(embedded, hidden)
        logits = self.output(output.squeeze(1))
        return logits, next_hidden


class Seq2Seq(nn.Module):
    """A baseline encoder-decoder without attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.vocab_size = vocab_size

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run teacher-forced training forward pass.

        ``tgt`` is expected to include ``<SOS>`` and ``<EOS>``. Returned logits
        have shape ``[batch, tgt_len - 1, vocab_size]`` and should be compared
        with ``tgt[:, 1:]``.
        """
        if src_lengths is None:
            src_lengths = (src != PAD_ID).sum(dim=1).clamp_min(1)

        hidden = self.encoder(src, src_lengths)
        # 输入 <SOS>..倒数第二个 token, 目标是第二个 token..<EOS>
        decoder_input = tgt[:, :-1]
        return self.decoder(decoder_input, hidden)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        max_len: int = 80,
    ) -> torch.Tensor:
        """Greedily generate token IDs from source ingredient IDs."""
        self.eval()
        if src_lengths is None:
            src_lengths = (src != PAD_ID).sum(dim=1).clamp_min(1)

        hidden = self.encoder(src, src_lengths)
        batch_size = src.size(0)
        device = src.device
        token = torch.full((batch_size,), SOS_ID, dtype=torch.long, device=device)
        generated = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits, hidden = self.decoder.step(token, hidden)
            token = logits.argmax(dim=-1)
            generated.append(token)
            finished |= token.eq(EOS_ID)
            if finished.all():
                break

        if not generated:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)
        return torch.stack(generated, dim=1)
