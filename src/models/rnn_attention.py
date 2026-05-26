"""GRU encoder-decoder with manually implemented Bahdanau attention."""

from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.preprocessing import EOS_ID, PAD_ID, SOS_ID


class BahdanauAttention(nn.Module):
    """Additive attention from Bahdanau et al. implemented with basic layers."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.decoder_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.encoder_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy = nn.Linear(hidden_size, 1, bias=False)

    # 这里手写 Bahdanau additive attention, 不调用任何现成 attention 模块。
    # 1) score: 对每个源位置 i, 把上一时刻 decoder hidden s_{t-1}
    #    和 encoder hidden h_i 分别做线性投影, 相加后过 tanh, 再用 v^T
    #    压成一个标量分数。decoder_hidden 原本是 (B,H), 需要 unsqueeze
    #    成 (B,1,H), 这样可以广播到所有源时间步 (B,T,H)。
    # 2) softmax: padding 位置不是实际 ingredient token, 不能参与归一化。
    #    因此先用 src_mask 把 padding 的 score 设成一个极小值, 再沿源
    #    序列维度做 softmax, 得到每个目标生成步对所有 ingredient token
    #    的注意力分布 alpha_{t,i}。
    # 3) context: attention_weights 是 (B,T), encoder_outputs 是 (B,T,H)。
    #    通过 batch matrix multiplication 做加权求和, 得到当前步上下文
    #    context_t = sum_i alpha_{t,i} h_i, shape 为 (B,H)。这个 context
    #    会与当前目标 token embedding 拼接后送入 decoder GRU。
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return context vector and attention weights for one decoder step."""
        projected_decoder = self.decoder_proj(decoder_hidden).unsqueeze(1)
        projected_encoder = self.encoder_proj(encoder_outputs)
        scores = self.energy(torch.tanh(projected_decoder + projected_encoder)).squeeze(-1)
        scores = scores.masked_fill(~src_mask, torch.finfo(scores.dtype).min)

        attention_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights


class Encoder(nn.Module):
    """Encode ingredient IDs into all hidden states plus the final state."""

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

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(encoder_outputs, final_hidden)``."""
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded,
            src_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src.size(1),
        )
        return outputs, hidden


class AttentionDecoder(nn.Module):
    """GRU decoder that attends over encoder outputs at every step."""

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
        self.attention = BahdanauAttention(hidden_size)
        self.rnn = nn.GRU(
            embed_size + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def step(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode one step and return ``(logits, next_hidden, weights)``."""
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))
        decoder_hidden = hidden[-1]
        context, attention_weights = self.attention(decoder_hidden, encoder_outputs, src_mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, next_hidden = self.rnn(rnn_input, hidden)
        logits = self.output(output.squeeze(1))
        return logits, next_hidden, attention_weights

    def forward(
        self,
        tgt_input: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a teacher-forced target sequence and return logits + weights."""
        logits_steps = []
        attention_steps = []
        current_hidden = hidden

        # 逐步循环是为了每个 target token 都能保存一份可视化用的 attention
        for t in range(tgt_input.size(1)):
            logits, current_hidden, attention_weights = self.step(
                tgt_input[:, t],
                current_hidden,
                encoder_outputs,
                src_mask,
            )
            logits_steps.append(logits)
            attention_steps.append(attention_weights)

        logits_all = torch.stack(logits_steps, dim=1)
        attention_all = torch.stack(attention_steps, dim=1)
        return logits_all, attention_all


class Seq2SeqAttention(nn.Module):
    """A GRU encoder-decoder with manually implemented Bahdanau attention."""

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
        self.decoder = AttentionDecoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.vocab_size = vocab_size

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Return True for real source tokens and False for padding."""
        return src.ne(PAD_ID)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run teacher-forced training forward pass."""
        if src_lengths is None:
            src_lengths = (src != PAD_ID).sum(dim=1).clamp_min(1)

        src_mask = self.make_src_mask(src)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        decoder_input = tgt[:, :-1]
        logits, attention_weights = self.decoder(decoder_input, hidden, encoder_outputs, src_mask)

        if return_attention:
            return logits, attention_weights
        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        max_len: int = 80,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Greedily generate token IDs and optionally attention weights."""
        self.eval()
        if src_lengths is None:
            src_lengths = (src != PAD_ID).sum(dim=1).clamp_min(1)

        src_mask = self.make_src_mask(src)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        batch_size = src.size(0)
        device = src.device
        token = torch.full((batch_size,), SOS_ID, dtype=torch.long, device=device)
        generated = []
        attention_steps = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits, hidden, attention_weights = self.decoder.step(
                token,
                hidden,
                encoder_outputs,
                src_mask,
            )
            token = logits.argmax(dim=-1)
            generated.append(token)
            attention_steps.append(attention_weights)
            finished |= token.eq(EOS_ID)
            if finished.all():
                break

        if generated:
            generated_ids = torch.stack(generated, dim=1)
            attention_all = torch.stack(attention_steps, dim=1)
        else:
            generated_ids = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            attention_all = torch.empty(batch_size, 0, src.size(1), device=device)

        if return_attention:
            return generated_ids, attention_all
        return generated_ids
