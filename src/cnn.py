"""TextCNN for the CMPSC 448 midterm project ("Who Wrote It?").

Data spec (from the preprocessing step, arena-human-preference-140k):
    files   : train.csv / validation.csv / test.csv
    columns : prompt_id, LLM_Input, LLM_output, LLM_name, label
    labels  : GPT=0, Claude=1, Gemini=2

Model interface (shared with the RNN so both use the same training loop):
    input : LongTensor (batch, seq_len) of token ids, right-padded with pad_idx (=0)
    output: FloatTensor (batch, NUM_CLASSES) of logits (no softmax;
            nn.CrossEntropyLoss applies it during training)

Draft version: hyperparameter defaults are starting points to be tuned.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = {"GPT": 0, "Claude": 1, "Gemini": 2}
ID2LABEL = {v: k for k, v in LABELS.items()}
NUM_CLASSES = len(LABELS)


class TextCNN(nn.Module):
    """embedding -> Conv1d per kernel width -> ReLU -> max over time -> concat -> linear"""

    def __init__(self, vocab_size, num_classes=NUM_CLASSES, emb_dim=128,
                 kernel_sizes=(3, 4, 5), num_filters=100, dropout=0.5, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.kernel_sizes = kernel_sizes
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (B, L)
        # the widest filter needs at least max(kernel_sizes) tokens
        min_len = max(self.kernel_sizes)
        if x.size(1) < min_len:
            x = F.pad(x, (0, min_len - x.size(1)), value=self.pad_idx)

        lengths = (x != self.pad_idx).sum(dim=1)        # (B,) real tokens per row
        emb = self.embedding(x).transpose(1, 2)         # (B, D, L)

        pooled = []
        for k, conv in zip(self.kernel_sizes, self.convs):
            h = F.relu(conv(emb))                       # (B, F, L-k+1)
            # ignore windows that run into padding, so pad tokens can't win the max
            # (keep position 0 valid even for texts shorter than k)
            last_valid = (lengths - k).clamp(min=0)     # (B,)
            pos = torch.arange(h.size(2), device=x.device)
            valid = pos.unsqueeze(0) <= last_valid.unsqueeze(1)   # (B, L-k+1)
            h = h.masked_fill(~valid.unsqueeze(1), 0.0)
            pooled.append(h.max(dim=2).values)          # (B, F)

        feat = torch.cat(pooled, dim=1)                 # (B, F * len(kernel_sizes))
        return self.fc(self.dropout(feat))              # (B, NUM_CLASSES)


if __name__ == "__main__":
    # shape checks with fake data
    model = TextCNN(vocab_size=20000)
    x = torch.randint(1, 20000, (8, 512))
    x[0, 10:] = 0                                       # a short, padded example
    print(model(x).shape)                               # torch.Size([8, 3])
    print(model(torch.randint(1, 20000, (2, 3))).shape) # shorter than widest kernel -> (2, 3)
    print(sum(p.numel() for p in model.parameters()), "parameters")
