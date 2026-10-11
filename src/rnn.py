"""BiLSTM for CMPSC 448 LLM-family classification.

Uses the same data.py loaders and model(x) interface as TextCNN.
Input: right-padded LongTensor (batch, seq_len), pad_idx=0.
Output: FloatTensor (batch, 3) logits; use nn.CrossEntropyLoss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

LABELS = {"GPT": 0, "Claude": 1, "Gemini": 2}
ID2LABEL = {v: k for k, v in LABELS.items()}
NUM_CLASSES = len(LABELS)


class TextRNN(nn.Module):
    """Embedding -> bidirectional LSTM -> final hidden states -> dropout -> linear."""

    def __init__(self, vocab_size, num_classes=NUM_CLASSES, emb_dim=128,
                 hidden_dim=128, num_layers=1, dropout=0.5, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # data.py right-pads with 0, so callers do not need a lengths argument.
        if x.size(1) == 0:
            x = F.pad(x, (0, 1), value=self.pad_idx)
        lengths = (x != self.pad_idx).sum(dim=1)
        # Packing requires positive lengths. Empty rows get one zero embedding.
        packed = pack_padded_sequence(
            self.embedding(x), lengths.clamp(min=1).cpu(),
            batch_first=True, enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        # Last layer's forward and backward states, in original batch order.
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        features = features.masked_fill((lengths == 0).unsqueeze(1), 0.0)
        return self.fc(self.dropout(features))


if __name__ == "__main__":
    model = TextRNN(vocab_size=20000)
    x = torch.randint(1, 20000, (8, 512))
    x[0, 10:] = 0
    print(model(x).shape)                              # torch.Size([8, 3])
    print(model(torch.randint(1, 20000, (2, 3))).shape) # torch.Size([2, 3])
    print(model(torch.zeros(2, 512, dtype=torch.long)).shape)
    print(sum(p.numel() for p in model.parameters()), "parameters")
