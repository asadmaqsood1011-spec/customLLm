"""Turn the from-scratch transformer into a sequence classifier.

Reuses the pretrained language-model backbone (token embeddings + transformer
blocks) and attaches a small classification head on top of the last token's
hidden state. This is transfer learning: pretraining learned general language
representations; the head learns the faithfulness decision.

Input format:  <|user|> {source} <|assistant|> {claim}
Output:        logits over {0: supported, 1: hallucinated}
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPT, GPTConfig


class GPTClassifier(nn.Module):
    def __init__(self, cfg: GPTConfig, n_classes=2):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        self.backbone = GPT(cfg)
        self.cls_head = nn.Linear(cfg.n_embd, n_classes)

    def load_backbone(self, state_dict):
        # load LM weights; ignore the (tied) LM head, keep our random cls head
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        return missing, unexpected

    def forward(self, idx, lengths, labels=None):
        feats = self.backbone.features(idx)                 # (B, T, C)
        last = feats[torch.arange(idx.size(0)), lengths - 1]  # last real token
        logits = self.cls_head(last)                        # (B, n_classes)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return logits, loss
