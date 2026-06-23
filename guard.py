"""HalluGuard as a drop-in faithfulness guardrail.

Wraps the trained classifier behind one function you can put on top of any LLM:
give it the source the model was supposed to use and the answer it produced, and
it returns whether the answer is supported, a 0..1 hallucination score, and which
answer words are missing from the source (a cheap, readable explanation).

    from guard import Guard
    g = Guard.load()                       # out/halluguard.pt + data/tokenizer.json
    g.check(source, answer)                # -> {"supported": bool, "score": float, ...}

Also runs as a tiny HTTP service:

    uvicorn guard:app --port 8000
    curl -s localhost:8000/check -H 'content-type: application/json' \
         -d '{"source": "...", "claim": "..."}'
"""

import os
import re

import torch

import chat
import hallu_data as hd
from bpe import BPETokenizer
from classifier import GPTClassifier
from model import GPTConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
_WORD = re.compile(r"[a-z]+")


def unsupported_words(source, claim):
    """Content words in the claim that never appear in the source."""
    src = set(_WORD.findall(source.lower()))
    seen, out = set(), []
    for w in _WORD.findall(claim.lower()):
        if len(w) >= 4 and w not in src and w not in seen:
            out.append(w)
            seen.add(w)
    return out


class Guard:
    def __init__(self, model, tok, block_size, device):
        self.model = model
        self.tok = tok
        self.block_size = block_size
        self.device = device
        self.pad_id = tok.special_tokens[chat.EOT]

    @classmethod
    def load(cls, ckpt=None, tokenizer=None, device=None):
        ckpt = ckpt or os.path.join(OUT_DIR, "halluguard.pt")
        tokenizer = tokenizer or os.path.join(DATA_DIR, "tokenizer.json")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = BPETokenizer().load(tokenizer)
        ck = torch.load(ckpt, map_location=device)
        cfg = GPTConfig(**ck["config"])
        model = GPTClassifier(cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        return cls(model, tok, ck["block_size"], device)

    @torch.no_grad()
    def check(self, source, claim):
        X, L, _ = hd.make_batch(
            self.tok, [{"source": source, "claim": claim, "label": 0}],
            self.block_size, self.pad_id, self.device,
        )
        logits, _ = self.model(X, L)
        score = torch.softmax(logits.float(), -1)[0, 1].item()   # P(hallucinated)
        return {
            "supported": score < 0.5,
            "score": round(score, 4),
            "verdict": "supported" if score < 0.5 else "unsupported",
            "unsupported_words": unsupported_words(source, claim),
        }


# --- optional HTTP service (only if FastAPI is installed) --------------------
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    class _Req(BaseModel):
        source: str
        claim: str

    app = FastAPI(title="HalluGuard")
    _guard = {"g": None}

    def _get():
        if _guard["g"] is None:
            _guard["g"] = Guard.load()
        return _guard["g"]

    @app.post("/check")
    def _check(req: _Req):
        return _get().check(req.source, req.claim)
except ImportError:
    app = None


if __name__ == "__main__":
    g = Guard.load()
    src = "The Eiffel Tower is in Paris and was completed in 1889."
    for claim in ["It was completed in 1889.", "It is located in Berlin."]:
        print(claim, "->", g.check(src, claim))
