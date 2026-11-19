# arch/text_adapter.py
# Generated using Claude Sonnet 5.5
import torch, torch.nn as nn, torch.nn.functional as F

NORMAL = ["a photo of a flawless {}.", "a photo of a perfect {}.",
          "a photo of an unblemished {}.", "a photo of a {} without defect."]
ANOMALOUS = ["a photo of a damaged {}.", "a photo of a broken {}.",
             "a photo of a {} with a defect.", "a photo of a {} with flaws."]

@torch.no_grad()
def encode_prompts(clip_model, tokenizer, category, device):
    """Frozen [2, 512]: row 0 = normal, row 1 = anomalous. Cache per category."""
    rows = []
    for templates in (NORMAL, ANOMALOUS):
        tok = tokenizer([t.format(category) for t in templates]).to(device)
        e = F.normalize(clip_model.encode_text(tok).float(), dim=-1)
        rows.append(F.normalize(e.mean(0), dim=-1))
    return torch.stack(rows)

class QueryConditionedTextAdapter(nn.Module):
    def __init__(self, vis_dim=768, text_dim=512, llm_dim=2048, hidden=512):
        super().__init__()
        self.text_dim = text_dim
        self.ctx = nn.Sequential(
            nn.LayerNorm(3 * vis_dim), nn.Linear(3 * vis_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * text_dim))
        nn.init.zeros_(self.ctx[-1].weight); nn.init.zeros_(self.ctx[-1].bias)
        self.to_llm = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, llm_dim))
        self.slot = nn.Parameter(torch.zeros(2, llm_dim))  # normal / anomalous slot ids

    def forward(self, xq, dx, text):
        # xq, dx: [B, N, C]; text: [B, 2, D] frozen
        g = torch.cat([xq.mean(1), dx.mean(1), dx.amax(1)], dim=-1)
        delta = self.ctx(g).view(-1, 2, self.text_dim)
        adapted = F.normalize(text + delta, dim=-1)               # [B, 2, D]
        return adapted, self.to_llm(adapted) + self.slot          # tokens: [B, 2, llm_dim]

def align_loss(pooled_proj, text, labels, scale=100.0):
    """pooled_proj [B, D] (CLIP pooled embedding, or learned DINOv2 projection)."""
    logits = scale * (F.normalize(pooled_proj, dim=-1).unsqueeze(1) * text).sum(-1)  # [B, 2]
    return F.cross_entropy(logits, labels)