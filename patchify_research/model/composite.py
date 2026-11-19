# patchify_research/model/composite.py  (reuses _multiscale_embeddings from patch_embeddings.py)
# Generated using Claude Sonnet 5.5
import torch
from patch_embeddings import _multiscale_embeddings

@torch.no_grad()
def composite_features(query_img, ref_imgs, model, preprocess,
                       scales=(224, 112, 64), weights=None):
    """
    query_img: PIL image; ref_imgs: list of K PIL images (K may be 0).
    Returns xq [N, C], xc [N, C], dx [N, C], best_sim [N]
    """
    visual, device = model.visual, next(model.parameters()).device
    q = preprocess(query_img.convert("RGB"))
    refs = [preprocess(r.convert("RGB")) for r in ref_imgs]
    imgs = torch.stack([q] + refs).to(device)                 # [1+K, 3, H, W]

    _, tokens = visual(imgs)                                  # [1+K, N, C] original patch features
    xq = tokens[0]
    if not refs:                                              # K = 0: dx = 0
        return xq, torch.zeros_like(xq), torch.zeros_like(xq), torch.zeros(xq.shape[0], device=device)

    match = _multiscale_embeddings(visual, imgs, scales, weights)   # [1+K, N, D], L2-normalized
    K, N, _ = match[1:].shape
    sim = torch.einsum("nd,kmd->nkm", match[0], match[1:]).reshape(N, K * N)
    best_sim, best = sim.max(dim=-1)                          # best over all refs and positions
    xc = tokens[1:].reshape(K * N, -1)[best]                  # gather ORIGINAL tokens
    return xq, xc, (xq - xc).abs(), best_sim