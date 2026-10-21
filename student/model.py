"""Reflex-1 student: a small ViT over native 256x144 frames with four task heads.

Backbone: patch-16 ViT (Ti or S), 16x9 = 144 patches + 1 CLS token.
Heads read the CLS token:
  - box:     4 normalized edges (x0,y0,x1,y1), sigmoid, direct regression.
  - steer:   3-way  (left / straight / right).
  - throttle:3-way  (gas / brake / coast).
  - mask_next: 4 independent bits (the next-frame key bitmask, values 0..15).

Direct box regression, not DFL: the car is ~95 px wide at source res and the
gate is edge err < 5 px, so a 4-scalar head trained with L1 + (1 - IoU) is the
shortest path that clears the gate. ponytail: if edge err stalls above 5 px,
swap the box head for a DFL distribution over ~36 bins per edge (upgrade path).
"""

import math

import mlx.core as mx
import mlx.nn as nn

# Native frame the eye hands us, and the patch grid it makes.
IMG_H, IMG_W = 144, 256
PATCH = 16
GRID_H, GRID_W = IMG_H // PATCH, IMG_W // PATCH  # 9 x 16
N_PATCH = GRID_H * GRID_W                        # 144
# Source resolution the labels (box px) live in.
SRC_W, SRC_H = 512, 288

CONFIGS = {
    # name:   (width, depth, heads, mlp_ratio)
    "ti": (192, 12, 3, 4.0),
    "s":  (384, 12, 6, 4.0),
}


class Block(nn.Module):
    """Pre-norm transformer block with fused SDPA attention."""

    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x):
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, N, D)
        x = x + self.proj(attn)
        h = self.norm2(x)
        h = self.fc2(nn.gelu(self.fc1(h)))
        x = x + h
        return x


class Student(nn.Module):
    def __init__(self, size="ti"):
        super().__init__()
        width, depth, heads, mlp_ratio = CONFIGS[size]
        self.size = size
        self.width = width
        self.patch = nn.Conv2d(3, width, kernel_size=PATCH, stride=PATCH)
        self.cls = mx.zeros((1, 1, width))
        self.pos = mx.zeros((1, N_PATCH + 1, width))
        self.blocks = [Block(width, heads, mlp_ratio) for _ in range(depth)]
        self.norm = nn.LayerNorm(width)
        # Task heads off the CLS token.
        self.box_head = nn.Linear(width, 4)
        self.steer_head = nn.Linear(width, 3)
        self.throttle_head = nn.Linear(width, 3)
        self.mask_head = nn.Linear(width, 4)

    def backbone(self, x):
        """Run the tower and return the full token sequence (B, N+1, width).

        Row 0 is the CLS token; rows 1..N are the 9x16 patch grid in row-major
        order. Stage 2 (Laya) projects the patch rows into its state sequence,
        so the grid is a first-class output, not only CLS. No extra params.
        """
        B = x.shape[0]
        h = self.patch(x)                       # B, GRID_H, GRID_W, width
        h = h.reshape(B, N_PATCH, self.width)
        cls = mx.broadcast_to(self.cls, (B, 1, self.width))
        h = mx.concatenate([cls, h], axis=1)    # B, N+1, width
        h = h + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return h                                # B, N+1, width

    def cls_feat(self, x):
        return self.backbone(x)[:, 0]           # B x width

    def patch_tokens(self, x):
        """The 9x16 patch grid for stage-2 projection: (B, N_PATCH, width)."""
        return self.backbone(x)[:, 1:]

    def __call__(self, x, return_tokens=False):
        seq = self.backbone(x)
        feat = seq[:, 0]
        box = mx.sigmoid(self.box_head(feat))   # normalized x0,y0,x1,y1 in [0,1]
        out = {
            "box": box,
            "steer": self.steer_head(feat),
            "throttle": self.throttle_head(feat),
            "mask": self.mask_head(feat),       # logits for 4 bits
        }
        if return_tokens:
            out["tokens"] = seq[:, 1:]          # B, N_PATCH, width
            out["cls"] = feat
        return out


def param_count(model):
    def walk(v):
        if isinstance(v, mx.array):
            return v.size
        if isinstance(v, dict):
            return sum(walk(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return sum(walk(x) for x in v)
        return 0
    return walk(model.parameters())


def init_pos_sincos(model):
    """Give the position embedding a sane 2D sin-cos start (CLS row stays zero)."""
    width = model.width
    pos = _sincos_2d(GRID_H, GRID_W, width)
    full = mx.concatenate([mx.zeros((1, width)), pos], axis=0)
    model.pos = full[None]


def _sincos_2d(gh, gw, dim):
    assert dim % 4 == 0
    d4 = dim // 4
    omega = mx.exp(-math.log(10000.0) * mx.arange(d4) / d4)
    ys = mx.arange(gh).reshape(gh, 1, 1)
    xs = mx.arange(gw).reshape(1, gw, 1)
    oy = ys * omega.reshape(1, 1, d4)
    ox = xs * omega.reshape(1, 1, d4)
    oy = mx.broadcast_to(oy, (gh, gw, d4))
    ox = mx.broadcast_to(ox, (gh, gw, d4))
    emb = mx.concatenate([mx.sin(oy), mx.cos(oy), mx.sin(ox), mx.cos(ox)], axis=-1)
    return emb.reshape(gh * gw, dim)


if __name__ == "__main__":
    for sz in ("ti", "s"):
        m = Student(sz)
        init_pos_sincos(m)
        n = param_count(m)
        out = m(mx.zeros((1, IMG_H, IMG_W, 3)))
        mx.eval(out)
        print(f"ViT-{sz.upper()}: {n/1e6:.2f}M params, box {out['box'].shape}, "
              f"steer {out['steer'].shape}, mask {out['mask'].shape}")
