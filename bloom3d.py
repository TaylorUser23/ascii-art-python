#!/usr/bin/env python3
"""
bloom3d.py - a 3D flower that blooms and spins, drawn entirely in glowing ASCII.

How it works
  1. Geometry: every petal is a curved, cupped surface built from a (u, v) grid.
     u runs from the base of the petal to the tip, v runs edge to edge.
     Petals sit in 5 layers; the "bloom" value (0 = bud, 1 = open) tilts each
     layer outward and curls the tips back.
  2. Projection: points are spun, tilted toward the camera and perspective-
     projected onto a grid of character cells. A z-buffer keeps the nearest point.
  3. Shading: surface normals + a light give a brightness per cell. Brightness
     picks both the character (from RAMP) and the colour (dark magenta -> white).
  4. Glow: the text frame is blurred (shrink + enlarge) and added back on top.

Install:  pip install pygame numpy pillow
Run:      python bloom3d.py
Keys:     SPACE pause   W word mode   B freeze bloom   C colour theme
          +/- zoom      S screenshot  ESC quit
Export:   python bloom3d.py --gif bloom.gif        (one full bloom cycle)
"""
import argparse
import math
import os
import sys

import numpy as np

RAMP = ".,:;-~=+*ox%#&@"          # dim -> bright (no space: every hit cell draws)
WORD = "bloom"

# (petals, length, width, bud angle, open angle, base radius, base height)
LAYERS = [
    (3, 0.50, 0.42, 0.08, 0.30, 0.00, 0.06),
    (5, 0.68, 0.44, 0.16, 0.65, 0.03, 0.02),
    (6, 0.84, 0.47, 0.22, 0.98, 0.06, -0.02),
    (8, 1.00, 0.50, 0.30, 1.28, 0.09, -0.06),
    (9, 1.12, 0.52, 0.38, 1.55, 0.12, -0.10),
]
NU, NV = 44, 28                    # samples per petal (length x width)

THEMES = {
    "rose":   [(25, 0, 12), (120, 10, 60), (215, 40, 120), (255, 120, 185), (255, 215, 235), (255, 255, 255)],
    "orchid": [(15, 0, 30), (70, 20, 140), (150, 60, 220), (200, 130, 255), (235, 210, 255), (255, 255, 255)],
    "ember":  [(25, 3, 0), (130, 25, 0), (230, 80, 10), (255, 160, 40), (255, 225, 150), (255, 255, 240)],
    "matrix": [(0, 15, 0), (0, 80, 20), (0, 170, 50), (60, 255, 110), (190, 255, 200), (255, 255, 255)],
}
LEVELS = 32


# ----------------------------------------------------------------- geometry --
def petal_table():
    rows = []
    for li, (n, L, W, a0, a1, r0, z0) in enumerate(LAYERS):
        stagger = 0.5 * li * (2 * math.pi / n)
        for k in range(n):
            phi = stagger + 2 * math.pi * k / n + 0.12 * math.sin(k * 2.3 + li * 1.7)
            jitter = 1 + 0.06 * math.sin(k * 3.1 + li)
            rows.append((phi, L * jitter, W, a0, a1, r0, z0, li / (len(LAYERS) - 1)))
    return np.array(rows, dtype=np.float64)


def build_flower(table, bloom):
    """Return points (M,3), normals (M,3), ambient occlusion (M,) for a bloom value 0..1."""
    phi, L, W, a0, a1, r0, z0, layer = (table[:, i][:, None, None] for i in range(8))
    u = np.linspace(0.0, 1.0, NU)[None, :, None]
    v = np.linspace(-1.0, 1.0, NV)[None, None, :]

    theta = a0 + (a1 - a0) * bloom                     # how far the layer leans out
    curl = 0.15 + 1.1 * bloom * (0.3 + 0.7 * layer)    # tips roll back as it opens
    a = theta + curl * u ** 2                          # lean angle along the spine

    # integrate the spine so the curl is smooth
    ds = L / (NU - 1)
    spine_r = np.cumsum(np.sin(a), axis=1) * ds
    spine_z = np.cumsum(np.cos(a), axis=1) * ds
    spine_r -= spine_r[:, :1]
    spine_z -= spine_z[:, :1]

    w = W * np.sin(np.pi * u ** 0.62) ** 0.75           # narrow base, round tip
    cup = (0.6 - 0.35 * bloom) * v ** 2 * w             # edges fold inward
    ripple = 0.025 * np.sin(v * 5 + phi * 3) * u        # organic wobble

    radial = r0 + spine_r - cup * np.cos(a)
    up = z0 + spine_z + cup * np.sin(a) + ripple
    side = v * w

    cp, sp = np.cos(phi), np.sin(phi)
    x = radial * cp - side * sp
    y = radial * sp + side * cp
    z = np.broadcast_to(up, x.shape)

    P = np.stack(np.broadcast_arrays(x, y, z), axis=-1)       # (petals, NU, NV, 3)
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=2)
    N = np.cross(du, dv)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9

    ao = np.broadcast_to(0.35 + 0.65 * u ** 0.55, x.shape)    # darker deep inside
    return P.reshape(-1, 3), N.reshape(-1, 3), ao.reshape(-1)


# ---------------------------------------------------------------- rendering --
def project(P, N, AO, spin, tilt, cols, rows, scale_px, cw, ch):
    """Return per-cell brightness grid (rows, cols); -1 means empty."""
    cs, sn = math.cos(spin), math.sin(spin)
    x = P[:, 0] * cs - P[:, 1] * sn
    y = P[:, 0] * sn + P[:, 1] * cs
    z = P[:, 2] - 0.35                                  # centre the flower vertically
    nx = N[:, 0] * cs - N[:, 1] * sn
    ny = N[:, 0] * sn + N[:, 1] * cs
    nz = N[:, 2]

    ct, st = math.cos(tilt), math.sin(tilt)
    up = z * ct + y * st
    toward = -y * ct + z * st                           # +ve = closer to camera
    n_up = nz * ct + ny * st
    n_tw = -ny * ct + nz * st

    # petals are two-sided: flip normals that face away from the camera
    flip = np.where(n_tw < 0, -1.0, 1.0)
    nx, n_up, n_tw = nx * flip, n_up * flip, n_tw * flip

    light = np.array([-0.45, 0.65, 0.62]); light /= np.linalg.norm(light)
    half = light + np.array([0, 0, 1.0]); half /= np.linalg.norm(half)
    diff = np.clip(nx * light[0] + n_up * light[1] + n_tw * light[2], 0, 1)
    spec = np.clip(nx * half[0] + n_up * half[1] + n_tw * half[2], 0, 1) ** 24
    rim = (1 - np.clip(n_tw, 0, 1)) ** 3
    bright = np.clip(AO * (0.10 + 0.72 * diff) + 0.40 * spec + 0.18 * rim * AO, 0, 1)

    D = 5.0
    f = D / (D - toward)
    col = (cols / 2 + x * f * scale_px / cw).astype(np.int32)
    row = (rows / 2 - up * f * scale_px / ch).astype(np.int32)
    ok = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    idx = (row * cols + col)[ok]
    depth = toward[ok]
    b = bright[ok]

    order = np.argsort(-depth)                          # nearest first
    cells, first = np.unique(idx[order], return_index=True)
    grid = np.full(rows * cols, -1.0)
    grid[cells] = b[order[first]]
    return grid.reshape(rows, cols)


def palette(stops, levels=LEVELS):
    stops = np.array(stops, dtype=float)
    t = np.linspace(0, len(stops) - 1, levels)
    i = np.minimum(t.astype(int), len(stops) - 2)
    frac = (t - i)[:, None]
    return [tuple(int(c) for c in col) for col in stops[i] * (1 - frac) + stops[i + 1] * frac]


class Renderer:
    def __init__(self, pygame, font, cols, rows, theme="rose"):
        self.pg = pygame
        self.font = font
        self.cols, self.rows = cols, rows
        self.cw, self.ch = font.size("M")
        self.w, self.h = cols * self.cw, rows * self.ch
        self.cache = {}
        self.set_theme(theme)

    def set_theme(self, name):
        self.theme = name
        self.colors = palette(THEMES[name])
        self.cache.clear()

    def glyph(self, c, k):
        key = (c, k)
        g = self.cache.get(key)
        if g is None:
            g = self.font.render(c, True, self.colors[k])
            self.cache[key] = g
        return g

    def draw(self, grid, word_mode):
        pg = self.pg
        text = pg.Surface((self.w, self.h))
        rs, cs = np.nonzero(grid >= 0)
        vals = grid[rs, cs]
        ks = np.minimum((vals * LEVELS).astype(int), LEVELS - 1)
        ramp_i = np.minimum((vals * len(RAMP)).astype(int), len(RAMP) - 1)
        blits = []
        for r, c, k, ri in zip(rs.tolist(), cs.tolist(), ks.tolist(), ramp_i.tolist()):
            ch = WORD[c % len(WORD)] if word_mode else RAMP[ri]
            blits.append((self.glyph(ch, max(k, 3) if word_mode else k), (c * self.cw, r * self.ch)))
        text.blits(blits, doreturn=False)

        out = pg.Surface((self.w, self.h))
        out.fill((0, 0, 0))
        for div, passes in ((10, 2), (4, 1)):            # wide soft glow, then tight glow
            small = pg.transform.smoothscale(text, (self.w // div, self.h // div))
            glow = pg.transform.smoothscale(small, (self.w, self.h))
            for _ in range(passes):
                out.blit(glow, (0, 0), special_flags=pg.BLEND_RGB_ADD)
        out.blit(text, (0, 0), special_flags=pg.BLEND_RGB_ADD)
        return out


def bloom_at(t, period):
    """Bud -> open -> hold -> close, smoothly."""
    p = (t % period) / period
    s = 0.5 - 0.5 * math.cos(2 * math.pi * p)
    return 0.08 + 0.92 * s * s * (3 - 2 * s)


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description="ASCII 3D blooming flower")
    ap.add_argument("--cols", type=int, default=150)
    ap.add_argument("--rows", type=int, default=72)
    ap.add_argument("--font-size", type=int, default=11)
    ap.add_argument("--period", type=float, default=10.0, help="seconds per bloom cycle")
    ap.add_argument("--theme", choices=THEMES, default="rose")
    ap.add_argument("--word", action="store_true", help="draw with the word 'bloom' instead of the ramp")
    ap.add_argument("--gif", help="render one bloom cycle to this GIF and exit")
    ap.add_argument("--gif-fps", type=int, default=15)
    ap.add_argument("--gif-width", type=int, default=540)
    args = ap.parse_args()

    if args.gif:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    pygame.init()
    font = pygame.font.SysFont("consolas,dejavusansmono,menlo,couriernew,monospace", args.font_size, bold=True)
    R = Renderer(pygame, font, args.cols, args.rows, args.theme)
    table = petal_table()
    scale_px = min(R.w, R.h) * 0.36
    word_mode = args.word

    def frame(t, bloom):
        P, N, AO = build_flower(table, bloom)
        spin = t * 0.45
        tilt = 0.72 + 0.12 * math.sin(t * 0.35)
        grid = project(P, N, AO, spin, tilt, args.cols, args.rows, scale_px * zoom, R.cw, R.ch)
        return R.draw(grid, word_mode)

    zoom = 1.0
    if args.gif:
        from PIL import Image
        n = int(args.period * args.gif_fps)
        h = int(R.h * args.gif_width / R.w)
        imgs = []
        for i in range(n):
            t = i / args.gif_fps
            surf = frame(t, bloom_at(t, args.period))
            arr = pygame.surfarray.array3d(surf).swapaxes(0, 1)
            imgs.append(Image.fromarray(arr).resize((args.gif_width, h), Image.LANCZOS))
            print(f"\rframe {i + 1}/{n}", end="", flush=True)
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.gif_fps), loop=0, optimize=True)
        print(f"\nsaved {args.gif}")
        return

    screen = pygame.display.set_mode((R.w, R.h))
    pygame.display.set_caption("bloom3d.py")
    clock = pygame.time.Clock()
    t = 0.0
    bloom_t = 0.0
    paused = bloom_frozen = False
    shot = 0
    themes = list(THEMES)

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_w:
                    word_mode = not word_mode
                elif e.key == pygame.K_b:
                    bloom_frozen = not bloom_frozen
                elif e.key == pygame.K_c:
                    R.set_theme(themes[(themes.index(R.theme) + 1) % len(themes)])
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    zoom *= 1.1
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    zoom /= 1.1
                elif e.key == pygame.K_s:
                    shot += 1
                    pygame.image.save(screen, f"bloom_{shot:03d}.png")

        dt = clock.tick(30) / 1000.0
        if not paused:
            t += dt
            if not bloom_frozen:
                bloom_t += dt
        screen.blit(frame(t, bloom_at(bloom_t, args.period)), (0, 0))
        pygame.display.flip()


if __name__ == "__main__":
    main()
