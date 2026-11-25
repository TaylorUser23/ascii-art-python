#!/usr/bin/env python3
"""
skyline_sunrise.py - a city of skyscrapers watching the sun come up, all in glowing ASCII.

The scene runs a full day loop: night -> sunrise -> morning -> sunset -> night.
  * Sky: colour keyframes driven by the sun's elevation, plus sun glow and a warm
    horizon band at dawn.
  * Clouds: tileable value noise, stretched sideways so it streaks, drifting
    slowly and lit pink/orange from underneath at sunrise.
  * Stars that twinkle and fade out as the sky brightens.
  * Three layers of procedurally generated skyscrapers (stepped tops,
    antennas with blinking beacons, spires). Distant layers fade into the haze.
  * Every window has its own "bedtime" threshold, so lights go out one by one
    as morning arrives. Building edges facing the sun pick up a rim of light.
  * A river mirrors everything with rippling distortion plus sun glitter.
  * Birds cross the morning sky.

Rendering trick: each frame is two numpy arrays, one of characters and one of
colours. Characters are blitted in white, then multiplied by the colour image, so
every cell can be any colour without caching thousands of coloured glyphs.

Install:  pip install pygame numpy pillow
Run:      python skyline_sunrise.py
Keys:     SPACE pause   LEFT/RIGHT scrub time   UP/DOWN speed
          N new city    S screenshot           ESC quit
Export:   python skyline_sunrise.py --gif sunrise.gif            (night -> morning)
          python skyline_sunrise.py --frames out_dir             (PNG frames for ffmpeg)
"""
import argparse
import math
import os
import sys

import numpy as np

# ------------------------------------------------------------------ chars --
CHARSET = " .,:;-=+*#%@~|/\\v^'"
CI = {c: i for i, c in enumerate(CHARSET)}
SKY_RAMP = np.array([CI[c] for c in " .,:;-=+*#"])
WALL_CHARS = [CI["."], CI[":"], CI[":"]]          # far, mid, near

NONE, WALL, WIN, EDGE_L, EDGE_R, ANT, BEACON, ROOF, SL, SR = range(10)

# sky keyframes: sun elevation -> (top, middle, horizon) colours
SKY_KEYS = [
    (-0.30, (2, 3, 12), (6, 8, 28), (14, 14, 40)),
    (-0.12, (6, 8, 32), (28, 22, 66), (80, 40, 90)),
    (-0.03, (18, 24, 75), (120, 55, 115), (235, 100, 70)),
    (0.05, (35, 55, 130), (205, 100, 125), (255, 155, 65)),
    (0.20, (55, 100, 190), (190, 150, 175), (255, 205, 140)),
    (0.45, (45, 120, 225), (120, 175, 235), (200, 225, 245)),
    (0.80, (40, 115, 225), (110, 170, 240), (185, 215, 245)),
]


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    return a + (b - a) * t


def sky_palette(e):
    keys = SKY_KEYS
    if e <= keys[0][0]:
        k = keys[0]
        return [np.array(c, float) for c in k[1:]]
    for k0, k1 in zip(keys, keys[1:]):
        if e <= k1[0]:
            t = (e - k0[0]) / (k1[0] - k0[0])
            t = t * t * (3 - 2 * t)
            return [mix(np.array(a, float), np.array(b, float), t) for a, b in zip(k0[1:], k1[1:])]
    return [np.array(c, float) for c in keys[-1][1:]]


def value_noise(h, w, cy, cx, rng, octaves=3):
    """Value noise that tiles horizontally (w must be a multiple of cx)."""
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        gh, gw = h // cy + 2, w // cx
        g = rng.random((gh, gw))
        y = np.arange(h) / cy
        x = np.arange(w) / cx
        y0, x0 = y.astype(int), x.astype(int)
        fy, fx = y - y0, x - x0
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        x1, x0 = (x0 + 1) % gw, x0 % gw
        a, b = g[y0][:, x0], g[y0][:, x1]
        c, d = g[y0 + 1][:, x0], g[y0 + 1][:, x1]
        top, bot = a + (b - a) * fx, c + (d - c) * fx
        out += amp * (top + (bot - top) * fy[:, None])
        total += amp
        amp *= 0.5
        cy, cx = max(1, cy // 2), max(1, cx // 2)
    return out / total


# ------------------------------------------------------------------- city --
def add_building(role, thr, x0, w, top, style, rng):
    H, cols = role.shape
    top = max(4, top)
    xa, xb = max(0, x0), min(cols, x0 + w)
    if xb - xa < 1:
        return
    role[top:H, xa:xb] = WALL
    if 0 <= x0 < cols:
        role[top:H, x0] = EDGE_L
    if 0 <= x0 + w - 1 < cols:
        role[top:H, x0 + w - 1] = EDGE_R
    role[top, xa:xb] = ROOF

    px = int(rng.choice([2, 3]))
    rr = np.arange(top + 2, H - 1, 2)
    cc = np.arange(x0 + 2, x0 + w - 2, px)
    cc = cc[(cc >= 0) & (cc < cols)]
    if len(rr) and len(cc):
        floor_bias = rng.random(len(rr))[:, None] * 0.35       # whole floors go dark together
        role[np.ix_(rr, cc)] = WIN
        thr[np.ix_(rr, cc)] = np.clip(rng.random((len(rr), len(cc))) * 0.75 + floor_bias, 0, 1)

    xc = x0 + w // 2
    if style == "stepped" and w >= 6:
        nw = max(3, int(w * rng.uniform(0.5, 0.7)))
        add_building(role, thr, x0 + (w - nw) // 2, nw, top - int(rng.integers(3, 8)),
                     rng.choice(["flat", "antenna"]), rng)
    elif style == "antenna" and 0 <= xc < cols:
        k = int(rng.integers(3, 8))
        role[max(0, top - k):top, xc] = ANT
        if top - k - 1 >= 0:
            role[top - k - 1, xc] = BEACON
    elif style == "spire":
        half = w // 2
        for i in range(1, half + 1):
            r, a, b = top - i, x0 + i, x0 + w - 1 - i
            if r < 1 or a > b:
                break
            role[r, max(0, a):min(cols, b + 1)] = WALL
            if 0 <= a < cols:
                role[r, a] = SL
            if 0 <= b < cols:
                role[r, b] = SR
        peak = top - half - 1
        if 0 <= xc < cols and peak > 5:
            role[peak - 4:peak + 1, xc] = ANT
            role[peak - 5, xc] = BEACON


def make_city(cols, H, rng, gap=(0.0, 0.0)):
    """gap: fraction-of-width range kept open (only low, distant buildings) so the sunrise shows."""
    s = max(0.6, cols / 200)
    ga, gb = int(gap[0] * cols), int(gap[1] * cols)
    specs = [  # wmin wmax  hmin  hmax  gap  fog
        (4, 10, 0.16, 0.36, 1, 0.58),
        (6, 14, 0.14, 0.44, 2, 0.30),
        (8, 18, 0.10, 0.40, 4, 0.05),
    ]
    layers = []
    for li, (wmin, wmax, hmin, hmax, gmax, fog) in enumerate(specs):
        role = np.zeros((H, cols), np.int8)
        thr = np.zeros((H, cols))
        x = -int(rng.integers(0, 6))
        while x < cols:
            w = max(3, int(rng.integers(wmin, wmax + 1) * s))
            h = rng.uniform(hmin, hmax)
            style = rng.choice(["flat", "stepped", "antenna", "spire"], p=[0.38, 0.27, 0.27, 0.08])
            in_gap = x + w > ga and x < gb
            if in_gap and li > 0:
                x = gb + int(rng.integers(0, 3))
                continue
            if in_gap:
                h, style = rng.uniform(0.05, 0.13), "flat"
            if li == 2 and rng.random() < 0.16:                   # landmark towers
                h, style = rng.uniform(0.52, 0.66), rng.choice(["spire", "antenna", "stepped"])
            add_building(role, thr, x, w, int(H * (1 - h)), style, rng)
            x += w + int(rng.integers(0, gmax + 1) * s)
        layers.append(dict(role=role, thr=thr, fog=fog, depth=li))
    return layers


# ------------------------------------------------------------------ scene --
class Scene:
    def __init__(self, cols, rows, cw, ch, seed=None):
        self.cols, self.rows, self.cw, self.ch = cols, rows, cw, ch
        self.H = int(rows * 0.70)                    # horizon row: sky above, river below
        self.reseed(seed)

    def reseed(self, seed=None):
        rng = np.random.default_rng(seed)
        cols, H = self.cols, self.H
        p_rise = 0.25 - math.asin(0.4) / (2 * math.pi)           # where elevation crosses 0
        rise_x = self.sun_path(p_rise)[1]
        self.layers = make_city(cols, H, rng, gap=(rise_x - 0.08, rise_x + 0.08))
        self.nw = int(math.ceil(cols * 3 / 64) * 64)
        self.noise = value_noise(H, self.nw, 4, 32, rng, octaves=3)
        self.star = (rng.random((H, cols)) < 0.018) * rng.uniform(0.3, 1.0, (H, cols))
        self.star_phase = rng.uniform(0, 2 * np.pi, (H, cols))
        self.star_rate = rng.uniform(1.0, 4.0, (H, cols))
        self.rr = np.arange(H)[:, None].astype(float)
        self.cc = np.arange(cols)[None, :].astype(float)
        self.h = (1 - (np.arange(H) + 0.5) / H)[:, None]   # 1 at the top, 0 at the horizon
        self.birds = [(rng.uniform(0, 30), rng.uniform(-3, 3), rng.uniform(0, 6)) for _ in range(7)]

    @staticmethod
    def sun_path(p):
        e = 0.5 * math.sin(2 * math.pi * (p - 0.25)) + 0.2        # elevation, fraction of sky
        x = 0.5 + 0.24 * math.sin(2 * math.pi * (p - 0.5))        # rises left of centre, sets right
        return e, x

    def frame(self, p, t):
        cols, H, rows, cw, ch = self.cols, self.H, self.rows, self.cw, self.ch
        rr, cc, h = self.rr, self.cc, self.h
        e, sxf = self.sun_path(p)
        sx, sy = sxf * cols, H - 1 - e * H

        dl = float(smoothstep(-0.08, 0.35, e))                    # daylight
        night = 1 - float(smoothstep(-0.12, 0.02, e))
        dawn = math.exp(-((e - 0.02) / 0.13) ** 2)                 # peaks at sunrise
        sun_col = mix(np.array([255, 105, 35.]), np.array([255, 245, 225.]), float(smoothstep(-0.05, 0.45, e)))

        # ---- sky gradient
        top, mid, hor = sky_palette(e)
        hh = h[:, :, None]
        grad = np.where(hh < 0.35, mix(hor, mid, hh / 0.35), mix(mid, top, (hh - 0.35) / 0.65))
        sky = np.broadcast_to(grad, (H, cols, 3)).copy()

        # ---- sun glow + warm horizon band
        dxp, dyp = (cc - sx) * cw, (rr - sy) * ch
        d = np.sqrt(dxp ** 2 + dyp ** 2) / (H * ch)
        vis = float(smoothstep(-0.25, 0.02, e))
        glow = (0.9 * np.exp(-(d / 0.11) ** 2) + (0.25 + 0.5 * dawn) * np.exp(-d / 0.28)) * vis
        sky += sun_col * glow[..., None]
        band = dawn * 0.85 * np.exp(-h / 0.16) * np.exp(-np.abs(cc - sx) / (cols * 0.55))
        sky += np.array([255, 95, 45.]) * band[..., None]

        # ---- clouds
        off = int(t * 1.2) % self.nw
        n = self.noise[:, (np.arange(cols) + off) % self.nw]
        dens = smoothstep(0.56, 0.78, n) * smoothstep(0.32, 0.48, h) * (1 - smoothstep(0.82, 0.96, h))
        cloud = mix(np.array([28, 26, 48.]), np.array([255, 135, 120.]), float(smoothstep(-0.16, 0.0, e)))
        cloud = mix(cloud, np.array([238, 236, 245.]), float(smoothstep(0.1, 0.4, e)))
        cloud = cloud + sun_col * (glow * 0.6)[..., None]
        sky = mix(sky, cloud, (dens * 0.85)[..., None])
        sky_bg = sky.copy()

        lum = (sky @ np.array([0.30, 0.59, 0.11])) / 255
        chars = SKY_RAMP[np.minimum((np.clip(lum, 0, 1) ** 0.85 * len(SKY_RAMP)).astype(int), len(SKY_RAMP) - 1)]
        chars = np.where(dens > 0.3, CI["~"], chars)

        # ---- stars
        sb = self.star * night * (0.55 + 0.45 * np.sin(t * self.star_rate + self.star_phase)) * (1 - dens)
        sm = sb > 0.12
        chars = np.where(sm, np.where(sb > 0.75, CI["*"], np.where(sb > 0.4, CI["+"], CI["."])), chars)
        sky = np.where(sm[..., None], sky + np.array([210, 220, 255.]) * sb[..., None], sky)

        # ---- sun disc
        dpx = np.sqrt(dxp ** 2 + dyp ** 2)
        R = 0.075 * H * ch
        disc = dpx < R
        if disc.any():
            core = mix(np.array([255, 255, 240.]), sun_col, (dpx / R)[..., None] ** 2)
            sky = np.where(disc[..., None], mix(core, cloud, (dens * 0.5)[..., None]), sky)
            chars = np.where(disc, np.where(dpx < R * 0.65, CI["@"], CI["#"]), chars)

        # ---- birds (morning)
        if dl > 0.25:
            for i, (ox, oy, ph) in enumerate(self.birds):
                bx = int((t * 6 + ox * 2) % (cols + 60)) - 30 + (i % 3) * 3
                by = int(H * 0.28 + oy + 1.2 * math.sin(t * 0.8 + ph) + (i % 4))
                if 0 <= bx < cols and 0 <= by < H:
                    chars[by, bx] = CI["v"] if math.sin(t * 9 + ph * 3) > 0 else CI["-"]
                    sky[by, bx] = (25, 25, 35)

        # ---- buildings, far to near
        base_wall = mix(np.array([10, 10, 22.]), np.array([72, 78, 98.]), dl)
        rim_k = (0.3 + 0.7 * dawn) * float(smoothstep(-0.06, 0.03, e))
        lit_cut = 0.32 + 0.66 * dl
        for L in self.layers:
            role, thr, fog, depth = L["role"], L["thr"], L["fog"], L["depth"]
            m = role > 0
            wall = base_wall * (0.55 if depth == 2 else 1.0)
            col = mix(wall, sky_bg, fog)
            rim = rim_k * (1 - fog)
            facing = ((role == EDGE_L) & (cc > sx)) | ((role == EDGE_R) & (cc < sx))
            col = col + sun_col * (rim * facing)[..., None]
            col = col + sun_col * (0.6 * rim * (role == ROOF))[..., None]

            win = role == WIN
            lit = win & (thr > lit_cut)
            warm = np.where((thr * 37) % 1 < 0.72, 1.0, 0.0)[..., None]
            lamp = mix(np.array([185, 210, 255.]), np.array([255, 185, 80.]), warm)
            lamp = lamp * (0.55 + 0.45 * ((thr * 13) % 1))[..., None] * (0.5, 0.75, 1.0)[depth]
            unlit = mix(wall * 1.5, sky_bg * 0.75, 0.25 + 0.3 * dl)
            glint = (dawn * (((thr * 97 + math.floor(t * 1.5) * 0.37) % 1) > 0.93))[..., None]
            unlit = unlit + sun_col * glint * 0.55 * (1 - fog)
            col = np.where(lit[..., None], lamp, np.where(win[..., None], unlit, col))

            blink = (np.sin(t * 4 + cc * 0.7) > 0.2) & (role == BEACON)
            beacon = np.array([255, 45, 40.]) * (0.35 + 0.65 * night)
            col = np.where((role == BEACON)[..., None], np.where(blink[..., None], beacon, wall), col)

            lch = np.full(role.shape, WALL_CHARS[depth])
            lch = np.where(lit, CI["#"], np.where(win, CI["."], lch))
            lch = np.where((role == EDGE_L) | (role == EDGE_R) | (role == ANT), CI["|"], lch)
            lch = np.where(role == ROOF, CI["="], lch)
            lch = np.where(role == SL, CI["/"], np.where(role == SR, CI["\\"], lch))
            lch = np.where(role == BEACON, np.where(blink, CI["*"], CI["."]), lch)

            chars = np.where(m, lch, chars)
            sky = np.where(m[..., None], col, sky)

        # ---- river reflection
        Wr = rows - H
        D = np.arange(Wr)[:, None].astype(float)
        src = np.clip(H - 1 - (D[:, 0] * 1.3).astype(int), 0, H - 1)
        shift = (np.sin(t * 1.6 + D * 0.9) * (0.6 + D * 0.12) + np.sin(t * 0.7 + D * 2.3) * 0.5
                 + 0.8 * np.sin(cc * 0.35 + t * 2 + D))
        idx = (cc + np.round(shift)).astype(int) % cols
        refl = sky[src[:, None], idx]
        fade = (0.82 * (1 - 0.4 * D / Wr))[..., None]
        water = refl * fade * np.array([0.78, 0.88, 1.0]) + np.array([4, 8, 18.]) * (0.5 + dl)
        wl = (water @ np.array([0.30, 0.59, 0.11])) / 255
        wave = np.sin(cc * 0.45 - t * 2.5 + D * 1.7 + np.sin(cc * 0.1 + D))
        wch = np.where(wave > 0.15, CI["~"], CI["-"])
        wch = np.where(wl < 0.10, np.where(wave > 0.6, CI["."], CI[" "]), wch)

        # sun glitter on the water
        if e > -0.04:
            hsh = (np.sin(cc * 12.9898 + D * 78.233 + math.floor(t * 8) * 3.1) * 43758.5453) % 1
            width = 2 + D * 0.7 + 6 * dl
            path = np.exp(-((cc - sx) / width) ** 2)
            g = (hsh > 0.72) * path * float(smoothstep(-0.04, 0.05, e)) * (1 - 0.5 * D / Wr)
            water = water + sun_col * (g * 0.9)[..., None]
            wch = np.where(g > 0.35, CI["*"], wch)

        # embankment
        water[0] = base_wall * 0.8 + sun_col * 0.15 * rim_k
        wch[0] = CI["="]

        all_chars = np.vstack([chars, wch]).astype(np.int32)
        all_cols = np.clip(np.vstack([sky, water]), 0, 255).astype(np.uint8)
        return all_chars, all_cols


# --------------------------------------------------------------- renderer --
class Renderer:
    def __init__(self, pygame, font, cols, rows, backdrop=0.14, glow=0.85):
        self.pg = pygame
        self.cols, self.rows = cols, rows
        self.cw, self.ch = font.size("M")
        self.w, self.h = cols * self.cw, rows * self.ch
        self.glyphs = [font.render(c, True, (255, 255, 255)) for c in CHARSET]
        self.backdrop = int(255 * backdrop)
        self.glow = int(255 * glow)
        self.xs = np.arange(cols) * self.cw
        self.ys = np.arange(rows) * self.ch

    def draw(self, chars, colors):
        pg = self.pg
        text = pg.Surface((self.w, self.h))
        rs, cs = np.nonzero(chars)
        g, xs, ys = self.glyphs, self.xs, self.ys
        text.blits([(g[k], (xs[c], ys[r])) for r, c, k in zip(rs.tolist(), cs.tolist(), chars[rs, cs].tolist())],
                   doreturn=False)

        pix = np.repeat(np.repeat(colors, self.ch, axis=0), self.cw, axis=1)
        csurf = pg.surfarray.make_surface(pix.swapaxes(0, 1))
        text.blit(csurf, (0, 0), special_flags=pg.BLEND_RGB_MULT)

        out = csurf.copy()                                   # faint colour wash behind the text
        out.fill((self.backdrop,) * 3, special_flags=pg.BLEND_RGB_MULT)
        out.blit(text, (0, 0), special_flags=pg.BLEND_RGB_ADD)
        for div, k in ((10, 1.0), (3, 0.45)):
            small = pg.transform.smoothscale(text, (self.w // div, self.h // div))
            glow = pg.transform.smoothscale(small, (self.w, self.h))
            glow.fill((int(self.glow * k),) * 3, special_flags=pg.BLEND_RGB_MULT)
            out.blit(glow, (0, 0), special_flags=pg.BLEND_RGB_ADD)
        return out


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description="ASCII skyline sunrise")
    ap.add_argument("--width", type=int, default=1280, help="window width in pixels")
    ap.add_argument("--height", type=int, default=760, help="window height in pixels")
    ap.add_argument("--font-size", type=int, default=12)
    ap.add_argument("--day", type=float, default=60.0, help="seconds for one full day")
    ap.add_argument("--start", type=float, default=0.10, help="time of day to start, 0..1 (0.18 = sunrise)")
    ap.add_argument("--seed", type=int, help="city seed (same seed = same skyline)")
    ap.add_argument("--gif", help="render night -> morning to this GIF and exit")
    ap.add_argument("--frames", help="render night -> morning as PNG frames into this folder and exit")
    ap.add_argument("--from", dest="p0", type=float, default=0.10)
    ap.add_argument("--to", dest="p1", type=float, default=0.36)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    exporting = bool(args.gif or args.frames)
    if exporting:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    pygame.init()
    font = pygame.font.SysFont("consolas,dejavusansmono,menlo,couriernew,monospace", args.font_size, bold=True)
    cw, ch = font.size("M")
    cols, rows = args.width // cw, args.height // ch
    R = Renderer(pygame, font, cols, rows)
    S = Scene(cols, rows, cw, ch, args.seed)

    if exporting:
        n = int((args.p1 - args.p0) * args.day * args.fps)
        imgs = []
        if args.frames:
            os.makedirs(args.frames, exist_ok=True)
        for i in range(n):
            t = i / args.fps
            surf = R.draw(*S.frame(args.p0 + t / args.day, t + 100))
            if args.frames:
                pygame.image.save(surf, os.path.join(args.frames, f"frame_{i:04d}.png"))
            if args.gif:
                from PIL import Image
                arr = pygame.surfarray.array3d(surf).swapaxes(0, 1)
                im = Image.fromarray(arr)
                imgs.append(im.resize((im.width // 2, im.height // 2), Image.LANCZOS))
            print(f"\rframe {i + 1}/{n}", end="", flush=True)
        if args.gif:
            imgs[0].save(args.gif, save_all=True, append_images=imgs[1:],
                         duration=int(1000 / args.fps), loop=0, optimize=True)
        print("\ndone")
        return

    screen = pygame.display.set_mode((R.w, R.h))
    pygame.display.set_caption("skyline_sunrise.py")
    clock = pygame.time.Clock()
    p, t, speed, paused, shot = args.start, 0.0, 1.0, False, 0

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_RIGHT:
                    p += 0.02
                elif ev.key == pygame.K_LEFT:
                    p -= 0.02
                elif ev.key == pygame.K_UP:
                    speed = min(speed * 1.5, 20)
                elif ev.key == pygame.K_DOWN:
                    speed = max(speed / 1.5, 0.1)
                elif ev.key == pygame.K_n:
                    S.reseed()
                elif ev.key == pygame.K_s:
                    shot += 1
                    pygame.image.save(screen, f"skyline_{shot:03d}.png")
        dt = clock.tick(30) / 1000
        t += dt                                   # waves, stars and clouds keep moving when paused
        if not paused:
            p = (p + dt * speed / args.day) % 1.0
        screen.blit(R.draw(*S.frame(p, t)), (0, 0))
        pygame.display.flip()


if __name__ == "__main__":
    main()
