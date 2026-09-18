[中文](README.zh.md) · **English** · [日本語](README.ja.md)

# manual-tracking

Real-time hand effects in pure Python: MediaPipe Hand Landmarker + OpenCV, pinning five effects to your own hands — an **origami mirror sheet, a glass box, a floating cube, a TouchDesigner-style banner and a neon wire skeleton**. One Mac, one built-in webcam, no GPU; 1080p at a steady 30 fps.

![Five styles](docs/img/styles.jpg)

(Rendered with synthetic hands on a procedurally generated background by `tools/gallery.py`. On a real camera the effects sit on your own hands and look a lot better than this.)

## Quick start

```bash
git clone https://github.com/Eureka0w0v0/manual-tracking.git
cd manual-tracking
./run.sh                 # creates the venv, installs deps, downloads the model (7.5 MB, sha256-checked), starts the floating cube
```

Allow camera access when macOS asks. Put your hands in front of the camera. `S` cycles styles, `Q` quits.
Double-clicking `start-live.command` does the same (it triggers the permission dialog for you first).

Requirements: macOS (tested) + Python 3.12–3.14. Linux / Windows are untested; the code has fallback paths, see the end of the tutorial.

**Full tutorial** (install, how to do each gesture, live tuning, recording, offline rendering, troubleshooting, architecture) → [`docs/TUTORIAL.en.md`](docs/TUTORIAL.en.md)

Code comments and the two deep-dive docs (`docs/GLASS_BOX_GEOMETRY.md`, `docs/TOUCHDESIGNER_SETUP.md`) are in Chinese; each has a short English summary at the top.

## The five styles

`S` cycles through them in this order; `--style` picks one directly. Legacy names (`fabric`/`frame`/`planes`/`fluid` → mirror, `track` → screen, `outline` → wire) still work.

- **`mirror` — origami mirror sheet.** Corners pinned to both thumbs and index fingertips. Flat, it is an inverted-mirror sheet (`clamp(283 − 0.56 × background)`); flip one hand and that half darkens and folds; cross the hands and it twists; pinch both hands shut and it collapses to a thin double line.
- **`screen` — glass box.** Two hands hold up a **parametric rigid cuboid**: the hands do not pin vertices, they only supply a few low-noise parameters (per hand: an anchor interpolated between palm centre and finger-arc midpoint → box axis and length; finger-arc spread → height; palm-width ratio → the axis's depth component; palm orientation → roll about the axis). The cuboid is built in 3D and projected with weak perspective, so rigidity and two-point perspective are constructed rather than corrected. Face visibility uses 3D outward normals + Lambert shading. **Bring the hands together → the box flattens away; pull apart → it grows back.** Six faces, six pixel treatments:

  | Face | Treatment | Source |
  |---|---|---|
  | top | inverted blue gradient map + horizontal glitch strips | measured pixel-by-pixel from the reference clip |
  | front | riso print (two colours + channel-offset fringes + ordered dither) | TD screen recording |
  | back | hard-threshold duotone (screen print / Polaroid) | TD screen recording |
  | bottom | point cloud / hologram (monocular approximation, local contrast as pseudo-depth) | TD screen recording |
  | left end | quadtree adaptive mosaic | TD screen recording |
  | right end | halftone dots (warm paper + ink) | original |

  How every constant was solved from the reference clip, and what was tried and rejected, is in [`docs/GLASS_BOX_GEOMETRY.md`](docs/GLASS_BOX_GEOMETRY.md) (Chinese, English summary at the top).

- **`cube` — floating cube.** The fundamental difference from `screen`: **it has its own pose**. The cube stores its own position, orientation and size; the hands are only controllers. Let go and it stays where it is, slowly turning (≈10°/s). Its six faces reuse the six treatments above.

  | Gesture | Effect |
  |---|---|
  | one hand **pinches on the cube** and drags | rotate: horizontal drag about the vertical axis, vertical drag about the horizontal axis. A **ripple** bursts from the pinch point the moment you grab it (your receipt); pinching in mid-air does nothing — grab = pinch ∧ pinch point on the cube. Once grabbed it follows you wherever you drag, survives ~0.25 s of lost detection, and a single-frame "release" spike does not count |
  | both hands **pinch on the cube** | translate (follows the midpoint) + scale (follows the distance) + **twist** (turn both hands like a steering wheel → roll about the screen normal, the third axis dragging cannot reach) |
  | **open hand, five fingers spread** | **exploded view**: the six faces fly out along their normals and hover — the only pose where all six treatments are visible at once; make a fist or drop the hand to close it |
  | let go | momentum carries: rotation keeps sliding, translation keeps drifting, **it bounces off the frame edges**, friction brings it back to its idle turn. With `F` (gravity) on, a release follows a **parabola**, bounces on the floor and rolls to a stop |

  Rotation is driven by **drag deltas**, not palm orientation, so dragging in one direction keeps turning forever instead of turning back the way `screen` does (`screen` is bound by the cosine-shaped saturation of the palm-orientation signal, see `docs/GLASS_BOX_GEOMETRY.md` §6). The feel is pinned by automated assertions: `tools/cube_check.py`.

- **`banner` — TouchDesigner-style banner.** Yellow-threshold head band / X-ray middle window / white separator / a red foot band hanging below the thumbs. All of it is a screen-space duotone of the camera image.
- **`wire` — neon wire skeleton.** Glow + core line + light dots flowing along the bones. No gesture needed; the cheapest style.

## Keys

| Key | Action |
|---|---|
| `S` | mirror / screen / cube / banner / wire |
| `D` | camera background / black background |
| `o` `p` | background brightness (darker / brighter) |
| `R` | record (at the measured frame rate, without the HUD) → `output/live_*.mp4` |
| `Q` / `Esc` | quit |

Live tuning for `screen` (current values shown on the HUD):

| Key | Parameter | Effect |
|---|---|---|
| `[` `]` | `roll_expo` | steepness of the flip curve (small = sensitive, large = dead centre but steadier at rest) |
| `;` `'` | `anchor_lift` | how high the box hangs (0 = palm centre, 1 = finger-arc midpoint) |
| `,` `.` | `depth_bias` | the fixed point of rotation about the axis (0 = front face, 0.5 = centre, 1 = back face) |
| `7` `8` | `roll_max_rate` | angular-rate cap in °/frame (blocks the jumps caused by the palm-orientation signal flipping sign in saturation; the main control against colour flicker) |
| `9` `0` | `roll_resp` | rotation responsiveness (large = follows the hand, small = smooth) |
| `-` `=` (or `_` `+`) | `anchor_resp` | anchor responsiveness (large = follows, small = steady but dull) |
| `<` `>` | `gap_shut` | how close the hands must get to fold the box away (normalised by palm width, so independent of distance to the camera) |
| `{` `}` | `box_edge_w` | edge line width in px (0 = seamless, default 0) |

Live tuning for `cube`:

| Key | Parameter | Effect |
|---|---|---|
| `9` `0` | `orbit_gain` | how far 1 px of drag turns it (large = a flick spins it around) |
| `g` `h` | `face_alpha` | glass transparency |
| `{` `}` | `box_edge_w` | edge line width |
| `F` | `gravity` | gravity on/off |
| `X` | — | reset the pose (when it's turned into a mess or pushed to the edge) |

The three tables are **gated by style**: press a key the current style cannot use and the terminal says `[ 只对 screen 有用 (当前 cube)` ("only for screen (currently cube)") instead of silently changing a field that does nothing. The startup banner and that message are generated from the same key table, never hand-copied.

The end of the HUD shows which faces currently face the camera (e.g. `顶+前`, top+front) — when tuning roll, this tells "the geometry didn't turn" apart from "it turned but you can't read it".

## Command line

```bash
./run.sh live                              # default style cube, 1920x1080, picks the built-in camera automatically
./run.sh live --style screen               # straight into the glass box
./run.sh live --fps 60                     # request 60 fps capture (needs camera support; detection takes 6.9 ms, it keeps up)
./run.sh live --window-scale 1.4           # bigger window (you can also drag its corners)
./run.sh run  -i in.mp4 -o out.mp4 --style screen   # render a video offline
./run.sh dump -i in.mp4 -o landmarks.json           # export the 21 landmarks per frame as JSON (After Effects / TD)
python main.py live --style mirror         # IDE-friendly entry point, same arguments
```

All flags and the JSON format: [`docs/TUTORIAL.en.md` §6](docs/TUTORIAL.en.md#6-command-line-reference).

**Do not** run `src/manual_tracking/__main__.py` directly — it uses relative imports and only works as `python -m manual_tracking`; `main.py` exists solely to get IDEs around that rule. The package is deliberately not pip-installed; `run.sh`, `main.py` and `.vscode` all set `PYTHONPATH=src` for you.

The first run downloads the MediaPipe hand model into `models/` (the official versioned URL, immutable content, sha256-verified). To use your own model: `--model /path/to/xxx.task` (an explicit path is **never** auto-downloaded).

## Where to change what

Every tunable lives at the top of the module that owns it, each with a one-line "why this number". The live keys change the same-named fields and the defaults come back on exit.

| What you want to change | File |
|---|---|
| the per-face pixel treatments (riso / duotone / point cloud / quadtree / halftone / blue-top glitch) | `effects.py` |
| geometry and filtering of the `screen` box | `glassbox.py` |
| how `cube` feels (gains, inertia, gravity, explode, grab test) | `floatcube.py` |
| the `mirror` / `banner` / `wire` styles | `sheet.py` / `banner.py` / `neon.py` |
| cuboid vertex order and weak-perspective projection (the **single** copy shared by `screen` and `cube`) | `boxgeom.py` |
| polygon fill and stroke primitives | `paint.py` |
| hand filtering (One Euro, tracks, handedness latch) | `tracker.py` |
| the detection thread and velocity extrapolation | `detect_service.py` |
| keys, recording, HUD, main loop | `live.py` |

`renderer.py` only keeps "dispatch by style + drawing `screen`/`cube` + background cache + left/right role hysteresis". Architecture tour: [`docs/TUTORIAL.en.md` §8](docs/TUTORIAL.en.md#8-how-it-works-architecture-tour).

## Development

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff check main.py src tools tests                # static checks (configured in pyproject.toml)
.venv/bin/pytest                                           # pure-logic contracts + render smoke, 163 tests, <1 s
PYTHONPATH=src .venv/bin/python tools/cube_check.py        # cube feel baseline, 27 assertions
PYTHONPATH=src .venv/bin/python tools/e2e_check.py         # screen feel baseline, 3 assertions (needs a sample clip, see below)
PYTHONPATH=src .venv/bin/python tools/e2e_check.py --sweep # sweep expo/cap to pick parameters
```

Three layers of verification, each catching a different class of breakage:

| Layer | Where | Catches | Needs |
|---|---|---|---|
| contracts | `tests/` | "changed it wrong, it crashes": face topology vs vertex order, effect size/immutability contracts, box rigidity, the alias table, handedness hysteresis, extrapolation and recording frame rate, key gating, entry-point defaults, model download verification, shutdown cleanup | nothing (synthetic hands, no media) |
| feel | `tools/cube_check.py` | can't drag it, it turns back, it drifts on release, it grows off-screen | nothing (synthetic hands) |
| regression | `tools/e2e_check.py` | "changed it worse": colour flicker, back face unreadable | sample clip + model |

The first two touch no camera, media or model, so they run on CI on every push (`.github/workflows/ci.yml`). **The third stays local** — a green CI does not mean no flicker. After touching `screen`'s geometry / mapping / filtering you must run `e2e_check`: it is the only thing that catches "visible faces switching 5 times a second".

`assets/sample.mp4` is **not in the repository** (it is the original creator's video; not ours to redistribute). To run the third layer, put your own 1080p/30fps clip of two hands flipping at that path. Every number in `docs/` was measured against the original clip, so with your own footage re-baseline with `--sweep` as described in `docs/GLASS_BOX_GEOMETRY.md` §3.7.

`tools/synth.py` is the synthetic hand shared by `tests/` and `cube_check`; `tools/gallery.py` renders the picture at the top with it.

## Docs

- [`docs/TUTORIAL.en.md`](docs/TUTORIAL.en.md) — the tutorial: install, every gesture, live tuning, recording, offline rendering and JSON, where to change what, architecture tour, adding a style, FAQ ([中文](docs/TUTORIAL.md) · [日本語](docs/TUTORIAL.ja.md))
- [`docs/GLASS_BOX_GEOMETRY.md`](docs/GLASS_BOX_GEOMETRY.md) — glass-box calibration: how each constant was solved from the reference clip, the measured filter/cap grids, the rejected attempts (Chinese, English summary on top)
- [`docs/TOUCHDESIGNER_SETUP.md`](docs/TOUCHDESIGNER_SETUP.md) — doing the same kind of effect in TouchDesigner with the torinmb/mediapipe-touchdesigner plugin (Chinese, English summary on top)

## Sources and credits

- Hand tracking: [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) (Google).
- Effect references: the origami mirror sheet and glass box from "manualtracking / AM" on Douyin (colours and geometry were reverse-measured pixel by pixel from the video), and the kind of TouchDesigner screen recordings shared as "GitHub TouchDesigner projects" on Douyin (the riso / duotone / point-cloud / quadtree faces). The TD base is [torinmb/mediapipe-touchdesigner](https://github.com/torinmb/mediapipe-touchdesigner).
- This repository contains **no** original video material.

## License

[MIT](LICENSE)
