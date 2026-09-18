[中文](TUTORIAL.md) · **English** · [日本語](TUTORIAL.ja.md)

# Tutorial

From zero to effects on your own hands, then on to tuning and adding styles. Read in order, or jump to the section you're stuck on.

- [0. What you need](#0-what-you-need)
- [1. Install and first run](#1-install-and-first-run)
- [2. What's on the screen](#2-whats-on-the-screen)
- [3. How to play each style](#3-how-to-play-each-style)
- [4. Live tuning](#4-live-tuning)
- [5. Recording](#5-recording)
- [6. Command-line reference](#6-command-line-reference)
- [7. Changing parameters: which file, which line](#7-changing-parameters-which-file-which-line)
- [8. How it works (architecture tour)](#8-how-it-works-architecture-tour)
- [9. Development: the four checks and CI](#9-development-the-four-checks-and-ci)
- [10. Adding a style](#10-adding-a-style)
- [11. FAQ](#11-faq)

---

## 0. What you need

| Item | Requirement | Notes |
|---|---|---|
| Computer | macOS (tested); Apple Silicon is the comfortable choice | On an M4 Pro at 1080p: detection 6.9 ms/frame, the heaviest style draws in 6.5 ms at p95, 30 fps with plenty of headroom. Intel Macs run it but were never measured. Linux / Windows are **untested**, see the end of [11](#11-faq) |
| Camera | the built-in one is fine | 1080p is best; 720p works too, effect sizes scale with the frame |
| Python | 3.12 – 3.14 | `mediapipe` wheels lag behind new Python releases; if it won't install, step back one version (CI pins 3.12, local development uses 3.14) |
| Network | first run only | installs dependencies and downloads the 7.5 MB MediaPipe hand model; offline after that |
| ffmpeg | optional | only used for the frame-rate fix when recording and the h264 remux of offline renders; everything runs without it |

Only three dependencies: `mediapipe`, `opencv-contrib-python`, `numpy` (version ranges in `requirements.txt`, upper bounds pinned at the next major, reasons in the file).

---

## 1. Install and first run

### 1.1 One command (recommended)

```bash
git clone https://github.com/Eureka0w0v0/manual-tracking.git
cd manual-tracking
./run.sh
```

The first time, `run.sh` creates `.venv/` → installs `requirements.txt` → sets `PYTHONPATH=src` → starts `live --style cube`.
Every time after that it only does the last step.

The first start also downloads the model to `models/hand_landmarker.task` (Google's official URL with a version number, so the content never changes; sha256 is verified after download and a mismatch deletes the file and raises — that guards against proxy hijacking and CDN error pages).

### 1.2 Camera permission (macOS)

The first time you get a dialog like "Terminal would like to access the camera". Click **OK**. If it never appeared or you clicked the wrong thing:
System Settings → Privacy & Security → Camera → enable your terminal (Terminal / iTerm / VS Code).

You can also **double-click `start-live.command`**: it grabs one frame from the auto-selected built-in camera first to trigger the permission dialog, then starts.

> With an iPhone connected via Continuity Camera, OpenCV device 0 may be the phone — selecting it lights the phone up and takes it over.
> The default `--camera -1` walks the `system_profiler` list and picks the first device that isn't an iPhone/iPad, which is normally the built-in camera.

### 1.3 Manual install (if you manage your own environments)

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m manual_tracking live --style cube
```

**Note**: the package is deliberately not `pip install -e .`-ed (see the first line of `pyproject.toml`), so `PYTHONPATH=src` is mandatory.
Also, `src/manual_tracking/__main__.py` uses relative imports and only runs as `python -m manual_tracking`; `python __main__.py` always fails with `attempted relative import with no known parent package`.

### 1.4 IDE

- **VS Code**: the repo ships `.vscode/`. `Cmd+Shift+B` = run the default style; `F5` = launch with the debugger (for the live loop prefer `Ctrl+F5` without it — debugpy drags the frame rate down); the test panel runs `tests/` directly.
- **Other IDEs**: open `main.py` at the repo root and hit run. It does exactly one thing — pushes `src/` onto `sys.path` and imports the package properly, to get IDEs around the relative-import rule. Arguments as usual: `python main.py live --style mirror`.

### 1.5 Quitting

Press `Q` or `Esc` with the window focused. `Ctrl+C` in the terminal also works — shutdown order is recording file → camera → window → detection thread → model,
and a failure in one step never blocks the others (so force-quitting mid-recording still leaves a playable mp4).

---

## 2. What's on the screen

### 2.1 Startup banner (terminal)

```
========================================================
  MANUAL TRACKING LIVE — 折纸镜面 / 彩色玻璃盒 / 悬浮立方体
  拇指+食指捏纸；翻转一只手拧麻花；捏死压成细线
  采集 1920x1080 (req 1920x1080)  帧率 30 (req 30)  推理边 全帧
  窗口 1920x1080 (可拖拽边角缩放)
  Q退出  S风格  D暗底  R录制  o p 底亮度
  screen 调参: [ ] 翻转曲线  ; ' 挂多高  , . 旋转轴  - = 锚点跟手  < > 收起距离  7 8 角速度上限
  screen/cube: 9 0 旋转跟手  g h 通透  { } 棱线
  cube  键位: F 重力开关  X 归位
  cube  手势: 捏在盒上拖=转 | 双手捏住=移动+缩放+拧 | 张开手=炸开
========================================================
```

The banner is Chinese; the key tables in [4](#4-live-tuning) are the translation.

- `采集 … (req …)` = the resolution/frame rate the camera **actually** delivers vs what you requested. The device decides based on its capabilities and the light; in low light it drops to 15 fps by itself.
- The key lines are not hand-written: they are generated from the key tables in `live.py`, the same tables that produce the "wrong style for this key" message at runtime.

### 2.2 HUD (the black bar at the top of the window)

```
FPS  30.1  det   6.9ms  draw  4.2ms  hands:2  cube  idle  size 180 grip1 open0.62  顶+前+右
```

| Field | Meaning |
|---|---|
| `FPS` | display frame rate (EMA; the first 5 frames are skipped — the cold-start frame is absurdly slow and must not seed it) |
| `det` | time of the latest MediaPipe detection (runs on a worker thread, never blocks the picture) |
| `draw` | render time (EMA) |
| `hands:N` | hands used this frame |
| style name | current style |
| `busy` / `idle` | whether the detection thread is inferring right now |
| `ERRn` | cumulative count of detection exceptions. **Any value is abnormal** — the first exception is already printed in the terminal; without this counter a dead detector would just look like `hands:0`, indistinguishable from "no hands" |
| `REC` + red dot top right | recording |

Then a style-specific part:

- **cube**: `size` edge length in px, `grip` number of hands holding the cube, `open` the spread ratio of the un-pinched hand (drives the explode, thresholds 0.70→1.05), then the **faces currently facing the camera** (e.g. `顶+前+右` = top+front+right) or `explode`.
- **screen**: the current values of the five knobs `expo lift bias resp cap`, `psi` the box's angle about its axis, `oL oR` the two palm orientations (the raw signal driving roll, −1..1), `dz` the depth component of the axis, then the visible faces.
  When tuning roll this tells "the geometry didn't turn" apart from "it turned but you can't read it".

The HUD uses OpenCV's Hershey font, which **only knows ASCII**, hence the abbreviations. Face names are single Chinese characters: 顶 top, 前 front, 背 back, 底 bottom, 左 left end, 右 right end.

### 2.3 The picture

- The camera background is dimmed to 55 % by default (`D` switches to black, `o`/`p` adjust brightness); the effect is the subject.
- Skeleton: gold is the first hand, orange the second, fingertip dots one size larger than the other joints.
- The picture is **mirrored** by default (like a mirror, so left/right match intuition). `--no-mirror` turns that off.

---

## 3. How to play each style

`S` cycles `mirror → screen → cube`. `--style` picks one directly.

First, gesture hygiene that applies to everything:

- Hands filling **1/4 to 1/2** of the frame height is the sweet spot. Too far and landmarks jitter; too close and fingers leave the frame.
- Face the camera with light coming from the front. Backlight and overhead light make MediaPipe lose hands.
- Busy backgrounds slow detection; if `det` goes above 15 ms, sit somewhere else.
- Don't stack the two hands for long: when they occlude each other MediaPipe sometimes sees only one. The program has an 8-frame grace period (two-hand mode) and keeps tracks for 250 ms, beyond that you re-establish.

### 3.1 cube — floating cube (default)

Fundamentally different from the other styles: **the cube has its own position, orientation and size**; the hands only push and turn it. Let go and it stays put, slowly turning (≈10°/s), like a real object hovering in front of you.

| You want to | Do this | What decides it |
|---|---|---|
| grab it | **pinch** thumb tip and index tip together with the pinch point on the cube | pinch = tip distance / palm width < 0.42 (release needs > 0.62; the gap in between is hysteresis so it doesn't chatter); pinch point = midpoint of the two tips, must fall within ±0.75 edge lengths of the centre — **pinching in mid-air controls nothing**. A ripple bursts from the pinch point the moment you grab, as a receipt |
| turn a face | grab with one hand and **drag** | horizontal drag rotates about the vertical axis, vertical about the horizontal, 1 px ≈ 0.009 rad (349 px for 180°). It uses drag deltas rather than palm orientation, so dragging one way keeps turning forever and never turns back on its own |
| move / scale / twist | grab it with **both hands** | the cube follows the midpoint 1:1 (no discount); pull the hands apart by some factor and the edge grows by that factor (limited to 6 %–60 % of the short side); turn both hands like a steering wheel → roll about the screen normal (the third axis dragging can't reach) |
| explode to see all six faces | release the pinch and **spread five fingers** | spread = index tip ↔ pinky tip distance / palm width; below 0.70 closed, above 1.05 fully exploded; the six faces fly out along their normals and hover. Make a fist or drop the hand to close |
| throw it | let go while dragging | momentum carries: rotation keeps sliding (friction 0.85/frame, stops in about a second), translation keeps drifting, **it bounces off the frame edges**. A hard fling is exactly one face period |
| gravity | press `F` | releases follow a parabola, bounce on the floor, roll to a stop under floor friction. Press again to turn it off |
| it's a mess / pushed to the edge | press `X` | pose resets to the centre |

Feel details (all tunable at the top of `floatcube.py`):

- Once grabbed it **follows wherever you drag**, even if the pinch point has left the cube (two-hand scaling inevitably pulls the hands outside). Only releasing the pinch breaks the grab.
- Lost detection for ~0.25 s does not break the grab; a single-frame "release" spike (the tips hidden behind your own hand while turning the wrist, landmarks jump for a frame) doesn't count — it takes 3 consecutive frames.
- If two-hand mode loses one hand, the cube **freezes** for an 8-frame grace period instead of dropping to one-hand mode and suddenly turning.

### 3.2 screen — glass box

**Open both hands as if holding an invisible cuboid.** The box is "constructed" from a few low-noise parameters of the two hands:

- the line between the two anchors (85 % of the way from palm centre to finger-arc midpoint) = the long axis, its length = box length;
- finger-arc spread (index tip ↔ pinky tip) × 1.14 = box height, depth = 1.47 × height;
- palm orientation (turning the wrist) → **roll** about the axis;
- the ratio of the two **palm widths** → the axis's depth component: push one hand towards the camera and the box points out of the screen, revealing an end face.

The hands never pin vertices, so the box is always rigid and the perspective always right (the old version had eight corners each chasing a fingertip, and wedge shapes were unfixable — the whole story is in [`GLASS_BOX_GEOMETRY.md`](GLASS_BOX_GEOMETRY.md)).

| You want to | Do this |
|---|---|
| make the box appear | both hands open, at least 40 px apart, fingers roughly up |
| flip to see the back | **turn the wrists** (palm ↔ back of hand towards the camera). The two orientations are averaged, so turning both hands together is cleanest |
| see an end face | push one hand forward, pull the other back |
| fold it away / bring it back | bring the hands until the nearest points are < 0.35 palm widths apart → the box flattens away in 0.17 s; pull apart to > 0.75 → it grows back (hysteresis, so it never flickers at the threshold) |

Six faces, six treatments, each recognisable at a glance:

| Face | Treatment |
|---|---|
| top | inverted blue gradient map + horizontal glitch strips |
| front | riso print (two colours + channel-offset fringes + ordered dither) |
| back | hard-threshold duotone (screen print) |
| bottom | point cloud / hologram (local contrast as pseudo-depth) |
| left end | quadtree adaptive mosaic |
| right end | halftone dots |

One **known limit**: palm orientation is a cosine-shaped signal that goes `0→+1→0→−1` around a full turn, so if you keep turning one way the box turns back at some angle. That is the nature of the signal; for unlimited turning use `cube`.

### 3.3 mirror — origami mirror sheet

**Each hand pinches a corner of the sheet between thumb and index finger**: left index tip = top-left, left thumb tip = bottom-left, mirrored on the right.

- Flat: a sheet of inverted mirror (`clamp(283 − 0.56 × background)`, the background shows through as a negative ghost).
- **Turn one hand**: that half darkens, cools towards grey, and its sampling point shifts outward — the paper folds.
- Offset the hands until the top edge crosses the bottom edge: the sheet twists into two triangular wings, the right one in front.
- **Pinch both hands shut** (gap < 16 px): the sheet turns edge-on, only two white lines with a slit between them.

---

## 4. Live tuning

Each key changes a **field in memory**; the module-level default comes back on exit. To change something permanently go to [7](#7-changing-parameters-which-file-which-line).

Press a key the current style cannot use and the terminal says `[ 只对 screen 有用 (当前 cube)` ("only for screen (currently cube)") instead of silently changing a field that does nothing.

### All styles

| Key | Action |
|---|---|
| `S` | cycle styles |
| `D` | camera background ↔ black |
| `o` `p` | background brightness (0.05 per step) |
| `R` | recording on/off |
| `Q` / `Esc` | quit |

### Shared by screen and cube

| Key | Parameter | In plain words |
|---|---|---|
| `g` `h` | `face_alpha` | how transparent the glass is. Opacity of the facing faces 0.25–1.0; the inner walls follow at 0.42× |
| `{` `}` | `box_edge_w` | edge line width in px, 0 = seamless (default). The reference clip has a 4 px white line, but once each face has its own texture the white line overpowers the faces |
| `9` `0` | screen: `roll_resp` / cube: `orbit_gain` | rotation responsiveness. screen steps additively by 0.05; cube spans an order of magnitude, so it steps multiplicatively by ×1.15 |

### screen only

| Key | Parameter | In plain words |
|---|---|---|
| `[` `]` | `roll_expo` | steepness of the flip curve: small = sensitive, large = dead centre but steadier at rest. **Not** a de-noise knob (see the next row) |
| `7` `8` | `roll_max_rate` | angular-rate cap in °/frame. This is the main control against "colour flicker": it blocks the jumps caused by the palm-orientation signal flipping sign in saturation. Default 10 = 300°/s, still above real hand speed; 8 would clip real motion |
| `;` `'` | `anchor_lift` | how high the box hangs: 0 = palm centre (low), 1 = finger-arc midpoint (where the reference clip has it), default 0.85 |
| `,` `.` | `depth_bias` | the fixed point of rotation about the axis: 0 = front face on the hands, 0.5 = centre (default, the box turns around the hands as a whole), 1 = back face |
| `-` `=` (or `_` `+`) | `anchor_resp` | anchor responsiveness: large = follows, small = steady but dull. Opens up automatically when the hands move fast (velocity-adaptive) |
| `<` `>` | `gap_shut` | how close the hands must get to fold the box away (normalised by palm width, so independent of distance to the camera). Reappear threshold = this + 0.40 |

### cube only

| Key | Action |
|---|---|
| `F` | gravity on/off |
| `X` | reset the pose |

---

## 5. Recording

- Press `R` to start, `R` again to stop. The file lands in `output/live_YYYYmmdd_HHMMSS.mp4`, **without the HUD**.
- `--record path.mp4` sends the first recording to that path; later presses of `R` auto-name their files and never overwrite the take you just finished.
- The container frame rate is the measured FPS at the moment you start (clamped to 10–60; during the first 15 warm-up frames it falls back to 30 — otherwise the ~6 fps of the first frame would be frozen into the container and the take would play at 67 % speed).
- On stop, if the real average frame rate differs from the container by more than 5 % (common: the camera drops to 15 fps in low light) and the take is at least 1 s long, ffmpeg **remuxes** it (`-c copy`, no re-encode) to the real rate. Without ffmpeg you get a note that it was skipped; the file still plays, just at the wrong speed.

---

## 6. Command-line reference

Three subcommands. No arguments = `live` (style from `live.DEFAULT_STYLE`, currently `cube`).

### `live` — real time

```bash
./run.sh live [--style S] [--camera N] [--width W --height H] [--fps F]
              [--window-scale K] [--infer-size N] [--source-dim D]
              [--no-source] [--no-mirror] [--no-filter] [--model PATH] [--record PATH]
```

| Flag | Default | Meaning |
|---|---|---|
| `--style` | `cube` | `mirror` / `screen` / `cube`; legacy names `fabric`→mirror, `track`→screen still work |
| `--camera` | `-1` | camera index; −1 = auto-pick the built-in one (skipping iPhone Continuity Camera) |
| `--width` `--height` | 1920 × 1080 | requested capture size. If the device delivers something else, frames are resized to this so the effect coordinate system stays consistent |
| `--fps` | 0 (=30) | **requested** frame rate. 60 needs camera support; detection takes 6.9 ms median, so 60 Hz is fed fine |
| `--window-scale` | 1.0 | initial window size factor. The window is resizable by dragging; this only affects display |
| `--infer-size` | 0 | longest side for MediaPipe inference; 0 = full frame. **Full frame is fastest and most accurate on Apple Silicon**; only set 640 or so on machines where inference is genuinely too slow |
| `--source-dim` | 0.55 | background brightness 0–1 |
| `--no-source` | | black background |
| `--no-mirror` | | no left/right mirroring |
| `--no-filter` | | turn off the One Euro temporal filter and show raw landmarks (jittery; for debugging; there are no cross-frame track ids in this mode) |
| `--model` | auto | a specific `.task` model; with an explicit path there is **no** auto-download, a missing file is an error |
| `--record` | | output path for the first recording after start |

### `run` — render a video offline

```bash
./run.sh run -i input.mp4 -o output.mp4 [--style S] [--source-dim 0.35] [--no-source] [--no-filter] [--max-frames N] [--model PATH]
```

Runs detection + rendering frame by frame and writes `mp4v`; with ffmpeg installed it is remuxed to h264 + `faststart` for player compatibility.
Offline defaults are style `mirror` and background 0.35 (in a finished clip the effect is the subject, the footage only the environment).
`cube` makes little sense offline — nobody pinches it, it just turns.

### `dump` — export the 21 landmarks as JSON

```bash
./run.sh dump -i input.mp4 -o landmarks.json [--model PATH]
```

For After Effects / TouchDesigner / your own scripts:

```json
{
  "width": 1920, "height": 1080, "fps": 30.0, "video": "input.mp4",
  "frame_count": 358,
  "frames": [
    {
      "frame": 0,
      "hands": [
        {"handedness": "Right", "score": 0.98,
         "points": [[x_px, y_px, z_norm], "... 21 in total, MediaPipe order (0 wrist … 4 thumb tip … 8 index tip … 20 pinky tip)"]}
      ]
    }
  ]
}
```

`x_px`/`y_px` are **full-frame pixel coordinates** (already One-Euro filtered); `z_norm` is MediaPipe's raw relative depth (only xy is filtered, z passes through so reference frames don't get mixed). Hands are sorted `Right` first, `Left` second.

---

## 7. Changing parameters: which file, which line

All tunables sit at the **top of the module that owns them**, each with a one-line "why this number".

| What you want to change | File | Example |
|---|---|---|
| how the cube feels | `src/manual_tracking/floatcube.py` | turns too fast → lower `ORBIT_GAIN`; slides too long after release → lower `SPIN_DAMP`; explodes too eagerly → raise `EXPLODE_LO/HI`; initial size → `CUBE_SIZE0` |
| glass-box geometry and filtering | `src/manual_tracking/glassbox.py` | box too flat → `BOX_H_GAIN`; depth → `BOX_DEPTH_RATIO`; colour flicker → `BOX_ROLL_MAX_RATE` (read `docs/GLASS_BOX_GEOMETRY.md` §3.7 first, don't guess) |
| the six face treatments | `src/manual_tracking/effects.py` | colours → `RISO_DARK` / `DUOTONE_*` / `PC_TINT`…; which face gets which treatment → the six lines of `BOX_FACES` |
| glass transparency / edges / lighting | `src/manual_tracking/renderer.py` | `BOX_FACE_ALPHA`, `BOX_BACK_ALPHA`, `BOX_EDGE_W`, `_LIGHT`, `SHADE_MIN` |
| the mirror sheet | `sheet.py` | brightness swing `B_SWING`, fold sampling shift `MIRROR_SHIFT`, pinch-shut threshold `PINCH_SHUT_PX` |
| hand filtering | `src/manual_tracking/tracker.py` | jittery → lower `OE_MIN_CUTOFF` to 0.6; laggy → raise to 1.5 |
| detection latency compensation | `src/manual_tracking/detect_service.py` | `EXTRAP_CAP_MS` / `EXTRAP_DAMP` / `EXTRAP_MAX_PX` |
| live keys | `src/manual_tracking/live.py` | the `_KNOBS` (±step knobs) and `_ACTIONS` (everything else) tables; a duplicate key blows up at startup |
| default background brightness | `live.SOURCE_DIM` (live 0.55) / `pipeline.SOURCE_DIM` (offline 0.35) | two scenarios, two values, each defined exactly once |
| default style | `live.DEFAULT_STYLE` / `pipeline.DEFAULT_STYLE` | every entry point reads these two; never copy the value elsewhere |

After a change, run the checks in [9](#9-development-the-four-checks-and-ci); after touching `screen`'s geometry / mapping / filtering, run the fourth one.

---

## 8. How it works (architecture tour)

### 8.1 Data flow

```
camera ──cap.read()──▶ main thread                                       window
                        │ mirror / resize                                  ▲
                        ├─submit(frame)──▶ detection thread (detect_service) │ imshow
                        │                    │ MediaPipe HandLandmarker
                        │                    │ tracker: track matching + One Euro + handedness latch
                        │                    └─▶ last two results (timestamped)
                        ├─latest_pair() ◀────┘
                        ├─extrapolate(): push landmark velocity to the current instant
                        ├─renderer.render(frame, hands)
                        │    ├─ cube:   floatcube.update → project → fill six faces
                        │    ├─ screen: glassbox.solve → fill six faces
                        │    ├─ mirror: sheet
                        │    └─ skeleton
                        ├─recorder.write(out)      ← recorded before the HUD is drawn
                        └─HUD → imshow → waitKey → key table dispatch
```

Two decisions make it "follow the hand":

1. **Detection never blocks the picture.** MediaPipe runs on a worker thread; the main thread submits the newest frame every frame with zero copies (a latest-wins mailbox: the worker only processes the newest frame, older ones are dropped). Detection rate = 1/max(detection time, frame interval), never slowed down by a queue.
2. **Extrapolation to display time.** Detection results are always 1–2 frames behind the picture; the main thread takes the last two results, matches hands by track id, computes velocities and pushes the landmarks to "now" (at most 80 ms, damping 0.7, 40 px per point max so nothing flies off). That removes the stair-stepping you get when detection rate is below display rate.

### 8.2 The tracking layer (`tracker.py`)

- Each hand is a **persistent track** (`track_id`): frame-to-frame matching is a 2×2 optimal assignment on wrist distance with a threshold adaptive to palm width; a briefly lost hand keeps its track for 250 ms (wall clock, not frame count — frame counts go wrong the moment the machine stalls).
- **One Euro** filtering instead of a fixed EMA: heavy smoothing at rest, opens up automatically during fast motion. Only xy is filtered, z passes through.
- **Handedness is latched to the track**: MediaPipe decides left/right independently every frame and flips for single frames while the wrist turns; the same track is the same hand, so the label only changes after 5 consecutive frames disagree. The first frame's label is not trusted (freshly detected hands are the ones most often mislabelled), so it starts one vote short.

### 8.3 Division of labour in rendering

```
paths ──▶ landmarks ──▶ tracker ──▶ handgeom ──▶ boxgeom ──▶ glassbox / floatcube
                                                     │                │
                                     effects ──▶ paint ──▶ sheet ──▶ renderer ──▶ detect_service ──▶ live ──▶ __main__
```

| Module | Does exactly one thing |
|---|---|
| `landmarks.py` | MediaPipe's 21 landmark indices and the bone connection table |
| `handgeom.py` | hand → a few low-noise scalars/anchors (palm centre, pinch, palm orientation, finger-arc spread, palm width). Touches no canvas, knows of no box |
| `boxgeom.py` | cuboid vertex order + weak-perspective projection. The **single** copy shared by `screen` and `cube` |
| `glassbox.py` | two-hand parameters → the 8 vertices of a rigid cuboid (with cross-frame filter state). Touches no canvas |
| `floatcube.py` | the cube with a pose: grab test, the drag / two-hand / inertia / gravity / explode state machine. Touches no canvas |
| `effects.py` | the six pixel treatments (pure `src → out` functions, parameters baked into lookup tables at construction). Knows nothing about "hands" |
| `paint.py` | fills a polygon with "the background after some treatment" (bbox-local window + mask + alpha + shading). Knows nothing about hands or boxes |
| `sheet.py` | the mirror sheet (the one style without a box) |
| `renderer.py` | dispatch by style; face ordering for `screen`/`cube` (3D outward-normal visibility + far-to-near + Lambert shading); background cache; left/right role hysteresis |
| `detect_service.py` | detection thread + extrapolation |
| `live.py` | camera, recording, HUD, keys, main loop |
| `pipeline.py` | offline video → video / JSON |

Dependencies only point to the right; ruff's isort rule pins the import order to that line as a side effect.

### 8.4 Performance (1920×1080, real landmarks, M4 Pro)

| Style | draw mean | p95 |
|---|---|---|
| mirror | 0.66 ms | 0.88 |
| cube | 3.55 | 5.21 |
| screen | 4.05 | 6.49 |

Detection (6.9 ms median) runs on the other thread. Less than a fifth of the 33 ms frame budget is used; there is nothing left worth optimising in the hot path — each effect's docstring records what it was optimised from and what was tried without success.

---

## 9. Development: the four checks and CI

```bash
.venv/bin/pip install -r requirements-dev.txt                # ruff + pytest
.venv/bin/ruff check main.py src tools tests                 # 1. static checks (incl. isort)
.venv/bin/pytest                                             # 2. contracts: 158 tests, <1 s, no camera / media / model
PYTHONPATH=src .venv/bin/python tools/cube_check.py          # 3. cube feel baseline: 27 assertions, synthetic hands
PYTHONPATH=src .venv/bin/python tools/e2e_check.py           # 4. screen regression: needs sample clip + model, ~1 min
```

| Layer | Catches |
|---|---|
| contracts (`tests/`) | "changed it wrong, it crashes": face topology vs vertex order, effect size/immutability contracts, box rigidity, the alias table, handedness hysteresis, extrapolation and recording frame rate, key gating, entry-point defaults, model download verification, shutdown cleanup |
| feel (`cube_check`) | can't drag it, it turns back, it drifts on release, it grows off-screen |
| regression (`e2e_check`) | "changed it worse": **colour flicker such as visible faces switching 5 times a second**, back face unreadable. A green `tests/` cannot see this |

The first three run on GitHub Actions on every push (`.github/workflows/ci.yml`, ubuntu + Python 3.12). The fourth stays local.

**About the sample clip**: `assets/sample.mp4` is not in the repository (it is the original creator's video). To run the fourth check, put your own 1080p/30fps clip of two hands flipping at that path. Every number in the docs was measured against the original clip, so after switching footage:

```bash
PYTHONPATH=src .venv/bin/python tools/e2e_check.py --sweep   # sweep the expo × cap grid
```

Look at the grid before deciding whether the three thresholds at the top of `tools/e2e_check.py` need to move — they guard "crossing the perceptual red line", not "identical to last time".

`tools/synth.py` is the synthetic hand shared by `tests/` and `cube_check`: all 21 landmarks laid out in real hand proportions (leave one at `(0,0)` and it drags the palm centre towards the top-left corner, the measured "responsiveness" is fake, and nothing errors).

Commit habits: split a batch of changes **by topic**, messages in Chinese conventional-commit style (`feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `build:` / `ci:` / `chore:`), the body explains **why** — what was broken, what the measurements said. `git log` is this project's design document.

---

## 10. Adding a style

Say it's called `glow`:

1. `src/manual_tracking/glow.py`: write a `draw(canvas, frame_bgr, ...)`. It only receives hands already sorted into left/right (role hysteresis belongs to the renderer); put tunables at the top with their provenance. Fill with `paint.fill`, write pixel treatments as `effects.FaceEffect` (pure `src → out` functions).
2. `renderer.py`: add the name to `STYLES` (order = `S` key cycle order) and a branch in `render()`. Old names go into `STYLE_ALIASES` if you want to keep them.
3. Style-specific keys? Add to `_KNOBS` (±step) or `_ACTIONS` (any other shape) in `live.py` with `styles=("glow",)`; the banner and the "wrong style" message follow automatically. Duplicate keys blow up at startup.
4. Tests: `tests/test_registry.py` covers the registry automatically; add a "synthetic hands in, no crash, picture actually changed" case to `tests/test_render_smoke.py`.
5. Add a row to the README's style list and key tables; run the four checks.

---

## 11. FAQ

**Can't open the camera / `打不开摄像头 #0`**
Permission ([1.2](#12-camera-permission-macos)); another program holding it (Zoom, OBS, a browser tab); try another index with `--camera 1`.
Indices shift when an iPhone is connected; the default auto-pick already skips it, watch out when specifying manually.

**`hands:0` although a hand is clearly in the frame**
Light (from the front), distance (hand 1/4–1/2 of the frame height), keep the hand off the edges. If the HUD shows `ERRn`, the detector isn't failing to see, it has **crashed** — the terminal has the first exception, usually a corrupt model file (delete `models/` and let it re-download).

**Stuttering / `FPS` drops to 15**
Read the HUD first: `det` high → slow machine, try `--infer-size 640` or `--width 1280 --height 720`; `draw` high → almost impossible (see [8.4](#84-performance-19201080-real-landmarks-m4-pro)), check whether a debugger is attached; neither high but FPS low → the camera dropped its rate in low light, turn on a lamp.

**`attempted relative import with no known parent package`**
You ran `src/manual_tracking/__main__.py` directly. Use `./run.sh`, `python main.py` or `python -m manual_tracking`.

**`ModuleNotFoundError: No module named 'manual_tracking'`**
`PYTHONPATH=src` is missing (`run.sh` / `main.py` / `.vscode` set it for you; set it yourself when running by hand).

**`模型校验失败: sha256 … ≠ 预期 …`** (model verification failed)
What was downloaded is not the official model (corporate proxy / campus network hijacking / a CDN error page); it has been deleted automatically. Try another network, download the URL from the comment by hand into `models/hand_landmarker.task`, or point `--model` at your own file.

**`import cv2` errors / weird behaviour**
`opencv-python` and `opencv-contrib-python` in the same environment overwrite each other: `pip uninstall opencv-python`, keep only contrib (it is what mediapipe depends on, and `live` needs its GUI). On Linux, `import cv2` complaining about `libGL.so.1`: `apt install libgl1 libglib2.0-0`.

**Recordings play too fast / too slow**
Install ffmpeg (`brew install ffmpeg`); the take is remuxed to the real frame rate when you stop. See [5](#5-recording).

**The box (screen) keeps flickering colours**
Press `7` to lower `roll_max_rate` to 8 and see whether it stops; if so, your hand speed / lighting is flipping the palm-orientation signal in its saturated region — `docs/GLASS_BOX_GEOMETRY.md` §3.7 has the full analysis. Don't touch `roll_expo`, it is not a de-noise knob.

**Does it run on Linux / Windows?**
Untested. The two macOS-specific spots both have fallbacks: the AVFoundation backend is guarded by `hasattr` and falls back to `CAP_ANY`; the automatic camera pick depends on `system_profiler` and falls back to index 0. The rest is plain OpenCV / MediaPipe. `start-live.command` is the macOS double-click entry; elsewhere use `run.sh` or `python main.py`. If it works for you, an issue saying so would be welcome.
