"""MediaPipe hand landmark indices and bone connections."""

from __future__ import annotations

# 21-point hand topology (MediaPipe Hand Landmarker)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS: dict[str, tuple[int, ...]] = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# Palm ring used for filled vector body
PALM_RING: tuple[int, ...] = (
    WRIST,
    THUMB_CMC,
    INDEX_MCP,
    MIDDLE_MCP,
    RING_MCP,
    PINKY_MCP,
)

CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

# Colors: BGR for OpenCV
STYLE_A = {
    "fill": (210, 140, 40),      # deep cyan-ish fill (BGR)
    "stroke": (255, 240, 220),   # almost white stroke
    "joint": (80, 255, 255),     # yellow joint
    "glow": (255, 180, 60),
}
STYLE_B = {
    "fill": (180, 80, 220),      # magenta fill
    "stroke": (255, 230, 255),
    "joint": (200, 255, 120),
    "glow": (200, 100, 255),
}
