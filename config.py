"""
config.py
=========
Single source of truth for all tunable constants in the application.
No magic numbers should appear anywhere else in the codebase — import
from here instead.

Sections:
    Display         — window and capture resolution, target FPS
    Colors          — BGR tuples for the teal palette
    ROOM            — sphere sizing, animation speeds, charge timing
    Gesture Timing  — hold durations and sequence window
    Cooldowns       — per-technique re-trigger lockout (seconds)
    Animations      — per-technique playback duration (seconds)
    Name Card       — HUD overlay timing
    Gesture Thresholds — landmark-level classifier parameters
    Particles       — particle system bounds
    Audio           — pygame.mixer init parameters
    Assets          — relative paths to sound files
"""

# ── Display ────────────────────────────────────────────────────────────────────
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
TARGET_FPS   = 30

# ── Colors (BGR) ───────────────────────────────────────────────────────────────
TEAL_BRIGHT = (165, 202,  93)   # #5DCAA5
TEAL_MID    = (117, 158,  29)   # #1D9E75
TEAL_DARK   = ( 86, 110,  15)   # #0F6E56
WHITE       = (255, 255, 255)
BLACK       = (  0,   0,   0)

# ── ROOM ───────────────────────────────────────────────────────────────────────
# Sphere radius expressed as a fraction of min(frame_width, frame_height)
ROOM_RADIUS_FRACTION     = 0.38

# Animation speeds expressed as multiples of the final radius per second
ROOM_EXPAND_SPEED        = 3.5   # how fast the sphere grows on activation
ROOM_COLLAPSE_SPEED      = 4.0   # how fast it shrinks on deactivation

# Alpha fade rates (0→1 and 1→0) in opacity units per second
ROOM_ALPHA_EXPAND_RATE   = 3.0
ROOM_ALPHA_COLLAPSE_RATE = 2.5

# How long the user must hold an open palm before ROOM activates (seconds)
ROOM_CHARGE_HOLD         = 0.5

# ── Gesture Timing (seconds) ───────────────────────────────────────────────────
# Maximum window in which a multi-step gesture sequence must be completed
SEQUENCE_WINDOW             = 4.0   # was 2.0 — extended to give more time between K-ROOM steps

# Per-gesture hold durations before the gesture is considered "fired"
GESTURE_HOLD_SHAMBLES_STEP1 = 0.3   # hold two-finger point before wrist flick
GESTURE_HOLD_GAMMA_KNIFE    = 0.4   # charge single-finger point before releasing
GESTURE_HOLD_MES            = 0.3   # hold chest-fist
GESTURE_HOLD_TAKT           = 0.3   # hold V-sign upward
GESTURE_HOLD_K_ROOM_STEP    = 0.4   # hold each step of the 3-sign K-ROOM sequence

# ── Cooldowns (seconds) ────────────────────────────────────────────────────────
# Minimum time between consecutive activations of the same technique.
COOLDOWN_ROOM        = 1.0
COOLDOWN_SHAMBLES    = 2.0
COOLDOWN_AMPUTATE    = 1.5
COOLDOWN_GAMMA_KNIFE = 2.0
COOLDOWN_MES         = 2.5
COOLDOWN_TAKT        = 1.5
COOLDOWN_K_ROOM      = 5.0

# ── Animation Durations (seconds) ─────────────────────────────────────────────
# How long each technique's visual effect plays before TECHNIQUE_COMPLETE fires.
ANIM_SHAMBLES    = 0.8
ANIM_AMPUTATE    = 0.6
ANIM_GAMMA_KNIFE = 1.0
ANIM_MES         = 1.2
ANIM_TAKT        = 1.5
ANIM_K_ROOM      = 3.0

# ── Name Card ──────────────────────────────────────────────────────────────────
# Duration the technique name overlay remains visible after activation (seconds)
NAME_CARD_DURATION = 2.0

# ── MediaPipe ──────────────────────────────────────────────────────────────────
MEDIAPIPE_MODEL_PATH            = "assets/models/hand_landmarker.task"
MEDIAPIPE_DETECTION_CONFIDENCE  = 0.7
MEDIAPIPE_PRESENCE_CONFIDENCE   = 0.5
MEDIAPIPE_TRACKING_CONFIDENCE   = 0.6

# ── Gesture Landmark Thresholds ────────────────────────────────────────────────
# All values are in MediaPipe's normalized [0, 1] coordinate space unless noted.

# A finger is "extended" when tip.y is strictly above its PIP joint (y increases
# downward, so tip.y < pip.y).  This offset can be tuned to require more clearance.
FINGER_EXTENSION_Y_MARGIN = 0.0

# Number of non-thumb fingers allowed to be extended and still count as a fist
FIST_MAX_FINGERS_EXTENDED = 0

# Minimum angle (degrees) between index and middle finger vectors for Takt V-sign
V_SIGN_MIN_SPREAD_DEGREES = 15.0   # was 20.0 — loosened for reliability

# Maximum normalized horizontal spread across the four blade fingers for Amputate
AMPUTATE_MAX_FINGER_SPREAD = 0.05

# Minimum wrist displacement (normalized) in one frame to count as a swipe
SWIPE_VELOCITY_THRESHOLD = 0.03   # was 0.05 — lowered for Shambles reliability

# Mes: hand y-coordinate (normalized) must be greater than this to count as "chest"
CHEST_REGION_Y_MIN = 0.40   # was 0.50 — lowered so mid-torso height counts

# AMPUTATE: max degrees the hand axis may deviate from horizontal
AMPUTATE_MAX_ANGLE_DEG = 45.0

# TAKT: (wrist_y_px - tip_y_px) / frame_height must exceed this value for "upward"
# (in pixel space y increases downward, so wrist_y > tip_y when pointing up)
TAKT_MIN_UPWARD_RATIO = 0.08   # was 0.12 — lowered for reliability

# TAKT: number of consecutive frames shape+orientation must be stable before
# the hold timer is considered "started" (reduces false-positive jitter)
TAKT_CONSECUTIVE_FRAMES = 3

# SHAMBLES: seconds from arming (two-finger hold satisfied) until flick window closes
SHAMBLES_FLICK_WINDOW = 2.5   # was 1.5 — extended for Shambles reliability

# Seconds to suppress FIST emission after GAMMA_KNIFE fires (prevents accidental
# ROOM collapse when the user curls their fingers to release the technique)
GAMMA_KNIFE_FIST_SUPPRESS = 0.5

# Minimum normalized distance between thumb tip (lm4) and thumb MCP (lm2) for
# the thumb to count as "extended / abducted" in open-palm detection
THUMB_ABDUCTION_MIN_DIST = 0.04

# Minimum mean x-distance between adjacent fingertips (lm8, lm12, lm16, lm20)
# for open_palm to be accepted.  Rejects fists where all fingers register as
# "extended" but remain physically bunched together.
MIN_PALM_FINGER_SPREAD = 0.04

# Consecutive no-hand frames before arm/hold state is reset (grace period for
# brief MediaPipe tracking drops that should not interrupt an in-progress gesture)
HAND_LOSS_GRACE_FRAMES = 4

# ── Particles ──────────────────────────────────────────────────────────────────
PARTICLE_MIN_SPEED = 1.5    # pixels per frame
PARTICLE_MAX_SPEED = 3.5
PARTICLE_MIN_LIFE  = 25     # frames
PARTICLE_MAX_LIFE  = 50
PARTICLE_MAX_COUNT = 120    # hard cap to protect frame budget

# ── Audio ──────────────────────────────────────────────────────────────────────
AUDIO_FREQUENCY = 44100
AUDIO_SIZE      = -16       # signed 16-bit
AUDIO_CHANNELS  = 2         # stereo
AUDIO_BUFFER    = 512       # samples; low latency on M4

# ── Asset Paths ────────────────────────────────────────────────────────────────
# Relative to the project root.  Audio manager will silently skip missing files.
SOUND_ROOM        = "assets/sounds/room_activate.wav"
SOUND_SHAMBLES    = "assets/sounds/shambles.wav"
SOUND_AMPUTATE    = "assets/sounds/amputate.wav"
SOUND_GAMMA_KNIFE = "assets/sounds/gamma_knife.wav"
SOUND_MES         = "assets/sounds/mes.wav"
SOUND_TAKT        = "assets/sounds/takt.wav"
SOUND_K_ROOM      = "assets/sounds/k_room.wav"
