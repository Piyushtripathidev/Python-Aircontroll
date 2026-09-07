import cv2
import mediapipe as mp
import math
import time
import os
import ctypes
import subprocess
from collections import deque
from dataclasses import dataclass


# ============================================================
# AIR CONTROLLER V6
# ============================================================
#
# Python: 3.14.7
# MediaPipe: Tasks API
#
# FEATURES
# ------------------------------------------------------------
# Index finger          -> Mouse movement
# Thumb + Index         -> Left click
# Thumb + Middle        -> Double click / Open
# Index finger          -> Air writing
# Four fingers          -> Take photo
#
# IMPORTANT V6 TRACKING
# ------------------------------------------------------------
# Index tip  -> GREEN
# Thumb tip  -> BLUE
# Middle tip -> YELLOW
#
# Ring and pinky are NOT used for:
#   - Mouse movement
#   - Writing
#   - Left click
#   - Double click
#
# MediaPipe still detects the complete hand internally.
# This allows the existing four-finger photo gesture to
# continue working exactly as before.
#
# KEYBOARD
# ------------------------------------------------------------
# M -> Mouse mode
# W -> Writing mode
# I -> Idle mode
# C -> Clear writing
# 1 -> Open Google
# 2 -> Open Notepad
# 3 -> Open Calculator
# S -> Open Google Search
# ESC -> Exit
#
# V6 IMPORTANT FIXES
# ------------------------------------------------------------
# 1. Writing uses the REAL index-tip pixel on the camera frame.
# 2. No second horizontal mirroring.
# 3. Writing does NOT convert camera -> screen -> camera.
# 4. Mouse movement is separated from writing movement.
# 5. Writing follows landmark 8 directly.
# 6. Large tracking jumps break the stroke.
# 7. Search never uses input(), so the camera won't freeze.
# 8. Important interaction fingers have separate colors.
# 9. Ring and pinky are visually de-emphasized.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

MODEL_PATH = "hand_landmarker.task"


# ============================================================
# MOUSE CONFIGURATION
# ============================================================

# Camera area used for mouse control.
# Smaller margins make screen edges easier to reach.
CAMERA_MARGIN_X = 0.025
CAMERA_MARGIN_Y = 0.025

EDGE_ZONE = 0.035

# Lower = slower/smoother mouse.
# This affects mouse movement only.
MOUSE_SENSITIVITY = 0.55

# Mouse smoothing history.
MOUSE_HISTORY_SIZE = 3

# Minimum screen movement before moving the mouse.
# Helps eliminate tiny jitter.
MOUSE_DEADZONE = 1.5


# ============================================================
# WRITING CONFIGURATION
# ============================================================

# IMPORTANT:
# Writing does NOT use mouse sensitivity.
#
# 1.0 = direct index-tip movement.
# This is intentionally high so the writing follows
# the actual finger instead of lagging behind it.
WRITE_FOLLOW = 1.0

# Tiny smoothing only.
# 0.0 = completely raw.
# 1.0 = no smoothing.
WRITE_SMOOTHING = 0.88

# Maximum distance allowed between two writing points.
# If the fingertip suddenly jumps this far, the stroke breaks.
MAX_WRITE_JUMP = 90

# Writing line thickness.
WRITE_THICKNESS = 4

# Writing point history.
# Keep this small because we want responsiveness.
WRITE_HISTORY_SIZE = 2


# ============================================================
# PINCH CONFIGURATION
# ============================================================

PINCH_THRESHOLD = 0.055

PINCH_START_FRAMES = 3
PINCH_RELEASE_FRAMES = 3


# ============================================================
# INDEX DETECTION
# ============================================================

INDEX_ACTIVATE_FRAMES = 2
INDEX_DEACTIVATE_FRAMES = 4


# ============================================================
# PHOTO GESTURE
# ============================================================

PHOTO_HOLD_FRAMES = 12


# ============================================================
# V6 IMPORTANT FINGER COLORS
# ============================================================
#
# OpenCV uses BGR format.
#
# INDEX  -> Green
# THUMB  -> Blue
# MIDDLE -> Yellow
#
# Ring and pinky do not receive special fingertip colors.
# ============================================================

INDEX_COLOR = (0, 255, 0)
THUMB_COLOR = (255, 0, 0)
MIDDLE_COLOR = (0, 255, 255)

LANDMARK_COLOR = (100, 100, 100)
SKELETON_COLOR = (100, 100, 100)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: hand_landmarker.task was not found.")
    print()
    print("Expected location:")
    print(os.path.abspath(MODEL_PATH))
    print()

    raise SystemExit


# ============================================================
# MEDIAPIPE
# ============================================================

mp_tasks = mp.tasks
mp_vision = mp.tasks.vision

BaseOptions = mp_tasks.BaseOptions


landmarker_options = mp_vision.HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=os.path.abspath(
            MODEL_PATH
        )
    ),

    running_mode=mp_vision.RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.35,

    min_hand_presence_confidence=0.30,

    min_tracking_confidence=0.30,
)


landmarker = mp_vision.HandLandmarker.create_from_options(
    landmarker_options
)


# ============================================================
# SCREEN INFORMATION
# ============================================================

user32 = ctypes.windll.user32

SCREEN_WIDTH = user32.GetSystemMetrics(0)
SCREEN_HEIGHT = user32.GetSystemMetrics(1)


# ============================================================
# MOUSE FUNCTIONS
# ============================================================

def set_mouse_position(x, y):

    x = int(
        max(
            0,
            min(
                SCREEN_WIDTH - 1,
                x
            )
        )
    )

    y = int(
        max(
            0,
            min(
                SCREEN_HEIGHT - 1,
                y
            )
        )
    )

    user32.SetCursorPos(
        x,
        y
    )


def get_mouse_position():

    point = ctypes.wintypes.POINT()

    user32.GetCursorPos(
        ctypes.byref(point)
    )

    return point.x, point.y


def mouse_left_click():

    LEFT_DOWN = 0x0002
    LEFT_UP = 0x0004

    user32.mouse_event(
        LEFT_DOWN,
        0,
        0,
        0,
        0
    )

    user32.mouse_event(
        LEFT_UP,
        0,
        0,
        0,
        0
    )


def mouse_double_click():

    mouse_left_click()

    time.sleep(0.055)

    mouse_left_click()


# ============================================================
# GEOMETRY
# ============================================================

def distance_3d(a, b):

    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z

    return math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )


def angle_3_points(a, b, c):

    abx = a.x - b.x
    aby = a.y - b.y
    abz = a.z - b.z

    cbx = c.x - b.x
    cby = c.y - b.y
    cbz = c.z - b.z

    ab_length = math.sqrt(
        abx * abx +
        aby * aby +
        abz * abz
    )

    cb_length = math.sqrt(
        cbx * cbx +
        cby * cby +
        cbz * cbz
    )

    if ab_length == 0 or cb_length == 0:
        return 0.0

    dot = (
        abx * cbx +
        aby * cby +
        abz * cbz
    )

    value = dot / (
        ab_length *
        cb_length
    )

    value = max(
        -1.0,
        min(
            1.0,
            value
        )
    )

    return math.degrees(
        math.acos(value)
    )


# ============================================================
# INDEX TRACKING
# ============================================================

def index_tracking_valid(hand):

    """
    Tolerant validation.

    Used for:
        - Mouse
        - Clicking
        - Writing tracking

    This is intentionally more tolerant than the old
    strict writing detector.

    The actual writing point is ALWAYS landmark 8.
    """

    wrist = hand[0]

    index_mcp = hand[5]
    index_pip = hand[6]
    index_tip = hand[8]

    wrist_to_tip = distance_3d(
        wrist,
        index_tip
    )

    mcp_to_tip = distance_3d(
        index_mcp,
        index_tip
    )

    pip_to_tip = distance_3d(
        index_pip,
        index_tip
    )

    mcp_to_pip = distance_3d(
        index_mcp,
        index_pip
    )

    if wrist_to_tip < 0.12:
        return False

    if mcp_to_tip < 0.055:
        return False

    if pip_to_tip < 0.030:
        return False

    if mcp_to_pip < 0.020:
        return False

    if mcp_to_tip <= pip_to_tip * 1.08:
        return False

    if wrist_to_tip <= distance_3d(
        wrist,
        index_mcp
    ) * 1.08:
        return False

    return True


# ============================================================
# INDEX EXTENSION
# ============================================================

def index_is_extended(hand):

    """
    More tolerant than the previous V3 detector.

    Used for deciding whether the user intends to
    control the index finger.

    The writing position itself still comes directly
    from landmark 8.
    """

    index_mcp = hand[5]
    index_pip = hand[6]
    index_dip = hand[7]
    index_tip = hand[8]

    mcp_to_tip = distance_3d(
        index_mcp,
        index_tip
    )

    mcp_to_pip = distance_3d(
        index_mcp,
        index_pip
    )

    pip_to_tip = distance_3d(
        index_pip,
        index_tip
    )

    if mcp_to_tip < 0.075:
        return False

    if mcp_to_pip < 0.020:
        return False

    if pip_to_tip < 0.022:
        return False

    pip_angle = angle_3_points(
        index_mcp,
        index_pip,
        index_dip
    )

    dip_angle = angle_3_points(
        index_pip,
        index_dip,
        index_tip
    )

    if pip_angle < 115:
        return False

    if dip_angle < 115:
        return False

    return True


# ============================================================
# FINGER EXTENSION
# ============================================================

def finger_is_extended(
    hand,
    tip_index,
    pip_index
):

    wrist = hand[0]

    tip = hand[tip_index]
    pip = hand[pip_index]

    wrist_to_tip = distance_3d(
        wrist,
        tip
    )

    wrist_to_pip = distance_3d(
        wrist,
        pip
    )

    return (
        wrist_to_tip >
        wrist_to_pip * 1.35
    )


def four_finger_gesture(hand):

    # --------------------------------------------------------
    # KEPT EXACTLY FROM V5.
    #
    # This preserves the existing photo gesture.
    # --------------------------------------------------------

    index = finger_is_extended(
        hand,
        8,
        6
    )

    middle = finger_is_extended(
        hand,
        12,
        10
    )

    ring = finger_is_extended(
        hand,
        16,
        14
    )

    pinky = finger_is_extended(
        hand,
        20,
        18
    )

    return (
        index and
        middle and
        ring and
        pinky
    )


# ============================================================
# POINT HISTORY
# ============================================================

class PointHistory:

    def __init__(self, size):

        self.points = deque(
            maxlen=size
        )

    def add(self, x, y):

        self.points.append(
            (x, y)
        )

    def clear(self):

        self.points.clear()

    def median(self):

        if not self.points:
            return None

        xs = sorted(
            p[0]
            for p in self.points
        )

        ys = sorted(
            p[1]
            for p in self.points
        )

        middle = len(xs) // 2

        if len(xs) % 2 == 0:

            x = (
                xs[middle - 1] +
                xs[middle]
            ) / 2

            y = (
                ys[middle - 1] +
                ys[middle]
            ) / 2

        else:

            x = xs[middle]
            y = ys[middle]

        return x, y


# ============================================================
# MOUSE SMOOTHING
# ============================================================

def mouse_smooth(
    old_x,
    old_y,
    new_x,
    new_y
):

    if old_x is None or old_y is None:
        return new_x, new_y

    dx = new_x - old_x
    dy = new_y - old_y

    movement = math.sqrt(
        dx * dx +
        dy * dy
    )

    # Ignore microscopic movement.
    if movement < MOUSE_DEADZONE:
        return old_x, old_y

    # Small movement = smooth.
    if movement < 8:
        alpha = 0.25

    # Normal movement.
    elif movement < 25:
        alpha = 0.55

    # Large movement should remain responsive.
    else:
        alpha = 0.85

    return (
        old_x + dx * alpha,
        old_y + dy * alpha
    )


# ============================================================
# WRITING SMOOTHING
# ============================================================

def writing_smooth(
    old_x,
    old_y,
    new_x,
    new_y
):

    if old_x is None or old_y is None:
        return new_x, new_y

    return (
        old_x +
        (new_x - old_x) *
        WRITE_SMOOTHING,

        old_y +
        (new_y - old_y) *
        WRITE_SMOOTHING
    )


# ============================================================
# CAMERA TO SCREEN
# ============================================================

def camera_to_screen(
    normalized_x,
    normalized_y
):

    # IMPORTANT:
    #
    # The camera frame has ALREADY been horizontally
    # flipped before MediaPipe.
    #
    # Therefore:
    #
    # DO NOT do:
    #
    # x = 1.0 - normalized_x
    #
    # That would mirror it a second time.

    x = (
        normalized_x -
        CAMERA_MARGIN_X
    ) / (
        1.0 -
        CAMERA_MARGIN_X * 2
    )

    y = (
        normalized_y -
        CAMERA_MARGIN_Y
    ) / (
        1.0 -
        CAMERA_MARGIN_Y * 2
    )

    x = max(
        0.0,
        min(
            1.0,
            x
        )
    )

    y = max(
        0.0,
        min(
            1.0,
            y
        )
    )

    # Edge snapping.

    if x < EDGE_ZONE:
        x = 0.0

    elif x > 1.0 - EDGE_ZONE:
        x = 1.0

    if y < EDGE_ZONE:
        y = 0.0

    elif y > 1.0 - EDGE_ZONE:
        y = 1.0

    return (
        x * (SCREEN_WIDTH - 1),
        y * (SCREEN_HEIGHT - 1)
    )


# ============================================================
# INDEX STATE
# ============================================================

@dataclass
class IndexState:

    active: bool = False

    good_frames: int = 0

    bad_frames: int = 0

    def update(self, detected):

        if detected:

            self.good_frames += 1
            self.bad_frames = 0

            if (
                self.good_frames >=
                INDEX_ACTIVATE_FRAMES
            ):

                self.active = True

        else:

            self.bad_frames += 1
            self.good_frames = 0

            if (
                self.bad_frames >=
                INDEX_DEACTIVATE_FRAMES
            ):

                self.active = False

    def reset(self):

        self.active = False
        self.good_frames = 0
        self.bad_frames = 0


# ============================================================
# PINCH STATE
# ============================================================

@dataclass
class PinchState:

    active: bool = False

    start_frames: int = 0

    release_frames: int = 0

    def update(self, touching):

        triggered = False

        if touching:

            self.release_frames = 0

            if not self.active:

                self.start_frames += 1

                if (
                    self.start_frames >=
                    PINCH_START_FRAMES
                ):

                    self.active = True
                    triggered = True

        else:

            self.start_frames = 0

            if self.active:

                self.release_frames += 1

                if (
                    self.release_frames >=
                    PINCH_RELEASE_FRAMES
                ):

                    self.active = False
                    self.release_frames = 0

        return triggered

    def reset(self):

        self.active = False
        self.start_frames = 0
        self.release_frames = 0


# ============================================================
# MOUSE STATE
# ============================================================

@dataclass
class MouseState:

    x: float | None = None
    y: float | None = None

    history: PointHistory | None = None

    def __post_init__(self):

        self.history = PointHistory(
            MOUSE_HISTORY_SIZE
        )

    def update(
        self,
        x,
        y
    ):

        self.history.add(
            x,
            y
        )

        point = self.history.median()

        if point is None:
            return None

        target_x, target_y = point

        self.x, self.y = mouse_smooth(
            self.x,
            self.y,
            target_x,
            target_y
        )

        return (
            self.x,
            self.y
        )

    def clear_history(self):

        self.history.clear()

    def reset(self):

        self.x = None
        self.y = None

        self.history.clear()


# ============================================================
# WRITING STATE
# ============================================================

@dataclass
class WritingState:

    drawing: bool = False

    last_x: float | None = None
    last_y: float | None = None

    history: PointHistory | None = None

    def __post_init__(self):

        self.history = PointHistory(
            WRITE_HISTORY_SIZE
        )

    def add_point(
        self,
        x,
        y
    ):

        self.history.add(
            x,
            y
        )

        point = self.history.median()

        if point is None:
            return None

        return (
            point[0] * WRITE_FOLLOW,
            point[1] * WRITE_FOLLOW
        )

    def start_stroke(self):

        self.drawing = True

    def break_stroke(self):

        self.drawing = False

        self.last_x = None
        self.last_y = None

        self.history.clear()

    def reset(self):

        self.break_stroke()


# ============================================================
# PHOTO STATE
# ============================================================

@dataclass
class PhotoState:

    hold_frames: int = 0

    active: bool = False

    def update(
        self,
        detected
    ):

        triggered = False

        if detected:

            self.hold_frames += 1

            if (
                self.hold_frames >=
                PHOTO_HOLD_FRAMES
            ):

                if not self.active:

                    self.active = True
                    triggered = True

        else:

            self.hold_frames = 0
            self.active = False

        return triggered

    def reset(self):

        self.hold_frames = 0
        self.active = False


# ============================================================
# PINCH DETECTION
# ============================================================

def thumb_index_pinch(hand):

    # ONLY thumb + index are used here.

    return (
        distance_3d(
            hand[4],
            hand[8]
        )
        < PINCH_THRESHOLD
    )


def thumb_middle_pinch(hand):

    # ONLY thumb + middle are used here.

    return (
        distance_3d(
            hand[4],
            hand[12]
        )
        < PINCH_THRESHOLD
    )


# ============================================================
# WRITING CANVAS
# ============================================================

canvas = None


def create_canvas(frame):

    global canvas

    if canvas is None:

        canvas = frame.copy()

        canvas[:] = (
            20,
            20,
            20
        )


def clear_canvas():

    global canvas

    canvas = None


def draw_writing_point(
    x,
    y,
    writing_state
):

    global canvas

    if canvas is None:
        return

    x = int(x)
    y = int(y)

    old_x = writing_state.last_x
    old_y = writing_state.last_y

    # --------------------------------------------------------
    # First point
    # --------------------------------------------------------

    if old_x is None or old_y is None:

        cv2.circle(
            canvas,
            (x, y),
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

        writing_state.last_x = x
        writing_state.last_y = y

        return

    # --------------------------------------------------------
    # Jump protection
    # --------------------------------------------------------

    jump = math.sqrt(
        (x - old_x) ** 2 +
        (y - old_y) ** 2
    )

    if jump > MAX_WRITE_JUMP:

        writing_state.break_stroke()

        writing_state.last_x = x
        writing_state.last_y = y

        return

    # --------------------------------------------------------
    # Draw line
    # --------------------------------------------------------

    cv2.line(
        canvas,
        (
            int(old_x),
            int(old_y)
        ),
        (
            x,
            y
        ),
        (255, 255, 255),
        WRITE_THICKNESS,
        cv2.LINE_AA
    )

    writing_state.last_x = x
    writing_state.last_y = y


# ============================================================
# HAND DRAWING
# ============================================================

HAND_CONNECTIONS = [

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
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


def draw_hand(
    frame,
    hand
):

    height, width = frame.shape[:2]

    points = []

    for landmark in hand:

        x = int(
            landmark.x *
            width
        )

        y = int(
            landmark.y *
            height
        )

        points.append(
            (x, y)
        )

    # --------------------------------------------------------
    # Skeleton
    # --------------------------------------------------------

    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            SKELETON_COLOR,
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Landmarks
    # --------------------------------------------------------
    #
    # Ring and pinky receive only a small neutral landmark.
    #
    # The important fingertips get their own colors below.
    # --------------------------------------------------------

    for i, point in enumerate(points):

        # Don't give ring/pinky special large markers.

        if i in (
            4,
            8,
            12
        ):
            continue

        cv2.circle(
            frame,
            point,
            2,
            LANDMARK_COLOR,
            -1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # INDEX TIP
    # --------------------------------------------------------
    #
    # GREEN
    #
    # This is the ACTUAL point used for writing and mouse
    # positioning.
    # --------------------------------------------------------

    cv2.circle(
        frame,
        points[8],
        11,
        INDEX_COLOR,
        2,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        points[8],
        5,
        INDEX_COLOR,
        -1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # THUMB TIP
    # --------------------------------------------------------
    #
    # BLUE
    #
    # Used for thumb + index click.
    # Used for thumb + middle double-click.
    # --------------------------------------------------------

    cv2.circle(
        frame,
        points[4],
        9,
        THUMB_COLOR,
        2,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        points[4],
        4,
        THUMB_COLOR,
        -1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # MIDDLE TIP
    # --------------------------------------------------------
    #
    # YELLOW
    #
    # Used with thumb for double-click/open.
    # --------------------------------------------------------

    cv2.circle(
        frame,
        points[12],
        8,
        MIDDLE_COLOR,
        2,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        points[12],
        4,
        MIDDLE_COLOR,
        -1,
        cv2.LINE_AA
    )


# ============================================================
# UI TEXT
# ============================================================

def draw_text(
    frame,
    text,
    x,
    y,
    scale=0.55,
    thickness=1
):

    cv2.putText(
        frame,
        text,
        (
            int(x),
            int(y)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# UI
# ============================================================

def draw_ui(
    frame,
    mode,
    fps,
    index_active,
    writing_active,
    click_active,
    open_active,
    photo_active
):

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # TOP BAR
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 78),
        (20, 20, 20),
        -1
    )

    draw_text(
        frame,
        f"FPS: {fps:.1f}",
        15,
        25
    )

    draw_text(
        frame,
        f"MODE: {mode}",
        120,
        25
    )

    index_text = (
        "INDEX OK"
        if index_active
        else
        "INDEX OFF"
    )

    draw_text(
        frame,
        index_text,
        15,
        55
    )

    if mode == "WRITE":

        status = (
            "WRITING"
            if writing_active
            else
            "READY"
        )

    elif mode == "MOUSE":

        if open_active:

            status = "OPEN"

        elif click_active:

            status = "CLICK"

        else:

            status = "MOUSE"

    else:

        status = "IDLE"

    draw_text(
        frame,
        status,
        140,
        55
    )

    # --------------------------------------------------------
    # BOTTOM BAR
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (
            0,
            height - 42
        ),
        (
            width,
            height
        ),
        (20, 20, 20),
        -1
    )

    controls = (
        "M Mouse | W Write | I Idle | "
        "C Clear | 1 Google | 2 Notepad | "
        "3 Calculator | S Search | ESC Exit"
    )

    draw_text(
        frame,
        controls,
        8,
        height - 15,
        0.36,
        1
    )

    if photo_active:

        draw_text(
            frame,
            "PHOTO!",
            width - 90,
            30,
            0.55,
            2
        )


# ============================================================
# APPLICATION FUNCTIONS
# ============================================================

def open_url(url):

    try:

        subprocess.Popen(
            [
                "cmd",
                "/c",
                "start",
                "",
                url
            ],
            shell=True
        )

        return True

    except Exception as error:

        print(
            "Could not open URL:",
            error
        )

        return False


def open_google():

    open_url(
        "https://www.google.com"
    )


def open_notepad():

    try:

        subprocess.Popen(
            ["notepad.exe"]
        )

        print(
            "Notepad opened."
        )

    except Exception as error:

        print(
            "Could not open Notepad:",
            error
        )


def open_calculator():

    try:

        subprocess.Popen(
            ["calc.exe"]
        )

        print(
            "Calculator opened."
        )

    except Exception as error:

        print(
            "Could not open Calculator:",
            error
        )


def open_search():

    # IMPORTANT:
    # No input() here.
    #
    # This simply opens Google Search.
    # The OpenCV loop therefore continues running.

    open_url(
        "https://www.google.com/search"
    )


# ============================================================
# PHOTO
# ============================================================

photo_number = 0


def take_photo(frame):

    global photo_number

    folder = "photos"

    os.makedirs(
        folder,
        exist_ok=True
    )

    photo_number += 1

    filename = os.path.join(
        folder,
        (
            f"photo_"
            f"{photo_number}_"
            f"{int(time.time())}.jpg"
        )
    )

    cv2.imwrite(
        filename,
        frame
    )

    print(
        f"Photo saved: {filename}"
    )


# ============================================================
# CHANGE MODE
# ============================================================

def change_mode(
    old_mode,
    new_mode,
    writing_state,
    mouse_state
):

    if old_mode == new_mode:
        return new_mode

    # Leaving writing mode.
    if old_mode == "WRITE":

        writing_state.reset()

    # Leaving mouse mode.
    if old_mode == "MOUSE":

        mouse_state.reset()

    return new_mode


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print()
    print(
        "ERROR: Could not open webcam."
    )
    print()

    landmarker.close()

    raise SystemExit


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

camera.set(
    cv2.CAP_PROP_FPS,
    TARGET_FPS
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# ============================================================
# CREATE STATES
# ============================================================

mode = "MOUSE"

index_state = IndexState()

click_state = PinchState()

open_state = PinchState()

mouse_state = MouseState()

writing_state = WritingState()

photo_state = PhotoState()


# ============================================================
# FPS
# ============================================================

fps = 0.0

fps_counter = 0

fps_timer = time.perf_counter()


# ============================================================
# MEDIAPIPE TIMESTAMP
# ============================================================

last_timestamp_ms = 0


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ====================================================
        # CAMERA
        # ====================================================

        success, frame = camera.read()

        if not success:

            print(
                "Could not read camera frame."
            )

            break

        # ----------------------------------------------------
        # Mirror image
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # Keep a clean frame for photos.
        original_frame = frame.copy()

        frame_height, frame_width = frame.shape[:2]

        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp_ms = int(
            time.perf_counter() * 1000
        )

        if timestamp_ms <= last_timestamp_ms:

            timestamp_ms = (
                last_timestamp_ms +
                1
            )

        last_timestamp_ms = timestamp_ms

        # ====================================================
        # HAND DETECTION
        # ====================================================

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # ------------------------------------------------
            # DRAW HAND
            # ------------------------------------------------
            #
            # V6 emphasizes only:
            #   Index
            #   Thumb
            #   Middle
            #
            # Ring/pinky remain internally available but
            # have no special interaction markers.
            # ------------------------------------------------

            draw_hand(
                frame,
                hand
            )

            # ------------------------------------------------
            # INDEX DETECTION
            # ------------------------------------------------

            tracking_valid = (
                index_tracking_valid(
                    hand
                )
            )

            index_extended = (
                index_is_extended(
                    hand
                )
            )

            index_state.update(
                index_extended
            )

            # =================================================
            # GET REAL INDEX PIXEL
            # =================================================
            #
            # Landmark 8 -> exact camera-frame pixel.
            #
            # No screen mapping.
            # No reverse mapping.
            # No second mirror.
            #
            # Therefore writing follows the visible fingertip.
            # =================================================

            index_tip = hand[8]

            raw_index_x = (
                index_tip.x *
                frame_width
            )

            raw_index_y = (
                index_tip.y *
                frame_height
            )

            # Draw a small marker exactly at the raw index
            # landmark.

            debug_x = int(raw_index_x)
            debug_y = int(raw_index_y)

            cv2.circle(
                frame,
                (
                    debug_x,
                    debug_y
                ),
                4,
                INDEX_COLOR,
                -1,
                cv2.LINE_AA
            )

            # =================================================
            # MOUSE
            # =================================================

            if (
                mode == "MOUSE"
                and
                tracking_valid
            ):

                screen_x, screen_y = (
                    camera_to_screen(
                        index_tip.x,
                        index_tip.y
                    )
                )

                # ------------------------------------------------
                # Lower mouse sensitivity.
                #
                # Instead of aggressively following every
                # fingertip pixel, movement is scaled relative
                # to the previous target.
                # ------------------------------------------------

                if mouse_state.x is None:

                    mouse_state.x = screen_x
                    mouse_state.y = screen_y

                else:

                    dx = (
                        screen_x -
                        mouse_state.x
                    )

                    dy = (
                        screen_y -
                        mouse_state.y
                    )

                    mouse_state.x += (
                        dx *
                        MOUSE_SENSITIVITY
                    )

                    mouse_state.y += (
                        dy *
                        MOUSE_SENSITIVITY
                    )

                set_mouse_position(
                    mouse_state.x,
                    mouse_state.y
                )

            # =================================================
            # WRITING
            # =================================================

            if mode == "WRITE":

                # ------------------------------------------------
                # Writing requires a usable index.
                # ------------------------------------------------

                if (
                    index_state.active
                    and
                    tracking_valid
                ):

                    # ------------------------------------------------
                    # DIRECT CAMERA PIXEL
                    # ------------------------------------------------
                    #
                    # DO NOT convert this to screen coordinates.
                    #
                    # This makes writing follow the fingertip.
                    # ------------------------------------------------

                    write_x = raw_index_x
                    write_y = raw_index_y

                    # Small history only.
                    point = (
                        writing_state.add_point(
                            write_x,
                            write_y
                        )
                    )

                    if point is not None:

                        target_x, target_y = point

                        # ------------------------------------------------
                        # Very light smoothing.
                        # ------------------------------------------------

                        smooth_x, smooth_y = (
                            writing_smooth(
                                writing_state.last_x,
                                writing_state.last_y,
                                target_x,
                                target_y
                            )
                        )

                        # ------------------------------------------------
                        # Start stroke.
                        # ------------------------------------------------

                        writing_state.start_stroke()

                        draw_writing_point(
                            smooth_x,
                            smooth_y,
                            writing_state
                        )

                else:

                    # Finger not usable.
                    # Break only the current stroke.
                    # Canvas remains.

                    writing_state.break_stroke()

            # =================================================
            # LEFT CLICK
            # =================================================
            #
            # ONLY THUMB + INDEX.
            # Ring and pinky have no role.
            # =================================================

            click_detected = (
                tracking_valid
                and
                thumb_index_pinch(
                    hand
                )
            )

            click_triggered = (
                click_state.update(
                    click_detected
                )
            )

            if click_triggered:

                mouse_left_click()

            # =================================================
            # DOUBLE CLICK / OPEN
            # =================================================
            #
            # ONLY THUMB + MIDDLE.
            # Ring and pinky have no role.
            # =================================================

            open_detected = (
                thumb_middle_pinch(
                    hand
                )
            )

            open_triggered = (
                open_state.update(
                    open_detected
                )
            )

            if open_triggered:

                mouse_double_click()

            # =================================================
            # PHOTO
            # =================================================
            #
            # PRESERVED FROM V5.
            #
            # The existing photo gesture still uses all four
            # fingers so no V5 functionality is removed.
            # =================================================

            photo_detected = (
                four_finger_gesture(
                    hand
                )
            )

            photo_triggered = (
                photo_state.update(
                    photo_detected
                )
            )

            if photo_triggered:

                take_photo(
                    original_frame
                )

        # ====================================================
        # NO HAND
        # ====================================================

        else:

            index_state.reset()

            click_state.reset()

            open_state.reset()

            photo_state.reset()

            mouse_state.reset()

            if mode == "WRITE":

                writing_state.break_stroke()

        # ====================================================
        # DISPLAY FRAME
        # ====================================================

        display_frame = frame

        if (
            mode == "WRITE"
            and
            canvas is not None
        ):

            display_frame = cv2.addWeighted(
                frame,
                0.35,
                canvas,
                0.65,
                0
            )

        # ====================================================
        # FPS
        # ====================================================

        fps_counter += 1

        current_time = time.perf_counter()

        elapsed = (
            current_time -
            fps_timer
        )

        if elapsed >= 1.0:

            fps = (
                fps_counter /
                elapsed
            )

            fps_counter = 0

            fps_timer = current_time

        # ====================================================
        # UI
        # ====================================================

        draw_ui(
            display_frame,
            mode,
            fps,
            index_state.active,
            writing_state.drawing,
            click_state.active,
            open_state.active,
            photo_state.active
        )

        # ====================================================
        # WINDOW
        # ====================================================

        cv2.imshow(
            "Air Controller V6",
            display_frame
        )

        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # ESC
        # ----------------------------------------------------

        if key == 27:

            break

        # ----------------------------------------------------
        # MOUSE MODE
        # ----------------------------------------------------

        elif key in (
            ord("m"),
            ord("M")
        ):

            mode = change_mode(
                mode,
                "MOUSE",
                writing_state,
                mouse_state
            )

            print(
                "Mode: MOUSE"
            )

        # ----------------------------------------------------
        # WRITING MODE
        # ----------------------------------------------------

        elif key in (
            ord("w"),
            ord("W")
        ):

            mode = change_mode(
                mode,
                "WRITE",
                writing_state,
                mouse_state
            )

            create_canvas(
                frame
            )

            writing_state.reset()

            print(
                "Mode: WRITE"
            )

        # ----------------------------------------------------
        # IDLE MODE
        # ----------------------------------------------------

        elif key in (
            ord("i"),
            ord("I")
        ):

            mode = change_mode(
                mode,
                "IDLE",
                writing_state,
                mouse_state
            )

            print(
                "Mode: IDLE"
            )

        # ----------------------------------------------------
        # CLEAR CANVAS
        # ----------------------------------------------------

        elif key in (
            ord("c"),
            ord("C")
        ):

            clear_canvas()

            writing_state.reset()

            print(
                "Canvas cleared."
            )

        # ----------------------------------------------------
        # GOOGLE
        # ----------------------------------------------------

        elif key == ord("1"):

            open_google()

        # ----------------------------------------------------
        # NOTEPAD
        # ----------------------------------------------------

        elif key == ord("2"):

            open_notepad()

        # ----------------------------------------------------
        # CALCULATOR
        # ----------------------------------------------------

        elif key == ord("3"):

            open_calculator()

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        elif key in (
            ord("s"),
            ord("S")
        ):

            open_search()


# ============================================================
# CLEANUP
# ============================================================

except KeyboardInterrupt:

    print()
    print(
        "Program stopped."
    )

finally:

    camera.release()

    cv2.destroyAllWindows()

    landmarker.close()

    print(
        "Air Controller V6 closed."
    )