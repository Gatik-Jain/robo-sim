"""Default simulation parameters for ROBO-SIM.

Every value here is in *simulation units* (abbreviated "u" in the interface).
They are not millimetres, centimetres or any physical measurement -- the whole
simulation is scale-free, so only the ratios between the numbers matter.

Keeping the defaults in one module means the web interface, the experiment
runner and the unit tests all start from exactly the same robot.
"""

# --- Robot geometry -------------------------------------------------------

BASE_X = 0.0
BASE_Y = 0.0

LINK1_LENGTH = 170.0
LINK2_LENGTH = 130.0

# Allowed range for the link-length inputs in the interface.
MIN_LINK_LENGTH = 20.0
MAX_LINK_LENGTH = 300.0

# --- Joint limits (degrees) ----------------------------------------------
# Joint 1 is measured from the positive X axis.
# Joint 2 is measured *relative to link 1*.

JOINT1_MIN = -180.0
JOINT1_MAX = 180.0
JOINT2_MIN = -150.0
JOINT2_MAX = 150.0

JOINT1_INITIAL = 35.0
JOINT2_INITIAL = -60.0

# --- Collision geometry ---------------------------------------------------

# Each link is modelled as a line segment swollen by this radius, i.e. a
# capsule. The radius is what gives the links their physical thickness.
LINK_RADIUS = 7.5

# The gripper at the tip is modelled as a disc.
END_EFFECTOR_RADIUS = 9.0

# Distances closer than this count as touching. Contact is treated as a
# collision, so the comparison is `distance <= radius + CONTACT_EPS`.
CONTACT_EPS = 1e-9

# --- Target ---------------------------------------------------------------

TARGET_X_INITIAL = 180.0
TARGET_Y_INITIAL = 120.0

# A planned motion counts as having reached the target when the end effector
# finishes within this distance of it. With a 5 degree grid the tip of a
# 300 unit arm moves up to ~26 units per step, so a tolerance below ~8 units
# frequently leaves no discrete goal state at all.
TARGET_TOLERANCE = 12.0
MIN_TARGET_TOLERANCE = 4.0
MAX_TARGET_TOLERANCE = 40.0

# --- Planning -------------------------------------------------------------

# Resolutions are restricted to values that divide both joint ranges exactly,
# so that the configured joint limits are themselves grid angles.
SUPPORTED_RESOLUTIONS = (3.0, 5.0, 10.0)
DEFAULT_RESOLUTION = 5.0

# Maximum change in any single joint angle between two collision checks when
# validating a transition. One degree is finer than the 3 degree planning grid,
# so every planned edge is sampled at least three times internally.
TRANSITION_STEP_DEG = 1.0

# Safety valve for the hosted application: refuse to build a search graph
# larger than this many states.
MAX_GRID_STATES = 30000

# --- Obstacles ------------------------------------------------------------

OBSTACLE_COORD_LIMIT = 600.0
MIN_OBSTACLE_SIZE = 1.0
MAX_OBSTACLES = 12

# --- Animation ------------------------------------------------------------

# Upper bound on the number of Plotly animation frames. Longer paths are
# sub-sampled for display only; validation always uses the full path.
MAX_ANIMATION_FRAMES = 60

# --- Visual palette -------------------------------------------------------
# A restrained engineering palette, reused by every figure.

COLOR_BACKGROUND = "#0f1419"
COLOR_PANEL = "#161c23"
COLOR_GRID = "#2a333d"
COLOR_AXIS = "#55636f"
COLOR_LINK1 = "#4c9be8"
COLOR_LINK2 = "#6fd3c7"
COLOR_JOINT = "#e8eef4"
COLOR_BASE = "#8e9aa6"
COLOR_END_EFFECTOR = "#ffc14d"
COLOR_TARGET = "#ff6b6b"
COLOR_OBSTACLE = "#7a4a52"
COLOR_OBSTACLE_LINE = "#c2616f"
COLOR_TRAJECTORY = "#b58cf0"
COLOR_WORKSPACE = "#3a4650"
COLOR_COLLISION = "#ff4d4d"
COLOR_TEXT = "#d8e0e8"
