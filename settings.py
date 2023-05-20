WORKSPACE = "workspaces/template"

SPRITESHEET_NAME = "sheet.png"

TURNTABLE_RAW_PATH    = f"{WORKSPACE}/turntable-image.png"
TURNTABLE_IMAGE_PATH  = f"{WORKSPACE}/turntable-cleaned.png"
TURNTABLE_POSE_PATH   = f"{WORKSPACE}/turntable-pose.png"

def get_src_pose_png_path(anim_folder):  return f"{WORKSPACE}/comps/{anim_folder}/source-pose.png"
def get_src_pose_json_path(anim_folder): return f"{WORKSPACE}/comps/{anim_folder}/source-pose.json"
def get_src_img_path(anim_folder):       return f"{WORKSPACE}/comps/{anim_folder}/source-image.png"
def get_src_stripped_path(anim_folder):  return f"{WORKSPACE}/comps/{anim_folder}/source-stripped.png"

COMPS_PATH   = f"{WORKSPACE}/comps"
ANIM_PATH    = f"{WORKSPACE}/anims"
RENDERS_PATH = f"{WORKSPACE}/renders"
TEMP_PATH    = f"{WORKSPACE}/tmp"
DEBUG_FOLDER = f"{WORKSPACE}/debug"

REVERSE_ARMS = False
REVERSE_LEGS = False

index_to_tag = [
    "mouth",
    "chest",
    "lshoulder" if REVERSE_ARMS else "rshoulder",
    "lelbow"    if REVERSE_ARMS else "relbow",
    "lhand"     if REVERSE_ARMS else "rhand",
    "rshoulder" if REVERSE_ARMS else "lshoulder",
    "relbow"    if REVERSE_ARMS else "lelbow",
    "rhand"     if REVERSE_ARMS else "lhand",
    "lhip"  if REVERSE_LEGS else "rhip",
    "lknee" if REVERSE_LEGS else "rknee",
    "lfoot" if REVERSE_LEGS else "rfoot",
    "rhip"  if REVERSE_LEGS else "lhip",
    "rknee" if REVERSE_LEGS else "lknee",
    "rfoot" if REVERSE_LEGS else "lfoot",
    "reye",
    "leye",
    "rear",
    "lear",
]

groups = {
    "larm": [
        "lshoulder",
        "lelbow",
        "lhand",
    ],
    "lleg": [
        "lhip",
        "lknee",
        "lfoot",
    ],
    "body": [
        "mouth",
        "rhip",
        "lhip",
    ],
    "rleg": [
        "rhip",
        "rknee",
        "rfoot",
    ],
    "rarm": [
        "rshoulder",
        "relbow",
        "rhand",
    ],
}
order = list(groups.keys())
arm = ["lower", "upper"]
leg = ["lower", "upper"]

def get_index(key):
    for i, name in enumerate(index_to_tag):
        if name == key:
            return i
    raise Exception(f"Could not find index for key '{key}'")

def get_point(key, data):
    for i, point in enumerate(data["keypoints"]):
        if index_to_tag[i] == key:
            return point
    raise Exception(f"Could not find point with key '{key}'")
