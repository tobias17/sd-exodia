from settings import *
from render import (
    run_through_sd,
    splice_images,
    get_temp_paths,
    full_payload,
    face_payload,
    SOURCE_W,
    SOURCE_H,
)

from tqdm import tqdm
import numpy as np
import cv2
import sys
import os

#=========================================================================================
# Settings                # Descriptions
#=========================================================================================
USE_CONTROLNET = True     # Whether the code will run Stable Diffusion with a ControlNet
INCLUDE_TURNTABLE = True  # Will be added to the very left
IMGS_LEFT  = 0            # How many images are placed to the left of the image being run
IMGS_RIGHT = 1            # Same as above, but to the right

run = [                   # The settings to run through Stable Diffusion
    {
        "type": "single",
        "payload": {
            "denoising_strength": 0.15,
            "steps": 40
        }
    },
    {
        "type": "face",
        "payload": {
            "denoising_strength": 0.10,
            "steps": 40
        }
    }
]




def run_folder(anim_folder):
    anim_root = f"{ANIM_PATH}/{anim_folder}"
    assert os.path.exists(anim_root), f"Could not find anim_root '{anim_root}', check anim_name param or make anim_folder and run render.py"
    render_root = f"{RENDERS_PATH}/{anim_folder}"
    assert os.path.exists(render_root), f"Could not find render_root '{render_root}', check anim_name param or run render.py"

    for iter in range(1, 1000):
        touch_file_path = f"{render_root}/.iter{iter:03}"
        if not os.path.exists(touch_file_path):  break
    iter_str = f"iter{iter:03}"
    print(iter_str)

    files = []
    turntable_img  = cv2.imread(TURNTABLE_IMAGE_PATH) if INCLUDE_TURNTABLE else None
    turntable_pose = cv2.imread(TURNTABLE_POSE_PATH)  if INCLUDE_TURNTABLE else None
    assert not INCLUDE_TURNTABLE or turntable_img is not None, f"Turntable image loaded to None when INCLUDE_TURNTABLE was active, searched '{TURNTABLE_IMAGE_PATH}'"
    assert not INCLUDE_TURNTABLE or turntable_pose is not None, f"Turntable pose loaded to None when INCLUDE_TURNTABLE was active, searched '{TURNTABLE_POSE_PATH}'"

    filenames = [f.replace(".json", "") for f in os.listdir(anim_root) if f.endswith(".json") and "settings" not in f]
    assert len(filenames) > 0, f"Could not find any animation files, searched: '{anim_root}'"
    for anim_name in filenames:
        if USE_CONTROLNET and not os.path.exists(f"{anim_root}/{anim_name}.png"):
            print(f"WARNING: could not find pose image {anim_name}.png with ControlNet active, skipping")
            continue

        rendered_path  = f"{render_root}/iter{iter - 1:03}_n_{anim_name}.png"
        body_mask_path = f"{render_root}/iter000_m_{anim_name}.png"
        face_mask_path = f"{render_root}/iter000_f_{anim_name}.png" if face_payload is not None else None
        pose_path      = f"{anim_root}/{anim_name}.png" if USE_CONTROLNET else None

        assert os.path.exists(rendered_path),  f"Could not find rendered image '{rendered_path}'"
        assert os.path.exists(body_mask_path), f"Could not find body mask image '{body_mask_path}'"
        assert os.path.exists(face_mask_path), f"Could not find face mask image '{face_mask_path}'"
        
        rendered_img  = cv2.imread(rendered_path)
        body_mask_img = cv2.imread(body_mask_path, cv2.IMREAD_GRAYSCALE)
        face_mask_img = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE)
        pose_img      = cv2.imread(pose_path) if USE_CONTROLNET else None

        files.append({
            "name": anim_name,
            "input": rendered_img,
            "body_mask": body_mask_img,
            "face_mask": face_mask_img,
            "pose": pose_img,
        })

    def wrap(x):
        l = len(files)
        while x <  0:  x += l
        while x >= l:  x -= l
        return x

    black_img = np.zeros((SOURCE_H, SOURCE_W))

    for i in tqdm(range(len(files))):
        anim_name = files[i]["name"]
        input_img = files[i]["input"]
        body_mask = files[i]["body_mask"]
        face_mask = files[i]["face_mask"]
        pose_img  = files[i]["pose"]

        for dy in range(-1, -(IMGS_LEFT + 1), -1):
            input_img = splice_images(files[wrap(i + dy)]["input"], input_img)
            body_mask = splice_images(black_img,                    body_mask)
            face_mask = splice_images(black_img,                    face_mask) if face_mask is not None else None
            pose_img  = splice_images(files[wrap(i + dy)]["pose"],  pose_img)  if pose_img  is not None else None
        if INCLUDE_TURNTABLE:
            input_img = splice_images(turntable_img,  input_img)
            body_mask = splice_images(black_img,      body_mask)
            face_mask = splice_images(black_img,      face_mask) if face_mask is not None else None
            pose_img  = splice_images(turntable_pose, pose_img)  if pose_img  is not None else None
        for dx in range(1, IMGS_RIGHT + 1):
            input_img = splice_images(input_img, files[wrap(i + dx)]["input"])
            body_mask = splice_images(body_mask, black_img                   )
            face_mask = splice_images(face_mask, black_img                   ) if face_mask is not None else None
            pose_img  = splice_images(pose_img,  files[wrap(i + dx)]["pose"] ) if pose_img  is not None else None
        
        paths = get_temp_paths(anim_folder, anim_name)
        cv2.imwrite(paths["img"], input_img)
        cv2.imwrite(paths["mask"], body_mask)
        if face_mask is not None:  cv2.imwrite(paths["face"], face_mask)
        if pose_img  is not None:  cv2.imwrite(paths["pose"], pose_img)
        
        x_split1 = SOURCE_W * (IMGS_LEFT + (1 if INCLUDE_TURNTABLE else 0))
        width_override = SOURCE_W * (1 + IMGS_LEFT + IMGS_RIGHT + (1 if INCLUDE_TURNTABLE else 0))
        run_through_sd(anim_folder, anim_name, paths, full_payload, face_payload, run, f"{render_root}/{iter_str}", x_split1, x_split1 + SOURCE_W, width_override=width_override)
    
    with open(touch_file_path, "w") as f:
        f.write("Thanks for checking out my software! (this file demarks that this iteration ran to completion)")

if __name__ == "__main__":
    assert len(sys.argv) == 2, f"{sys.argv[0]} requires 1 argument: <anim_name>, {len(sys.argv) - 1} arguments provided"
    run_folder(sys.argv[1])
