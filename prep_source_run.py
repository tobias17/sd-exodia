from settings import *
from render import splice_images, overlay_images

import subprocess
import numpy as np
import cv2
import os
import sys

def main(anim_folder):
    assert os.path.exists(TURNTABLE_POSE_PATH),  f"Could not find turntable pose image, settings.py says it should be at TURNTABLE_POSE_PATH='{TURNTABLE_POSE_PATH}'"
    assert os.path.exists(TURNTABLE_RAW_PATH),   f"Could not find turntable generated image, settings.py says it should be at TURNTABLE_RAW_PATH='{TURNTABLE_RAW_PATH}'"
    src_pose_png_path = get_src_pose_png_path(anim_folder)
    assert os.path.exists(src_pose_png_path), f"Could not find source pose image, settings.py says it should be at src_pose_png_path='{src_pose_png_path}'"

    root_dir = f"{TEMP_PATH}/{anim_folder}"
    if not os.path.exists(TEMP_PATH):  os.mkdir(TEMP_PATH)
    if not os.path.exists(root_dir):   os.mkdir(root_dir)

    stripped_path = f"{root_dir}/turntable-stripped.png"
    subprocess.call(["rembg", "i", TURNTABLE_RAW_PATH, stripped_path])
    stripped_tt   = cv2.imread(stripped_path, cv2.IMREAD_UNCHANGED)
    gray_img      = np.zeros(stripped_tt.shape)
    gray_img[:,:] = (127, 127, 127, 255)
    cleaned_tt    = overlay_images(gray_img.copy(), stripped_tt)
    cv2.imwrite(TURNTABLE_IMAGE_PATH, cleaned_tt)

    comb_image = splice_images(cleaned_tt, gray_img.copy())
    comb_pose  = splice_images(cv2.imread(TURNTABLE_POSE_PATH), cv2.imread(src_pose_png_path))

    cv2.imwrite(f"{root_dir}/image.png", comb_image)
    cv2.imwrite(f"{root_dir}/pose.png",  comb_pose)

if __name__ == "__main__":
    assert len(sys.argv) == 2, f"{sys.argv[0]} requires 1 argument: <anim_folder>, {len(sys.argv) - 1} arguments provided"
    main(sys.argv[1])