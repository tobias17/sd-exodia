from settings import *
from render import split_image

import subprocess
import cv2
import os
import sys

def main(anim_folder):
    input_path = f"{TEMP_PATH}/{anim_folder}/source.png"
    assert os.path.exists(input_path), f"Could not find source image to split, expected path '{input_path}'"

    img = cv2.imread(input_path)
    img = split_image(img, img.shape[1]//2)
    src_img_path = get_src_img_path(anim_folder)
    cv2.imwrite(src_img_path, img)

    src_stripped_path = get_src_stripped_path(anim_folder)
    subprocess.call(["rembg", "i", src_img_path, src_stripped_path])

if __name__ == "__main__":
    assert len(sys.argv) == 2, f"{sys.argv[0]} requires 1 argument: <anim_folder>, {len(sys.argv) - 1} arguments provided"
    main(sys.argv[1])
