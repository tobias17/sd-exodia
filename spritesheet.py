from settings import *

import numpy as np
import cv2
import os
import math
import subprocess
from tqdm import tqdm

anims_to_join = [ "idle", "srun-interp", "frun-interp" ]
SPRITESHEET_RESCALE = 0.5        # 
SPRITESHEET_MAX_WIDTH = 16384    # 

DO_ALL = True
RE_STRIP = False

def create_spritesheet(imgs):
    row_count = 1
    row_size = len(imgs)
    h, w = imgs[0].shape[0], sum([img.shape[1] for img in imgs])
    dy = h

    new_w = w
    while new_w > SPRITESHEET_MAX_WIDTH:
        row_count *= 2
        row_size = math.ceil(len(imgs) / float(row_count))
        new_w = row_size * imgs[0].shape[0]
    h, w = h * row_count, new_w

    print(f"Final spritesheet size: {row_size}x{row_count} ({w}x{h})")

    output_img = np.zeros((h, w, imgs[0].shape[2]))
    img_i = 0
    for row_i in range(row_count):
        x = 0
        for _ in range(row_size):
            img = imgs[img_i]
            dx = img.shape[1]

            output_img[dy*row_i:dy*(row_i+1),x:x+dx] = img

            x += dx
            img_i += 1
            if img_i >= len(imgs):  break
        if img_i >= len(imgs):  break

    return output_img

def main():
    assert os.path.exists(RENDERS_PATH), f"Could not find renders root, run render.py or check settings, settings.py says it is RENDERS_PATH='{RENDERS_PATH}'"
    
    sprites = []
    for anim_folder in anims_to_join:
        render_root = f"{RENDERS_PATH}/{anim_folder}"
        assert os.path.exists(render_root), f"Could not find render_root '{render_root}', check anims_to_join variable or run render.py"

        for iter in range(1, 1000):
            touch_file_path = f"{render_root}/.iter{iter:03}"
            if not os.path.exists(touch_file_path):  break
        iter_str = f"iter{(iter - 1):03}"
        print(f"{anim_folder}: {iter_str}")

        files = [f for f in os.listdir(render_root) if (DO_ALL or f"{iter_str}_n_" in f) and ("_s.png" not in f and "_s_" not in f)]
        print(f"  {files}")
        for file in tqdm(files):
            stripped = f"{render_root}/" + (file.replace(".png", "_s.png") if DO_ALL else file.replace(f"{iter_str}_n_", f"{iter_str}_s_"))
            assert file not in stripped, "did not change filename"
            if RE_STRIP or not os.path.exists(stripped):
                subprocess.call(["rembg", "i", f"{render_root}/{file}", stripped])
            img = cv2.imread(stripped, cv2.IMREAD_UNCHANGED)
            assert img is not None, f"Stripped img was not loaded properly, tried with path '{stripped}'"
            sprites.append(cv2.resize(img, (int(SPRITESHEET_RESCALE*img.shape[1]), int(SPRITESHEET_RESCALE*img.shape[0]))))
    
    sheet = create_spritesheet(sprites)
    cv2.imwrite(f"{WORKSPACE}/{SPRITESHEET_NAME}", sheet)

if __name__ == "__main__":
    main()