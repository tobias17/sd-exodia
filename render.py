from settings import *

from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import json
import math
import os
import io
import sys
import base64
import requests

#=========================================================================================
# Settings                        # Descriptions
#=========================================================================================
SD_URL = "http://127.0.0.1:7860"  # The Automatic1111 webui API url
USE_CONTROLNET = True             # Whether the code will run Stable Diffusion with a ControlNet
SOURCE_W, SOURCE_H = 512, 512     # The width and height of the images, make sure your poses follow this too

char_desc  =  "female scientist, (white skin:1.4), (blonde hair:1.3), wearing a white lab coat, gray pants, white shoes, (hair in a bun:1.4)"
pos_prompt = f"((best quality)), ((masterpiece)), (detailed), charturnerv2, character reference sheet, {char_desc}, solid gray background, same character, identical characters, identical outfit, identical hair, identical face, multiple of the same person, (perfect hands:1.2), (perfect face:1.2), 4k, 8k, ultrarealism, (photorealistic:1.3)"
neg_prompt =  "<lora:EasyNegative:1.5>, ((hands in pocket)), ((hidden hands)), text, watermark, logo, blurry, weapon, sword, shield, holding things, items, spears, blades, clubs, handles, lighting, edge lighting, harsh lighting, deformed face, sexy, skimpy, revealing clothes, (multiple legs:1.2), (multiple limbs:1.2), (cape:1.3), (draping clothing:1.3), (wrist bands:1.3), (wearing things on wrist:1.3)"

default_payload = {               # The default payload that gets added to every run, values can get overwritten
    "prompt": pos_prompt,
    "negative_prompt": neg_prompt,
    "cfg_scale": 8,

    "alwayson_scripts": {
        "ControlNet": {
            "args": [
                {
                    "module": "none",
                    "model": "controlnet11Models_openpose [73c2b67d]",
                    "weight": 1,
                    "lowvram": True,
                },
            ],
        },
    },

    "width": 2*SOURCE_W,
    "height":  SOURCE_H,
    "sampler_index": "Euler a",
    "inpainting_fill": 1,
    "inpaint_full_res": False,
}

full_payload = {                  # The default "full" payload (single, median) to add to default_payload
}

face_payload = {                  # The default face payload to add to default_payload
    "inpaint_full_res": True,
    "inpaint_full_res_padding": 16,
}

run = [] # will be loaded from disk






#############################
#                           #
#    Helper Funcs Classes   #
#                           #
#############################

DEBUG = False
SHOW_INTERM = False

def to_circle(angle, normalize_to=0):
    while angle >  math.pi + normalize_to:  angle -= 2*math.pi
    while angle < -math.pi + normalize_to:  angle += 2*math.pi
    return angle

def overlay_images(bg, fg):
    abg = bg[:,:,3]/255.0
    afg = fg[:,:,3]/255.0
    for c in range(3):
        bg[:,:,c] = afg * fg[:,:,c]  +  abg * bg[:,:,c] * (1-afg)
    bg[:,:,3] = (1 - (1 - afg) * (1 - abg)) * 255
    return bg

def splice_images(left_img, right_img):
    assert left_img is not None,  "Left image was None when passed into splice_images"
    assert right_img is not None, "Right image was None when passed into splice_images"
    assert len(left_img.shape) == len(right_img.shape), f"Left and Right images must have the shape length, got {len(left_img.shape)} vs {len(right_img.shape)} -> {{ {left_img.shape}, {right_img.shape} }}"
    assert left_img.shape[0]   == right_img.shape[0],   f"Left and Right images must have same height to splice, got {left_img.shape[0]} != {right_img.shape[0]} -> {{ {left_img.shape}, {right_img.shape} }}"
    if len(left_img.shape) >= 3:
        assert left_img.shape[2] == right_img.shape[2], f"Left and Right images must have same color depth to splice, got {left_img.shape[2]} != {right_img.shape[2]}"
        full_img = np.zeros((left_img.shape[0], left_img.shape[1] + right_img.shape[1], left_img.shape[2]))
        full_img[:,:left_img.shape[1],:] = left_img
        full_img[:,left_img.shape[1]:,:] = right_img
    else:
        full_img = np.zeros((left_img.shape[0], left_img.shape[1] + right_img.shape[1]))
        full_img[:,:left_img.shape[1]] = left_img
        full_img[:,left_img.shape[1]:] = right_img
    return full_img

def split_image(full_img, x_split1, x_split2=None):
    if x_split2 is None:
        right_img = np.zeros((full_img.shape[0], full_img.shape[1] - x_split1, full_img.shape[2]))
        right_img[:,:,:] = full_img[:,x_split1:,:]
        return right_img
    else:
        right_img = np.zeros((full_img.shape[0], x_split2 - x_split1, full_img.shape[2]))
        right_img[:,:,:] = full_img[:,x_split1:x_split2,:]
        return right_img

def read_encode_img(img_path):
    with open(img_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def save_encoded_img(enc_img, save_path):
    image = Image.open(io.BytesIO(base64.b64decode(enc_img.split(",",1)[0])))
    image.save(save_path)

def call_sd(payload_mod, img, mask, control):
    payload = {}
    payload.update(default_payload)
    payload["init_images"] = [img]
    payload["mask"] = mask
    if control is not None:  payload["alwayson_scripts"]["ControlNet"]["args"][0]["input_image"] = control
    else:                    payload.pop("alwayson_scripts")
    payload.update(payload_mod)

    response = requests.post(url=f'{SD_URL}/sdapi/v1/img2img', json=payload)
    resp_json = response.json()

    if 'images' not in resp_json:
        print(resp_json)
        raise RuntimeError("Could not find key 'images' in response from server.")
    
    return resp_json['images']

class Vector2:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]
    def point(self):       return [self.x, self.y]
    def point_int(self):   return [int(self.x), int(self.y)]
    def __sub__(self, o):  return Vector2([self.x - o.x, self.y - o.y])
    def __add__(self, o):  return Vector2([self.x + o.x, self.y + o.y])
    def angle(self):       return math.atan2(self.y, self.x)
    def length(self):      return math.sqrt(self.x**2 + self.y**2)
    def __str__(self):     return f"Point(x={self.x},y={self.y})"

    def __mul__(self, o):
        if type(o) == type(self):  return Vector2([self.x * o.x, self.y * o.y])
        else:                      return Vector2([self.x * o,   self.y * o  ])

    def normalize(self, length=1):
        assert length != 0, "Can not normalize a vector to length 0!"
        assert self.x != 0 or self.y != 0, "Can not normalize a vector of length 0!"
        div = self.length() / float(length)
        return Vector2([self.x / div, self.y / div])

class TransformBoundary:
    dist = 1024

    def __init__(self, comp1, comp2):
        self.comp1 = comp1
        self.comp2 = comp2

        self.diff = to_circle(comp1.vec.angle() - comp2.vec.angle()) / 2
        self.angle = math.pi/2 + comp1.vec.angle() - self.diff
        self.vec = Vector2([math.cos(self.angle), math.sin(self.angle)])
    
    def draw_mask_on(self, img, color):
        angle = self.angle
        mod_dist = Vector2([math.cos(angle - math.pi/2) * self.dist, math.sin(angle - math.pi/2) * self.dist])
        cv2.line(img, (self.comp1.p2 + mod_dist - self.vec*self.dist).point_int(), (self.comp1.p2 + mod_dist + self.vec*self.dist).point_int(), color, int(self.dist*2))

    def draw_debug_on(self, img, p1, length, color, size):
        cv2.line(img, (p1 + (self.vec * length)).point_int(), (p1 - (self.vec * length)).point_int(), color, size)
        cv2.circle(img, (p1 + (self.vec * length)).point_int(), 5, (0, 0, 255, 255), -1)

class TransformComponent:
    transform_offset = 40
    tp_1, tp_2 = None, None

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.vec = p2 - p1
    
    def draw_boundary_on(self, draw_img, ref_img, angle):
        root = self.p1
        start_angle = self.vec.angle()*180/math.pi + (90 if angle > 0 else 270)
        end_angle = start_angle + angle
        move_vec = Vector2([math.cos(start_angle*math.pi/180), math.sin(start_angle*math.pi/180)])
        for i in range(1, 50+1):
            point = (root + (move_vec * i)).point_int()
            color = [int(v) for v in list(ref_img[point[1], point[0]])]
            cv2.ellipse(draw_img, root.point_int(), (i,i), 0, start_angle, end_angle, color, 2)

    def get_transform_points(self):
        points = []
        for is_left in [True, False]:
            for is_first in [True, False]:
                dir_vec = (self.p1 - self.p2 if is_first else self.p2 - self.p1).normalize(self.transform_offset)
                dir_vec = Vector2( [(1 if is_left else -1) * dir_vec.y, (-1 if is_left else 1) * dir_vec.x] )
                points.append(((self.p1 if is_first else self.p2) + dir_vec).point())
        return np.float32(points)

    def transform_image(self, img, comp, size):
        self.tp_1, comp.tp_2 = self.get_transform_points(), comp.get_transform_points()
        M = cv2.getPerspectiveTransform(self.tp_1, comp.tp_2)
        return cv2.warpPerspective(img, M, size)

    def draw_debug_on(self, img, color):
        cv2.line(img, self.p1.point_int(), (self.p1 + self.vec).point_int(), color, 4)
        # if self.tp_1 is not None:
        #     for x in self.tp_1:  cv2.circle(img, (int(x[0]), int(x[1])), 2, (0,255,0,255), 5)
        # if self.tp_2 is not None:
        #     for x in self.tp_2:  cv2.circle(img, (int(x[0]), int(x[1])), 2, (0,127,0,255), 5)



#############################
#                           #
#     Actual Computation    #
#                           #
#############################

def run_body(anim_folder, anim_name, body_part="body"):
    part_path = f"{COMPS_PATH}/{anim_folder}/{body_part}.png"
    assert os.path.exists(part_path), f"Could not find image for {body_part}, searched for {part_path}"
    input_img = cv2.imread(part_path, cv2.IMREAD_UNCHANGED)
    size = (input_img.shape[1], input_img.shape[0],)

    src_pos_json_path = get_src_pose_json_path(anim_folder)
    with open(src_pos_json_path)                             as f:  start_data = json.load(f)
    with open(f"{ANIM_PATH}/{anim_folder}/{anim_name}.json") as f:  end_data   = json.load(f)

    s_mouth, s_hip1, s_hip2 = [Vector2(get_point(groups[body_part][i], start_data)) for i in range(3)]
    e_mouth, e_hip1, e_hip2 = [Vector2(get_point(groups[body_part][i], end_data  )) for i in range(3)]

    s_hip, e_hip = (s_hip1 + s_hip2)*0.5, (e_hip1 + e_hip2)*0.5

    s_comp = TransformComponent(s_mouth, s_hip)
    e_comp = TransformComponent(e_mouth, e_hip)

    output_img = input_img.copy()
    output_img = s_comp.transform_image(output_img, e_comp, size)

    if DEBUG == True:
        cv2.imwrite(f"{DEBUG_FOLDER}/{anim_folder}/{anim_name}_{body_part}.png", output_img)
    return output_img

def run_limb(anim_folder, anim_name, body_part):
    part_path = f"{COMPS_PATH}/{anim_folder}/{body_part}.png"
    assert os.path.exists(part_path), f"Could not find image for {body_part}, searched for {part_path}"
    input_img = cv2.imread(part_path, cv2.IMREAD_UNCHANGED)
    size = (input_img.shape[1], input_img.shape[0],)

    src_pos_json_path = get_src_pose_json_path(anim_folder)
    with open(src_pos_json_path)                             as f:  start_data = json.load(f)
    with open(f"{ANIM_PATH}/{anim_folder}/{anim_name}.json") as f:  end_data   = json.load(f)

    s_p = [Vector2(get_point(groups[body_part][i], start_data)) for i in range(3)]
    e_p = [Vector2(get_point(groups[body_part][i], end_data))   for i in range(3)]

    s_comp1 = TransformComponent(s_p[0], s_p[1])
    s_comp2 = TransformComponent(s_p[1], s_p[2])
    s_bound = TransformBoundary(s_comp1, s_comp2)

    e_comp1 = TransformComponent(e_p[0], e_p[1])
    e_comp2 = TransformComponent(e_p[1], e_p[2])
    e_bound = TransformBoundary(e_comp1, e_comp2)

    mask = np.zeros((input_img.shape[1], input_img.shape[0]), np.uint8)
    s_bound.draw_mask_on(mask, (255,))
    if DEBUG == True:
        cv2.imwrite(f"{DEBUG_FOLDER}/{anim_folder}/{anim_name}_{body_part}_zmask.png", mask)

    upper_img = input_img.copy()
    upper_img[mask > 127] = (0, 0, 0, 0)
    lower_img = input_img.copy()
    lower_img[mask < 127] = (0, 0, 0, 0)

    s_comp2.draw_boundary_on(lower_img, input_img, e_bound.diff*180/math.pi*2.5)

    upper_img = s_comp1.transform_image(upper_img, e_comp1, size)
    lower_img = s_comp2.transform_image(lower_img, e_comp2, size)

    mask[:,:] = (0,)
    e_bound.draw_mask_on(mask, (255,))
    upper_img[mask > 127] = (0, 0, 0, 0)
    if DEBUG == True:
        cv2.imwrite(f"{DEBUG_FOLDER}/{anim_folder}/{anim_name}_{body_part}_zmask2.png", mask)

    lo = arm if anim_name.endswith("arm") else leg
    if   lo[0] == "upper" and lo[1] == "lower":
        full_img = upper_img.copy()
        overlay_images(full_img, lower_img)
    elif lo[0] == "lower" and lo[1] == "upper":
        full_img = lower_img.copy()
        overlay_images(full_img, upper_img)
    else:
        raise RuntimeError(f"Malformed limb ordering for '{anim_name}', expected 'upper' and 'lower', got {lo}")

    if DEBUG == True:
        for image in [input_img, upper_img, lower_img]:
            s_bound.draw_debug_on(image, s_comp1.p2, 20, (0, 0, 0, 255), 4)
            s_comp2.draw_debug_on(image, (0, 0, 255, 255))
            s_comp1.draw_debug_on(image, (0, 0, 255, 255))
            e_bound.draw_debug_on(image, e_comp1.p2, 20, (0, 0, 0, 255), 4)
            e_comp2.draw_debug_on(image, (255, 0, 0, 255))
            e_comp1.draw_debug_on(image, (255, 0, 0, 255))

        cv2.imwrite(f"{DEBUG_FOLDER}/{anim_folder}/{anim_name}_{body_part}_lines.png", input_img)
        cv2.imwrite(f"{DEBUG_FOLDER}/{anim_folder}/{anim_name}_{body_part}_upper.png", upper_img)
        cv2.imwrite(f"{DEBUG_FOLDER}/{anim_folder}/{anim_name}_{body_part}_lower.png", lower_img)
        cv2.imwrite(f"{DEBUG_FOLDER}/{anim_folder}/{anim_name}_{body_part}.png", full_img)

    return full_img

def draw_face_mask(img, color, anim_folder, anim_name):
    with open(f"{ANIM_PATH}/{anim_folder}/{anim_name}.json") as f:  data = json.load(f)
    mouth, ear1, ear2 = [Vector2(get_point(part, data)) for part in ["mouth", "rear", "lear"]]
    vec1, vec2 = ear1 - mouth, ear2 - mouth
    vec = vec1 if vec1.length() > vec2.length() else vec2

    cv2.circle(img, (mouth + (vec * 0.3)).point_int(), int(vec.length()), color, -1)
    return img

def draw_limb_mask(limb_img):
    limb_mask = np.zeros((limb_img.shape[0], limb_img.shape[1]))
    limb_mask[:,:] = limb_img[:,:,3]
    _, limb_mask = cv2.threshold(limb_mask, 50, 255, cv2.THRESH_BINARY)
    kernel = make_dilate_kernel()
    return cv2.dilate(limb_mask, kernel, iterations=1)

def get_temp_paths(anim_folder, anim_name):
    keys = [ "img", "pose", "mask", "face" ] + list(groups.keys())
    return { k: f"{TEMP_PATH}/{anim_folder}/{anim_name}_comb_{k}.png" for k in keys }

def run_through_sd(anim_folder, anim_name, paths, full_payload, face_payload, run_info, save_prefix, x_split1=SOURCE_W, x_split2=None, width_override=None):
    enc_img  = read_encode_img(paths["img"])
    enc_mask = read_encode_img(paths["mask"])
    enc_face = read_encode_img(paths["face"]) if face_payload is not None else None
    enc_pose = read_encode_img(paths["pose"]) if USE_CONTROLNET else None

    final_path = f"{TEMP_PATH}/{anim_folder}/{anim_name}_output.png"

    gen_count = 0
    if SHOW_INTERM:
        f = f"{DEBUG_FOLDER}/{anim_folder}"
        if not os.path.exists(f):
            os.mkdir(f)

    for entry in run_info:
        payload = {}
        to_median = False
        run_face = False
        median_count = 1

        assert 'type' in entry, "key 'type' needs to be in every run entry"
        assert 'payload' in entry, "key 'payload' needs to be in every run entry"
        payload.update(entry['payload'])
        if entry['type'] == 'median':
            # print("\nRunning median payload")
            payload.update(full_payload)
            assert 'count' in entry, "median payloads must have 'count' key and value"
            median_count = entry['count']
            payload['n_iter'] = median_count
            to_median = True
        elif entry['type'] == 'single':
            # print("\nRunning single payload")
            payload.update(full_payload)
        elif entry['type'] == 'face':
            # print("\nRunning face payload")
            payload.update(face_payload)
            run_face = True
        else:
            raise RuntimeError(f"Found unexpected value for run entry payload '{entry['type']}'")
        
        if width_override is not None:
            payload["width"] = width_override

        enc_imgs = call_sd(payload, enc_img, enc_face if run_face else enc_mask, enc_pose)

        if to_median:
            enc_imgs = enc_imgs[:median_count] # strip pose
            # print(f"Median-ing {median_count} images")

            master_img = np.zeros((SOURCE_H, 2*SOURCE_W, 3, median_count))
            for index, img in enumerate(enc_imgs):
                if SHOW_INTERM:
                    save_encoded_img(img, f"{DEBUG_FOLDER}/{anim_folder}/{anim_name}_interm_{gen_count:02}.png")
                    gen_count += 1
                save_encoded_img(img, final_path)
                master_img[:,:,:,index] = cv2.imread(final_path)
            median = np.median(master_img, axis=3)
            cv2.imwrite(final_path, median)
            enc_img = read_encode_img(final_path)
        else:
            enc_img = enc_imgs[0]
            if SHOW_INTERM:
                save_encoded_img(enc_img, f"{DEBUG_FOLDER}/{anim_folder}/{anim_name}_interm_{gen_count:02}.png")
                gen_count += 1
    
    save_encoded_img(enc_img, final_path)

    if x_split1 is not None:
        final_img = split_image(cv2.imread(final_path), x_split1, x_split2)
        final_path = f"{save_prefix}_n_{anim_name}.png"
        cv2.imwrite(final_path, final_img)
    return final_path

def make_dilate_kernel():
    k_size = 15
    kernel = np.zeros((k_size, k_size))
    kernel[k_size//2, k_size//2] = 1
    kernel = cv2.GaussianBlur(kernel, (k_size, k_size), 1)
    return kernel

def run_folder(anim_folder):
    anim_root = f"{ANIM_PATH}/{anim_folder}"
    assert os.path.exists(anim_root), f"Could not find anim_root: {anim_root}"

    settings_path = f"{anim_root}/settings.json"
    assert os.path.exists(settings_path), f"Could not find the settings folder for {anim_folder}, searched '{settings_path}'"
    with open(settings_path) as f:
        data = json.load(f)
        global order, arm, leg, run
        if "order" in data:  order = data["order"]
        if "arm"   in data:  arm   = data["arm"  ]
        if "leg"   in data:  leg   = data["leg"  ]

        assert 'run' in data, f"Expected key 'run' to be in {settings_path}"
        assert type(data['run']) is list, f"data['run'] needs to be type list in {settings_path}, got {type(data['run']).__name__}"
        assert len(data['run']) > 0, f"data['run'] must have at least 1 entry, got 0 from {settings_path}"
        for index, entry in enumerate(data['run']):
            assert type(entry) is dict, f"data['run'][{index}] needs to be type dict in {settings_path}, got {type(entry).__name__}"
        run = data['run']

    render_root = f"{RENDERS_PATH}/{anim_folder}"
    if not os.path.exists(RENDERS_PATH):  os.mkdir(RENDERS_PATH)
    if not os.path.exists(render_root):   os.mkdir(render_root)

    touch_file_path = f"{render_root}/.iter000"
    assert not os.path.exists(touch_file_path), f"ERROR: this animation has already been rendered, delete '{touch_file_path}' and rerun to overwrite"

    if DEBUG == True:
        debug_root = f"{DEBUG_FOLDER}/{anim_folder}"
        if not os.path.exists(DEBUG_FOLDER):  os.mkdir(DEBUG_FOLDER)
        if not os.path.exists(debug_root):    os.mkdir(debug_root)

    kernel = make_dilate_kernel()

    files = [f.replace(".json", "") for f in os.listdir(anim_root) if f.endswith(".json") and "settings" not in f]
    print(files)
    for anim_name in tqdm(files):
        if USE_CONTROLNET and not os.path.exists(f"{anim_root}/{anim_name}.png"):
            print(f"WARNING: could not find pose image {anim_name}.png, skipping")
            continue

        # Generate the inital image
        master_img = np.zeros((SOURCE_H, SOURCE_W, 4))
        for limb in order:
            limb_img = run_body(anim_folder, anim_name) if limb == "body" else run_limb(anim_folder, anim_name, limb)
            overlay_images(master_img, limb_img)

        sd_mask_img = np.zeros((master_img.shape[0], master_img.shape[1]))
        sd_mask_img[:,:] = master_img[:,:,3]
        sd_mask_img = cv2.dilate(sd_mask_img, kernel, iterations=1)
        sd_face_img = draw_face_mask(np.zeros((SOURCE_H, SOURCE_W)), (255,), anim_folder, anim_name)

        cv2.imwrite(f"{render_root}/iter000_m_{anim_name}.png", sd_mask_img)
        cv2.imwrite(f"{render_root}/iter000_f_{anim_name}.png", sd_face_img)

        bg_img = np.zeros(master_img.shape)
        bg_img[:,:] = (127, 127, 127, 255)
        master_img = overlay_images(bg_img, master_img)
        if DEBUG == True:
            cv2.imwrite(f"{debug_root}/{anim_name}_zpre.png", master_img)

        # Run through Stable Diffusion
        if not os.path.exists(TEMP_PATH):                     os.mkdir(TEMP_PATH)
        if not os.path.exists(f"{TEMP_PATH}/{anim_folder}"):  os.mkdir(f"{TEMP_PATH}/{anim_folder}")

        paths = get_temp_paths(anim_folder, anim_name)

        cv2.imwrite( paths["img" ], splice_images(cv2.imread(TURNTABLE_IMAGE_PATH), master_img[:,:,:3]) )
        cv2.imwrite( paths["mask"], splice_images(np.zeros((SOURCE_H, SOURCE_W)),   sd_mask_img       ) )
        cv2.imwrite( paths["face"], splice_images(np.zeros((SOURCE_H, SOURCE_W)), sd_face_img       ) )

        if USE_CONTROLNET:
            assert os.path.exists(TURNTABLE_POSE_PATH), f"Turntable pose image does not exist, settings.py said it was TURNTABLE_POSE_PATH='{TURNTABLE_POSE_PATH}'"
            comb_pose = splice_images(cv2.imread(TURNTABLE_POSE_PATH), cv2.imread(f"{ANIM_PATH}/{anim_folder}/{anim_name}.png"))
            cv2.imwrite(paths["pose"], comb_pose)

        run_through_sd(anim_folder, anim_name, paths, full_payload, face_payload, run, f"{render_root}/iter000")

    with open(touch_file_path, "w") as f:
        f.write("Thanks for checking out my software! (this file demarks that this iteration ran to completion)")

if __name__ == "__main__":
    assert len(sys.argv) == 2, f"{sys.argv[0]} requires 1 argument: <anim_name>, {len(sys.argv) - 1} arguments provided"
    run_folder(sys.argv[1])
