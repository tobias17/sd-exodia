# Stable Diffusion Exodia

Stable Diffusion Exodia (sd-exodia) is a utility program for generating high quality animation spritesheets with relatively low effort. It does this by taking in a "turntable" and "source" images, and using those to "render" out a full animation, as declared by the poses fed in.

# Install

pip install the following packages:
```
pip install opencv-python
pip install numpy
pip install rembg
```
NOTE: `rembg` is not being used directly in python, but being called by the cmd command. The `rembg` package has not been updated past python 3.9. If you are using a later python, you will need to install `rembg` from an earlier python version, but can use the later python version to run this software.

# Usage

## Create a Workspace

1. Create a folder within the `workspaces` folder and name it what you like.
2. Go into `settings.py` and, in the first line, change the `WORKSPACE` variable to point to your new folder.
3. Your workspace will need some pose information. Either copy all of the contents of `workspaces/template` into your new workpsace, or create your own poses from scratch.

## Create a Turntable

In the root of your workspace, you should find a turntable-pose.png image (copied from `workspaces/template`). You will need to make an image through stable diffusion with that pose. This will be used in all of the Stable Diffusion runs, not being modified itself, but to act as a reference for how the characters in your animation should look. It is important to spend some time making this look nice and exactly how you want your resulting character to look, as everything else will be based on this turntable image.

Once you have a turntable image that you are happy with, place it in the root of your workspace (next to `turntable-pose.png`) and rename the new image to `turntable-image.png`.

## Creating a Source

NOTE: This and the following steps will have to be done per animation folder (`srun`, `brun`, `frun`, etc.)

1. Call `python prep_source_run.py <anim_name>` where `<anim_name>` is the animation you are doing next (e.g. `python prep_source_run.py frun`)
2. You should see a `tmp` folder appear in your workspace. Navigate into that folder as well as the new folder created with your `<anim_name>` from step (1).
3. Use `image.png`, `pose.png`, and `turntable-stripped.png` to create your source image. The source image should look like `image.png`, but has your character in the right with `pose.png` pose.
4. Once your source image has been made, throw it into `tmp/<anim_name>` next to `image.png` and rename the source image to `source.png`.

## Creating Components

1. Call `python split_source_run.py <anim_name>`, where `<anim_name>` is the same name you used in the previous step.
2. Navigate to `comps/<anim_name>` and you should see new `source-image.png` and `source-stripped.png` images.
3. Open `source-stripped.png` in an image editing program (like Photoshop, I use Krita myself).
4. Extract each limb into a seperate layer and save these images. You should have 5 total - 2 arms, 2 legs, and a body. These have to be name `larm.png`, `rarm.png`, `lleg.png`, `rleg.png`, `body.png`. These go in the `comps/<anim_name>` folder next to `source-stripped.png`.

## Rendering out an Animation

1. Make sure your Automatic1111 webui is open and select the model you want to use (this program calls into the API).
2. Open `render.py` and modify the `char_desc`, `pos_prompt`, and `neg_prompt` variables to contain your prompt information. Note that the `char_desc` gets put inside `pos_prompt` by default.
3. Call `python render.py <anim_name>` and you should see images appearing in `renders/<anim_name>`.
4. If you want to change settings for how these are run, open `anims/<anim_name>/settings.py`. In here you can change what order images are spliced together and what stable diffusion settings are used (steps, denoising strength, etc.).

## Iterating an Animation

1. This tooling also has an option to run these images again, but with more animations pulled into the context.
2. To change settings, edit the top of `iterate.py`. The `IMGS_LEFT` and `IMGS_RIGHT` variables will pull in more images, but not that this will require more VRAM.
3. Call `python iterate.py <anim_name>` to run an iteration. This can be done as many times as you like, always iterating on the previous iteration.
4. To "rerun" an iteration, delete the `.iterXXX` file in `renders/<anim_name>`. E.g. if you have run `render.py` and then ran `iterate.py` twice, you should see `.iter000`, `.iter001`, and `.iter002`. If you delete the `.iter002` file and run `iterate.py`, it will rerun your latest run, overwriting the results.

## Making the Spritesheet

Once the above steps have been done for all of the animations, you can generate a spritesheet from the results.

1. Open `spritesheet.py` and make sure all of the settings are correct. `ANIMS_TO_JOIN` should have all of your folders, and `DO_ALL` should be False if you have not done anything else (like interpolation).
2. If you want to alter what and how the spritesheet is constructed, you need to extract all of the images you want in it into seperate folders in the `render` directory. These folders need to contain only the images you want in the spritesheet. Set `DO_ALL` to True and set `ANIMS_TO_JOIN` to the folder in `render` containing the images.
3. Call `python spritesheet.py` and you should see the result in your workspace root named `sheet.png`.
