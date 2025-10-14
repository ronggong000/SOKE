import sys, os
sys.path.append(os.path.join(os.path.abspath(os.getcwd()),'..')) # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import bpy, bmesh
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
cwd = os.getcwd()

## initialize blender
imgRes_x = 768
imgRes_y = 768 
numSamples = 40 
exposure = 1.5 
bpy.context.scene.cycles.device = "GPU"


def render(name2idx, dataset):
    for vid, f_idx_lst in name2idx.items():
        mesh_dir = f'visualize/compare_{dataset}/{vid}/mesh'
        gt_files, base_files, ours_files = [], [], []
        for f in sorted(os.listdir(mesh_dir)):
            if f.endswith('gt.ply'):
                gt_files.append(f)
            elif f.endswith('base.ply'):
                base_files.append(f)
            elif f.endswith('ours.ply'):
                ours_files.append(f)

        op_dir = f'visualize/compare_{dataset}/{vid}/blender'
        os.makedirs(op_dir, exist_ok=True)

        # source = ['base', 'ours']
        if f_idx_lst == []:
            f_idx_lst = list(range(len(ours_files)))
        source = ['ours', 'base', 'gt']
        dir_lst = [ours_files, base_files, gt_files]

        for dir, src in zip(dir_lst, source):
            if len(dir) == 0:
                continue
            for f_idx in tqdm(sorted(f_idx_lst)):
                bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

                ## read mesh (choose either readPLY or readOBJ)
                meshPath = os.path.join(mesh_dir, f"{dir[min(f_idx, len(dir)-1)]}")
                location = (1.12, -0, 1.0) # (UI: click mesh > Transform > Location)
                # rotation = (90, 0, 227) # (UI: click mesh > Transform > Rotation)
                rotation = (90, 180, -90) # (UI: click mesh > Transform > Rotation)
                scale = (1.5,1.5,1.5) # (UI: click mesh > Transform > Scale)
                mesh = bt.readMesh(meshPath, location, rotation, scale)

                ## set shading (uncomment one of them)
                # bpy.ops.object.shade_smooth() 

                ## subdivision
                bt.subdivision(mesh, level = 2)

                # # set material
                # colorObj(RGBA, H, S, V, Bright, Contrast)
                meshColor = bt.colorObj(bt.derekBlue, 0.5, 1.0, 1.0, 0.0, 2.0)
                AOStrength = 0.0
                bt.setMat_balloon(mesh, meshColor, AOStrength)

                ## set invisible plane (shadow catcher)
                bt.invisibleGround(shadowBrightness=0.9)

                ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
                camLocation = (3.5, 0, 1.2)
                # camLocation = (3, 0, -2)
                lookAtLocation = (0, 0, 0.6)
                focalLength = 45 # (UI: click camera > Object Data > Focal Length)
                cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

                ## set light
                lightAngle = (6, -30, -155) 
                strength = 2
                shadowSoftness = 0.3
                sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

                ## set ambient light
                bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

                ## set gray shadow to completely white with a threshold 
                bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

                ## save blender file so that you can adjust parameters in the UI
                # bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')

                ## save rendering
                fname = f'{f_idx:03d}_{src}.png'
                outputPath = os.path.join(op_dir, fname)
                bt.renderImage(outputPath, cam)

        #         break
        #     break
        # break


if __name__ == '__main__':
    # for SLG
    name2idx = {'S000291_P0000_T00': []}
    render(name2idx, 'csl')