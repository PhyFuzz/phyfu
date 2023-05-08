import os
import shutil

def vis_mpm(model, save_dir, num_steps):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    for s in range(num_steps):
        model.mpm.visualize(s * model.substeps, save_dir, save_name=f"{s:03d}")
    os.system(f"ffmpeg -y -loglevel quiet -framerate 20 -i '{save_dir}/%03d.png' '{save_dir}/out.mp4'")

