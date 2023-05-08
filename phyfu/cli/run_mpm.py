import os, subprocess
import random
from time import sleep
import shlex
from omegaconf import OmegaConf

from phyfu.utils.log_utils import cur_time
from phyfu.utils.path_utils import ModulePath, LogPathUtils


def main():
    mp = ModulePath("taichi", "mpm")
    cfg = OmegaConf.load(mp.mutate_config_path)

    while True:
        while os.path.exists(os.path.join(mp.results_dir, cur_time())):
            sleep(random.randint(1, 30))
        n_times = 0
        if os.path.exists(mp.results_dir):
            time_stamp_list = mp.get_all_time_stamps()
            for time_stamp in time_stamp_list:
                lp = LogPathUtils(mp, False, time_stamp)
                n_times += len(lp.get_all_test_time_dirs())
            print("Current total fuzzing iterations:", n_times)
            if n_times >= cfg.test_times:
                break

        p = subprocess.Popen(shlex.split(f"phyfu.run taichi mpm fuzz"), env=os.environ)
        returncode = p.wait()
        print("returncode is", returncode)
        next_seed = random.randint(1, 100000)
        cfg.seed = next_seed

        print("Next seed number is:", next_seed)

        OmegaConf.save(cfg, mp.mutate_config_path)
