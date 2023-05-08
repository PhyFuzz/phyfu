import os

from phyfu.utils.log_utils import LogPathUtils


def get_exec_time(lp: LogPathUtils):
    d = lp.get_all_test_time_dirs()
    d = [int(s) for s in d]
    d.sort()
    start_time = os.path.getctime(lp.get_test_time_dir(d[0]))
    end_time = os.path.getctime(lp.get_test_time_dir(d[-1]))
    return int(end_time - start_time)


def seconds_to_readable(seconds):
    s = int(seconds)
    h = s // 3600
    s %= 3600
    m = s // 60
    s %= 60
    return f"{h:02d}hr {m:02d}min {s:02d}sec"
