import re
from dataclasses import dataclass
import time
from typing import Any, Tuple

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.utils.path_utils import LogPathUtils, ModulePath
from phyfu.utils.printer_utils import BufferedWriter, to_readable


def cur_time():
    return time.strftime("%m%d_%H%M", time.localtime(time.time()))


@dataclass
class MetaInfo:
    root_state: Any
    seed_action: Any
    seed_init: Any
    seed_final: Any
    mut_dev_before: Any
    mut_init_before: Any
    mut_final_before: Any
    min_loss: float
    mut_dev_after: Any
    mut_init_after: Any
    mut_final_after: Any


@dataclass
class ReadableMetaInfo:
    test_time: int
    mut_dev_before: Any
    loss_before: float
    min_loss: float
    min_loss_item: Any
    stop_message: str

    def __str__(self):
        return f"""==================
test_time: {self.test_time}
mut_dev_before: {to_readable(self.mut_dev_before)}
loss_before: {to_readable(self.loss_before)}
min_loss: {to_readable(self.min_loss)}
min_loss_item: {to_readable(self.min_loss_item)}
stop_message: {self.stop_message}"""

    @staticmethod
    def load(file_path):
        info_list = []
        cur_info = {}
        with open(file_path, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]
        for line in lines:
            if line.startswith("========"):
                if cur_info:
                    info_list.append(cur_info)
                cur_info = {}
                continue
            comma_idx = line.index(":")
            cur_info.update({line[:comma_idx]: line[comma_idx+2:]})
        info_list.append(cur_info)
        return info_list


@dataclass
class OptInfo:
    opt_iter: int
    opt_dev: Any
    loss_value: float
    grads: Any

    def __str__(self):
        return f"iter: {self.opt_iter} loss: {to_readable(self.loss_value)} " \
               f"opt_dev: {to_readable(self.opt_dev)} grads: {to_readable(self.grads)}"


class StringLogWriter:
    def __init__(self, disable):
        self.disable = disable

    def write_str(self, file_path, str_to_write):
        if self.disable:
            return
        with open(file_path, 'w') as f:
            f.write(str_to_write)


class DataAnalysisLogger:
    def __init__(self):
        self._loss_w = "loss_too_large"
        self._dev_w = "deviated_init_state"
        self._crash_w = "crashes"
        self._loss_p = re.compile(self._hint_pattern(self._loss_w) + "(\d+)")
        self._dev_p = re.compile(self._hint_pattern(self._dev_w) + "(\d+)")
        self._crash_p = re.compile(self._hint_pattern(self._crash_w) + "(\d+)")

    def load(self, log_path_utils: LogPathUtils) -> Tuple[int, int, int]:
        with open(log_path_utils.data_analysis_path, 'r') as f:
            lines = [line.rstrip() for line in f.readlines() if line.startswith("#")]
        n_loss = int(self._loss_p.match(lines[-2]).group(1))
        n_dev = int(self._dev_p.match(lines[-1]).group(1))
        n_crashes = 0
        if len(lines) > 2:
            m = self._crash_p.match(lines[-3])
            if m is not None:
                n_crashes = int(m.group(1))
        return  n_loss, n_dev, n_crashes

    @staticmethod
    def _hint_pattern(hint):
        return f"#{hint}: "

    def to_summary(self, n_loss, n_dev, n_crashes):
        s = f"#{self._loss_w}: {n_loss}\n" \
            f"#{self._dev_w}: {n_dev}\n"
        if n_crashes > 0:
            s = f"#{self._crash_w}: {n_crashes}\n" + s
        return s


class FuzzingLogger:
    def __init__(self, disable, module_path_utils: ModulePath = None,
                 array_utils: ArrayUtils = None, label=None):
        self.disable = disable
        if not disable:
            if label is not None:
                self.time_stamp = label
            else:
                self.time_stamp = cur_time()
            self.path_handler = LogPathUtils(module_path_utils, not disable, self.time_stamp)
            self.meta_str_writer = BufferedWriter(self.path_handler.meta_readable_path, 100)
            self.log_root = self.path_handler.log_root
            self.array_utils = array_utils
        self.opt_info_list = []
        self.num_iter = 0

    def set_num_iter(self, num_iter):
        self.num_iter = num_iter
        self.opt_info_list = []

    def log_meta_info(self, meta_info: MetaInfo, readable_meta: ReadableMetaInfo):
        if self.disable:
            return
        self.meta_str_writer.write(str(readable_meta))
        self.array_utils.save(self.path_handler.meta_info_per_opt_path(self.num_iter),
                              [meta_info])

    def log_opt_info(self, opt_info):
        self.opt_info_list.append(opt_info)

    def dump_test_iter(self):
        if self.disable:
            return
        self.array_utils.save(self.path_handler.opt_arr_path(self.num_iter),
                              self.opt_info_list)
        with open(self.path_handler.opt_readable_path(self.num_iter), 'w') as f:
            f.write("\n".join([str(i) for i in self.opt_info_list]))
            f.write("\n")

    def dump_summary(self):
        if self.disable:
            return
        self.meta_str_writer.close()
