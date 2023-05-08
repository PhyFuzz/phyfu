import os
import re
from typing import Union, List, Tuple

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(ROOT_DIR), "results")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
STAT_DIR = os.path.join(CONFIG_DIR, "statistics")

TIME_STAMPS_PATH = os.path.join(STAT_DIR, "time_stamps.yaml")
FINDINGS_PATH = os.path.join(STAT_DIR, "findings.yaml")
CAUSE_RESULTS_PATH = os.path.join(STAT_DIR, "cause.yaml")
PV_THRESHOLD_PATH = os.path.join(STAT_DIR, "pv_threshold.yaml")


class ModulePath:
    instance_dict = {}
    def __new__(cls, module_name, model_name):
        if not module_name in cls.instance_dict.keys() or \
            model_name not in cls.instance_dict[module_name].keys():
            instance = super(ModulePath, cls).__new__(cls)
            instance._module_name = module_name
            instance._model_name = model_name
            instance._module_dir = os.path.join(ROOT_DIR, f"{module_name}_mutate")
            instance._results_dir = os.path.join(RESULTS_DIR, module_name, model_name)
            if module_name not in cls.instance_dict.keys():
                cls.instance_dict[module_name] = {}
            cls.instance_dict[module_name][model_name] = instance

        return cls.instance_dict[module_name][model_name]

    @property
    def results_dir(self):
        return self._results_dir

    @results_dir.setter
    def results_dir(self, value):
        self._results_dir = value

    @property
    def config_dir(self):
        return os.path.join(CONFIG_DIR, "fuzzing", self.module, self.model_name)

    @property
    def module(self):
        return self._module_name

    @property
    def model_name(self):
        return self._model_name

    @property
    def mutate_config_path(self):
        return os.path.join(self.config_dir, f"mutate.yaml")

    @property
    def analysis_config_path(self):
        return os.path.join(self.config_dir, f"analysis.yaml")

    def get_all_time_stamps(self):
        pat = re.compile(r"\d{4}_\d{4}")
        return [t for t in os.listdir(self.results_dir)
                if os.path.isdir(os.path.join(self.results_dir, t)) and
                pat.match(t)]


class LogPathUtils:
    """
    Accessing the paths of log files should go through this class
    """
    def __init__(self, module_path_utils: ModulePath, make_dir: bool,
                 extra_paths: Union[List[str], str, Tuple[str]]):
        """
        Create a manager for accessing paths of log files
        :param module_path_utils: the path handler for your module and model
        :param extra_paths: the extra paths to be appended onto module_path_utils.results_dir
        :param make_dir: whether to make directories if the log dir does not exist
        """
        if isinstance(extra_paths, str):
            self.log_root = os.path.join(module_path_utils.results_dir, extra_paths)
        elif (isinstance(extra_paths, list) or isinstance(extra_paths, tuple)) \
                and isinstance(extra_paths[0], str):
            self.log_root = os.path.join(module_path_utils.results_dir, *extra_paths)
        else:
            raise RuntimeError(
                "The type of parameter extra_paths should be str or List[str] or Tuple[str]")
        if make_dir:
            os.makedirs(self.log_root)
        self.make_dir = make_dir

    def get_all_test_time_dirs(self):
        pat = re.compile(r"\d+")
        return [d for d in os.listdir(self.log_root)
                if os.path.isdir(os.path.join(self.log_root, d)) and pat.match(d)]

    def get_test_time_dir(self, opt_num):
        return os.path.join(self.log_root, str(opt_num))

    @property
    def meta_arr_path(self):
        return os.path.join(self.log_root, "meta.npz")

    def meta_info_per_opt_path(self, test_time):
        return os.path.join(self.opt_dir_path(test_time), "meta.npz")

    @property
    def meta_readable_path(self):
        return os.path.join(self.log_root, "summary.txt")

    def opt_dir_path(self, opt_iter):
        opt_dir = os.path.join(self.log_root, str(opt_iter))
        if not os.path.exists(opt_dir) and self.make_dir:
            os.mkdir(opt_dir)
        return opt_dir

    def opt_arr_path(self, opt_iter):
        return os.path.join(self.opt_dir_path(opt_iter), "opt.npz")

    def opt_readable_path(self, opt_iter):
        return os.path.join(self.opt_dir_path(opt_iter), "opt.txt")

    @property
    def data_analysis_path(self):
        return os.path.join(self.log_root, "data_analysis.txt")
