from omegaconf import DictConfig, OmegaConf

from phyfu.common.registry import populate_registry
from phyfu.common.registry import Registry
from phyfu.utils.cli_utils import create_parser


def entry(cfg: DictConfig) -> None:
    populate_registry(cfg)
    if cfg.operation == "fuzz" or cfg.operation == "both":
        from phyfu.common import fuzz
        time_stamp = fuzz.fuzz()
    if cfg.operation == "find_errors" or cfg.operation == "both":
        if cfg.operation == "both":
            Registry.bug_oracle.oracle_cfg.time_stamp = time_stamp
        if cfg.model_name == "mpm":
            from phyfu.data_ana import mpm_error_finder
            mpm_error_finder.check_all(
                Registry.module_path_utils, Registry.array_utils, Registry.bug_oracle)
        else:
            from phyfu.data_ana import error_finder
            error_finder.check(
                Registry.module_path_utils, Registry.array_utils, Registry.bug_oracle)


def main():
    parser = create_parser()
    parser.add_argument("--operation", choices=["fuzz", "find_errors", "both"],
                        default="both")
    parser.add_argument("--test_times", "-n", type=int)
    parser.add_argument("--time_stamp", "-t", type=str)
    parser.add_argument("--seed_getter", "-s", type=str, choices=['random', 'art'],
                        default="art")
    args = parser.parse_args()
    extra_opts = {}
    if args.test_times:
        extra_opts.update({"test_times": args.test_times})
    if args.time_stamp:
        extra_opts.update({"time_stamp": args.time_stamp})
    if args.seed_getter:
        extra_opts.update({"seed_getter": {"type": args.seed_getter}})
    if extra_opts:
        conf = OmegaConf.create({"module": args.module, "model_name": args.model_name,
                                 "operation": args.operation,
                                 "extra_opts": extra_opts})
    else:
        conf = OmegaConf.create({"module": args.module, "model_name": args.model_name,
                                 "operation": args.operation})
    entry(conf)


if __name__ == "__main__":
    main()
