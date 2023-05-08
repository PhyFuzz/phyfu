from phyfu.utils.cli_utils import create_parser
from phyfu.common.registry import populate_registry, Registry
from phyfu.data_ana.mock_guided_timing import get_scheduling_overhead
from phyfu.data_ana.grad_type import classify_backward_errors
from phyfu.data_ana.get_findings import get_total_num_errors
from phyfu.data_ana.forward_type import classify_forward_errors


def main():
    args = create_parser().parse_args()
    cfg = {"module": args.module, "model_name": args.model_name, "operation": "both"}
    if args.model_name == "mpm":
        cfg["extra_opts"] = {"extra_paths": "art", "seed_getter": {"type": "art"}}

    populate_registry(cfg)
    mp = Registry.module_path_utils
    print(f"Module: {mp.module}, model_name: {mp.model_name}")

    get_total_num_errors()
    get_scheduling_overhead()
    classify_forward_errors()
    classify_backward_errors()
