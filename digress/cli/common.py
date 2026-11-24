"""
new file cli/common.py: model CLIs read dataset from checkpoint, 
baseline CLIs take --dataset; Baseline results saved to a folder per dataset, since
they depend on the data.
"""

from digress.config import BASELINES_DIR
from digress.data.dataset import get_dataset
from digress.data.info import DATASETS


def add_dataset_args(parser):
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="qm9")
    parser.add_argument("--subset-size", type=int, help="QM9: 50000 = Milestone 1 data; MOSES: train subset")
    parser.add_argument("--aromatic", action="store_true", help="aromatic bonds as an additional edge class")


def dataset_from_args(args):
    return get_dataset(args.dataset, args.subset_size, args.aromatic)


def baselines_dir(data):
    #Baseline results depend on data they were estimated from, save to folder
    sub = data.meta.get("subset_size")
    return BASELINES_DIR / (data.info.key + (f"_sub{sub}" if sub else ""))
