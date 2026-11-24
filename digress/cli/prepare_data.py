"""
new file cli/prepare_data.py

building the cache once so not required to rebuild each time
"""

import argparse
import os

from digress.data.dataset import cache_path, get_dataset, load_cache
from digress.data.info import DATASETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="moses")
    parser.add_argument("--subset-size", type=int, help="QM9 only (MOSES subsets are always views of full cache)")
    parser.add_argument("--aromatic", action="store_true", help="aromatic bonds as an additional edge class")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="RDKit worker processes")
    args = parser.parse_args()

    load_cache(args.dataset, args.subset_size, args.aromatic, workers=args.workers)
    data = get_dataset(args.dataset, args.subset_size, args.aromatic)
    print(f"{cache_path(args.dataset, args.subset_size, args.aromatic)}: {len(data)} molecules "
          f"(train {len(data.train_idx)}, val {len(data.val_idx)}, test {len(data.test_idx)}), "
          f"max {data.info.max_atoms} atoms, skipped {data.meta['skipped']}")


if __name__ == "__main__":
    main()
