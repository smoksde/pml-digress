"""
New file: moses.py for MOSES dataset

lioke in DiGress authors' code, using test_scaffolds as the validation set.
(train ~1.58M, test&test_scaffolds ~176k molecules)


"""
import csv
import urllib.request
from pathlib import Path

from digress.config import MOSES_DIR

_BASE = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/"
MOSES_FILES = {"train": "train.csv", "test": "test.csv", "test_scaffolds": "test_scaffolds.csv"}


def ensure_moses_files(directory=MOSES_DIR):
    """Returns `directory` once it holds all MOSES csv files downloading only the missing ones."""
    directory = Path(directory)
    missing = [name for name in MOSES_FILES.values() if not (directory / name).is_file()]
    if not missing:
        return directory
    try:
        if not directory.is_dir():
            directory.mkdir(parents=True)
        for name in missing:
            path = directory / name
            print(f"downloading {_BASE + name} -> {path}", flush=True)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(_BASE + name, tmp)
            tmp.rename(path)
    except OSError as err:  # apptainer --bind ... :ro:: read-only folder
        raise FileNotFoundError(
            f"MOSES files {missing} are missing in {directory} and could not be downloaded there ({err}). "
            f"Place {', '.join(MOSES_FILES.values())} in that folder manually, or set MOSES_DIR to some writable "
            "folder or use --bind for Apptainer that allows for writing") from err
    return directory


def load_moses_smiles(split, directory=MOSES_DIR):
    path = ensure_moses_files(directory) / MOSES_FILES[split]
    with open(path, newline="") as f:
        return [row["SMILES"] for row in csv.DictReader(f)]
