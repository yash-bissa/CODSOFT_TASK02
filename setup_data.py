from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from shutil import copyfileobj
from zipfile import BadZipFile, ZipFile

DATASET = 'hijest/genre-classification-dataset-imdb'
FILENAMES = ['train_data.txt', 'test_data_solution.txt']
ROOT = Path(__file__).resolve().parent
URL = "https://www.kaggle.com/api/v1/datasets/download/" + DATASET


def main():
    destination = ROOT / "data"
    if all((destination / name).is_file() and (destination / name).stat().st_size
           for name in FILENAMES):
        print("Dataset already present in data/.")
        return

    destination.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=destination) as temp:
        archive_path = Path(temp) / "dataset.zip"
        print("Downloading the dataset from Kaggle...")
        request = Request(URL, headers={"User-Agent": "CodSoft-ML-Demo/1.0"})
        with urlopen(request, timeout=120) as response, archive_path.open("wb") as archive_file:
            copyfileobj(response, archive_file)
        with ZipFile(archive_path) as archive:
            members = {Path(name).name: name for name in archive.namelist()}
            for filename in FILENAMES:
                staged = Path(temp) / filename
                with archive.open(members[filename]) as source, staged.open("wb") as target:
                    copyfileobj(source, target)
                staged.replace(destination / filename)
    print("Dataset ready in data/.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, BadZipFile, KeyError) as error:
        raise SystemExit(f"Download failed ({error}). Use the manual download steps in README.md.")
