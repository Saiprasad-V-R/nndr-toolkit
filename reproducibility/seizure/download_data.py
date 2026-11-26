"""Download just the representative public CHB-MIT recording, not the cohort."""
import argparse
from pathlib import Path
from urllib.request import urlopen
import shutil

BASE = 'https://physionet.org/files/chbmit/1.0.0/chb01/'


def download(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('chb01_03.edf', 'chb01-summary.txt'):
        dest = directory / name
        if dest.exists():
            print('Exists:', dest)
            continue
        temp = dest.with_suffix(dest.suffix + '.part')
        try:
            with urlopen(BASE + name, timeout=90) as response, temp.open('wb') as f:
                shutil.copyfileobj(response, f)
            if name.endswith('.edf') and temp.stat().st_size < 1000000:
                raise ValueError('Unexpectedly small EDF download')
            temp.replace(dest)
            print('Downloaded:', dest)
        finally:
            if temp.exists():
                temp.unlink()
    return directory / 'chb01_03.edf'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    download(parser.parse_args().data_dir)
