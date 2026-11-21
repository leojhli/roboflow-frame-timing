"""Get the public test video (Roboflow's own 'vehicles' sample clip)."""
from pathlib import Path

from supervision.assets import VideoAssets, download_assets

DATA_DIR = Path("data")


def get_test_video() -> str:
    """Download vehicles.mp4 into data/ the first time; afterwards reuse it."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / VideoAssets.VEHICLES.filename
    if not path.exists():
        download_assets(VideoAssets.VEHICLES, directory=DATA_DIR)
    return str(path)
