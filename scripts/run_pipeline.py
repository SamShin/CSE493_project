import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.download_video import main as run_downloads

if __name__ == "__main__":
    run_downloads()
