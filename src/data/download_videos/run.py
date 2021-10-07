import os

# Download the videos, then mask them, then convert to miniclips saved in .npz format
# This assumes a Python 3.7 virtual environment has been created and requirements.txt is pip installed
os.system('python download_videos.py')
os.system('python mask.py')
os.system('python to_npz.py')