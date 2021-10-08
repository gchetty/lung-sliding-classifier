0. cd to src/data/ in terminal
1. Specify database credentials and tweak parameters in config.yml
2. Make venv (HAS TO BE PYTHON 3.7.X!) and install requirements.txt
3. To perform all, simply do python run.py

(Optional) any of download_videos.py, mask.py, and to_npz.py should run standalone, as long as mask.py has videos 
downloaded in the path specificed by UNMASKED_VIDEOS in the config file, and to_npz.py has videos in MASKED_VIDEOS.