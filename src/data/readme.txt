cd to src/data/ in terminal.
Make venv (HAS TO BE PYTHON 3.7.X!), activate it, and pip install -r requirements.txt (the requirements.txt in src/data/).
Tweak parameters in config.yml, located in the project root folder, according to your specifications. Ensure the MASKING_TOOL path is set to the downloaded file (you need this file).

In the project root folder, create a file called database_config.yml with the following contents:
HOST: 'host'
USERNAME: 'your_db_username'
PASSWORD: 'your_db_password'
DATABASE: 'data'

To perform all operations, simply do python run.py in your virtual environment.

(Optional) any of download_videos.py, mask.py, and to_npz.py should run standalone, as long as mask.py has videos 
downloaded in the path specificed by UNMASKED_VIDEOS in the config file, and to_npz.py has videos in MASKED_VIDEOS.