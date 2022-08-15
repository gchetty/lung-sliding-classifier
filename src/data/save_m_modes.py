import os
import numpy as np
import yaml
import cv2
import glob
import random
import shutil
from utils import refresh_folder

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "..\\..\\config.yml"), 'r'))['PREPROCESS']
cfg_full = yaml.full_load(open(os.path.join(os.getcwd(), "..\\..\\config.yml"), 'r'))

mmode_dir = cfg_full['EXTERNAL_VAL']['PATHS']['MMODES']
npz_dir = cfg['PATHS']['NPZ']

if not os.path.exists(mmode_dir):
    os.makedirs(mmode_dir)

if cfg_full['EXTERNAL_VAL']['REFRESH_FOLDERS']:
    refresh_folder(mmode_dir)


def save_m_modes(src, dest):
    '''
    Processes .npz files from src as M-mode images; saves resulting images in dest. Images are stored as .jpg files.
    :param src: directory storing .npz files.
    :param dest: directory into which to save M-mode images.
    '''
    for npz_class in os.listdir(src):
        npz_class_dir = os.path.join(src, npz_class)
        mmode_class_dir = os.path.join(dest, npz_class)
        for npz in os.listdir(npz_class_dir):
            npz_file_path = os.path.join(npz_class_dir, npz)
            n = np.load(npz_file_path)
            arr = (n['mmode'])
            cv2.imwrite(os.path.join(mmode_class_dir, npz[:-4]) + '.jpg', arr)


def get_clip_length(video_path):
    '''
    Returns the length of an .mp4 video file (in seconds).
    :param video: name of .mp4 file.
    '''
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_ct = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_ct / fps
    return duration


def external_m_mode_peek(n, npz_dir, save_dir, pull_by_loc=True):
    '''
    Saves n of the M-modes into save_dir. Pulls approximately the same number of M-modes from each multi-center location
    if pull_by_loc is set to True. Note: The M-modes are pulled from npz_dir.
    :param n: Number of M-mode images to pull.
    :param npz_dir: Directory from which to retrieve .npz files. These files are converted to M-modes. Absolute path.
    :param save_dir: Directory to which to save the n M-modes. Absolute path.
    :param pull_by_loc: Defaults to True. If True, then M-modes are evenly pulled by multi-center location.
    '''
    npzs = []

    # If pull_by_loc is True, then pull M-modes evenly by location.
    if pull_by_loc:
        num_clips = []
        locs = cfg_full['EXTERNAL_VAL']['LOCATIONS']
        num_loc = len(locs)
        n_per_loc = n // num_loc
        for i in range(num_loc):
            num_clips.append(n_per_loc)
        num_clips[0] += n - n_per_loc * num_loc

        # Get n random .npzs to convert to M-modes. Do so evenly by location.
        for i in range(len(num_clips)):
            print(os.path.join(npz_dir, locs[i]))
            for input_class in ['sliding', 'no_sliding']:
                npzs_for_loc = glob.glob(os.path.join(npz_dir, locs[i], input_class, '*.npz'))
                random.shuffle(npzs)
                npzs_to_convert = npzs_for_loc[:(num_clips[i] // 2)]
                npzs.extend(npzs_to_convert)

    else:
        for loc in cfg_full['EXTERNAL_VAL']['LOCATIONS']:
            for input_class in ['sliding', 'no_sliding']:
                npzs.extend(glob.glob(os.path.join(npz_dir, loc, input_class, '*.npz')))
        random.shuffle(npzs)
        npzs = npzs[:n]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        refresh_folder(save_dir)

    # Temporarily store the .npzs in save_dir.
    for file in npzs:
        filename = file.split('\\')[-1]
        shutil.copy(file, os.path.join(save_dir, filename))

    for file in glob.glob(os.path.join(save_dir, '*.npz')):
        n = np.load(file)
        arr = (n['mmode'])
        path_parts = file.split('\\')
        save_path = os.path.join('\\'.join(path_parts[:-1]), path_parts[-1][:-4] + '.jpg')
        cv2.imwrite(save_path, arr)

    # Remove .npzs from M-mode directory.
    for file in glob.glob(os.path.join(save_dir, '*.npz')):
        os.remove(file)


if __name__ == '__main__':
    project_dir = '\\'.join(os.getcwd().split('\\')[:-2])
    external_m_mode_peek(30, project_dir + cfg_full['EXTERNAL_VAL']['PATHS']['NPZS'],
                         project_dir + '\\mmode_sample')
    # raw_clips_dir = cfg['PATHS']['MASKED_VIDEOS']
    # short_clips = []
    # for clip_class in os.listdir(raw_clips_dir):
    #     clip_class_dir = os.path.join(raw_clips_dir, clip_class)
    #     for clip in os.listdir(clip_class_dir):
    #         # Get clip length.
    #         l = get_clip_length(os.path.join(clip_class_dir, clip))
    #         if l < 3.0:
    #             short_clips.append(os.path.join(clip_class_dir, clip))
    # # Show the filenames of all clips less than the mini-clip window specified in the config file.
    # for clip in short_clips:
    #     print(clip)
    # print(len(short_clips))
    #
    # save_m_modes(npz_dir, mmode_dir)
