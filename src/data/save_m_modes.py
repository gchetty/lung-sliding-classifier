import os
import numpy as np
import yaml
import cv2

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "..\\..\\config.yml"), 'r'))['PREPROCESS']
cfg_full = yaml.full_load(open(os.path.join(os.getcwd(), "..\\..\\config.yml"), 'r'))

mmode_dir = cfg_full['GENERALIZE']['PATHS']['MMODES']
npz_dir = cfg['PATHS']['NPZ']

if not os.path.exists(mmode_dir):
    os.makedirs(mmode_dir)


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


def length(video):
    '''
    Returns the length of an .mp4 video file (in seconds).
    :param video: .mp4 file.
    '''
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_ct = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_ct / fps
    return duration


if __name__ == '__main__':
    raw_clips_dir = cfg['PATHS']['MASKED_VIDEOS']
    short_clips = []
    for clip_class in os.listdir(raw_clips_dir):
        clip_class_dir = os.path.join(raw_clips_dir, clip_class)
        for clip in os.listdir(clip_class_dir):
            # Get clip length.
            l = length(os.path.join(clip_class_dir, clip))
            if l < 3.0:
                short_clips.append(os.path.join(clip_class_dir, clip))
    # Show the filenames of all clips less than the mini-clip window specified in the config file.
    for clip in short_clips:
        print(clip)
    print(len(short_clips))

    save_m_modes(npz_dir, mmode_dir)
