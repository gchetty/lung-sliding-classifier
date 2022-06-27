from PIL import Image
import os
import numpy as np
import yaml
import subprocess
from moviepy.editor import *
import cv2

npz_dir = 'C:\\Users\\hsmle\\Documents\\Deep_Breathe\\lung-sliding-classifier\\results\\data\\npzs\\'
mmode_dir = 'C:\\Users\\hsmle\\Documents\\Deep_Breathe\\lung-sliding-classifier\\results\\data\\mmodes'
cfg = yaml.full_load(open(os.path.join(os.getcwd(), "..\\..\\config.yml"), 'r'))['PREPROCESS']

if not os.path.exists(mmode_dir):
    os.makedirs(mmode_dir)

for npz_class in os.listdir(npz_dir):
    npz_class_dir = os.path.join(npz_dir, npz_class)
    mmode_class_dir = os.path.join(mmode_dir, npz_class)
    if not os.path.exists(mmode_class_dir):
        os.makedirs(mmode_class_dir)
    for npz in os.listdir(npz_class_dir):
        npz_file_path = os.path.join(npz_class_dir, npz)
        n = np.load(npz_file_path)
        arr = (n['mmode'])
        cv2.imwrite(os.path.join(mmode_class_dir, npz[:-4]) + '.jpg', arr)
        # img = Image.fromarray(arr[:, :, 0])
        # img.save(os.path.join(mmode_class_dir, npz[:-4]) + '.jpg', 'jpeg')


def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


def length(video):
    clip = VideoFileClip(video)
    return clip.duration


if __name__=='__main__':
    raw_clips_dir = cfg['PATHS']['MASKED_VIDEOS']
    short_clips = []
    for clip_class in os.listdir(raw_clips_dir):
        clip_class_dir = os.path.join(raw_clips_dir, clip_class)
        for clip in os.listdir(clip_class_dir):
            l = length(os.path.join(clip_class_dir, clip))
            if l < 3.0:
                print(l)
                short_clips.append(os.path.join(clip_class_dir, clip))
    for clip in short_clips:
        print(clip)
    print(len(short_clips))
