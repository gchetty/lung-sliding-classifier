import pandas as pd
import cv2
import os

csv = 'csvs/sliding_boxes.csv'
df = pd.read_csv(csv)

vid_dir = 'masked_videos/sliding/'

ids = []
heights = []
widths = []
for index, row in df.iterrows():
    s = row['id']
    ids.append(s[s.rindex('/')+1:s.rindex('.mp4')])

    # Add original dimensions
    cap = cv2.VideoCapture(vid_dir + s + '.mp4')
    heights.append(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    widths.append(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

df['id'] = ids
df['height'] = heights
df['width'] = widths

df.to_csv(csv)
