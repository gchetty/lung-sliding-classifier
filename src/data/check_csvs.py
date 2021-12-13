import os
import pandas as pd

df = pd.read_csv('csvs/no_sliding_flow_mini_clips.csv')

bad_count = 0

for index, row in df.iterrows():
    if not os.path.exists(os.path.join('flow_npzs/no_sliding/', row['id'])):
        bad_count += 1

print(bad_count)
