import os
import pandas as pd

df = pd.read_csv('csvs/no_sliding_flow_mini_clips.csv')

indices = []

for index, row in df.iterrows():
    if not (os.path.exists(os.path.join('flow_npzs/no_sliding/', row['id'] + '.npz')) or os.path.exists(os.path.join('flow_npzs/orig_no_sliding/', row['id'] + '.npz'))):
        indices.append(index)

print('Dropping ' + str(len(indices)) + ' bad entries')
df = df.drop(indices)

df.to_csv('csvs/no_sliding_flow_mini_clips.csv', index=False)
