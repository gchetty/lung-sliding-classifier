import os
import pandas as pd

df = pd.read_csv('csvs/sliding_flow_mini_clips.csv')

indices = []

for index, row in df.iterrows():
    if not os.path.exists(os.path.join('flow_npzs/sliding/', row['id'] + '.npz')):
        indices.append(index)

print('Dropping ' + str(len(indices)) + ' bad entries')
df = df.drop(indices)

df.to_csv('csvs/sliding_flow_mini_clips.csv', index=False)
