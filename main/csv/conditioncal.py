import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np

import pandas as pd
from itertools import combinations



output_file = "cause_effect_pairs.csv"



pairs_df = pd.read_csv(output_file)
pairs_df['cause_date'] = pd.to_datetime(pairs_df['cause_date'].str[:10], format="%Y-%m-%d")
pairs_df['effect_date'] = pd.to_datetime(pairs_df['effect_date'].str[:10], format="%Y-%m-%d")


pairs_df['time_lag'] = (pairs_df['effect_date'] - pairs_df['cause_date']).dt.days
pairs_df = pairs_df[pairs_df['time_lag'] >= 0]  

lag_bins = [0, 7, 14, 30, 60, 90, 180, 365, 730, np.inf]
lag_labels = [f"{lag_bins[i]}-{lag_bins[i+1]}" for i in range(len(lag_bins)-1)]


pairs_df['lag_bin'] = pd.cut(pairs_df['time_lag'], bins=lag_bins, labels=lag_labels, right=True)


pair_counts = pairs_df.groupby(['cause', 'effect', 'lag_bin', 'cause_type', 'effect_type']).size().reset_index(name='pair_count')


cause_counts = pairs_df.groupby(['cause', 'cause_type']).size().reset_index(name='cause_count')


prob_df = pd.merge(pair_counts, cause_counts, on=['cause', 'cause_type'])
prob_df['conditional_probability'] = prob_df['pair_count'] / prob_df['cause_count']


prob_df = prob_df.drop(['pair_count', 'cause_count'], axis=1)
prob_df = prob_df[prob_df['conditional_probability'] > 0.0]


prob_df.to_csv("conditional_probabilities_all_events.csv", index=False)

print(f"Total Pairs Saved: {prob_df.shape[0]}")
print(prob_df.head())
















































































