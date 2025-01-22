import pandas as pd
import os
from glob import glob
import ast

device_probabilities = {}

lookup_table_path = 'All_Device_Lookup_merged_with_ports.csv'
lookup_table = pd.read_csv(lookup_table_path)

device_files = glob(os.path.join('**', '*_causal_pairs.csv'), recursive=True)

print(device_files)
for file in device_files:
    device_df = pd.read_csv(file)
    device_name = device_df['device'].iloc[0] if 'device' in device_df.columns else os.path.basename(file).split('_')[0]

    print(f"Processing {device_name}...")
    generalized_entries = device_df[
        (device_df['cause_type'].str.lower().isin(['condition', 'immunization'])) |
        (device_df['effect_type'].str.lower().isin(['condition', 'immunization']))
    ]

    for idx, row in generalized_entries.iterrows():
        cause = row['cause']
        effect = row['effect']
        cause_type = row['cause_type']
        effect_type = row['effect_type']
        if cause_type.lower() == 'condition':
            device_df.loc[idx, 'cause_type'] = cause
        if effect_type.lower() == 'condition':
            device_df.loc[idx, 'effect_type'] = effect
        if cause_type.lower() == 'immunization':
            device_df.loc[idx, 'cause_type'] = cause
        if effect_type.lower() == 'immunization':
            device_df.loc[idx, 'effect_type'] = effect
        print(f"Updated: cause_type='{cause}' effect_type='{effect}' for device {device_name} (Row {idx})")
   grouped = (
        device_df
        .groupby(['cause_type', 'effect_type'])['conditional_probability']
        .apply(list) 
        .reset_index()
    )
   device_probabilities[device_name] = grouped


def attach_probability_list(row):
    device_name = row['device']
    
    if device_name in device_probabilities:
        prob_df = device_probabilities[device_name]
        cause_types = set(ast.literal_eval(row['cause_type']))
        effect_types = set(ast.literal_eval(row['effect_type']))
        matched_probs = prob_df[
            prob_df.apply(lambda x: x['cause_type'] in cause_types and x['effect_type'] in effect_types, axis=1)
        ]['conditional_probability'].tolist()
    
        if matched_probs:
            return [prob for sublist in matched_probs for prob in sublist]
        
    return []


lookup_table['probability_list'] = lookup_table.apply(attach_probability_list, axis=1)
empty_probs = lookup_table[lookup_table['probability_list'].apply(lambda x: len(x) == 0)]
if not empty_probs.empty:
    print("Rows with empty probability lists:")
    print(empty_probs)

lookup_table.to_csv('All_Device_Lookup_with_Probabilities.csv', index=False)
print("Probability lists attached to lookup table.")
