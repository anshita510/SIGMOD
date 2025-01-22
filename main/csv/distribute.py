import pandas as pd
import os
from collections import defaultdict


prob_df = pd.read_csv("conditional_probabilities_all_events.csv")

condition_keywords = {
    'Blood Pressure': ['hypertension', 'hypotension', 'bp'],
    'Glucose': ['diabetes', 'hyperglycemia', 'hypoglycemia'],
    'Heart Rate': ['tachycardia', 'bradycardia', 'arrhythmia'],
    'Body Weight': ['obesity', 'underweight'],
    'Creatinine': ['kidney', 'renal', 'creatinine'],
    'ALT (Elevated)': ['liver', 'alt', 'hepatitis'],
    'AST (Elevated)': ['ast'],
    'Sodium': ['sodium', 'natremia'],
    'Respiratory Rate': ['asthma', 'pneumonia', 'copd', 'wheezing'],
    'Calcium': ['hypocalcemia', 'hypercalcemia'],
    'Hemoglobin': ['anemia', 'polycythemia'],
    'Platelet Count': ['thrombocytopenia', 'thrombocytosis'],
    'Oxygen Saturation': ['hypoxemia', 'respiratory failure']
}



device_mapping = {}
device_usage_count = defaultdict(int)
next_device_id = 1


event_types = pd.concat([prob_df['cause_type'], prob_df['effect_type']]).unique()


for event_type in event_types:
    if event_type not in device_mapping:
        device_name = f"Device_{next_device_id}"
        device_mapping[event_type] = device_name
        print(f"Assigned {device_name} to handle '{event_type}' events.")
        next_device_id += 1
    
    device_usage_count[device_mapping[event_type]] += len(
        prob_df[(prob_df['cause_type'] == event_type) | (prob_df['effect_type'] == event_type)]
    )


if 'condition' not in device_mapping:
    device_mapping['condition'] = f"Device_{next_device_id}"
    next_device_id += 1

if 'immunization' not in device_mapping:
    device_mapping['immunization'] = f"Device_{next_device_id}"
    next_device_id += 1


blood_pressure_device = f"Device_{next_device_id}"
device_mapping['Blood Pressure'] = blood_pressure_device
print(f"Assigned {blood_pressure_device} to handle Blood Pressure events.")
next_device_id += 1


device_mapping['Systolic Blood Pressure'] = blood_pressure_device
device_mapping['Diastolic Blood Pressure'] = blood_pressure_device


assigned_conditions = set()


for condition in prob_df[prob_df['cause_type'] == 'condition']['cause'].unique():
    assigned = False
    
    
    for obs, keywords in condition_keywords.items():
        if any(keyword.lower() in condition.lower() for keyword in keywords):
            
            if obs == 'Blood Pressure':
                if condition not in assigned_conditions:
                    device_mapping[condition] = blood_pressure_device
                    print(f"Condition '{condition}' mapped to {blood_pressure_device} (Blood Pressure).")
                    assigned_conditions.add(condition)
                    assigned = True
                    break
            else:
                
                if condition not in assigned_conditions:
                    device_mapping[condition] = device_mapping[obs]
                    print(f"Condition '{condition}' mapped to {device_mapping[obs]} (from {obs}).")
                    assigned_conditions.add(condition)
                    assigned = True
                    break
    
    
    if not assigned:
        device_mapping[condition] = device_mapping['condition']


print("\nFinal Device Mapping:")
for cause_type, device in device_mapping.items():
    print(f"{cause_type}: {device}")


device_partitions = defaultdict(list)

for _, row in prob_df.iterrows():
    
    if row['cause_type'] == 'condition':
        assigned_device = device_mapping.get(row['cause'], device_mapping['condition'])
    else:
        assigned_device = device_mapping[row['cause_type']]
    
    device_partitions[assigned_device].append(row)


output_dir = "device_partitions"
os.makedirs(output_dir, exist_ok=True)

for device, data in device_partitions.items():
    device_df = pd.DataFrame(data)
    file_path = os.path.join(output_dir, f"{device}_causal_pairs.csv")
    device_df.to_csv(file_path, index=False)
    print(f"{device} stores {len(device_df)} causal pairs. Saved to {file_path}.")
