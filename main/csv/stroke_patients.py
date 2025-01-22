import pandas as pd


conditions = pd.read_csv('conditions.csv')


condition_terms = {
    'Stroke': ['stroke'],
    'Heart Attack': ['Myocardial infarction (disorder)'],
    'Lung Cancer': ['lung cancer'],
    'Seizure': ['Seizure disorder']
}


filtered_patients = pd.DataFrame()


for condition, terms in condition_terms.items():
    mask = conditions['DESCRIPTION'].str.contains(
        '|'.join(terms), case=False, na=False
    )
    if condition == 'Heart Attack':  
        mask = conditions['DESCRIPTION'].str.strip().str.lower().isin(
            [term.lower() for term in terms]
        )
    
    
    patients = conditions[mask].copy()
    patients['CONDITION'] = condition
    filtered_patients = pd.concat([filtered_patients, patients])


filtered_patients[['PATIENT', 'DESCRIPTION', 'CODE', 'CONDITION']].to_csv('filtered_patients.csv', index=False)
print("Filtered patients saved to filtered_patients.csv")
