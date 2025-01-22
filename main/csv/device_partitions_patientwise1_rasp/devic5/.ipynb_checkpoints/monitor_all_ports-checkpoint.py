import subprocess
import os

monitoring_dir = 'monitoring_scripts/'

for port in range(6000, 6027):
    script_path = os.path.join(monitoring_dir, f'monitor_{port}.py')
    command = f"python3 {script_path} --port {port} --initiate Stroke --patient_id 8a2ab9dc-d34e-9f31-9a98-14bcf27330c7"
    
    subprocess.Popen(command, shell=True)

print("Monitoring started for all ports (6000-6026).")

