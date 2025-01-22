import subprocess
import os

monitoring_dir = 'monitoring_scripts/'

for port in range(6000, 6027):
    script_path = os.path.join(monitoring_dir, f'monitor_{port}.py')

    if not os.path.exists(script_path):
        with open(os.path.join('monitor_template.py'), 'r') as template:
            with open(script_path, 'w') as monitor_script:
                monitor_script.write(template.read())

    command = f"python3 {script_path} --port {port} --patient_id 8a2ab9dc --disease Stroke"
    subprocess.Popen(command, shell=True)

print("Monitoring started for all ports (6000-6026).")
