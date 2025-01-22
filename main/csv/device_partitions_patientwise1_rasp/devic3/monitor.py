import os
import time
import pandas as pd
import psutil
import csv
from datetime import datetime


base_directory = "Device3"


output_csv = "device_metrics_analysis26.csv"


if not os.path.exists(output_csv):
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Device", "File", "Load Time (s)", "Memory Usage (MB)",
            "CPU Usage (%)", "RAM Usage (MB)", "Energy Consumption (J)"
        ])

def calculate_energy(cpu_usage):
    return cpu_usage * 0.1  

device_folder = base_directory
print(device_folder )
if os.path.isdir(device_folder):
    
    for file_name in os.listdir(device_folder):
        if file_name.endswith("causal_pairs.csv"):
            
            file_path = os.path.join(device_folder, file_name)
            print(file_path)
            
            start_time = time.time()
            data = pd.read_csv(file_path)
            load_time = time.time() - start_time

            
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 2)  
            cpu_usage = process.cpu_percent(interval=0.1)  
            ram_usage = psutil.virtual_memory().used / (1024 ** 2)  
            energy_consumption = calculate_energy(cpu_usage)
            
            with open(output_csv, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    device_folder, file_name, round(load_time, 4), round(memory_usage, 2),
                    round(cpu_usage, 2), round(ram_usage, 2), round(energy_consumption, 4)
                ])

            print(f"Metrics logged for {device_folder} - {file_name}")

print(f"Metrics analysis completed. Results stored in {output_csv}.")
