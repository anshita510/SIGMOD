device_lookup_path = '/Users/user_name/PhD/HealthDS/synthea/output/csv/device_partitions_patientwise1_rasp/newAll_Device_Lookup_with_Probabilities.csv'

initiator = None
final_device=None
total_time=0
experiment_id=None
disease_name=None
visited_devices=None
chain=None
t1=0
t2=0
t_fallback=0
t_dash=0
start_time=0
startdevicetimer=0
import subprocess
import networkx as nx
import os
from custom_protocol import CustomProtocol
import pandas as pd
import argparse
import threading
import logging
import socket
from contextlib import closing
import csv
import os
import psutil
import time
import socket
from datetime import datetime
protocol=[]

filtered_patients_path = '/Users/user_name/PhD/HealthDS/synthea/output/csv/filtered_patients_updated.csv'
filtered_patients = pd.read_csv(filtered_patients_path)

monitorlog_dir = '/Users/user_name/PhD/HealthDS/synthea/output/csv/logs'
os.makedirs(monitorlog_dir, exist_ok=True)

log_dir = '/Users/user_name/PhD/HealthDS/synthea/output/csv/device_partitions_patientwise1_rasp/'
os.makedirs(log_dir, exist_ok=True)

def monitor_resources(port, experiment_id, device_folder, disease_name, patient_id, final_device, visited_devices, chain, total_time, t1, t2,
            t_fallback, t_dash, interval=1):
    process = None
    device_log_dir = os.path.join(log_dir, device_folder)
    os.makedirs(device_log_dir, exist_ok=True)

    csv_file_path = os.path.join(device_log_dir, f"{device_folder}_{disease_name}_{patient_id}_metrics_log.csv")

    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
        'Timestamp', 'Disease Name', 'Patient ID', 'Initiator Device',
        'Final Device', 'Experiment ID', 'No. of Devices Accessed',
        'Chain Data', 'Total Time (T)',
        'Cached TPHG Load Time (t1)', 'Backward Viterbi Time (t2)',
        'Fallback Path Time (t_fallback)', 'Time per Device (t_dash)', 'Memory Usage (MB)', 'CPU Usage (%)',
        'RAM Usage (MB)', 'Energy Consumption (J)'
            ])
    process = find_process_by_port(port)
    print(port)
    print(process)
    if process:
        memory_usage = process.memory_info().rss / 1024 ** 2  
        cpu_usage = process.cpu_percent(interval=0.1)
        ram_usage = psutil.virtual_memory().used / 1024 ** 2  
        energy_consumption = cpu_usage * 0.1  
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = [
            timestamp, disease_name, patient_id, initiator,
            final_device, experiment_id, visited_devices,
            str(chain), total_time, t1, t2,
            t_fallback, t_dash, memory_usage,
            cpu_usage, ram_usage, energy_consumption
        ]

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_entry)

        print(f"[{device_folder}] Metrics logged for {patient_id} - {disease_name}.")
    else:
        print(f"No process found for port {port}. Retrying...")



def find_process_by_port(port):
    try:
        result = subprocess.run(
            ['lsof', '-i', f'UDP:{port}'],
            stdout=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            print(f"Process detected for UDP port {port}:")
            print(result.stdout)
            
            lines = result.stdout.splitlines()
            if len(lines) > 1:
                pid = int(lines[1].split()[1])  
                return psutil.Process(pid)
        else:
            print(f"No UDP process found on port {port}.")
            return None
    except Exception as e:
        print(f"Error running lsof: {str(e)}")
        return None




def get_relevant_devices(patient_id):
    relevant_devices = filtered_patients[
        (filtered_patients['PATIENT'] == patient_id)
    ]['RELEVANT_DEVICES'].tolist()
    
    print(f"Relevant devices for Patient {patient_id}: {relevant_devices}")
    return relevant_devices


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


device_folder = 'Device14'
device_partitions_patientwise = 'Device14/'
lookup_table = pd.read_csv(device_lookup_path)

device_row = lookup_table[lookup_table['device'] == device_folder]

if not device_row.empty:
    port = int(device_row['port'].iloc[0])
    ip_address = device_row['ip'].iloc[0]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((ip_address, port))
    except OSError:
        print(f"Port {port} is in use. Allocating a new port...")
        port = find_free_port()
        lookup_table.loc[lookup_table['device'] == device_folder, 'port'] = port
        lookup_table.to_csv(device_lookup_path, index=False)
else:
    ip_address = '172.20.10.2'
    port = find_free_port()
    new_row = {'device': device_folder, 'ip': ip_address, 'port': port}
    lookup_table = pd.concat([lookup_table, pd.DataFrame([new_row])], ignore_index=True)
    lookup_table.to_csv(device_lookup_path, index=False)



device_log_dir = os.path.join(log_dir, device_folder)
os.makedirs(device_log_dir, exist_ok=True)  

logging.basicConfig(
    filename=os.path.join(device_log_dir, f'{port}.txt'),  
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import joblib  

def load_tphg(patient_id):
    cache_file = os.path.join(log_dir, device_partitions_patientwise, f'{patient_id}_tphg.pkl')
    print(f"Loading TPHG from: {cache_file}")
    
    if os.path.exists(cache_file):
        print(f"[Device14] TPHG cache loaded for patient {patient_id}")
        
        loaded_graph = joblib.load(cache_file)
        
        if isinstance(loaded_graph, (nx.Graph, nx.MultiDiGraph)):
            return loaded_graph  
        else:
            raise ValueError(f"Unexpected data format in {cache_file}")
    
    print(f"[Device14] No TPHG cache found for patient {patient_id}.")
    return None

def random_walk_expansion(tphg, start_node, max_depth=10):
    import random

    if not tphg.has_node(start_node):
        print(f"Node '{start_node}' not found in TPHG. Random Walk cannot proceed.")
        return []

    current_node = start_node
    path = [current_node]
    path_probability = 1.0

    for _ in range(max_depth):
        neighbors = list(tphg.neighbors(current_node))
        if not neighbors:
            print(f"No neighbors found for node '{current_node}'. Ending Random Walk.")
            break

        next_node = random.choice(neighbors)
        edge_data = tphg.get_edge_data(current_node, next_node)

        if edge_data:
            step_probability = edge_data[0].get('probability', 1.0)
            path_probability *= step_probability

        path.append(next_node)
        current_node = next_node

        print(f"Random Walk Step: {path[-2]} -> {path[-1]} (P: {step_probability})")

    return path, path_probability

def test_random_walk(patient_id, start_event):
    tphg = load_tphg(patient_id)
    if tphg:
        print(f"Testing Random Walk Expansion from '{start_event}'")
        start_time = time.time()
        random_walk_path, random_walk_probability = random_walk_expansion(tphg, start_event)
        total_time = time.time() - start_time

        print(f"Random Walk Path: {random_walk_path}")
        print(f"Random Walk Path Probability: {random_walk_probability}")
        print(f"Random Walk Time Taken: {total_time:.2f} seconds")

        monitor_resources(
            port=0,  
            experiment_id=f"RandomWalk_{start_event}",
            device_folder=device_folder,
            disease_name="RandomWalkTest",
            patient_id=patient_id,
            final_device=random_walk_path[-1] if random_walk_path else None,
            visited_devices=len(random_walk_path),
            chain=random_walk_path,
            total_time=total_time,
            t1=0, t2=0, t_fallback=0, t_dash=total_time
        )
    else:
        print(f"TPHG not found for patient {patient_id}. Unable to perform Random Walk.")
