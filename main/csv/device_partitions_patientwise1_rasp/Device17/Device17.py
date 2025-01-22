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
        'RAM Usage (MB)', 'Energy Consumption (J)', 'current_device'
            ])
    process = find_process_by_port(port)
    print(port)
    print(process)
    if process:
        memory_usage = process.memory_info().rss / 1024 ** 2  
        cpu_usage = process.cpu_percent(interval=0.1)
        ram_usage = psutil.virtual_memory().used / 1024 ** 2  
        energy_consumption = cpu_usage * 0.1  
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]

        log_entry = [
            timestamp, disease_name, patient_id, initiator,
            final_device, experiment_id, visited_devices,
            str(chain), total_time, t1, t2,
            t_fallback, t_dash, memory_usage,
            cpu_usage, ram_usage, energy_consumption, device_folder
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


device_folder = 'Device17'
device_partitions_patientwise = 'Device17/'
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
        print(f"[Device17] TPHG cache loaded for patient {patient_id}")
        
        loaded_graph = joblib.load(cache_file)
        
        if isinstance(loaded_graph, (nx.Graph, nx.MultiDiGraph)):
            return loaded_graph  
        else:
            raise ValueError(f"Unexpected data format in {cache_file}")
    
    print(f"[Device17] No TPHG cache found for patient {patient_id}.")
    return None

def fallback_path_expansion(tphg, max_depth=10):
    paths = []
    for node in tphg.nodes:
        for target, path in nx.single_source_shortest_path(tphg, node, cutoff=max_depth).items():
            if len(path) > 1:  
                lag = sum(tphg[path[i]][path[i+1]][0].get('lag_bin', float('inf'))
                          for i in range(len(path)-1))
                paths.append((path, lag))
                print(f"Fallback Path: {path} (Lag: {lag})")
                
        if not paths:
            paths.append(([node], 0))
            print(f"Trivial Path (no outgoing edges): {[node]} (Lag: 0)")

    return paths

def viterbi_expand_backward(event, tphg, max_depth=10):
    from heapq import heappush, heappop
    global t_fallback
    global start_fallback
    pq = []
    completed_paths = {}
    prob_cache = {}

    if tphg.has_node(event):
        print(f"Event '{event}' found in TPHG. Beginning expansion.")
        heappush(pq, (0, 1.0, (event,)))
    else:
        print(f"Event '{event}' not found in TPHG. Triggering fallback path expansion.")
        start_fallback=time.time()
        fallback_paths = fallback_path_expansion(tphg, max_depth)
        t_fallback=time.time()-start_fallback
        for path, lag in fallback_paths:
            heappush(pq, (lag, 1.0, tuple(path)))
            print(f"Fallback Path added: {path} (Lag: {lag})")

    while pq:
        current_lag, current_prob, current_path = heappop(pq)
        last_node = current_path[0]

        if len(current_path) > max_depth:
            continue

        if last_node in prob_cache and prob_cache[last_node] >= current_prob:
            continue

        completed_paths[last_node] = current_path
        prob_cache[last_node] = current_prob

        if not tphg.has_node(last_node):
            continue

        expanded = False
        for pred in tphg.predecessors(last_node):
            if tphg.has_edge(pred, last_node):
                edge_data = tphg.edges[pred, last_node, 0]
                lag_bin = edge_data.get('lag_bin', float('inf'))
                edge_prob = edge_data.get('probability', 1.0)
                new_prob = current_prob * edge_prob
                new_path = (pred,) + current_path

                heappush(pq, (current_lag + lag_bin, new_prob, new_path))
                print(f"Path expanded: {new_path} (Lag: {current_lag + lag_bin}, P: {new_prob})")
                expanded = True

        if not expanded:
            print(f"No further expansion for '{last_node}'. Adding fallback path.")
            if last_node not in completed_paths:
                completed_paths[last_node] = current_path
                prob_cache[last_node] = current_prob

    print(f"Completed Paths: {completed_paths}")
    print(f"Probability Cache: {prob_cache}")

    if not completed_paths:
        print("No paths found. Returning fallback terminal path.")
        return [(event,)]

    return sorted(completed_paths.values(), key=lambda x: prob_cache.get(x[-1], 0), reverse=True)



def expand_causal_chain(event, chain, visited, patient_id):
    global t1,t2
    tphg_load_time=time.time()
    tphg = load_tphg(patient_id)
    t1=time.time()-tphg_load_time
    local_paths = []
    edge_set = set()

    if tphg:
        print("Entering viterbi")
        viterbi_start=time.time()
        viterbi_path = viterbi_expand_backward(event, tphg)
        t2=time.time()-viterbi_start
        print("Viterbi Paths: ", viterbi_path)
    
        if not viterbi_path:
            print("No paths found during viterbi expansion.")
            return local_paths
        
        for path_data in viterbi_path:
            print(f"Path data structure: {path_data}")

            if isinstance(path_data, tuple) and len(path_data) > 1:
                path = path_data
                for i in range(len(path) - 1):
                    pred = path[i]
                    effect = path[i + 1]
                    edge = (pred, effect)

                    if pred == effect:
                        print(f"Self-loop at {pred}. Adding to chain.")
                        chain.append([pred, effect, 1.0, 0, device_folder])
                        continue

                    if not tphg.has_edge(pred, effect):
                        print(f"No edge exists between {pred} and {effect}")
                        continue

                    if edge not in edge_set:
                        edge_set.add(edge)
                        edges = tphg[pred][effect]
                        for key, data in edges.items():
                            prob = data.get('probability', 1.0)
                            lag = data.get('lag_bin', float('inf'))
                            
                            chain.append([pred, effect, prob, lag, device_folder])
                            print("final_chain", chain)
                            
                            local_paths.append((pred, effect, prob, lag, device_folder))
                            visited.add((pred, effect))
                            print(f"Path added to chain: {pred} -> {effect} (P: {prob}, Lag: {lag})")
                    else:
                        print(f"Skipping duplicate edge: {pred} -> {effect}")

            elif isinstance(path_data, tuple) and len(path_data) == 1:
                node = path_data[0]
                print(f"Fallback path expansion ended at: {node}")

                chain.append([node, node, 1.0, 0, device_folder])  
                print(f"Single-node fallback path added: {node} -> {node}")

                print("Triggering chain forwarding after single-node fallback...")
                forward_chain(chain, patient_id, device_folder)

    return local_paths
import re


def return_to_initiator(chain, initiator, patient_id):
    print(f"Returning final chain to initiator {initiator}")
    print(lookup_table.loc[lookup_table['device'] == initiator, 'port'].values[0])
    final_effects = {effect for _, effect, _, _, _ in chain}
    
    protocol.send_packet({
        'effects': list(final_effects),
        'patient_id': patient_id,
        'chain': chain,
        'initiator': 'Device20',
        'final_packet': True  
    }, '172.20.10.2', 6019)


device_effect_map = {}  

import ast


def normalize_list(lst):
    return [str(e).strip().replace("'", "").lower() for e in lst]


import ast

def find_next_device(effect, patient_id, sender_device=None, visited_devices=None):
    global device_effect_map
    global filtered_lookup_table
    escaped_effect = effect.strip().lower()
    best_device = None
    max_prob = -1  
    fallback_candidates = []

    if visited_devices is None:
        visited_devices = set()

    effect_found = False

    relevant_devices = get_relevant_devices(patient_id)
    
    relevant_devices = relevant_devices[0].split(',')
    relevant_devices = [dev.strip() for dev in relevant_devices]
    lookup_table['device'] = lookup_table['device'].astype(str).str.strip()
    relevant_devices = [str(dev).strip() for dev in relevant_devices]

    filtered_lookup_table = lookup_table[lookup_table['device'].isin(relevant_devices)]
    print(filtered_lookup_table)
  
    
    
    
    for idx, row in filtered_lookup_table.iterrows():
        
        try:
            cause_list = ast.literal_eval(row['cause_type'])
            
            effect_list = ast.literal_eval(row['effect_type'])
        except (ValueError, SyntaxError):
            cause_list = str(row['cause_type']).strip("[]'").split(", ")
            effect_list = str(row['effect_type']).strip("[]'").split(", ")

        cause_list = [str(c).strip().lower() for c in cause_list]
        effect_list = [str(e).strip().lower() for e in effect_list]
        next_device = row['device']
        next_port = row['port']
        next_ip = row['ip']
        print(visited_devices)
        print(device_folder.lower())
        
        print("next_device", next_device)
        if next_device in visited_devices or next_device == device_folder:
            continue

        if escaped_effect in effect_list:
            effect_found = True
            probability_list = row['probability_list']
            try:
                probability_list = ast.literal_eval(probability_list)
                probability_list = [float(prob) for prob in probability_list]
            except (ValueError, SyntaxError):
                probability_list = []

            if probability_list:
                effect_index = effect_list.index(escaped_effect)

                for i, cause in enumerate(cause_list):
                    prob_index = i * len(effect_list) + effect_index

                    if prob_index < len(probability_list):
                        effect_prob = probability_list[prob_index]
                    else:
                        effect_prob = 0

                    if effect_prob > max_prob:
                        max_prob = effect_prob
                        best_device = (next_device, next_port, next_ip, effect_prob)

    if best_device:
        next_device, next_port, next_ip, highest_prob = best_device
        device_effect_map[next_device] = {
            'port': next_port,
            'ip': next_ip,
            'effects': {effect},
            'probability': highest_prob
        }
        print(f"Forwarding to best device '{next_device}' (P: {highest_prob}) for effect '{effect}' on port {next_port}")
        return

    if not effect_found:
        print(f"No direct match for effect '{effect}'. Checking cause_type for fallback...")
        print(device_folder)
        current_device_row = lookup_table[lookup_table['device'] == device_folder]
        if not current_device_row.empty:
            try:
                cause_type_list = ast.literal_eval(current_device_row['cause_type'].iloc[0])
            except (ValueError, SyntaxError):
                cause_type_list = str(current_device_row['cause_type'].iloc[0]).strip("[]'").split(", ")
           
            normalized_causes = [c.strip().lower() for c in cause_type_list]
            print(f"Using cause_type: {normalized_causes} for fallback search...")

            for idx, row in filtered_lookup_table.iterrows():
                try:
                    effect_list = ast.literal_eval(row['effect_type'])
                except (ValueError, SyntaxError):
                    effect_list = str(row['effect_type']).strip("[]'").split(", ")

                effect_list = [str(e).strip().lower() for e in effect_list]
                next_device = row['device']
                next_port = row['port']
                next_ip = row['ip']
                if next_device in visited_devices or next_device == device_folder:
                    continue

                matching_effects = set(normalized_causes).intersection(effect_list)
                if matching_effects:
                    probability_list = ast.literal_eval(row['probability_list'])
                    try:
                        highest_prob = max(probability_list)
                    except (ValueError, SyntaxError):
                        highest_prob = 0

                    fallback_candidates.append((next_device, next_port, next_ip, highest_prob))
                    print(f"Fallback match: Device '{next_device}' for effects '{matching_effects}'.")

    if fallback_candidates:
        best_fallback = max(fallback_candidates, key=lambda x: x[2])
        next_device, next_port, next_ip, highest_prob = best_fallback
        print(best_fallback)
        visited_devices.add(next_device)
        device_effect_map.clear()
        device_effect_map[next_device] = {
            'port': next_port,
            'ip': next_ip,
            'effects': {effect},
            'probability': highest_prob
        }

        print(f"Selected fallback device '{next_device}' for effect '{effect}' on port {next_port}")
    else:
        print(f"No fallback candidates found for '{effect}'. Halting expansion.")


def find_best_fallback_device_viterbi(effect, patient_id):
    tphg = load_tphg(patient_id)
    print("findingbest fallabck")
    
    if tphg is not None:
        viterbi_paths = viterbi_expand_backward(effect, tphg)
        
        if viterbi_paths:
            for path in viterbi_paths:
                for node in path:
                    current_device_row = filtered_lookup_table[filtered_lookup_table['device'] == device_folder]
                    if current_device_row.empty:
                        continue
                    
                    cause_type_list = ast.literal_eval(current_device_row['cause_type'].iloc[0])

                    matching_rows = filtered_lookup_table[
                            filtered_lookup_table['effect_type'].apply(
                            lambda x: any(cause in ast.literal_eval(x) for cause in cause_type_list)
                        )
                    ]

                    if not matching_rows.empty:
                        best_match = None
                        max_prob = -1

                        for idx, row in matching_rows.iterrows():
                            prob_list = ast.literal_eval(row['probability_list'])
                            highest_prob = max(prob_list) if prob_list else 0

                            if highest_prob > max_prob:
                                max_prob = highest_prob
                                best_match = (row['device'], row['port'], row['ip'], highest_prob)

                        if best_match:
                            next_device, next_port, next_ip, highest_prob = best_match
                            print(f"Fallback to device '{next_device}' (P: {highest_prob}) for cause '{cause_type_list}'")
                            return next_device, next_port, next_ip

    print("No valid fallback device found via Viterbi.")
    return None, None





def forward_chain(chain, patient_id, sender_device=None):
    global protocol, t_dash, visited_devices
    device_effect_map.clear()

    visited_devices = {device for _, _, _, _, device in chain} if chain else set()
    print("Visited Devices (Forward Chain):", visited_devices)

    forwarded = False
    max_prob = 0  

    if chain:
        last_effect = chain[-1][0]  
        print(last_effect, "last effect")
        find_next_device(last_effect, patient_id, sender_device, visited_devices)

    for next_device, info in device_effect_map.items():
        next_port = info['port']
        next_ip=info['ip']
        effects = list(info['effects'])
        max_prob = max(info.get('probability', 0), max_prob)  

        print(f"[{device_folder}] Forwarding chain to '{next_device}' on port {next_port} for effects: {effects}")
        t_dash=time.time()-startdevicetimer
        monitor_resources(port, experiment_id, device_folder, disease_name, patient_id, final_device, visited_devices, chain, total_time, t1, t2, t_fallback, t_dash, interval=1)
        protocol.send_packet({
            'effects': effects,
            'patient_id': patient_id,
            'chain': chain,
            'initiator': initiator,
            'final_packet': False
        }, next_ip, next_port)
        forwarded = True

    if not device_effect_map:
        print("No device found in direct lookup. Performing Viterbi fallback...")
        
        if chain:
            last_effect = chain[-1][0]
            next_device, next_port, next_device = find_best_fallback_device_viterbi(last_effect, patient_id)
            
            if next_device:
                print(f"Fallback to device '{next_device}' (Port: {next_port})")
                if next_device not in device_effect_map:
                    device_effect_map[next_device] = {'port': next_port, 'ip': next_ip, 'effects': set()}
                device_effect_map[next_device]['effects'].add(last_effect)
                forwarded = True
            else:
                print("Fallback failed. No device to handle effect.")

    if max_prob < 0.0001:
        print(f"Low probability ({max_prob}). Returning chain to initiator.")
        return_to_initiator(chain, sender_device, patient_id)
        return

    if not forwarded:
        if len(set([d for _, _, _, _, d in chain])) == 1:
            print("Cycle detected. Returning to initiator.")
        elif len(chain) > 20:
            print("Maximum chain depth reached. Returning to initiator.")
        elif not device_effect_map:
            print("All fallback paths exhausted. Returning to initiator.")
        else:
            print("No device found to handle effect. Returning chain to initiator.")
            
        return_to_initiator(chain, sender_device, patient_id)
        return

def handle_request(protocol):
    global startdevicetimer
    global initiator
    global experiment_id
    global final_device, start_time, total_time
    while True:
        try:
            print("Waiting to receive")
            packet, addr = protocol.receive_packet()

            if packet:
                
                event = packet.get('effects')
                patient_id = packet.get('patient_id')
                initiator = packet.get('initiator')
                chain = packet.get('chain', [])
                final_packet = packet.get('final_packet', False)

                print(f"[{device_folder}] Received packet for event '{event}' (Final: {final_packet})")
                experiment_id = f"Exp_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                print("starting to log")
                monitor_resources(port, experiment_id, device_folder, event, patient_id, final_device, visited_devices, chain, total_time, t1, t2, t_fallback, t_dash, interval=1)
                if final_packet:
                    print(f"[{device_folder}] Final packet received. Chain processing complete.")
                    total_time = time.time() - start_time
                    final_device=device_folder
                    monitor_resources(port, experiment_id, device_folder, event, patient_id, final_device, visited_devices, chain, total_time, t1, t2, t_fallback, t_dash, interval=1)
                    print("Final chain",chain)
                    return  
                
                else:
                    if event:
                        print(f"[{device_folder}] Received event '{event}' for patient '{patient_id}' from {initiator}")
                        startdevicetimer=time.time()
                        tphg = load_tphg(patient_id)
                        print("Edges in the graph:", list(tphg.edges(data=True)))
                        if tphg:

                            expand_causal_chain(event, chain, set(), patient_id)
                            
                            print(chain)
                            if chain:
                                print(f"[{device_folder}] Expanding and forwarding chain for event '{event}'")
                                forward_chain(chain, patient_id, initiator)
                        else:
                            print(f"[{device_folder}] No TPHG found. Returning chain to initiator {initiator}")
                            return_to_initiator(chain, initiator, patient_id)

        except Exception as e:
            logging.error(f"[{device_folder}] Error during packet handling: {e}")


def initiate_chain(event, patient_id):
    global protocol,initiator
    chain = []
    visited = set()
    device_effect_map.clear()  
    initiator="Device20"
    print(f"[{device_folder}] Initiating chain for event '{event}'")
    tphg = load_tphg(patient_id)
    local_paths = expand_causal_chain(event, chain, visited, patient_id)
    print(chain)
    if chain:
        print(f"[{device_folder}] Expanding and forwarding chain for event '{event}'")
        forward_chain(chain, patient_id, initiator)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start Device Service')
    parser.add_argument('--initiate', type=str, help='Trigger event for causal chain')
    parser.add_argument('--patient_id', type=str, help='Patient ID for processing')
    args = parser.parse_args()

    protocol = CustomProtocol(ip_address, port)
   

    if args.initiate and args.patient_id:
        initiate_chain(args.initiate, args.patient_id)
    threading.Thread(target=handle_request, args=(protocol,)).start()
