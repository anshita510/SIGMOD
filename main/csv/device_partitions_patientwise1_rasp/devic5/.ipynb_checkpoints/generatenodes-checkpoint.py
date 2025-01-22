import pandas as pd
import os
from contextlib import closing
import socket

device_lookup_path = '/Users/user_name/PhD/HealthDS/synthea/output/csv/device_partitions_patientwise1/All_Device_Lookup_effects_with_ports.csv'
base_dir = '/Users/user_name/PhD/HealthDS/synthea/output/csv/device_partitions_patientwise1/'

lookup_table = pd.read_csv(device_lookup_path)

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

device_template = """
import networkx as nx
import os
from custom_protocol import CustomProtocol
import pandas as pd
import argparse
import threading
import logging
import socket
from contextlib import closing
import joblib

log_dir = '{log_dir}'
device_folder = '{device_folder}'
device_partitions_patientwise = '{device_folder}/'
protocol = []

os.makedirs(log_dir, exist_ok=True)

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

lookup_table = pd.read_csv('{lookup_table_path}')

device_row = lookup_table[lookup_table['device'] == device_folder]

if not device_row.empty:
    port = int(device_row['port'].iloc[0])
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
    except OSError:
        print(f"Port {port} is in use. Allocating a new port...")
        port = find_free_port()
        lookup_table.loc[lookup_table['device'] == device_folder, 'port'] = port
        lookup_table.to_csv('{lookup_table_path}', index=False)
else:
    port = find_free_port()
    new_row = {{'device': device_folder, 'port': port}}
    lookup_table = pd.concat([lookup_table, pd.DataFrame([new_row])], ignore_index=True)
    lookup_table.to_csv('{lookup_table_path}', index=False)

logging.basicConfig(
    filename=os.path.join(log_dir, device_folder, f'{port}.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_tphg(patient_id):
    cache_file = os.path.join(log_dir, device_partitions_patientwise, f'{patient_id}_tphg.pkl')
    print(f"Loading TPHG from: {{cache_file}}")

    if os.path.exists(cache_file):
        loaded_graph = joblib.load(cache_file)
        if isinstance(loaded_graph, (nx.Graph, nx.MultiDiGraph)):
            return loaded_graph
        else:
            raise ValueError(f"Unexpected data format in {{cache_file}}")
    print(f"[{{device_folder}}] No TPHG cache found for patient {patient_id}.")
    return None

def get_best_edge(tphg, pred, effect):
    if tphg.has_edge(pred, effect):
        edges = tphg[pred][effect]
        best_edge = min(edges.items(), key=lambda x: x[1].get('lag_bin', float('inf')))
        return best_edge
    return None

def viterbi_expand_backward(event, tphg, max_depth=10):
    from heapq import heappush, heappop
    pq = []
    completed_paths = {}
    lag_cache = {}

    heappush(pq, (0, (event,)))

    while pq:
        current_lag, current_path = heappop(pq)
        last_node = current_path[0]

        if len(current_path) > max_depth:
            continue

        if last_node in lag_cache and lag_cache[last_node] <= current_lag:
            continue

        completed_paths[last_node] = current_path
        lag_cache[last_node] = current_lag

        for pred in tphg.predecessors(last_node):
            if pred == last_node:
                continue

            edge_data = tphg.edges[pred, last_node, 0]
            lag_bin = edge_data.get('lag_bin', float('inf'))
            new_lag = current_lag + lag_bin
            new_path = (pred,) + current_path

            heappush(pq, (new_lag, new_path))
            print(f"Backward Path found: {new_path} (Lag: {new_lag})")

    for node in tphg.nodes:
        if node not in completed_paths:
            recursive_backtrack(node, tphg, completed_paths)

    return list(completed_paths.values())


def recursive_backtrack(node, tphg, completed_paths, path=None, max_depth=10):
    if path is None:
        path = (node,)

    if len(path) > max_depth:
        return

    if node in completed_paths:
        return

    for pred in tphg.predecessors(node):
        if pred != node:
            new_path = (pred,) + path
            completed_paths[node] = new_path
            print(f"Recursively adding path: {new_path}")
            recursive_backtrack(pred, tphg, completed_paths, new_path)
def expand_causal_chain(event, chain, visited, patient_id):
    tphg = load_tphg(patient_id)
    print("Edges in the graph:", list(tphg.edges(data=True)))

    local_paths = []
    edge_set = set()

    if tphg:
        viterbi_path = viterbi_expand_backward(event, tphg)

        for path_data in viterbi_path:
            print(f"Path data structure: {path_data}")

            if isinstance(path_data, tuple) and len(path_data) > 1:
                path = path_data

                for i in range(len(path) - 1):
                    pred = path[i]
                    effect = path[i + 1]
                    edge = (pred, effect)

                    if pred == effect:
                        print(f"Skipping self-loop at {pred}")
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

                            chain.append((pred, effect, prob, device_folder))
                            local_paths.append((pred, effect, prob, device_folder))
                            visited.add((pred, effect))
                            print(f"Path added to chain: {pred} -> {effect} (P: {prob}, Lag: {lag})")
                    else:
                        print(f"Skipping duplicate edge: {pred} -> {effect}")

    return local_paths



import re



device_effect_map = {}  

def find_next_device(effect, sender_device=None):
    global device_effect_map
    escaped_effect = re.escape(effect)

    for idx, row in lookup_table.iterrows():
        effect_list = str(row['effect_type']).split(",")
        next_device = row['device']
        next_port = row['port']

        if sender_device and next_device == sender_device:
            continue

        if any(re.search(escaped_effect, eff, re.IGNORECASE) for eff in effect_list):
            if next_device not in device_effect_map:
                device_effect_map[next_device] = {'port': next_port, 'effects': set()}
            
            device_effect_map[next_device]['effects'].add(effect)
            print(f"Device '{next_device}' handles effect '{effect}' on port {next_port}")

def forward_chain(chain, patient_id, sender_device=None):
    global protocol
    device_effect_map.clear()

    for _, effect, _, _ in chain:
        find_next_device(effect, sender_device)

    for next_device, info in device_effect_map.items():
        next_port = info['port']
        effects = list(info['effects'])

        print(f"[{device_folder}] Forwarding chain to '{next_device}' on port {next_port} for effects: {effects}")
        protocol.send_packet({
            'effects': effects,
            'patient_id': patient_id,
            'chain': chain,
            'initiator': device_folder
        }, 'localhost', next_port)

def handle_request(protocol):
    while True:
        try:
            packet, addr = protocol.receive_packet()
            if packet:
                event = packet.get('event')
                patient_id = packet.get('patient_id', 'unknown')
                initiator = packet.get('initiator')
                chain = packet.get('chain', [])

                if event:
                    print(f"[{device_folder}] Received event '{event}' for patient '{patient_id}' from {initiator}")

                    tphg = load_tphg(patient_id)
                    if tphg:
                        expand_causal_chain(event, chain, set(), patient_id)

                    if chain:
                        print(f"[{device_folder}] Expanding and forwarding chain for event '{event}'")
                        forward_chain(chain, patient_id, initiator)
        except Exception as e:
            logging.error(f"[{device_folder}] Error during packet handling: {e}")


def initiate_chain(event, patient_id):
    global protocol
    chain = []
    visited = set()
    device_effect_map.clear()  

    print(f"[{device_folder}] Initiating chain for event '{event}'")
    
    local_paths = expand_causal_chain(event, chain, visited, patient_id)
    print(local_paths)

    if local_paths:
        print(f"[{device_folder}] Paths found: {local_paths}")
    else:
        print(f"[{device_folder}] No paths found. Forwarding to next device.")

    for _, effect, _, _ in chain:
        find_next_device(effect)
    
    for next_device, info in device_effect_map.items():
        next_port = info['port']
        effects = list(info['effects'])

        print(f"Forwarding chain to '{next_device}' on port {next_port} for effects: {effects}")
        protocol.send_packet({
            'effects': effects,
            'patient_id': patient_id,
            'chain': chain,
            'initiator': device_folder
        }, 'localhost', next_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start Device Service')
    parser.add_argument('--initiate', type=str, help='Trigger event for causal chain')
    parser.add_argument('--patient_id', type=str, help='Patient ID for processing')
    args = parser.parse_args()

    protocol = CustomProtocol('localhost', port)
    threading.Thread(target=handle_request, args=(protocol,)).start()
    if args.initiate and args.patient_id:
        initiate_chain(args.initiate, args.patient_id)
"""

for _, row in lookup_table.iterrows():
    device_name = row['device']
    device_folder = os.path.join(base_dir, device_name)
    os.makedirs(device_folder, exist_ok=True)

    script = device_template.format(
        device_folder=device_name,
        log_dir=device_folder,
        lookup_table_path=device_lookup_path,
        port='{port}',  
        patient_id='{patient_id}'  
    )



    script_path = os.path.join(device_folder, f"{device_name}.py")
    with open(script_path, 'w') as f:
        f.write(script)

    print(f"Script for {device_name} generated at {script_path}.")
