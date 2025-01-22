
device_lookup_path = '/Users/user_name/PhD/HealthDS/synthea/output/csv/device_partitions_patientwise1/All_Device_Lookup_causes_with_ports.csv'

import networkx as nx
import os
from custom_protocol import CustomProtocol
import pandas as pd
import argparse
import threading
import logging
import socket
from contextlib import closing

log_dir = '/Users/user_name/PhD/HealthDS/synthea/output/csv/device_partitions_patientwise1/'
os.makedirs(log_dir, exist_ok=True)

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

device_folder = '.ipynb_checkpoints'
device_partitions_patientwise = '.ipynb_checkpoints/'
lookup_table = pd.read_csv(device_lookup_path)

device_row = lookup_table[lookup_table['device'] == device_folder]

if not device_row.empty:
    port = int(device_row['port'].iloc[0])
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
    except OSError:
        logging.warning(f"Port {port} is in use. Allocating a new port...")
        port = find_free_port()
        lookup_table.loc[lookup_table['device'] == device_folder, 'port'] = port
        lookup_table.to_csv(device_lookup_path, index=False)
else:
    port = find_free_port()
    new_row = {'device': device_folder, 'port': port}
    lookup_table = pd.concat([lookup_table, pd.DataFrame([new_row])], ignore_index=True)
    lookup_table.to_csv(device_lookup_path, index=False)

logging.basicConfig(
    filename=os.path.join(log_dir, device_folder, f'{port}.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_tphg(patient_id):
    cache_file = os.path.join(device_partitions_patientwise, f'tphg_cache_{patient_id}.gpickle')
    if os.path.exists(cache_file):
        logging.info(f"[.ipynb_checkpoints] TPHG cache loaded for patient {patient_id}")
        return nx.read_gpickle(cache_file)
    logging.warning(f"[.ipynb_checkpoints] No TPHG cache found for patient {patient_id}.")
    return None

def viterbi_expand(event, tphg):
    if not tphg or event not in tphg.nodes:
        logging.warning(f"[.ipynb_checkpoints] Event '{event}' not found in TPHG.")
        return []

    paths = {event: [(event, 1.0)]}
    predecessors = list(nx.bfs_edges(tphg.reverse(), event))
    
    for pred, effect in predecessors:
        edge_data = tphg.edges[pred, effect, 0]
        prob = edge_data['probability']

        if effect in paths:
            for path, path_prob in paths[effect]:
                cumulative_prob = path_prob * prob
                new_path = [(pred, effect, device_folder)] + path
                
                if pred not in paths or cumulative_prob > paths[pred][0][1]:
                    paths[pred] = [(new_path, cumulative_prob)]

    start_path = paths.get(event, [])
    return max(start_path, key=lambda x: x[1]) if start_path else []

def expand_causal_chain(event, chain, visited, patient_id):
    tphg = load_tphg(patient_id)
    local_paths = []
    if tphg:
        viterbi_path = viterbi_expand(event, tphg)
        for pred, effect, device in viterbi_path:
            if (pred, effect) not in visited:
                prob = tphg.edges[pred, effect, 0]['probability']
                chain.append((pred, effect, prob, device))
                local_paths.append((pred, effect, prob, device))
                visited.add((pred, effect))
                expand_causal_chain(pred, chain, visited, patient_id)
    return local_paths

def find_next_device(effect):
    next_device_row = lookup_table[lookup_table['effect_type'].str.contains(effect, case=False)]
    if not next_device_row.empty:
        next_device = next_device_row.iloc[0]['device']
        next_port = next_device_row.iloc[0]['port']
        return next_device, next_port
    return None, None

def handle_request(protocol):
    while True:
        try:
            packet, addr = protocol.receive_packet()
            if packet:
                event = packet.get('event')
                patient_id = packet.get('patient_id', 'unknown')
                initiator = packet.get('initiator')
                chain = packet.get('chain', [])
                local_paths = []

                if event:
                    logging.info(f"[.ipynb_checkpoints] Received event '{event}' for patient '{patient_id}' from {initiator}")

                    tphg = load_tphg(patient_id)
                    if tphg:
                        local_paths = expand_causal_chain(event, chain, set(), patient_id)
                        if local_paths:
                            logging.info(f"[.ipynb_checkpoints] Local expansion found paths for '{event}'")

                    chain.sort(key=lambda x: -x[2])
                    for path in chain:
                        logging.info(f"Path Expanded: {path[0]} -> {path[1]} (P: {path[2]:.2f}) via {path[3]}")
                    
                    last_effect = chain[-1][1] if chain else event
                    next_device, next_port = find_next_device(last_effect)

                    if next_device:
                        logging.info(f"[.ipynb_checkpoints] Forwarding to {next_device} on port {next_port}")
                        next_packet = {
                            'event': last_effect,
                            'patient_id': patient_id,
                            'chain': chain,
                            'initiator': initiator
                        }
                        protocol.send_packet(next_packet, 'localhost', next_port)
        except Exception as e:
            logging.error(f"[.ipynb_checkpoints] Error during packet handling: {e}")

def initiate_chain(event, patient_id):
    protocol = CustomProtocol('localhost', port)
    chain = []

    initiator_packet = {
        'event': event,
        'initiator': device_folder,
        'patient_id': patient_id,
        'chain': chain
    }

    try:
        next_device, next_port = find_next_device(event)
        if next_device:
            logging.info(f"[.ipynb_checkpoints] Initiating causal chain for '{event}' (Patient: {patient_id}) to {next_device}")
            protocol.send_packet(initiator_packet, 'localhost', next_port)
        else:
            logging.warning(f"[.ipynb_checkpoints] No relevant device found for '{event}'")
    except Exception as e:
        logging.error(f"[.ipynb_checkpoints] Error initiating causal chain: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start Device Service')
    parser.add_argument('--initiate', type=str, help='Trigger event for causal chain')
    parser.add_argument('--patient_id', type=str, help='Patient ID for processing')
    args = parser.parse_args()

    protocol = CustomProtocol('localhost', port)
    threading.Thread(target=handle_request, args=(protocol,)).start()

    if args.initiate and args.patient_id:
        initiate_chain(args.initiate, args.patient_id)

