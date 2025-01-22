import psutil
import time
import csv
import os
import argparse
from datetime import datetime

def find_process_by_port(port):
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port and conn.type in (psutil.SOCK_STREAM, psutil.SOCK_DGRAM):
                    print(f"Found process: {proc.info['name']} (PID: {proc.info['pid']}) using port {port}")
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None


def get_network_io(proc):
    try:
        net_io = proc.io_counters()
        return net_io.write_bytes / 1024 ** 2, net_io.read_bytes / 1024 ** 2  
    except AttributeError:
        return 0, 0


def monitor_resources(port, patient_id, disease_name):
    log_dir = f'logs/Device_{port}'
    os.makedirs(log_dir, exist_ok=True)
    csv_file_path = os.path.join(log_dir, f"metrics_log_{port}.csv")

    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Timestamp', 'CPU Usage (%)', 'RAM Usage (MB)',
                'Memory Usage (MB)', 'Energy Consumption (J)',
                'Upload Rate (MB)', 'Download Rate (MB)', 'Patient ID',
                'Disease Name', 'Port'
            ])

    while True:
        process = find_process_by_port(port)
        if process:
            print(f"[{port}] Monitoring process (PID: {process.pid})...")
            while process.is_running():
                try:
                    memory_usage = process.memory_info().rss / 1024 ** 2
                    cpu_usage = process.cpu_percent(interval=1)
                    ram_usage = psutil.virtual_memory().used / 1024 ** 2
                    energy_consumption = cpu_usage * 0.1
                    upload_rate, download_rate = get_network_io(process)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = [
                        timestamp, cpu_usage, ram_usage, memory_usage,
                        energy_consumption, upload_rate, download_rate,
                        patient_id, disease_name, port
                    ]

                    with open(csv_file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(log_entry)

                    print(f"[{port}] Metrics logged.")
                except psutil.NoSuchProcess:
                    print(f"[{port}] Process ended. Waiting for new process...")
                    break
        else:
            print(f"[{port}] Waiting for process...")
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor resources by port.')
    parser.add_argument('--port', required=True, type=int, help='Port to monitor')
    parser.add_argument('--patient_id', required=True, help='Patient ID')
    parser.add_argument('--disease', required=True, help='Disease name')
    args = parser.parse_args()

    monitor_resources(args.port, args.patient_id, args.disease)
