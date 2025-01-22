import threading
import time
from collections import deque
import psutil
import csv
import os

monitorlog_dir = '/Users/user_name/PhD/HealthDS/synthea/output/csv/logs/'
os.makedirs(monitorlog_dir, exist_ok=True)

def calculate_transfer_rate(rate, interval=3, interface="lo0"):
    start_time = time.time()
    network_stats = psutil.net_io_counters(pernic=True)[interface]
    total_data = (network_stats.bytes_sent, network_stats.bytes_recv)

    while True:
        previous_data = total_data
        time.sleep(interval)
        network_stats = psutil.net_io_counters(pernic=True)[interface]
        current_time = time.time()
        total_data = (network_stats.bytes_sent, network_stats.bytes_recv)
        
        upload_speed, download_speed = [
            (current - previous) / (current_time - start_time) / 1000.0
            for current, previous in zip(total_data, previous_data)
        ]
        rate.append((upload_speed, download_speed))
        start_time = time.time()

        print(f"[{interface}] Upload: {upload_speed:.2f} kB/s, Download: {download_speed:.2f} kB/s")

def save_rate_to_csv(rate):
    csv_file_path = os.path.join(monitorlog_dir, "localhost_transfer_rate.csv")

    
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Upload Rate (kB/s)", "Download Rate (kB/s)"])

    while True:
        if rate:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, rate[-1][0], rate[-1][1]])
        time.sleep(5)

if __name__ == "__main__":
    print("Starting localhost network monitoring...")

    transfer_rate_queue = deque(maxlen=1)

    
    monitor_thread = threading.Thread(
        target=calculate_transfer_rate,
        args=(transfer_rate_queue, 3, "lo0")
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    
    csv_thread = threading.Thread(
        target=save_rate_to_csv,
        args=(transfer_rate_queue,)
    )
    csv_thread.daemon = True
    csv_thread.start()

    
    while True:
        time.sleep(10)
