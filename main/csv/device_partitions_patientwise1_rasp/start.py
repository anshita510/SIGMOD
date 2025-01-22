import os
import socket
import subprocess
import signal
import logging

log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, 'device_execution.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

root_device_dir = '/Users/user_name/PhD/HealthDS/synthea/output/csv/device_partitions_patientwise1'
patient_id = '8a2ab9dc-d34e-9f31-9a98-14bcf27330c7'

subprocess_list = []

def is_port_free(port):
    """Check if the port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def run_device_scripts():
    for device_folder in os.listdir(root_device_dir):
        device_folder_path = os.path.join(root_device_dir, device_folder)
        
        if not os.path.isdir(device_folder_path) or not device_folder.startswith('Device'):
            continue
        
        device_scripts = [f for f in os.listdir(device_folder_path) if f.endswith('.py')]
        
        if not device_scripts:
            logging.warning(f"No Python files found in {device_folder}.")
            continue
        
        for script in device_scripts:
            script_path = os.path.join(device_folder_path, script)
            
            try:
                device_number = int(device_folder.replace('Device', ''))
                port = 6000 + device_number
            except ValueError:
                logging.error(f"Failed to extract device number from {device_folder}")
                continue

            try:
                if not is_port_free(port):
                    logging.warning(f"Port {port} already in use. Skipping {script}.")
                    continue

                if 'Device20' in script_path:
                    process = subprocess.Popen([
                        'osascript', '-e',
                        f'''
                        tell application "Terminal"
                            do script "python3 {script_path} --initiate Stroke --patient_id {patient_id}"
                        end tell
                        '''
                    ])
                else:
                    process = subprocess.Popen([
                        'osascript', '-e',
                        f'''
                        tell application "Terminal"
                            do script "python3 {script_path}"
                        end tell
                        '''
                    ])
                
                subprocess_list.append(process)
                logging.info(f"Started {script} in {device_folder} (PID: {process.pid})")

            except Exception as e:
                logging.error(f"Failed to execute {script_path}. Error: {str(e)}")

def terminate_all_processes(signum, frame):
    logging.info("Termination signal received. Shutting down all terminals...")
    for process in subprocess_list:
        try:
            process.terminate()
            logging.info(f"Terminated process with PID: {process.pid}")
        except Exception as e:
            logging.error(f"Failed to terminate process {process.pid}. Error: {str(e)}")
    os._exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, terminate_all_processes)
    run_device_scripts()
    logging.info("All device scripts initiated. Press Ctrl+C to terminate.")
    signal.pause()
