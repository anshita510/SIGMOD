
csv_file="filtered_patients_updated.csv"


device20_script="device_partitions_patientwise1_rasp/Device20/Device20.py"

python_interpreter="python3"


device20_port=6019  

cleanup_port() {
    local port=$1
    echo "Checking if port $port is in use..."
    while lsof -i udp:$port > /dev/null; do
        echo "Port $port is in use. Terminating the process..."
        pid=$(lsof -i udp:$port | awk 'NR>1 {print $2}' | head -n 1)
        if [ -n "$pid" ]; then
            kill -9 "$pid" && echo "Terminated process using port $port (PID: $pid)."
        fi
        sleep 2  
    done
    echo "Port $port is now free."
}


process_patient_disease() {
    local patient=$1
    local disease_name=$2

    echo "Starting initiator (Device20) for Patient: $patient, Disease: $disease_name..."

    
    cleanup_port "$device20_port"

    
    $python_interpreter "$device20_script" --initiate "$disease_name" --patient_id "$patient" &
    device20_pid=$!

    echo "Device20 started (PID: $device20_pid) on port $device20_port."

    
}


while IFS=',' read -r PATIENT DESCRIPTION CODE CONDITION RELEVANT_DEVICES; do
    
    if [[ "$PATIENT" == "PATIENT" ]]; then
        continue
    fi

    
    disease_name=$(echo "$DESCRIPTION" | tr -d '[:space:]')
    patient=$(echo "$PATIENT" | tr -d '[:space:]')

    
    process_patient_disease "$patient" "$disease_name"

    
    echo "Waiting for 10 seconds before starting the next pair..."
    sleep 20
done < "$csv_file"


echo "All initiator processes triggered."
