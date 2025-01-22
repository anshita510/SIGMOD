
import socket
import json
import threading
import logging

class CustomProtocol:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = None

        logging.basicConfig(
            filename=f'./logs/protocol_log.txt',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.start_server()

    def is_port_in_use(self, port):
       
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, port)) == 0

    def start_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  
            self.server_socket.bind((self.host, self.port))
            logging.info(f"Server started at {self.host}:{self.port}")
        except Exception as e:
            logging.error(f"Failed to start server at {self.host}:{self.port}. Error: {str(e)}")
            raise


    def send_packet(self, packet, target_host, target_port):
        try:
            serialized_packet = json.dumps(packet).encode('utf-8')
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(serialized_packet, (target_host, target_port))
            logging.info(f"Packet sent to {target_host}:{target_port} - {packet}")
        except Exception as e:
            logging.error(f"Failed to send packet to {target_host}:{target_port}. Error: {str(e)}")

    def receive_packet(self, timeout=None):
        if timeout:
            self.server_socket.settimeout(timeout)
        try:
            data, addr = self.server_socket.recvfrom(65535)
            packet = json.loads(data.decode('utf-8'))
            logging.info(f"Packet received from {addr[0]}:{addr[1]} - {packet}")
            return packet, addr
        except socket.timeout:
            logging.warning("Timeout reached while waiting for packet.")
            return None, None
        except Exception as e:
            logging.error(f"Error receiving packet: {str(e)}")
            return None, None

    def close(self):
        if self.server_socket:
            self.server_socket.close()
            logging.info(f"Server socket closed at {self.host}:{self.port}")

