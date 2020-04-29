import socket
import numpy as np

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 64645        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	s.connect((HOST, PORT))

	while True:
		input_line = input(">>>")
		if input_line == "exit":
			break

		s.sendall(bytes(input_line, encoding='utf8'))
		data = s.recv(1024)

		try:
			data = np.frombuffer(data, dtype="float32")
		except ValueError as e:
			data = data.decode("utf-8")
			print(e)

		print(data)
