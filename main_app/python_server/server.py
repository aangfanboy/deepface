import os
import types
import socket
import selectors
import numpy as np

from tensorflow_worker import Engine


class Server:
    def process_input(self, input_command):
        output_command = input_command
        if input_command == b"clean up":
            os.system("cls")
            output_command = bytes(np.array([1.]))
        elif b"get_features" in input_command:
            input_command = input_command.decode("utf-8")
            output_command = self.tensorflow_engine.go_for_image_features(input_command.strip().split(" ")[-1].strip())
        elif b"compare" in input_command:
            input_command = input_command.decode("utf-8")
            paths = input_command.strip().split(" ")[1:]
            output_command = self.tensorflow_engine.compare_two(paths[0].strip(), paths[1].strip())
        elif b"save_outputs" in input_command:
            input_command = input_command.decode("utf-8")
            output_command = self.tensorflow_engine.save_outputs_to_json(input_command.strip().split(" ")[-1].strip())
        elif b"add_to_db" in input_command:
            input_command = input_command.decode("utf-8")
            path = input_command.strip().split(" ")[-2].strip()
            name = input_command.strip().split(" ")[-1].strip()
            output_command = self.tensorflow_engine.add_to_database(path, name)
        elif b"find_who" in input_command:
            input_command = input_command.decode("utf-8")
            output_command = self.tensorflow_engine.find_who(input_command.strip().split(" ")[-1].strip())
        elif b"give_face" in input_command:
            input_command = input_command.decode("utf-8")
            output_command = self.tensorflow_engine.get_only_face_and_save(input_command.strip().split(" ")[-1].strip())
        elif b"create_2d_space" == input_command:
            output_command = self.tensorflow_engine.db_manager.get_2d_space()
        elif b"reset_database" == input_command:
            output_command = self.tensorflow_engine.db_manager.reset_database()
        elif b"go_for_webcam" in input_command:
            input_command = input_command.decode("utf-8")
            output_command = self.tensorflow_engine.go_full_webcam(path=input_command.strip().split(" ")[-1].strip())
        elif b"go_for_video" in input_command:
            input_command = input_command.decode("utf-8")
            output_command = self.tensorflow_engine.go_full_webcam(path=input_command.strip().split(" ")[-1].strip())
        elif b"test_deepfake" in input_command:
            input_command = input_command.decode("utf-8")
            output_command = self.tensorflow_engine.is_deepfake(path=input_command.strip().split(" ")[-1].strip())
        elif b"update_ase" in input_command:
            input_command = input_command.decode("utf-8")
            aaa = input_command.strip().split(" ")
            path_g = aaa[-2].strip()
            person_id_g = aaa[-1].strip()
            output_command = self.tensorflow_engine.update_ASE(path=path_g, person_id=int(person_id_g))

        return output_command

    def __init__(self, host: str, port: int):
        self.HOST = host
        self.PORT = port

        self.tensorflow_engine = Engine("arcface_final.h5")
        print("TensorFlow Engine is ready")

        self.sel = selectors.DefaultSelector()
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.bind((self.HOST, self.PORT))
        lsock.listen()
        print("server streaming on", (self.HOST, self.PORT))
        lsock.setblocking(False)
        self.sel.register(lsock, selectors.EVENT_READ, data=None)

    def __call__(self, *args, **kwargs):
        while True:
            events = self.sel.select(timeout=None)
            for key, mask in events:
                if key.data is None:
                    self.accept_wrapper(key.fileobj)
                else:
                    self.service_connection(key, mask)

    def accept_wrapper(self, sock):
        conn, addr = sock.accept()  # Should be ready to read
        print("accepted connection from", addr)
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b'', outb=b'')
        _events = selectors.EVENT_READ | selectors.EVENT_WRITE
        self.sel.register(conn, _events, data=data)

    def service_connection(self, _key, _mask):
        sock = _key.fileobj
        data = _key.data
        if _mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)  # Should be ready to read
            if recv_data:
                output_line = self.process_input(recv_data)
                data.outb += output_line
            else:
                print('closing connection to', data.addr)
                self.sel.unregister(sock)
                sock.close()
        if _mask & selectors.EVENT_WRITE:
            if data.outb:
                sent = sock.send(data.outb)  # Should be ready to write
                data.outb = data.outb[sent:]


if __name__ == '__main__':
    ser = Server("127.0.0.1", 64645)
    ser()
