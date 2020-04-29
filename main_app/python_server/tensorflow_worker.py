import os
import cv2
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys

sys.path.append("../../")
from face_detection import mtcnn_detector
from glob import glob
from shutil import rmtree
from sklearn.decomposition import PCA


class Utils:
	def __init__(self):
		os.makedirs("saved_outputs", exist_ok=True)
		os.makedirs("faces", exist_ok=True)

		os.makedirs("faces2display", exist_ok=True)
		rmtree("faces2display")
		os.makedirs("faces2display", exist_ok=True)

	@staticmethod
	def get_json_id():
		json_files = glob("saved_outputs/*.json")
		max_id = 0
		for file in json_files:
			max_id = max(max_id, int(os.path.split(file.rstrip(".json"))[-1]))

		return str(max_id + 1)

	@staticmethod
	def get_faces2display_id():
		json_files = glob("faces2display/*.jpg")
		max_id = 0
		for file in json_files:
			max_id = max(max_id, int(os.path.split(file.rstrip(".jpg"))[-1]))

		return str(max_id + 1)

	def save_outputs(self, path, outputs):
		file_id = self.get_json_id()
		file_path = os.path.join("saved_outputs/", file_id + ".json")
		file_dict = {"path": path, "outputs": outputs.tolist()}

		with open(file_path, 'w') as outfile:
			json.dump(file_dict, outfile)

		return file_id


class DataBaseManager:
	def save(self):
		with open(self.database_path, 'w') as outfile:
			json.dump(self.data, outfile)

	def get_new_id(self):
		all_ids = list(set(list(self.data.keys())))
		if len(all_ids) == 0:
			all_ids = [0]

		return int(all_ids[-1]) + 1

	def get_2d_space(self):
		try:
			outputs = []
			y_data = []
			for key in self.data:
				outputs.append(self.data[key]["output"])
				y_data.append(int(key))

			pc_all = self.pca.fit_transform(outputs)

			fig, ax = plt.subplots(figsize=(10, 10))
			fig.patch.set_facecolor('white')
			for l in np.unique(y_data)[:10]:
				ix = np.where(y_data == l)
				ax.scatter(pc_all[:, 0][ix], pc_all[:, 1][ix])

			plt.savefig("2d_space.jpg")
		except Exception as e:
			print(e)
			return bytes(np.array([-1], dtype=np.float32))

		return bytes(np.array([1], dtype=np.float32))

	def __init__(self, distance_metric):
		self.database_path = "database.json"
		self.distance_metric = distance_metric
		self.pca = PCA(n_components=2)  # compress 512-D data to 2-D, we need to do that if we want to display data.

		if not os.path.exists(self.database_path):
			q = open(self.database_path, "w+")
			q.write("{}")
			q.close()

		with open(self.database_path, "r") as read_file:
			self.data = json.load(read_file)

	def find_match_in_db(self, output, th: float = 1.0):
		min_im = (th, -1, "none")
		for key in self.data:
			output_db = tf.convert_to_tensor(self.data[key]["output"])
			dist = self.distance_metric(output_db, output).numpy()
			if dist < min_im[0]:
				min_im = (dist, int(key), self.data[key]["name"])

		return min_im

	def add_to_db(self, output, name, face_frames):
		new_id = self.get_new_id()
		if os.path.exists(f"faces/{new_id}.jpg"):
			new_id += 1

		cv2.imwrite(f"faces/{new_id}.jpg", Engine.turn_rgb(tf.cast((face_frames * 128.) + 127., tf.uint8))[0].numpy())
		self.data[new_id] = {"name": name, "output": output.tolist(), "face": os.path.join(os.getcwd(), "faces", f"{new_id}.jpg")}

		self.save()

	def reset_database(self):
		self.data = {}
		self.save()

		return bytes(np.array([1], dtype=np.float32))


class Engine:
	@staticmethod
	def turn_rgb(images):
		b, g, r = tf.split(images, 3, axis=-1)
		images = tf.concat([r, g, b], -1)

		return images

	@staticmethod
	def set_face(face):
		face = tf.image.resize(face, (112, 112), method="nearest")

		return (tf.cast(face, tf.float32) - 127.5) / 128.

	def find_who(self, path):
		outputs, face_frames = self.go_for_image_features(path, to_bytes=False)
		match = self.db_manager.find_match_in_db(outputs, th=1.0)

		return bytes(np.array(match[1], dtype=np.float32))

	def add_to_database(self, path, name):
		outputs, face_frames = self.go_for_image_features(path, to_bytes=False)
		self.db_manager.add_to_db(outputs, name=name, face_frames=face_frames)

		return bytes(np.array([1], dtype=np.float32))

	def get_only_face_and_save(self, path):
		image = self.detector.load_image(path)
		faces = self.detector.get_faces_from_image(image)
		boxes = self.detector.get_boxes_from_faces(faces)
		face_frames = self.turn_rgb(self.detector.take_faces_from_boxes(image, boxes))
		face_frames = tf.convert_to_tensor([self.set_face(n) for n in face_frames])

		face_id = self.utils.get_faces2display_id()
		cv2.imwrite(f"faces2display/{face_id}.jpg", Engine.turn_rgb(tf.cast((face_frames * 128.) + 127., tf.uint8))[0].numpy())

		return bytes(np.array([face_id], dtype=np.float32))

	def __init__(self, model_path: str):
		self.cos_dis = lambda x, y: tf.norm(x - y)  # tf.keras.losses.CosineSimilarity()

		self.utils = Utils()
		self.db_manager = DataBaseManager(self.cos_dis)

		self.model = tf.keras.models.load_model(model_path)
		self.data = {}
		self.detector = mtcnn_detector.Engine()

		self.find_who("init.jpg")
		self.db_manager.get_2d_space()

	def get_output(self, images):
		return tf.nn.l2_normalize(self.model(images, training=False))

	def go_full_webcam(self, path=0):
		try:
			path = int(path)
		except:
			pass

		cap = cv2.VideoCapture(path)

		if not cap.isOpened():
			print("No webcam!")
			return bytes(np.array([0], dtype=np.float32))

		color_map = {}

		while True:
			try:
				ret, frame = cap.read()

				faces = self.detector.get_faces_from_image(frame)
				boxes = self.detector.get_boxes_from_faces(faces)
				if len(boxes) > 0:
					face_frames = [self.turn_rgb(n) for n in self.detector.take_faces_from_boxes(frame, boxes)]
					face_frames = tf.convert_to_tensor([self.set_face(n) for n in face_frames])
					output = self.get_output(face_frames).numpy()

					colors = []
					names = [self.db_manager.find_match_in_db(out, th=1.0)[-1] for out in output]
					for name in names:
						if not name in color_map.keys():
							color_map[name] = self.detector.generate_color()

						colors.append(color_map[name])

					frame = self.detector.draw_faces_and_labels_on_image(frame, boxes, names, color=colors)

				cv2.imshow('Input', frame)

				c = cv2.waitKey(1)
				if c == 27 or ret is False:
					break

			except Exception as e:
				if "not valid" in str(e):
					break
				continue

		cap.release()
		cv2.destroyAllWindows()

		return bytes(np.array([1], dtype=np.float32))

	def go_for_image_features(self, path, to_bytes: bool = True):
		print(f"Getting Features for: {path}")
		image = self.detector.load_image(path)
		faces = self.detector.get_faces_from_image(image)
		boxes = self.detector.get_boxes_from_faces(faces)
		face_frames = self.turn_rgb(self.detector.take_faces_from_boxes(image, boxes))
		face_frames = tf.convert_to_tensor([self.set_face(n) for n in face_frames])
		output = self.get_output(face_frames)[0].numpy()

		if to_bytes:
			output = output.tobytes()

			return output

		else:
			return output, face_frames

	def compare_two(self, path1, path2, to_bytes: bool = True):
		output1, face_frames1 = self.go_for_image_features(path1, to_bytes=False)
		output2, face_frames2 = self.go_for_image_features(path2, to_bytes=False)
		dist = self.cos_dis(output1, output2).numpy()
		print(f"Distance between {path1} - {path2} --> {dist}")

		if to_bytes:
			dist = tf.cast([dist], tf.float32).numpy().tostring()

		return dist

	def save_outputs_to_json(self, path):
		outputs, face_frames = self.go_for_image_features(path, to_bytes=False)
		file_id = self.utils.save_outputs(path, outputs)

		return bytes(np.array([file_id], dtype=np.float32))


if __name__ == '__main__':
	e = Engine("arcface_final.h5")
	e.go_full_webcam()
