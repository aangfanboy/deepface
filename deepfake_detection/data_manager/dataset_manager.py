import os
import json
import tensorflow as tf

from glob import glob
from tqdm import tqdm


class DataEngineTypical:
	@staticmethod
	def image_loader(image):
		image = tf.io.read_file(image)
		image = tf.io.decode_jpeg(image, channels=3)
		image = tf.image.resize(image, (112, 112), method="nearest")
		image = tf.image.random_flip_left_right(image)

		return (tf.cast(image, tf.float32) - 127.5) / 128.

	def mapper(self, path, label):
		return self.image_loader(path), label

	def save_key_real_fake_map(self):
		with open(os.path.join(self.main_path, "key_real_fake_map.json"), 'w') as f:
			json.dump(self.key_real_fake, f)

	def load_key_real_fake_map(self):
		if os.path.exists(os.path.join(self.main_path, "key_real_fake_map.json")):
			with open(os.path.join(self.main_path, "key_real_fake_map.json")) as f:
				self.key_real_fake = json.load(f)

			return True

		else:
			return False

	def __init__(self, main_path: str, mode: str = "id", batch_size: int = 16, buffer_size: int = 10000, epochs: int = 1,
	             reshuffle_each_iteration: bool = False, map_to: bool = True):
		self.main_path = main_path
		self.label_map = {}
		self.i = 0

		self.mode = mode
		if self.mode != "rf" and self.mode != "id":
			raise Exception(f"\"mode\" must be either 'rf' or 'id'. 'rf' stands for real-fake classes, id for face recognition.")

		self.load_key_real_fake_map()
		self.key_real_fake = {}

		self.x_train, self.y_train, self.x_test, self.y_test = self.load_images()

		self.dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
		self.dataset_test = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
		if buffer_size > 0:
			self.dataset = self.dataset.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=42)
			self.dataset_test = self.dataset_test.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=42)
		if map_to:
			self.dataset = self.dataset.map(self.mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			self.dataset_test = self.dataset_test.map(self.mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)

		self.dataset = self.dataset.batch(batch_size, drop_remainder=True).repeat(epochs)
		self.dataset_test = self.dataset_test.batch(batch_size, drop_remainder=True).repeat(epochs)

	def load_images(self):
		x_data, y_data = [], []
		x_test, y_test = [], []

		label_fake_real_map = {"real": 0, "fake": 1}

		for label in ["real", "fake"]:
			for path in tqdm(glob(f"{self.main_path}/train/{label}/*.png"), f"Reading {label} from train"):
				x_data.append(path)
				label_m = path.split("\\")[-1].split("_")[0]

				if not label_m in self.label_map.keys():
					self.label_map[label_m] = self.i
					self.key_real_fake[label_m] = label_fake_real_map[label]
					self.i += 1

				if self.mode == "rf":
					y_data.append(self.key_real_fake[label_m])
				if self.mode == "id":
					y_data.append(self.label_map[label_m])

		for label in ["real", "fake"]:
			for path in tqdm(glob(f"{self.main_path}/test/{label}/*.png"), f"Reading {label} from test"):
				x_test.append(path)
				label_m = path.split("\\")[-1].split("_")[0]

				if not label_m in self.label_map.keys():
					self.label_map[label_m] = self.i
					self.key_real_fake[label_m] = label_fake_real_map[label]
					self.i += 1

				if self.mode == "rf":
					y_test.append(self.key_real_fake[label_m])
				if self.mode == "id":
					y_test.append(self.label_map[label_m])

		self.save_key_real_fake_map()
		return x_data, y_data, x_test, y_test


if __name__ == '__main__':
	print("go check README.md")
