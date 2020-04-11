import os
import json
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm

from data_manager.dataset_manager import DataEngineTypical


class Mapper:
	def __init__(self, json_path: str):
		self.json_path = json_path

		with open(self.json_path, 'r') as f:
			self.data = json.load(f)

		self.keys = list(self.data.keys())
		self.values = self.data.values()

		self.values_all = []
		self.values_all_keys = []

		for i, list_big in enumerate(self.values):
			for n in list_big:
				self.values_all.append(n)
				self.values_all_keys.append(self.keys[i])

	def get_info(self, input_data):
		input_data_without_extension, extension = os.path.splitext(input_data)
		input_data_without_extension_single = input_data_without_extension.split("/")[-1]

		input_data_for_json = input_data_without_extension_single.split("_")[0] + ".mp4"

		try:
			main_video = self.values_all_keys[self.values_all.index(input_data_for_json)]
			reality = False
			side_videos = self.data[main_video]
		except:
			main_video = None
			reality = True
			side_videos = []

		return main_video, side_videos, reality, input_data_without_extension_single


class SpecialDataEngine(DataEngineTypical):
	def path_yielder(self):
		for x, y in zip(self.x_data, self.y_data):
			yield (x, y)

	def make_label_map(self):
		return False

	def get_triplet_examples_from_batch(self):
		for x, y in self.dataset:
			triplets = []

			all_classes, counts = np.unique(y, return_counts = True)
			positives = list(np.where(counts > 1)[0].tolist())
			negatives = list(np.where(counts == 1)[0].tolist())
			x = x.numpy()

			while len(positives) > 0 and len(negatives) > 0:
				p_0 = positives.pop()
				n_0 = negatives.pop()

				negative_image = x[np.where(y == all_classes[n_0])[0]]
				positives_images = x[np.where(y == all_classes[p_0])[0][:2]]

				triplets.append([positives_images[0], positives_images[1], negative_image[0]])

			for _ in range(8):
				an, ps, ng = self.get_same_triplet()
				a_path, p_path, n_path = self.load_triplet_from_ids(an, ps, ng)
				triplets.append([self.image_loader(a_path), self.image_loader(p_path), self.image_loader(n_path)])

			yield tf.convert_to_tensor(triplets)


	def load_triplet_from_ids(self, a_id, p_id, n_id):
		a_y_where = np.random.choice(np.where(self.y_data_more == a_id)[0], 1)[0]
		p_y_where = np.random.choice(np.where(self.y_data_more == p_id)[0], 1)[0]
		n_y_where = np.random.choice(np.where(self.y_data_more == n_id)[0], 1)[0]

		return self.x_data[a_y_where], self.x_data[p_y_where], self.x_data[n_y_where]

	def get_same_triplet(self, main_id: str = None):
		if main_id is None:
			main_id = np.random.choice(self.mapper_for_data.keys, 1)[0]
                
		try:
			main_keymap_id = self.label_map[main_id.strip(".mp4")]
		except KeyError:
			return self.get_same_triplet()
        
		main_real = np.random.choice([True, False], 1)[0]
        
		fake_videos = self.mapper_for_data.data[main_id]   
    
		if len(fake_videos) == 1:
			fake_videos = [fake_videos[0], fake_videos[0]]
        
		fake_ids = []
		for n in fake_videos:
			try:
				fake_ids.append(self.label_map[n.strip(".mp4")])
			except KeyError:
				pass
    
		np.random.shuffle(fake_ids)
		if len(fake_ids) < 2:
			return self.get_same_triplet()
		elif main_real:
			return [main_keymap_id, main_keymap_id, fake_ids[0]]
		else:
			return [fake_ids[0], fake_ids[1], main_keymap_id]

	def __init__(self, main_path: str, batch_size: int = 16, buffer_size: int = 10000, epochs: int = 1, reshuffle_each_iteration: bool = False, test_batch = 64,
		map_to: bool = True, **kwargs):

		self.main_path = main_path
		self.mapper_for_data = Mapper(os.path.join("/".join(self.main_path.split("/")[:-2]), "orig.json"))

		self.label_map = {}
		self.reverse_label_map = {}
		self.i = len(self.label_map)

		self.x_data, self.y_data_more, self.y_data = self.read_images()
		self.y_data = np.array(self.y_data)
		self.y_data_more = np.array(self.y_data_more)
		self.x_data = np.array(self.x_data)

		self.batch_size = batch_size

		super(SpecialDataEngine, self).__init__(
			main_path=main_path, 
			batch_size=batch_size, 
			buffer_size=buffer_size, 
			epochs=epochs, 
			reshuffle_each_iteration=reshuffle_each_iteration, 
			test_batch=test_batch, 
			map_to=map_to, 
			**kwargs
		)

	def read_images(self):
		x_data_train, y_data_train, y_data_real = [], [], []

		for label in ["fake", "real"]:
			for path in tqdm(glob(f"{self.main_path}/train/{label}/*.png"), f"Reading {label} from train"):
				x_data_train.append(path)

				if label == "fake":
					main_video, side_videos, reality, input_data_without_extension_single = self.mapper_for_data.get_info(path)
					input_data_without_extension_single = input_data_without_extension_single.split("_")[0]
				else:
					input_data_without_extension_single = os.path.splitext(path)[0].split("/")[-1].split("_")[0].split(".")[0]	

				if input_data_without_extension_single not in self.label_map.keys():
					self.i += 1
					self.label_map[input_data_without_extension_single] = self.i

				y_data_train.append(self.label_map[input_data_without_extension_single])

				if label == "real":
					y_data_real.append(self.label_map[input_data_without_extension_single])

		for label in ["fake", "real"]:
			for path in tqdm(glob(f"{self.main_path}/test/{label}/*.png"), f"Reading {label} from test"):
				x_data_train.append(path)
				main_video, side_videos, reality, input_data_without_extension_single = self.mapper_for_data.get_info(path)
				input_data_without_extension_single = input_data_without_extension_single.split("_")[0]

				if input_data_without_extension_single not in self.label_map.keys():
					self.i += 1
					self.label_map[input_data_without_extension_single] = self.i

				y_data_train.append(self.label_map[input_data_without_extension_single])

				if label == "real":
					y_data_real.append(self.label_map[input_data_without_extension_single])

		self.depth = len(self.label_map)
		self.reverse_label_map =  {v: k for k, v in self.label_map.items()}

		return x_data_train, y_data_train, y_data_real


if __name__ == '__main__':
	print("go check README.md")
