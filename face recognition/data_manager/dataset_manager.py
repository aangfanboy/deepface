import numpy as np
import tensorflow as tf


class MainEngineClass:
	def __init__(self):
		pass

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

			if len(triplets) > 0:
				yield tf.convert_to_tensor(triplets)

	def create_triplet_loss_dataset(self):
		self.triplet_dataset = tf.data.Dataset.from_generator(self.get_triplet_examples_from_batch, tf.float32).repeat(-1)
		self.triplet_dataset = self.triplet_dataset.prefetch(tf.data.experimental.AUTOTUNE)

		return self.triplet_dataset


class DataEngineTypical(MainEngineClass):
	def make_label_map(self):
		self.label_map = {}

		for i, class_name in enumerate(tf.io.gfile.listdir(self.main_path)):
			self.label_map[class_name] = i

		self.reverse_label_map =  {v: k for k, v in self.label_map.items()}

	def path_yielder(self):
		for class_name in tf.io.gfile.listdir(self.main_path):
			for path_only in tf.io.gfile.listdir(self.main_path+class_name):
				yield (self.main_path+class_name+"/"+path_only, self.label_map[class_name])

	def image_loader(self, image):
		image = tf.io.read_file(image)
		image = tf.io.decode_jpeg(image, channels=3)
		image = tf.image.resize(image, (112, 112), method="nearest")
		image = tf.image.random_flip_left_right(image)

		return (tf.cast(image, tf.float32) - 127.5) / 128.

	def mapper(self, path, label):
		return (self.image_loader(path), label)

	def __init__(self, main_path: str, batch_size: int = 16, buffer_size: int = 10000, epochs: int = 1, reshuffle_each_iteration: bool = False, test_batch = 64):
		super(DataEngineTypical, self).__init__()
		self.main_path = main_path.rstrip("/") + "/"
		self.make_label_map()

		reshuffle_each_iteration = False
		print(f"reshuffle_each_iteration set to False to create a appropriate test set, this may cancelled if tf.data will fixed.")

		self.dataset = tf.data.Dataset.from_generator(self.path_yielder, (tf.string, tf.int64))
		if buffer_size > 0:
			self.dataset = self.dataset.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=42)

		self.dataset = self.dataset.map(self.mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).repeat(epochs)

		self.dataset_test = self.dataset.take(int(test_batch))
		self.dataset = self.dataset.skip(int(test_batch))


class DataEngineTFRecord(MainEngineClass):
	def image_loader(self, image_raw):
		image = tf.image.decode_jpeg(image_raw, channels=3)
		image = tf.image.random_flip_left_right(image)

		return (tf.cast(image, tf.float32) - 127.5) / 128.

	def mapper(self, tfrecord_data):
		features = {'image_raw': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.int64)}
		features = tf.io.parse_single_example(tfrecord_data, features)

		return self.image_loader(features['image_raw']),  tf.cast(features['label'], tf.int64)

	def __init__(self, tf_record_path: str, batch_size: int = 16, epochs: int = 10, buffer_size: int = 50000, reshuffle_each_iteration: bool = True,
	 test_batch = 64):
		super(DataEngineTFRecord, self).__init__()

		reshuffle_each_iteration = False
		print(f"reshuffle_each_iteration set to False to create a appropriate test set, this may cancelled if tf.data will fixed.")
		self.tf_record_path = tf_record_path

		self.dataset = tf.data.TFRecordDataset(self.tf_record_path)
		if buffer_size > 0:
			self.dataset = self.dataset.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=42)

		self.dataset = self.dataset.map(self.mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).repeat(epochs)

		self.dataset_test = self.dataset.take(int(test_batch))
		self.dataset = self.dataset.skip(int(test_batch))

if __name__ == '__main__':
	print("go check README.md")