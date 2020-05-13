import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
sys.path.append("../")

from tqdm import tqdm
from sklearn.decomposition import PCA
from face_recognition.data_manager.dataset_manager import DataEngineTFRecord as DET


class Engine:
	def __init__(self):
		self.TDOM = DET(
			"../datasets/fashion_mnist_data.tfrecords",
			batch_size=64,
			epochs=1,
			buffer_size=70000,
			reshuffle_each_iteration=True,
			test_batch=0,
			map_to=True
		)

		self.pca = PCA(n_components=2)

		self.model = tf.keras.models.load_model("arcface_final.h5")
		self.x_data, self.y_data = self.get_outputs()
		self.vis()

	def get_outputs(self):
		x_data, y_data = [], []
		for x, y in tqdm(self.TDOM.dataset):
			outputs = self.model(x, training=False)
			for (xx, yy) in zip(outputs, y):
				x_data.append(xx)
				y_data.append(yy)

		return tf.convert_to_tensor(x_data).numpy(), tf.convert_to_tensor(y_data).numpy()

	def vis(self):
		pc_all = self.pca.fit_transform(self.x_data)

		fig, ax = plt.subplots(figsize=(10, 10))
		fig.patch.set_facecolor('white')
		for l in np.unique(self.y_data)[:10]:
			ix = np.where(self.y_data == l)
			ax.scatter(pc_all[:, 0][ix], pc_all[:, 1][ix])

		plt.show()


if __name__ == '__main__':
	e = Engine()
