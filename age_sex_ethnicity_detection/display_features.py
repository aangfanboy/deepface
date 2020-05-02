import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA
from UTKdata_engine import DataEngineTypical as DET


class Engine:
	def __init__(self, model_path: str, data_engine: DET):
		self.data_engine = data_engine
		self.pca = PCA(n_components=2)  # compress 512-D data to 2-D, we need to do that if we want to display data.

		self.model = tf.keras.models.load_model(model_path)

	def get_outputs(self, n: int = -1, by: str = "sex", save: bool = False):
		x_data, y_data = [], []

		for i, (x, y) in tqdm(enumerate(self.data_engine.dataset), "getting outputs from UTK"):
			if i > n > 0:
				break

			age, sex, gender = tf.split(y, 3, axis=1)
			output = self.model(x, training=False)
			feature = gender
			if by == "age":
				feature = age
			elif by == "sex":
				feature = sex

			for out, label in zip(output, feature):
				x_data.append(out)
				y_data.append(int(label))

		if save:
			np.save("x_data.npy", tf.convert_to_tensor(x_data).numpy())
			np.save("y_data.npy", tf.convert_to_tensor(y_data).numpy())

		return x_data, y_data

	def get_outputs_all(self, n: int = -1, save: bool = False):
		x_data, y_data_age, y_data_sex, y_data_eth = [], [], [], []

		for i, (x, y) in tqdm(enumerate(self.data_engine.dataset), "getting outputs from UTK"):
			if i > n > 0:
				break

			age, sex, eth = tf.split(y, 3, axis=1)
			output = self.model(x, training=False)

			for out, label_age, label_sex, label_eth in zip(output, age, sex, eth):
				x_data.append(out)
				y_data_age.append(int(label_age))
				y_data_sex.append(int(label_sex))
				y_data_eth.append(int(label_eth))

		if save:
			np.save("features_numpy/x_data.npy", tf.convert_to_tensor(x_data).numpy())
			np.save("features_numpy/y_data_age.npy", tf.convert_to_tensor(y_data_age).numpy())
			np.save("features_numpy/y_data_sex.npy", tf.convert_to_tensor(y_data_sex).numpy())
			np.save("features_numpy/y_data_eth.npy", tf.convert_to_tensor(y_data_eth).numpy())

		return x_data, y_data_age, y_data_sex, y_data_eth

	def __call__(self, n: int = -1, by: str = "sex"):
		x_data, y_data = self.get_outputs(n=n, by=by)
		x_data, y_data = tf.convert_to_tensor(x_data).numpy(), tf.convert_to_tensor(y_data).numpy()

		pc_all = self.pca.fit_transform(x_data)

		fig, ax = plt.subplots(figsize=(10, 10))
		fig.patch.set_facecolor('white')
		for l in np.unique(y_data)[:10]:
			ix = np.where(y_data == l)
			ax.scatter(pc_all[:, 0][ix], pc_all[:, 1][ix])

		plt.show()


if __name__ == '__main__':
	UTK_DET = DET("../datasets/UTKFace/", buffer_size=0, test_batch=0)

	main_engine = Engine("models_all/arcface_final.h5", UTK_DET)
	main_engine.get_outputs_all(n=-1, save=True)
