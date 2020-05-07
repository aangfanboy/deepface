import numpy as np
import lightgbm as lgbm
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from data_manager.dataset_manager import DataEngineTypical


class DataSideEngine4LGBM:
	def load_outputs(self):
		self.X_data, self.Y_data = np.load("features_numpy/X_data.npy"), np.load("features_numpy/Y_data.npy")
		self.X_data_test, self.Y_data_test = np.load("features_numpy/X_data_test.npy"), np.load("features_numpy/Y_data_test.npy")

	def save_outputs(self):
		np.save("features_numpy/X_data.npy", self.X_data)
		np.save("features_numpy/Y_data.npy", self.Y_data)
		np.save("features_numpy/X_data_test.npy", self.X_data_test)
		np.save("features_numpy/Y_data_test.npy", self.Y_data_test)

	def __init__(self):
		self.data_engine = DataEngineTypical("../datasets/dataset_V4/", mode="rf", batch_size=64, epochs=1)
		self.pca = PCA(n_components=2)  # compress 512-D data to 2-D, we need to do that if we want to display data.

		self.Y_data, self.Y_data_test = [], []
		self.X_data, self.X_data_test = [], []
		self.num_classes = 2

		try:
			self.load_outputs()
			print("[*] Outputs loaded from features_numpy folder")
		except FileNotFoundError:
			# self.keras_model = tf.keras.models.load_model("arcface_final.h5")
			self.keras_model = tf.keras.models.load_model("deepfake_final.h5")
			self()
			print("[*] Outputs predicted from arcface_final.h5 model and saved to features_numpy folder")

	def __call__(self):
		for xx, yy in tqdm(self.data_engine.dataset):
			outputs = self.keras_model(xx, training=False)
			for x, y in zip(outputs, yy):
				self.X_data.append(x)
				self.Y_data.append(y)

		for xx, yy in tqdm(self.data_engine.dataset_test):
			outputs = self.keras_model(xx, training=False)
			for x, y in zip(outputs, yy):
				self.X_data_test.append(x)
				self.Y_data_test.append(y)

		self.X_data, self.Y_data = tf.convert_to_tensor(self.X_data).numpy(), tf.convert_to_tensor(self.Y_data).numpy()
		self.X_data_test, self.Y_data_test = tf.convert_to_tensor(self.X_data_test).numpy(), tf.convert_to_tensor(self.Y_data_test).numpy()

		self.save_outputs()

	def display_features(self):
		x_data, y_data = np.concatenate([self.X_data, self.X_data_test]), np.concatenate([self.Y_data, self.Y_data_test])
		pc_all = self.pca.fit_transform(x_data)

		fig, ax = plt.subplots(figsize=(10, 10))
		fig.patch.set_facecolor('white')
		for label in tqdm(np.unique(y_data)):
			ix = np.where(y_data == label)
			ax.scatter(pc_all[:, 0][ix], pc_all[:, 1][ix])

		plt.show()


class TrainModel:
	@staticmethod
	def test_model_with_data(lgbm_model, x_data, y_data):
		return accuracy_score(y_data, np.argmax(lgbm_model.predict(x_data), axis=-1))

	def __init__(self, x_data, y_data, x_data_test, y_data_test, model_path, number_of_classes: int = 2):
		self.model_path = model_path
		self.num_classes = number_of_classes

		self.x_train, self.x_test, self.y_train, self.y_test = x_data, x_data_test, y_data, y_data_test

		self.lgbm_train = lgbm.Dataset(data=self.x_train, label=self.y_train)
		self.lgbm_test = lgbm.Dataset(data=self.x_test, label=self.y_test)

	def load_model(self):
		lgbm_model = lgbm.Booster(model_file=self.model_path)

		return lgbm_model

	def train_model(self, save: bool = True):
		params = {
			'task': 'train',
			'boosting_type': 'gbdt',
			'objective': 'multiclassova',
			'metric': 'multiclass',
			'num_leaves': 256,
			'learning_rate': 0.1,
			'num_class': self.num_classes,
			'num_iterations': 100,
			'tree_learner': 'feature',
		}

		lgbm_model = lgbm.train(params, self.lgbm_train)

		if save:
			lgbm_model.save_model(self.model_path)

		return lgbm_model

	def test_model(self, lgbm_model):
		y_train_pred = np.argmax(lgbm_model.predict(self.x_train), axis=-1)
		y_test_pred = np.argmax(lgbm_model.predict(self.x_test), axis=-1)

		train_score = accuracy_score(self.y_train, y_train_pred)
		test_score = accuracy_score(self.y_test, y_test_pred)

		return train_score, test_score


if __name__ == '__main__':
	DE = DataSideEngine4LGBM()
	DE.display_features()  # as you can see, LGBM is not the best method, but i will try it anyways

	trainer = TrainModel(DE.X_data, DE.Y_data, DE.X_data_test, DE.Y_data_test, "models_all/lgbm_model.txt", number_of_classes=DE.num_classes)
	model = trainer.train_model(save=True)
	model = trainer.load_model()
	train_acc, test_acc, = trainer.test_model(model)
	print(f"Train Acc --> {train_acc}")
	print(f"Test Acc --> {test_acc}")
