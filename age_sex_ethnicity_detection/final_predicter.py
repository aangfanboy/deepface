import numpy as np
import lightgbm as lgbm
import tensorflow as tf


class TesterLGBM:
	def __init__(self, model_sex_path: str, model_age_path: str, model_ethnicity_path: str):
		self.model_sex = lgbm.Booster(model_file=model_sex_path)
		self.model_age = lgbm.Booster(model_file=model_age_path)
		self.model_ethnicity = lgbm.Booster(model_file=model_ethnicity_path)

	def predict_sex(self, outputs, round_pred: bool = False):
		pred = self.model_sex.predict(outputs)
		if round_pred:
			pred = np.argmax(pred, axis=-1).astype(int)

		return pred

	def predict_age(self, outputs, round_pred: bool = False):
		pred = self.model_age.predict(outputs)
		if round_pred:
			pred = np.argmax(pred, axis=-1).astype(int)

		return pred

	def predict_ethnicity(self, outputs, round_pred: bool = False):
		pred = self.model_ethnicity.predict(outputs)
		if round_pred:
			pred = np.argmax(pred, axis=-1).astype(int)

		return pred


class Tester:
	def __init__(self, model_sex_path: str, model_age_path: str, model_ethnicity_path: str):
		self.model_sex = tf.keras.models.load_model(model_sex_path)
		self.model_age = tf.keras.models.load_model(model_age_path)
		self.model_ethnicity = tf.keras.models.load_model(model_ethnicity_path)

	def predict_sex(self, face):
		pred = tf.argmax(tf.nn.softmax(self.model_sex(tf.reshape(face, (-1, 112, 112, 3)), training=False)), axis=-1).numpy()

		return pred

	def predict_age(self, face):
		pred = tf.argmax(tf.nn.softmax(self.model_age(tf.reshape(face, (-1, 112, 112, 3)), training=False)), axis=-1).numpy()+1

		return pred

	def predict_ethnicity(self, face):
		pred = tf.argmax(tf.nn.softmax(self.model_ethnicity(tf.reshape(face, (-1, 112, 112, 3)), training=False)), axis=-1).numpy()

		return pred

