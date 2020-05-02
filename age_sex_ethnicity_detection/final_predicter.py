import numpy as np
import lightgbm as lgbm


class Tester:
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
