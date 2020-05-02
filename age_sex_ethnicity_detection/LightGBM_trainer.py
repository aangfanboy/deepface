import numpy as np
import lightgbm as lgbm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class TrainModel:
    @staticmethod
    def test_model_with_data(lgbm_model, x_data, y_data):
        return accuracy_score(y_data, np.argmax(lgbm_model.predict(x_data), axis=-1))

    def __init__(self, x_data, y_data, model_path, number_of_classes: int = 2):
        self.model_path = model_path
        self.num_classes = number_of_classes
        self.x_data, self.y_data = x_data, y_data

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data, test_size=0.1, random_state=42)

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
            'num_iterations': 10,
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
    X_data, Y_data = np.load("features_numpy/x_data.npy"), np.load("features_numpy/y_data_eth.npy")
    num_classes = np.max(Y_data) + 1

    trainer = TrainModel(X_data, Y_data, "models_all/lgbm_age_model.txt", number_of_classes=num_classes)
    model = trainer.train_model(save=True)
    model = trainer.load_model()
    train_acc, test_acc, = trainer.test_model(model)
    print(f"Train Acc --> {train_acc}")
    print(f"Test Acc --> {test_acc}")
