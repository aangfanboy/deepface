import os
import tensorflow as tf

from data_manager import dataset_manager as DSM
from side_model_scripts import TensorBoardCallback as TBH
from side_model_scripts import ModelEngine as ME


class Trainer:
	@staticmethod
	def calculate_accuracy(y_real, y_pred):
		return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y_pred), axis=1), tf.cast(y_real, tf.int64)), dtype=tf.float32))

	def __init__(self, dataset_engine: DSM, tensorboard_engine: TBH,
	             learning_rate: float = 0.01,
	             model_path: str = "classifier_model.tf",
	             lr_step_dict: dict = None,
	             optimizer: str = "ADAM", fr_model_path: str = "arcface_final.h5", base_trainable: bool = True):
		self.model_path = model_path
		self.dataset_engine = dataset_engine
		self.tensorboard_engine = tensorboard_engine
		self.lr_step_dict = lr_step_dict

		self.num_classes = 2  # 2 for rf mode, 39090 for id mode
		tf.io.gfile.makedirs("/".join(self.model_path.split("/")[:-1]))

		self.tb_delete_if_exists = True

		if self.lr_step_dict is not None:
			print("[*] LEARNING RATE WILL BE CHECKED WHEN step\\alfa_divided_ten == 0")
			learning_rate = list(self.lr_step_dict.values())[0]

		self.model_engine = ME(optimizer=optimizer, learning_rate=learning_rate, num_classes=self.num_classes)
		if os.path.exists(self.model_path):
			self.model_engine.load_model_from_df(self.model_path)
			print(f"[*] Model loaded from deepfake classifier model at {self.model_path}")

		if not os.path.exists(self.model_path):
			if not os.path.exists(fr_model_path):
				raise Exception(f"You must have either DeepFake Classifier Model or Arcface Face Recognition Model. Training from scratch is not a smart idea.")
			self.model_engine.create_model_from_fr(fr_model_path=fr_model_path, base_trainable=base_trainable, bn_before_cl=False)
			print(f"[*] Model created from facial recognition model at {fr_model_path}")

	def __call__(self, max_iteration: int = None, alfa_step=1000, qin: int = 10):
		if max_iteration is not None and max_iteration <= 0:
			max_iteration = None

		alfa_divided_ten = int(alfa_step / 10)
		alfa_multiplied_qin = int(alfa_step * qin)

		print(f"[*] Possible maximum step: {tf.data.experimental.cardinality(self.dataset_engine.dataset)}\n")

		acc_mean = tf.keras.metrics.Mean()
		loss_mean = tf.keras.metrics.Mean()

		self.tensorboard_engine.initialize(
			delete_if_exists=self.tb_delete_if_exists
		)
		print(f"[*] TensorBoard initialized on {self.tensorboard_engine.logdir}")

		for i, (x, y) in enumerate(self.dataset_engine.dataset):
			logits, loss, reg_loss = self.model_engine.train_step_reg(x, y)
			print(logits)
			accuracy = self.calculate_accuracy(y, logits)
			acc_mean(accuracy)
			loss_mean(loss)

			self.tensorboard_engine({"loss": loss, "reg_loss": reg_loss, "accuracy": accuracy})

			if i % alfa_divided_ten == 0:
				if i % alfa_step == 0 and i > 10:
					self.model_engine.model.save(self.model_path)
					print(f"[{i}] Model saved to {self.model_path}")

				print(f"[{i}] Loss: {loss_mean.result().numpy()} || Reg Loss: {reg_loss.numpy()} || Accuracy: %{acc_mean.result().numpy()} || LR: {self.model_engine.optimizer.learning_rate.numpy()}")
				acc_mean.reset_states()
				loss_mean.reset_states()
				if self.lr_step_dict is not None:
					lower_found = False
					for key in self.lr_step_dict:
						if i < int(key):
							lower_found = True
							lr_should_be = self.lr_step_dict[key]
							if lr_should_be != self.model_engine.last_lr:
								self.model_engine.change_learning_rate_of_optimizer(lr_should_be)
								print(f"[{i}] Learning Rate set to --> {lr_should_be}")

							break

					if not lower_found:
						print(f"[{i}] Reached to given maximum steps in 'lr_step_dict'({list(self.lr_step_dict.keys())[-1]})")
						self.model_engine.model.save(self.model_path)
						print(f"[{i}] Model saved to {self.model_path}, end of training.")
						break

				if i % alfa_multiplied_qin == 0 and self.dataset_engine.dataset_test is not None and i > 10:
					for x_test, y_test in self.dataset_engine.dataset_test:
						logits, loss, reg_loss = self.model_engine.test_step_reg(x_test, y_test)
						accuracy = self.calculate_accuracy(y, logits)

						self.tensorboard_engine({"val. loss": loss, "val. accuracy": accuracy})

						acc_mean(accuracy)
						loss_mean(loss)

					print(f"[{i}] Val. Loss --> {loss_mean.result().numpy()} || Val. Accuracy --> %{acc_mean.result().numpy()}")
					acc_mean.reset_states()
					loss_mean.reset_states()

				if max_iteration is not None and i >= max_iteration:
					print(f"[{i}] Reached to given maximum iteration({max_iteration})")
					self.model_engine.model.save(self.model_path)
					print(f"[{i}] Model saved to {self.model_path}, end of training.")
					break

		if max_iteration is None:
			print(f"[*] Reached to end of dataset")
			self.model_engine.model.save(self.model_path)
			print(f"[*] Model saved to {self.model_path}, end of training.")


if __name__ == '__main__':
	TDOM = DSM.DataEngineTypical(
		main_path="../datasets/dataset_V4/",
		mode="id",
		batch_size=16,
		buffer_size=30000,
		epochs=-1,
		reshuffle_each_iteration=True,
		map_to=True
	)

	TBE = TBH(
		logdir="deepfake_classifier_tensorboard"
	)  # TBE for "TensorBoard Engine"

	k_value: float = 0.5
	trainer = Trainer(
		dataset_engine=TDOM,
		tensorboard_engine=TBE,
		learning_rate=0.004,
		model_path="DeepFakeModel.h5",
		optimizer="SGD",
		lr_step_dict={
			int(40000 * k_value): 0.004,
			int(60000 * k_value): 0.0005,
			int(80000 * k_value): 0.0003,
			int(120000 * k_value): 0.0001,
		},
		fr_model_path="arcface_final.h5",
		base_trainable=False
	)

	trainer(max_iteration=-1, alfa_step=5000, qin=2)
