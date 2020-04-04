import tensorflow as tf

from data_manager import dataset_manager as DSM
from model_scripts import tensorboard_helper as TBH
from model_scripts import main_model_architect as MMA


class Trainer:
	@staticmethod
	def calculate_accuracy(y_real, y_pred):
		return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y_pred), -1), y_real), tf.float32))

	def only_test(self, dataset_test = None):
		if dataset_test is None:
			if self.dataset_engine.dataset_test is None:
				raise Exception("there is no defined test dataset")

			dataset_test = self.dataset_engine.dataset_test

		acc_mean = tf.keras.metrics.Mean()
		loss_mean = tf.keras.metrics.Mean()

		for i, (x, y) in enumerate(dataset_test):
			logits, features, loss, reg_loss = self.model_engine.test_step_reg(x, y)
			accuracy = self.calculate_accuracy(y, logits)

			acc_mean(accuracy)
			loss_mean(loss)

			print(f"Step {i}, Accuracy --> %{accuracy} || Loss --> {loss}")


		print(f"\n\n Accuracy Mean --> %{acc_mean.result().numpy()} || Loss Mean --> {loss_mean.result().numpy()}")

		return acc_mean, loss_mean


	def __init__(self, model_engine: MMA, dataset_engine: DSM, tensorboard_engine: TBH, use_arcface: bool, learning_rate: float = 0.01,
	 model_path: str = "classifier_model.tf"):
		self.model_path = model_path
		self.model_engine = model_engine
		self.dataset_engine = dataset_engine
		self.tensorboard_engine = tensorboard_engine

		self.tb_delete_if_exists = False

		if tf.io.gfile.exists(self.model_path):
			self.model_engine.model = tf.keras.models.load_model(self.model_path)

			self.model_engine(
				input_shape=(112, 112, 3),
				weights=None,
				num_classes=10575,  # 85742 for MS1MV2
				learning_rate=learning_rate,
				regularizer_l=5e-4,
				pooling_layer=tf.keras.layers.GlobalAveragePooling2D,
				create_model=False,
				use_arcface=use_arcface,
			)

		else:
			self.model_engine(
				input_shape=(112, 112, 3),
				weights=None,
				num_classes=10575,  # 85742 for MS1MV2
				learning_rate=learning_rate,
				regularizer_l=5e-4,
				pooling_layer=tf.keras.layers.GlobalAveragePooling2D,
				create_model=True,
				use_arcface=use_arcface,
			)

			self.tb_delete_if_exists = True

		self.model_engine.model.summary()

	def __call__(self, max_iteration: int = None, alfa_step=1000):
		alfa_divided_ten = int(alfa_step/10)
		alfa_multiplied_ten = int(alfa_step*10)

		print(f"Possible maximum step: {tf.data.experimental.cardinality(self.dataset_engine.dataset)}\n")

		acc_mean = tf.keras.metrics.Mean()
		loss_mean = tf.keras.metrics.Mean()

		self.tensorboard_engine.initialize(
						delete_if_exists=self.tb_delete_if_exists
					)
		print(f"TensorBoard initialized on {self.tensorboard_engine.logdir}")

		for i, (x, y) in enumerate(self.dataset_engine.dataset):
			logits, features, loss, reg_loss = self.model_engine.train_step_reg(x, y)
			accuracy = self.calculate_accuracy(y, logits)

			self.tensorboard_engine({"loss": loss, "reg_loss": reg_loss, "accuracy": accuracy})

			if i % alfa_divided_ten == 0:
				if i % alfa_step == 0:
					self.model_engine.model.save(self.model_path)
					print(f"Model saved to {self.model_path}, step --> {i}")

				print(f"Step: {i} || Loss: {round(loss.numpy(), 4)} || Reg Loss: {reg_loss.numpy()} || Accuracy: {accuracy.numpy()}")

				if i % alfa_multiplied_ten == 0:
					print(f"Testing on casia validation, this may take a while. Possible maximum step: {tf.data.experimental.cardinality(self.dataset_engine.dataset)}")
					for x, y in self.dataset_engine.dataset_test:
						logits, features, loss, reg_loss = self.model_engine.test_step_reg(x, y)
						accuracy = self.calculate_accuracy(y, logits)

						self.tensorboard_engine({"val. loss": loss, "val. reg_loss": reg_loss, "val. accuracy %": accuracy})

						acc_mean(accuracy)
						loss_mean(loss)

					print(f"Val. Loss --> {loss_mean.result().numpy()} || Val. Accuracy --> %{acc_mean.result().numpy()} || Val. Regression Loss: {reg_loss.numpy()}")
					acc_mean.reset_states()
					loss_mean.reset_states()

				if max_iteration is not None and i >= max_iteration:
					print(f"Reached to given maximum iteration({max_iteration}), {i} steps trained.")
					self.model_engine.model.save(self.model_path)
					print(f"Model saved to {self.model_path}, end of training.")
					break

		if max_iteration is None:
			print(f"Reached to end of dataset, {i} steps trained.")
			self.model_engine.model.save(self.model_path)
			print(f"Model saved to {self.model_path}, end of training.")


if __name__ == '__main__':
	TDOM = DSM.DataEngineTFRecord(
		"../datasets/faces_casia/tran.tfrecords", 
		batch_size = 64, 
		epochs = 1, 
		buffer_size = 20000,  
		reshuffle_each_iteration = False
	)  # TDO for "Tensorflow Dataset Object Manager"

	TBE = TBH.TensorBoardCallback(
		logdir="classifier_tensorboard"
	)  # TBE for TensorBoard Engine

	ME = MMA.ResNet50()  # ME for "Model Engine"

	trainer = Trainer(
		model_engine=ME,
		dataset_engine=TDOM,
		tensorboard_engine=TBE,
		use_arcface=False,
		learning_rate=0.001,
		model_path="classifier_model.h5",
		)

	trainer(
		max_iteration=None,
	)