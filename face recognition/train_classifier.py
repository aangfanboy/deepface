import tensorflow as tf

from data_manager import dataset_manager as DSM
from model_scripts import tensorboard_helper as TBH
from model_scripts import main_model_architect as MMA

from test_with_lfw import get_val_data, get_lfw_data, perform_val_arcface


class Trainer:
	@staticmethod
	def get_wrong(y_real, y_pred):
		return tf.where(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y_pred), -1), y_real), tf.float32) == 00)

	@staticmethod
	def calculate_accuracy(y_real, y_pred):
		return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y_pred), axis=1), y_real), dtype=tf.float32))

	def only_test(self, dataset_test=None, display_wrong_images: bool = False):
		if dataset_test is None:
			if self.dataset_engine.dataset_test is None:
				raise Exception("there is no defined test dataset")

			dataset_test = self.dataset_engine.dataset_test

		acc_mean = tf.keras.metrics.Mean()
		loss_mean = tf.keras.metrics.Mean()
		wrong_images = []

		for i, (x, y) in enumerate(dataset_test):
			logits, features, loss, reg_loss = self.model_engine.test_step_reg(x, y)
			accuracy = self.calculate_accuracy(y, logits)
			if accuracy < 1.0:
				images = x.numpy()[self.get_wrong(y, logits).numpy()][0]
				[wrong_images.append(image) for image in images]

			acc_mean(accuracy)
			loss_mean(loss)

			print(f"[*] Step {i}, Accuracy --> %{accuracy} || Loss --> {loss} || Reg Loss --> {reg_loss}")

		if display_wrong_images and len(wrong_images) > 0:
			self.tensorboard_engine.initialize(delete_if_exists=False)
			print(f"[*] TensorBoard initialized on {self.tensorboard_engine.logdir}")

			self.tensorboard_engine.add_images(f"wrong images from 'only_test' function", tf.convert_to_tensor(wrong_images), 0)
			print(f"[*] Wrong images({len(wrong_images)}) added to TensorBoard")

		print(f"\n\n[*] Accuracy Mean --> %{acc_mean.result().numpy()} || Loss Mean --> {loss_mean.result().numpy()}")

		return acc_mean, loss_mean, wrong_images

	def __init__(self, model_engine: MMA, dataset_engine: DSM, tensorboard_engine: TBH, use_arcface: bool,
	             learning_rate: float = 0.01,
	             model_path: str = "classifier_model.tf",
	             pooling_layer: tf.keras.layers.Layer = tf.keras.layers.GlobalAveragePooling2D,
	             lr_step_dict: dict = None,
	             optimizer: str = "ADAM", test_only_lfw: bool = True):
		self.model_path = model_path
		self.model_engine = model_engine
		self.dataset_engine = dataset_engine
		self.tensorboard_engine = tensorboard_engine
		self.use_arcface = use_arcface
		self.lr_step_dict = lr_step_dict

		self.num_classes = 85742  # 85742 for MS1MV2, 10575 for Casia, 105 for MINE
		tf.io.gfile.makedirs("/".join(self.model_path.split("/")[:-1]))

		self.tb_delete_if_exists = True
		if self.use_arcface:
			if not test_only_lfw:
				self.lfw, self.agedb_30, self.cfp_fp, self.lfw_issame, self.agedb_30_issame, self.cfp_fp_issame = get_val_data("../datasets/")
			else:
				self.lfw, self.lfw_issame = get_lfw_data("../datasets/")

		if self.lr_step_dict is not None:
			print("[*] LEARNING RATE WILL BE CHECKED WHEN step\\alfa_divided_ten == 0")
			learning_rate = list(self.lr_step_dict.values())[0]

		self.model_engine(
			input_shape=(112, 112, 3),
			weights=None,  # "imagenet" or None, not available for InceptionResNetV1
			num_classes=self.num_classes,  # 85742 for MS1MV2, 10575 for Casia, 105 for MINE
			learning_rate=learning_rate,
			regularizer_l=5e-4,  # weight decay, train once with 5e-4 and then try something lower such 1e-5
			pooling_layer=pooling_layer,  # Recommended: GlobalAveragePooling
			create_model=True,  # if you have a H5 file with config set this to zero and load model to self.model_engine.model
			use_arcface=self.use_arcface,  # set False if you want to train it as regular classification
			weight_path=self.model_path,  # paths of weights file(h5 or tf), it is okay if doesn't exists
			optimizer=optimizer  # Recommended: SGD
		)

	def test_on_val_data(self, is_ccrop: bool = False, step_i: int = 1, alfa_multiplied_ten: int = 1):
		step = int(alfa_multiplied_ten / step_i)

		print("-----------------------------------")
		acc_lfw, best_th = perform_val_arcface(512, 16, self.model_engine.model, self.lfw, self.lfw_issame, is_ccrop=is_ccrop)
		print(f"[*] Results on LFW, Accuracy --> {acc_lfw} || Best Threshold --> {best_th}")
		print("-----------------------------------")
		self.tensorboard_engine.add_with_step({"LFW": acc_lfw}, step=step)

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
			logits, features, loss, reg_loss = self.model_engine.train_step_reg(x, y)
			accuracy = self.calculate_accuracy(y, logits)
			acc_mean(accuracy)
			loss_mean(loss)

			self.tensorboard_engine({"loss": loss, "reg_loss": reg_loss, "accuracy": accuracy})

			if i % alfa_divided_ten == 0:
				if i % alfa_step == 0 and i > 10:
					self.model_engine.model.save_weights(self.model_path)
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
						self.model_engine.model.save_weights(self.model_path)
						print(f"[{i}] Model saved to {self.model_path}, end of training.")
						break

				if i % alfa_multiplied_qin == 0 and self.dataset_engine.dataset_test is not None and i > 10:
					for x_test, y_test in self.dataset_engine.dataset_test:
						logits, features, loss, reg_loss = self.model_engine.test_step_reg(x_test, y_test)
						accuracy = self.calculate_accuracy(y, logits)

						self.tensorboard_engine({"val. loss": loss, "val. accuracy": accuracy})

						acc_mean(accuracy)
						loss_mean(loss)

					print(f"[{i}] Val. Loss --> {loss_mean.result().numpy()} || Val. Accuracy --> %{acc_mean.result().numpy()}")
					acc_mean.reset_states()
					loss_mean.reset_states()

				if i % alfa_multiplied_qin == 0 and self.use_arcface and i > 10:
					self.test_on_val_data(False, i, alfa_multiplied_qin)

				if max_iteration is not None and i >= max_iteration:
					print(f"[{i}] Reached to given maximum iteration({max_iteration})")
					self.model_engine.model.save_weights(self.model_path)
					print(f"[{i}] Model saved to {self.model_path}, end of training.")
					break

		if max_iteration is None:
			print(f"[*] Reached to end of dataset")
			self.model_engine.model.save_weights(self.model_path)
			print(f"[*] Model saved to {self.model_path}, end of training.")

	def save_final_model(self, path: str = "arcface_final.h5", n: int = -4):
		m = tf.keras.models.Model(self.model_engine.model.layers[0].input, self.model_engine.model.layers[n].output)
		m.summary()

		m.save(path)
		print(f"[*] Final feature extractor saved to {path}")


if __name__ == '__main__':
	TDOM = DSM.DataEngineTFRecord(
		"../datasets/faces_emore/tran.tfrecords",
		batch_size=16,
		epochs=-1,
		buffer_size=30000,
		reshuffle_each_iteration=True,
		test_batch=0
	)  # TDOM for "Tensorflow Dataset Object Manager"

	TBE = TBH.TensorBoardCallback(
		logdir="classifier_tensorboard"
	)  # TBE for "TensorBoard Engine"

	ME = MMA.InceptionResNetV1()  # ME for "Model Engine"

	k_value: float = 4.  # recommended --> (512 / TDOM.batch_size)
	trainer = Trainer(
		model_engine=ME,
		dataset_engine=TDOM,
		tensorboard_engine=TBE,
		use_arcface=True,
		learning_rate=0.004,
		model_path="ArcFaceModel/model.tf",
		optimizer="SGD",
		lr_step_dict={
			int(60000 * k_value): 0.004,
			int(80000 * k_value): 0.0005,
			int(100000 * k_value): 0.0003,
			int(120000 * k_value): 0.0001,
		}
	)

	trainer(max_iteration=-1, alfa_step=5000, qin=2)
	trainer.save_final_model(path="arcface_final.h5")
