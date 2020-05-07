import os
import sys

sys.path.append("../")
import tensorflow as tf

from UTKdata_engine import DataEngineTypical as DET
from face_recognition.model_scripts import main_model_architect as MMA
from face_recognition.model_scripts import tensorboard_helper as TBH


class MainModel(MMA.MainModel):
	@property
	def __name__(self):
		return "FromArcFace"

	@tf.function
	def test_step_reg(self, x, y):
		age_pred, sex_pred, eth_pred = self.model([x, y], training=False)
		loss = self.loss_function(y, age_pred, sex_pred, eth_pred)

		reg_loss = tf.add_n(self.model.losses)

		return age_pred, sex_pred, eth_pred, loss, reg_loss

	@tf.function
	def train_step_reg(self, x, y):
		with tf.GradientTape() as tape:
			age_pred, sex_pred, eth_pred = self.model([x, y], training=True)

			loss = self.loss_function(y, age_pred, sex_pred, eth_pred)
			reg_loss = tf.add_n(self.model.losses)

			loss_all = tf.add(loss, reg_loss)

		gradients = tape.gradient(loss_all, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		return age_pred, sex_pred, eth_pred, loss, reg_loss

	def ASE_loss_function(self, y_real, age_pred, sex_pred, eth_pred):
		age_real, sex_real, eth_real = tf.split(y_real, 3, axis=1)

		age_loss = self.scc_loss(age_real, age_pred)
		sex_loss = self.scc_loss(sex_real, sex_pred)
		eth_loss = self.scc_loss(eth_real, eth_pred)

		return tf.add_n([age_loss, sex_loss, eth_loss])

	def __init__(self):
		super(MainModel, self).__init__()
		self.scc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		# self.mse_loss = tf.keras.losses.MeanSquaredError()
		self.loss_function = self.ASE_loss_function

	def __call__(self, input_shape, weights: str = None, arcface_model_path: str = None,
	             ASE_model_path: str = None, num_classes: int = 10,
	             regularizer_l: float = 5e-4, use_arcface: bool = True, optimizer="ADAM", learning_rate: float = 0.1,
	             pooling_layer: tf.keras.layers.Layer = tf.keras.layers.GlobalAveragePooling2D):

		self.last_lr = learning_rate

		if optimizer == "ADAM":
			self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
			print("[*] ADAM chosen as optimizer")
		elif optimizer == "SGD":
			self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
			print("[*] SGD chosen as optimizer")
		elif optimizer == "MOMENTUM":
			self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
			# MomentumOptimizer is not recommended, it is from TF 1.x makes problem at learning rate change, i will update if TF 2.x version comes out
			print("[*] MomentumOptimizer chosen as optimizer")
		else:
			raise Exception(f"{optimizer} is not a valid name! Go with either ADAM, SGD or MOMENTUM")

		if ASE_model_path is not None and os.path.exists(ASE_model_path):
			self.model = tf.keras.models.load_model(ASE_model_path, custom_objects={"ArcFaceLayer": MMA.ArcFaceLayer})
			self.model.trainable = True
			self.change_regularizer_l(regularizer_l)

			self.model.summary()
			print(f"[*] ASE model loaded from {ASE_model_path}")

		elif arcface_model_path is not None and os.path.exists(arcface_model_path):
			self.model = tf.keras.models.load_model(arcface_model_path)
			self.model.trainable = True
			self.change_regularizer_l(regularizer_l)

			x1 = self.model.layers[-1].output
			x = MMA.BatchNormalization(momentum=0.9, scale=False, epsilon=2e-5, name="bn_ase_sub")(x1)

			label_input_layer = tf.keras.layers.Input((None,), dtype=tf.int64)
			if use_arcface:
				x_age = MMA.ArcFaceLayer(num_classes=24, arc_m=0.5, arc_s=10., regularizer_l=regularizer_l,
				                         name="ageArcFace")(x, label_input_layer)
				x_sex = MMA.ArcFaceLayer(num_classes=2, arc_m=0.5, arc_s=10., regularizer_l=regularizer_l,
				                         name="sexArcFace")(x, label_input_layer)
				x_eth = MMA.ArcFaceLayer(num_classes=5, arc_m=0.5, arc_s=10., regularizer_l=regularizer_l,
				                         name="ethArcFace")(x, label_input_layer)
			else:
				x_age = tf.keras.layers.Dense(24, activation=None, name="ageClassificationLayer",
				                              kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)
				x_sex = tf.keras.layers.Dense(2, activation=None, name="sexClassificationLayer",
				                              kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)
				x_eth = tf.keras.layers.Dense(5, activation=None, name="ethClassificationLayer",
				                              kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)

			self.model = tf.keras.models.Model([self.model.layers[0].input, label_input_layer], [x_age, x_sex, x_eth],
			                                   name=f"{self.__name__}-ASE")
			self.model.summary()

			print(f"[*] Model created from ArcFace Model that placed in {arcface_model_path}")

		else:
			label_input_layer = tf.keras.layers.Input((None,), dtype=tf.int64)
			self.model = self.get_model(input_shape=input_shape, weights=weights)
			self.model.trainable = True

			self.change_regularizer_l(regularizer_l)
			# ACCORDING TO ARCFACE PAPER
			x = pooling_layer()(self.model.layers[-1].output)
			x = MMA.BatchNormalization(momentum=0.9, epsilon=2e-5)(x)
			x = tf.keras.layers.Dropout(0.4)(x)
			x1 = tf.keras.layers.Dense(512, activation=None, name="features_without_bn", use_bias=True,
			                           kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)
			x = MMA.BatchNormalization(momentum=0.9, scale=False, epsilon=2e-5)(x1)

			if use_arcface:
				x_age = MMA.ArcFaceLayer(num_classes=24, arc_m=0.5, arc_s=64., regularizer_l=regularizer_l,
				                         name="ageArcFace")(x, label_input_layer)
				x_sex = MMA.ArcFaceLayer(num_classes=2, arc_m=0.5, arc_s=64., regularizer_l=regularizer_l,
				                         name="sexArcFace")(x, label_input_layer)
				x_eth = MMA.ArcFaceLayer(num_classes=5, arc_m=0.5, arc_s=64., regularizer_l=regularizer_l,
				                         name="ethArcFace")(x, label_input_layer)
			else:
				x_age = tf.keras.layers.Dense(24, activation=None, name="ageClassificationLayer",
				                              kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)
				x_sex = tf.keras.layers.Dense(2, activation=None, name="sexClassificationLayer",
				                              kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)
				x_eth = tf.keras.layers.Dense(5, activation=None, name="ethClassificationLayer",
				                              kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)

			self.model = tf.keras.models.Model([self.model.layers[0].input, label_input_layer], [x_age, x_sex, x_eth],
			                                   name=f"{self.__name__}-ASE")
			self.model.summary()

			print("[*] Model structure created from scratch")


class ResNet50(MainModel, MMA.ResNet50):
	pass


class Trainer:
	@staticmethod
	def get_wrong(y_real, y_pred):
		return tf.where(
			tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y_pred), -1), tf.cast(y_real, tf.int64)), tf.float32) == 00)

	@staticmethod
	def calculate_accuracy(y_real, age_pred, sex_pred, eth_pred):
		age_real, sex_real, eth_real = tf.split(y_real, 3, axis=1)

		acc_age = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(age_pred), axis=-1), tf.cast(age_real, tf.int64)), dtype=tf.float32))
		acc_sex = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(sex_pred), axis=-1), tf.cast(sex_real, tf.int64)), dtype=tf.float32))
		acc_eth = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(eth_pred), axis=-1), tf.cast(eth_real, tf.int64)), dtype=tf.float32))

		return acc_age, acc_sex, acc_eth

	def only_test(self, dataset_test=None):
		if dataset_test is None:
			if self.dataset_engine.dataset_test is None:
				raise Exception("there is no defined test dataset")

			dataset_test = self.dataset_engine.dataset_test

		acc_age_mean = tf.keras.metrics.Mean()
		acc_sex_mean = tf.keras.metrics.Mean()
		acc_eth_mean = tf.keras.metrics.Mean()
		loss_mean = tf.keras.metrics.Mean()

		for i, (x, y) in enumerate(dataset_test):
			age_pred, sex_pred, eth_pred, loss, reg_loss = self.model_engine.test_step_reg(x, y)
			acc_age, acc_sex, acc_eth = self.calculate_accuracy(y, age_pred, sex_pred, eth_pred)

			acc_age_mean(acc_age)
			acc_sex_mean(acc_sex)
			acc_eth_mean(acc_eth)
			loss_mean(loss)

			print(f"[*] Step {i}, Age Acc --> %{acc_age}  || Sex Acc --> %{acc_sex} || Eth Acc --> %{acc_eth}"
			      f" || Loss --> {loss} || Reg Loss --> {reg_loss}")

		print(f"\n\n[*] Age Acc Mean --> %{acc_age_mean.result().numpy()} || Sex Acc Mean --> %{acc_sex_mean.result().numpy()} || Eth Acc Mean --> %{acc_eth_mean.result().numpy()}"
		      f" || Loss Mean --> {loss_mean.result().numpy()}")

		return acc_age_mean, acc_sex_mean, acc_eth_mean, loss_mean

	def __init__(self, model_engine: MMA, dataset_engine: DET, tensorboard_engine: TBH, use_arcface: bool,
	             learning_rate: float = 0.01,
	             model_path: str = "classifier_model.tf", arcface_model_path: str = "arcface_final.h5",
	             pooling_layer: tf.keras.layers.Layer = tf.keras.layers.GlobalAveragePooling2D,
	             lr_step_dict: dict = None,
	             optimizer: str = "ADAM", regularizer_l: float = 5e-4):
		self.model_path = model_path
		self.model_engine = model_engine
		self.dataset_engine = dataset_engine
		self.tensorboard_engine = tensorboard_engine
		self.use_arcface = use_arcface
		self.lr_step_dict = lr_step_dict

		self.num_classes = 2  # 2 for deepfake
		tf.io.gfile.makedirs("/".join(self.model_path.split("/")[:-1]))

		self.tb_delete_if_exists = True
		if self.lr_step_dict is not None:
			print("[*] LEARNING RATE WILL BE CHECKED WHEN step\\alfa_divided_ten == 0")
			learning_rate = list(self.lr_step_dict.values())[0]

		self.model_engine(
			input_shape=(112, 112, 3),
			num_classes=self.num_classes,
			learning_rate=learning_rate,
			regularizer_l=regularizer_l,  # weight decay, train once with 5e-4 and then try something lower such 1e-5
			pooling_layer=pooling_layer,  # Recommended: GlobalAveragePooling
			use_arcface=self.use_arcface,  # set False if you want to train it as regular classification
			ASE_model_path=self.model_path,  # path of keras model(h5) , it is okay if doesn't exists
			arcface_model_path=arcface_model_path,  # path of arcface keras model(h5) , it is okay if doesn't exists
			optimizer=optimizer  # Recommended: SGD
		)

	def __call__(self, max_iteration: int = None, alfa_step=1000, qin: int = 10):
		if max_iteration is not None and max_iteration <= 0:
			max_iteration = None

		alfa_divided_ten = int(alfa_step / 10)
		alfa_multiplied_qin = int(alfa_step * qin)

		print(f"[*] Possible maximum step: {tf.data.experimental.cardinality(self.dataset_engine.dataset)}\n")

		acc_age_mean = tf.keras.metrics.Mean()
		acc_sex_mean = tf.keras.metrics.Mean()
		acc_eth_mean = tf.keras.metrics.Mean()
		loss_mean = tf.keras.metrics.Mean()

		self.tensorboard_engine.initialize(
			delete_if_exists=self.tb_delete_if_exists
		)
		print(f"[*] TensorBoard initialized on {self.tensorboard_engine.logdir}")

		for i, (x, y) in enumerate(self.dataset_engine.dataset):
			age_pred, sex_pred, eth_pred, loss, reg_loss = self.model_engine.train_step_reg(x, y)
			acc_age, acc_sex, acc_eth = self.calculate_accuracy(y, age_pred, sex_pred, eth_pred)

			acc_age_mean(acc_age)
			acc_sex_mean(acc_sex)
			acc_eth_mean(acc_eth)
			loss_mean(loss)

			self.tensorboard_engine({"loss": loss, "reg_loss": reg_loss,
			                         "age_acc": acc_age, "acc_sex": acc_sex, "acc_eth": acc_eth})

			if i % alfa_divided_ten == 0:
				if i % alfa_step == 0 and i > 10:
					self.model_engine.model.save(self.model_path)
					print(f"[{i}] Model saved to {self.model_path}")

				print(f"[{i}] Loss: {loss_mean.result().numpy()} || Reg Loss: {reg_loss.numpy()} ||"
				      f" Age Acc: %{acc_age_mean.result().numpy()} || Sex Acc: %{acc_sex_mean.result().numpy()} || Eth Acc: %{acc_eth_mean.result().numpy()} ||"
				      f" LR: {self.model_engine.optimizer.learning_rate.numpy()}")
				acc_age_mean.reset_states()
				acc_sex_mean.reset_states()
				acc_eth_mean.reset_states()
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
					print("[*] Calculating validation loss and accuracy, this may take some time")
					acc_age_mean.reset_states()
					acc_sex_mean.reset_states()
					acc_eth_mean.reset_states()
					loss_mean.reset_states()
					for x_test, y_test in self.dataset_engine.dataset_test:
						age_pred, sex_pred, eth_pred, loss, reg_loss = self.model_engine.test_step_reg(x_test, y_test)
						acc_age, acc_sex, acc_eth = self.calculate_accuracy(y_test, age_pred, sex_pred, eth_pred)

						self.tensorboard_engine({"val. loss": loss, "val. age_acc": acc_age, "val. acc_sex": acc_sex, "val. acc_eth": acc_eth})

						acc_age_mean(acc_age)
						acc_sex_mean(acc_sex)
						acc_eth_mean(acc_eth)
						loss_mean(loss)

					print(f"[{i}] Val. Loss --> {loss_mean.result().numpy()} || "
					      f"Val. Age Acc: %{acc_age_mean.result().numpy()} || Val. Sex Acc: %{acc_sex_mean.result().numpy()} || Val. Eth Acc: %{acc_eth_mean.result().numpy()} ||")
					acc_age_mean.reset_states()
					acc_sex_mean.reset_states()
					acc_eth_mean.reset_states()
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

	def save_final_model(self, path: str = "deepfake_final.h5", n: int = -1, sum_it: bool = True):
		m = tf.keras.models.Model(self.model_engine.model.layers[0].input, self.model_engine.model.layers[n].output)
		if sum_it:
			m.summary()

		m.save(path)
		print(f"[*] Final feature extractor saved to {path}")


if __name__ == '__main__':
	TDOM = DET(
		"../datasets/UTKFace/",  # tfrecord path
		batch_size=16,
		epochs=-1,  # set to "-1" so it can stream forever
		buffer_size=100000,
		reshuffle_each_iteration=True,  # set True if you set test_batch to 0
	)  # TDOM for "Tensorflow Dataset Object Manager"

	TBE = TBH.TensorBoardCallback(
		logdir="classifier_tensorboard"  # folder to write TensorBoard
	)  # TBE for "TensorBoard Engine"

	ME = ResNet50()  # ME for "Model Engine"

	k_value: float = 0.5
	trainer = Trainer(
		model_engine=ME,
		dataset_engine=TDOM,
		tensorboard_engine=TBE,
		use_arcface=False,  # set False if you want to train a normal classification model
		learning_rate=0.004,  # it doesn't matter if you set lr_step_dict to anything but None
		model_path="ASE_model.h5",  # it will save only weights, you can chose "h5" as extension too
		arcface_model_path="models_all/arcface_final.h5",
		optimizer="SGD",  # SGD, ADAM or MOMENTUM. MOMENTUM is not recommended
		lr_step_dict={
			int(60000 * k_value): 0.01,
			int(80000 * k_value): 0.001,
			int(100000 * k_value): 0.0005,
			int(140000 * k_value): 0.0001,
		},
		regularizer_l=5e-4  # "l" parameter for l2 regularizer
	)

	trainer(max_iteration=-1, alfa_step=5000, qin=2)
