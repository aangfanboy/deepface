import tensorflow as tf

from data_manager import dataset_manager as DSM
from model_scripts import tensorboard_helper as TBH
from model_scripts import main_model_architect as MMA


class Trainer:
	@staticmethod
	def calculate_accuracy(y_real, y_pred):
		return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y_pred), -1), y_real), tf.float32))

	def __init__(self, model_engine: MMA, dataset_engine: DSM, tensorboard_engine: TBH, learning_rate: float = 0.01, model_path: str = "triplet_model.tf",
		path_for_pretrained_arcface_wolast_model: str = "arcface_model.tf"):
		self.model_path = model_path
		self.model_engine = model_engine
		self.dataset_engine = dataset_engine
		self.tensorboard_engine = tensorboard_engine

		self.model_engine.loss_function = self.model_engine.triplet_loss

		self.model_engine(
			input_shape=(112, 112, 3),
			learning_rate=learning_rate,
			create_model=False
		)

		if tf.io.gfile.exists(self.model_path):
			self.model_engine.model = tf.keras.models.load_model(self.model_path)
		else:
			self.model_engine.model = tf.keras.models.load_model(path_for_pretrained_arcface_wolast_model)
			self.model_engine.model = tf.keras.models.Model([self.model_engine.model.layers[0].input, tf.keras.layers.Input((None, ))],
			 [self.model_engine.model.layers[-3].output, self.model_engine.model.layers[-3].output])

		self.tensorboard_engine.initialize(
			delete_if_exists=False
		)

	def __call__(self, max_iteration: int = None, alfa_step=1000):
		alfa_divided_ten = int(alfa_step/10)

		try:
			for i, x in enumerate(self.dataset_engine.triplet_dataset):
				x = tf.reshape(x, (-1, 112, 112, 3))
				logits, features, loss = self.model_engine.train_step(x, tf.convert_to_tensor(1.))

				self.tensorboard_engine({"loss": loss})

				if i % alfa_divided_ten == 0:
					if i % alfa_step == 0:
						self.model_engine.model.save(self.model_path)
						print(f"Model saved to {self.model_path}, step --> {i}")

					print(f"Step: {i} || Loss: {round(loss.numpy(), 4)}")

					if max_iteration is not None and i >= max_iteration:
						print(f"Reached to given maximum iteration({max_iteration}), {i} steps trained.")
						self.model_engine.model.save(self.model_path)
						print(f"Model saved to {self.model_path}, end of training.")
						break

		except KeyboardInterrupt:
			self.model_engine.model.save(self.model_path)
			print("model saved, quiting")



if __name__ == '__main__':
	TDOM = DSM.DataEngineTFRecord(
		"../datasets/faces_emore/tran.tfrecords", 
		batch_size = 64, 
		epochs = 1, 
		buffer_size = 20000,  
		reshuffle_each_iteration = False,
		test_batch=0
	)  # TDO for "Tensorflow Dataset Object Manager"
	TDOM.create_triplet_loss_dataset()

	TBE = TBH.TensorBoardCallback(
		logdir="triplet_tensorboard"
	)  # TBE for TensorBoard Engine

	ME = MMA.ResNet50()  # ME for "Model Engine"

	trainer = Trainer(
		model_engine=ME,
		dataset_engine=TDOM,
		tensorboard_engine=TBE,
		learning_rate=0.005,
		model_path="triplet_model.h5",
		path_for_pretrained_arcface_wolast_model="ResNet50LastClassifier.h5"
		)

	trainer(
		max_iteration=30000
	)