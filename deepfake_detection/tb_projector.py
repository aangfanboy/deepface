import csv
import sys
import tensorflow as tf
sys.path.append("../")

from tqdm import tqdm
from data_manager.dataset_manager import DataEngineTypical as DSM


class ProjectorEngine:
	@staticmethod
	def flip_batch(batch):
		return batch[:, :, ::-1, :]

	def __init__(self, data_engine):
		self.data_engine = data_engine
		self.model = tf.keras.models.load_model("models_all/arcface_final.h5")

		tf.io.gfile.mkdir("projector_tensorboard")

		self.data_engine.reverse_label_map = {v: k for k, v in self.data_engine.label_map.items()}

	def __call__(self, flip: bool = False):
		metadata_file = open('projector_tensorboard/metadata.tsv', 'w')
		metadata_file.write('Class\tName\n')
		with open("projector_tensorboard/feature_vecs.tsv", 'w') as fw:
			csv_writer = csv.writer(fw, delimiter='\t')

			for x, y in tqdm(self.data_engine.dataset):
				outputs = self.model(x, training=False)
				if flip:
					outputs += self.model(self.flip_batch(x), training=False)

				csv_writer.writerows(outputs.numpy())
				for label in y.numpy():
					name = self.data_engine.reverse_label_map[label]
					metadata_file.write(f'{label}\t{name}\n')

		metadata_file.close()


if __name__ == '__main__':
	TDOM = DSM(
		"../datasets/dataset_V4/",  # tfrecord path
		mode="id",
		batch_size=32,
		epochs=1,  # set to "-1" so it can stream forever
		buffer_size=0,
		reshuffle_each_iteration=False,  # set True if you set test_batch to 0
	)  # TDOM for "Tensorflow Dataset Object Manager"

	p_e = ProjectorEngine(
		data_engine=TDOM
	)

	p_e()
