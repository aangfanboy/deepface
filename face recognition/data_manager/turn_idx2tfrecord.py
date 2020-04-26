# I forgot where i took this code from, but it is not mine. If it is your code, please open an issue and i will fix.

import mxnet as mx
import tensorflow as tf


def mx2tfrecords(imgidx, imgrec, data_name: str = "faces_emore"):
    output_path = f"../datasets/{data_name}/tran.tfrecords"
    writer = tf.compat.v1.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()


def main(data_name: str = "faces_emore"):
    input("\nPRESS ENTER TO GO THROUGH THIS PROCESS")

    imgrec = mx.recordio.MXIndexedRecordIO(f"../datasets/{data_name}/train.idx", f"../datasets/{data_name}/train.rec", 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    print(header.label)
    imgidx = list(range(1, int(header.label[0])))

    mx2tfrecords(imgidx, imgrec)    


if __name__ == '__main__':
    print("go check README.md")
