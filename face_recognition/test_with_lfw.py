"""
This script was modified from https://github.com/peteryuX/arcface-tf2/blob/66d0afb124c74b6c1e0b79a464325472ede6fb45/modules/evaluations.py
"""
import os
import cv2
import bcolz
import numpy as np
import tensorflow as tf
import tqdm
from sklearn.model_selection import KFold


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_lfw_data(data_path):
    """get validation data"""
    _lfw, _lfw_issame = get_val_pair(data_path, 'lfw_align_112/lfw')

    return _lfw, _lfw_issame


def get_val_data(data_path):
    """get validation data"""
    _lfw, _lfw_issame = get_val_pair(data_path, 'lfw_align_112/lfw')
    _agedb_30, _agedb_30_issame = get_val_pair(data_path, 'AgeDB/agedb_30')
    _cfp_fp, _cfp_fp_issame = get_val_pair(data_path, 'cfp_align_112/cfp_fp')

    return _lfw, _agedb_30, _cfp_fp, _lfw_issame, _agedb_30_issame, _cfp_fp_issame


def hflip_batch(imgs):
    return imgs[:, :, ::-1, :]


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    _acc = float(tp + tn) / dist.size
    return tpr, fpr, _acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds,))
    best_thresholds = np.zeros((nrof_folds,))
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds,))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)

        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = \
                calculate_accuracy(threshold,
                                   dist[test_set],
                                   actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
        nrof_folds=nrof_folds)

    return tpr, fpr, accuracy, best_thresholds


def perform_val_arcface(embedding_size, batch_size, model,
                        carray, issame, nrof_folds=10, is_ccrop=False, is_flip=True):
    """perform val"""
    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm.tqdm(range(0, len(carray), batch_size)):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1])
        b, g, r = tf.split(batch, 3, axis=-1)
        batch = tf.concat([r, g, b], -1)
        if is_flip:
            flipped = hflip_batch(batch)
            emb_batch = model([batch, tf.ones((batch.shape[0],), dtype=tf.int64)], training=False)[-1] + model([flipped, tf.ones((batch.shape[0],), dtype=tf.int64)], training=False)[-1]
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            emb_batch = model([batch, tf.ones((batch.shape[0],), dtype=tf.int64)], training=False)[-1]
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)

    tpr, fpr, accuracy, best_thresholds = evaluate(
        embeddings, issame, nrof_folds)

    return accuracy.mean(), best_thresholds.mean()


def perform_val(embedding_size, batch_size, model,
                carray, issame, nrof_folds=10, is_ccrop=False, is_flip=True):
    """perform val"""
    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm.tqdm(range(0, len(carray), batch_size)):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1])
        b, g, r = tf.split(batch, 3, axis=-1)
        batch = tf.concat([r, g, b], -1)
        if is_flip:
            flipped = hflip_batch(batch)
            emb_batch = model(batch, training=False) + model(flipped, training=False)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            emb_batch = model(batch, training=False)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)

    tpr, fpr, accuracy, best_thresholds = evaluate(
        embeddings, issame, nrof_folds)

    return accuracy.mean(), best_thresholds.mean()


if __name__ == '__main__':
    lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = get_val_data("../datasets/")

    model_ai = tf.keras.models.load_model("arcface_final.h5")

    print("-----------------------------------")
    print("Testing on LFW...")
    acc, best_th = perform_val(512, 32, model_ai, lfw, lfw_issame, is_ccrop=False)
    print(f"Results on LFW, Accuracy --> {acc} || Best Threshold --> {best_th}")
    print("-----------------------------------")

    print("-----------------------------------")
    print("Testing on AgeDB 30...")
    acc, best_th = perform_val(512, 32, model_ai, agedb_30, agedb_30_issame, is_ccrop=False)
    print(f"Results on AgeDB 30, Accuracy --> {acc} || Best Threshold --> {best_th}")
    print("-----------------------------------")

    print("-----------------------------------")
    print("Testing on CFP...")
    acc, best_th = perform_val(512, 32, model_ai, cfp_fp, cfp_fp_issame, is_ccrop=True)
    print(f"Results on CFP, Accuracy --> {acc} || Best Threshold --> {best_th}")
    print("-----------------------------------")
