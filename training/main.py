import tensorflow as tf
from utils import *
from pathlib import Path
import argparse
from unet import define_model
import datetime

# Tensorflow Warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"

def createArgParser():
    parser = argparse.ArgumentParser(prog='leave-one-out.py',
                        description='leave-one-out evaluation script.')
    parser.add_argument('-v','--verbose', action='store_true',
                         help='See print messages')
    parser.add_argument('-b','--batch', type=int, default=1,
                         help='batch size to use')
    parser.add_argument('-a','--alpha', type=int, default=10,
                         help='Weight Loss Coefficient')
    parser.add_argument('-e','--epoch', type=int, default=10,
                         help='Number of Training Epochs')
    parser.add_argument('weights', choices=['unet','pot'], type=str,
                         help='Pixel weight function')
    return parser.parse_args()

def main():
    # argparser
    args = createArgParser()

    # Settings
    VERBOSE = args.verbose
    WEIGHTS = args.weights
    BATCH_SIZE = args.batch
    ALPHA = args.alpha
    EPOCH = args.epoch

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Paths
    _root = Path.cwd().parent.parent
    _data_root = _root / 'data'
    _records_root = _data_root / 'records' # TFRecords Folder
    _tb_log = _root / 'src/logs/' / WEIGHTS / time_stamp # TensorBoard files
    _savedModel_root = _root / 'saved-models' / WEIGHTS
    _saved_model_dir = _savedModel_root / time_stamp

    # Get a list of Records
    samples_list = find_samples(_records_root)

    # Get train/test split
    (tr_list, te_list) = training_split(_records_root, samples_list)

    # Split Strategy
    strategy = tf.distribute.MirroredStrategy()  # Muli-GPU Strategy
    with strategy.scope():
        # Global Batch size
        global_batch_size = (BATCH_SIZE *
                             strategy.num_replicas_in_sync)
        if VERBOSE:
            print(f'\tBATCH_SIZE: {global_batch_size}')

        # Define DataSets
        _input_fcn = mapping_function(WEIGHTS)
        TR_DS = tf.data.TFRecordDataset(tr_list,
                compression_type="GZIP").map(_input_fcn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch() \
                .batch(global_batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE) \
                .filter(filter_fcn)
        TE_DS = tf.data.TFRecordDataset(te_list,
                compression_type="GZIP").map(_input_fcn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch() \
                .batch(global_batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE) \
                .filter(filter_fcn)

        if VERBOSE:
            for count, elem in enumerate(TR_DS):
                pass
            print(f"\tNumber of Training Batches Found: {count}" )
            for count, elem in enumerate(TE_DS):
                pass
            print(f"\tNumber of Testing Batches Found: {count}" )

        # Define a New Model
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss = weighted_loss(ALPHA)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=str(_tb_log),
                                                     write_graph=False,
                                                     profile_batch=0,
                                                     update_freq='batch')

        stop_callback = tf.keras.callbacks.EarlyStopping(min_delta=1e-3,
                                                        patience=0,
                                                        monitor='val_loss',
                                                        mode='min',
                                                        restore_best_weights=True)

        model = define_model()
        model.compile(optimizer=opt, loss=loss)

        class_weights = {'0':1, '1':5}

        # Train Model
        model.fit(TR_DS, validation_data=TE_DS, epochs=EPOCH, verbose=1, class_weight=class_weights,
                  use_multiprocessing=True, callbacks=[tb_callback, stop_callback])

        if not _saved_model_dir.is_dir():
            _saved_model_dir.mkdir(parents=True)

        # Save Model
        model.save( str(_saved_model_dir / 'model.h5') )
        with open( str(_saved_model_dir/'test_list.txt'), 'w') as fid:
            for i in te_list:
                fid.write(f"{i}\n")

if __name__ == "__main__":
    main()
