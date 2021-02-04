import tensorflow as tf
import argparse
import utils
import numpy as np
from scipy.io import savemat
import sys

def createArgParser():
    parser = argparse.ArgumentParser(prog='main.py',
                          description='make 3D predictions')
    parser.add_argument('-v','--verbose', action='store_true',
                          help='See Print Messages')
    parser.add_argument('-m','--model', type=str, default='model.h5',
                          help='which model to use')
    parser.add_argument('directory', type=str,
                        help='Directory of Images to make prediction')
    return parser.parse_args()

def main(args):
    # Settings
    VERBOSE = args.verbose
    DIR = args.directory
    MODEL = args.model

    # Load Model
    model = tf.keras.models.load_model( MODEL, compile=False )

    # Create Data Pipeline
    pipe = utils.DataPipe( DIR )
    assert len(pipe.files) > 0, "No tif files found"
    for name, DS in pipe.image_as_DS():

        if VERBOSE:
            print( f"Size of DS is currently: {sys.getsizeof(DS)}")

        # Reconstruct I
        initial=True
        for elem in DS:
            if initial:
                I = elem
                initial=False
            else:
                I = np.append(I, elem, axis=0)

        # Predict
        P = model.predict(DS)

        # Reconstruct
        P = utils.reconstruct(P,name)
        I = utils.reconstruct(I,name)

        if VERBOSE:
            print( f"Size of P is currently: {sys.getsizeof(P)}")
            print( f"Size of I is currently: {sys.getsizeof(I)}")

        # Save
        savemat( name.replace('tfrecord.gz','mat'), {'I':I, 'P':P})
        if VERBOSE:
            print( f"Saving: {name.replace('tfrecord.gz','mat')}" )

if __name__ == "__main__":
    args = createArgParser()

    main( args )
