import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from scipy.io import savemat, loadmat
from skimage.util import view_as_windows
from pathlib import Path
import numpy as np
from random import shuffle

def _sliding_window(I,M):
    '''
    Parses a 3D image into size [128, 128, 32] with no overlap
        Used for training images where boundary abnormalities are of no concern
    '''
    I = tf.extract_volume_patches(I, ksizes=[1,128,128,32,1],
                                     strides=[1,128,128,32,1],
                                     padding='VALID')
    I = tf.reshape(I, [-1,128,128,32,1])
    M = tf.extract_volume_patches(M, ksizes=[1,128,128,32,1],
                                     strides=[1,128,128,32,1],
                                     padding='VALID')
    M = tf.reshape(M, [-1,128,128,32,2])
    return I,M

def _sliding_window_overlap(I,M):
    '''
    Parses a 3D image into sizes [128,128,32] with a 50% overlap
        Used for evaluation images to remove boundary abnormalities
    '''
    I = tf.pad(I, ((0,0),(64,64),(64,64),(16,16),(0,0)))
    M = tf.pad(M, ((0,0),(64,64),(64,64),(16,16),(0,0)))

    I = tf.extract_volume_patches(I, ksizes=[1,128,128,32,1],
                                     strides=[1,64,64,16,1],
                                     padding='VALID')
    I = tf.reshape(I, [-1,128,128,32,1])

    M = tf.extract_volume_patches(M, ksizes=[1,128,128,32,1],
                                     strides=[1,64,64,16,1],
                                     padding='VALID')
    M = tf.reshape(M, [-1,128,128,32,2])
    return I,M

def mapping_function(weight_map, training=True):
    '''
    Function wrapper to return the model's input processing function based
    on the desired potential field to be used:
        weight_map: {'pot','unet'} describes which potential field to use for
            training
        training: {True, False} will use no-overlap or 50% ovelap, respectively

    Note: This input function was designed around using TFRecords with the
        following prototype:
            feature = {
            'image': _bytes_feature(I.tobytes('C')),
            'mask': _bytes_feature(M.tobytes('C')),
            'potential': _bytes_feature(P.tobytes('C')),
            'unet': _bytes_feature(U.tobytes('C')),
            'x': _int64_feature(x),
            'y': _int64_feature(y),
            'z': _int64_feature(z),
            }
    '''
    def _parse_image_function(example_proto):
        image_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
            'potential': tf.io.FixedLenFeature([], tf.string),
            'unet': tf.io.FixedLenFeature([], tf.string),
            'x': tf.io.FixedLenFeature([], tf.int64),
            'y': tf.io.FixedLenFeature([], tf.int64),
            'z': tf.io.FixedLenFeature([], tf.int64),
        }
        features = tf.io.parse_single_example(example_proto, image_feature_description)

        x,y,z = features['x'], features['y'], features['z']
        if not (x%128)==0:
            new_x = tf.cast(128-(x%128), tf.int32)
        else:
            new_x=0
        if not (y%128)==0:
            new_y = tf.cast(128-(y%128), tf.int32)
        else:
            new_y=0
        if not (z%32)==0:
            new_z = tf.cast(32-(z%32), tf.int32)
        else:
            new_z=0

        features['image'] = tf.io.decode_raw(features['image'], out_type=tf.float16)
        features['image'] = tf.reshape(features['image'], [x,y,z])
        features['image'] = tf.pad(features['image'], [(0,new_x),(0,new_y),(0,new_z)])

        features['mask'] = tf.io.decode_raw(features['mask'], out_type=tf.float16)
        features['mask'] = tf.reshape(features['mask'], [x,y,z])
        # features['mask'] = tf.cast(features['mask'],tf.float16)

        if weight_map=='unet':
            features['weights'] = tf.io.decode_raw(features['unet'], out_type=tf.float16)
        elif weight_map=='pot':
            features['weights'] = tf.io.decode_raw(features['potential'], out_type=tf.float16)
        features['weights'] = tf.reshape(features['weights'], [x,y,z])
        features['mask'] = tf.stack((features['mask'],features['weights']), axis=-1)
        features['mask'] = tf.pad(features['mask'], [(0,new_x),(0,new_y),(0,new_z),(0,0)])

        features['image'] = tf.expand_dims(features['image'], axis=0)
        features['mask'] = tf.expand_dims(features['mask'], axis=0)

        if training:
            I,M = _sliding_window(tf.expand_dims(features['image'], axis=-1), features['mask'])
            return I,M
        else:
            I,M = _sliding_window_overlap(tf.expand_dims(features['image'], axis=-1), features['mask'])
            return I,M
    return _parse_image_function

def find_samples( dir ):
    '''
    Return a unique list of sample povided in the given dir.
        Assumes specific naming convention:
        <cell_line>_<harvest>_<sampleID>_<augmentID>.tfrecord.gz
    '''
    records = [str.split(str(i.name),'.')[0] for i in dir.iterdir()]
    filenames = ['_'.join(str.split(i, '_')[:3]) for i in records]
    filenames = list(set(filenames))
    # Remove Poorly stained images
    rm_list = ['mcf7_day2_e','mcf7_day5_b','mcf7_day5_d',
               'mcf7_day7_a','mcf7_day7_d','mcf7_day7_f']
    for i in rm_list:
        filenames.remove(i)
    return list(set(filenames))

def training_split( dir, filenames ):
    '''
    Divides the examples into training/testing splits of 60/40, respectfully.
        (Split by 3D image)
    '''
    shuffle(filenames)
    N = len( filenames )
    training = filenames[:int(0.6*N)]
    testing = filenames[int(0.6*N):]
    train_list = list([])
    test_list = list([])
    for sample in training:
        for i in dir.glob(f'{sample}*.gz'):
            train_list.append( str(i) )
    for sample in testing:
        for i in dir.glob(f'{sample}.tfrecord.gz'):
            test_list.append( str(i) )
    return train_list, test_list

def filter_fcn(I,M):
    '''
    Boolean function to determine if a patch should be used for training.
        An empty ground truth volume will be discarded
    '''
    return tf.keras.backend.max(M[:,:,:,:,0])>0

def weighted_loss(alpha):
    '''
    Function wrapper that returns the cutom loss function.
        * The potential field is encoded in the mask
        * alpha is used as a scalar for the regularization term
        * returned function follows TF requirements (y_true, y_pred)
    '''
    def compute_loss(y_true, y_pred):
        M = tf.convert_to_tensor( y_true[:,:,:,:,0])
        W = tf.convert_to_tensor( y_true[:,:,:,:,1])
        y_pred = tf.convert_to_tensor( y_pred )
        y1 = sparse_categorical_crossentropy(M, y_pred)
        y2 = tf.math.multiply(W,y1)
        y1 = tf.reduce_mean( y1 )
        y2 = tf.reduce_mean( y2 )
        y = y1 + alpha*y2
        return y
    return compute_loss

def reconstruct( I,M,P ):
    '''
    Reconstructs evaluation examples for image-wise performance metrics.
        * from 50% overlap views of [128,128,32]
    Returns reformated Image, Mask, and Probabilities
    '''
    I = I[:,32:96,32:96,8:24,:]
    M = M[:,32:96,32:96,8:24,:]
    P = P[:,32:96,32:96,8:24,:]

    I = np.reshape(I, [5,5,-1,64,64,16,1])
    M = np.reshape(M, [5,5,-1,64,64,16,2])
    P = np.reshape(P, [5,5,-1,64,64,16,2])

    I_p = np.zeros([I.shape[0]*64, I.shape[1]*64, I.shape[2]*16,1])
    M_p = np.zeros([I.shape[0]*64, I.shape[1]*64, I.shape[2]*16,2])
    P_p = np.zeros([I.shape[0]*64, I.shape[1]*64, I.shape[2]*16,2])
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            for k in range(I.shape[2]):
                I_p[i*64:(i+1)*64,
                    j*64:(j+1)*64,
                    k*16:(k+1)*16,:] = I[i,j,k,:,:,:,:]
                M_p[i*64:(i+1)*64,
                    j*64:(j+1)*64,
                    k*16:(k+1)*16,:] = M[i,j,k,:,:,:,:]
                P_p[i*64:(i+1)*64,
                    j*64:(j+1)*64,
                    k*16:(k+1)*16,:] = P[i,j,k,:,:,:,:]
    return I_p,M_p,P_p

def save_sample(save_path, filename, I, M, P):
    '''
    Saves an image in matlab format with all components
    '''
    if not Path(save_path).is_dir():
        Path(save_path).mkdir(parents=True)
    savemat( save_path + '/' + filename + '.mat', {'P':P,'M':M, 'I':I} )
