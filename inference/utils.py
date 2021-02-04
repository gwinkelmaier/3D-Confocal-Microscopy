import tensorflow as tf
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
import numpy as np


def get_filenames( directory ):
    return [str(i) for i in Path( directory ).glob('*.tif')]

# TF Records
class TFRecordWriter:
    '''
    Custom TFRecord Writer
    Initialized with save directory, overwrite boolean and input data type
    create_record function converts all file in dir to TFRecord and returns a
    list of record names
    '''
    def __init__(self, save_dir, overwrite=True, type='tif'):
        self.save_dir = Path(save_dir)
        self.overwrite = overwrite
        self.data_type = type

    def create_record(self, file):
        self.save_name = Path(file).name.replace(self.data_type, 'tfrecord.gz')
        if not self.overwrite and (self.save_dir / self.save_name).is_file():
            return str(self.save_dir/self.save_name)

        self.image = np.transpose(imread(file), [2,3,0,1])[:,:,:,1]
        self.image = np.subtract(self.image, np.min(self.image))
        self.image = np.divide(self.image, np.max(self.image))
        # self.image = self.image[::4, ::4, :] # isomorphic: 0.25x0.25x1 microns -> 1x1x1 microns
        self.image = resize(self.image,
                           (self.image.shape[0]//2, self.image.shape[1]//2, self.image.shape[2]*2),
                           preserve_range=True) # isomorphic: 0.25x0.25x1 microns -> 1x1x1 microns
        self.image = self.image.astype(np.float16)
        self._serialize_example()
        self.save()

        return str(self.save_dir/self.save_name)

    def _bytes_feature(self, value):
      """Returns a bytes_list from a string / byte."""
      if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _serialize_example(self):
        x,y,z = self.image.shape
        feature = {
            'image': self._bytes_feature(self.image.tobytes('C')),
            'x': self._int64_feature(x),
            'y': self._int64_feature(y),
            'z': self._int64_feature(z),
        }
        self.example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return 0

    def save(self):
        name = str(self.save_dir / self.save_name)
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(name, options=options) as writer:
            writer.write(self.example_proto.SerializeToString())
        return 0

class DataPipe:
    '''
    Custom data_pipe to all files of a dir into TFRecords and returns the record
    list for creating a tf.data.Dataset
    Initialized with the directory to pull data from and the data type
    '''
    def __init__(self, directory, data_type='tif'):
        self.dir = directory
        self.type = data_type
        self.files = [str(i) for i in Path(self.dir).iterdir() if self.type in str(i)]
        self.batch_size = 2

    def image_as_DS(self):
        self.make_records()
        for record in self.records:
            yield (record,
                   tf.data.TFRecordDataset( record,
                    compression_type="GZIP").map( self._input_fcn,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch() \
                    .batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
                  )

    def make_records(self):
        TF = TFRecordWriter(self.dir, overwrite=True, type=self.type)
        self.records = []
        for file in self.files:
            self.records.append( TF.create_record(file) )
        return 0

    def _input_fcn(self, example_proto):
        image_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
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
        features['image'] = tf.expand_dims(features['image'], axis=0)

        return self._sliding_window(tf.expand_dims(features['image'], axis=-1))

    def _sliding_window(self, I):
        I = tf.pad(I, ((0,0),(64,64),(64,64),(16,16),(0,0)))

        I = tf.extract_volume_patches(I, ksizes=[1,128,128,32,1],
                                         strides=[1,64,64,16,1],
                                         padding='VALID')
        return tf.reshape(I, [-1,128,128,32,1])


def reconstruct( P, name ):
    I = imread(name.replace('tfrecord.gz','tif'))
    I = np.transpose(I, [2,3,0,1] )
    I = resize(I,
               (I.shape[0]//2, I.shape[1]//2, I.shape[2]*2),
               preserve_range=True) # isomorphic: 0.25x0.25x1 microns -> 1x1x1 microns
    x,y,z = I.shape[:3]
    if not (x%128)==0:
        add_x = 128-(x%128)
    else:
        add_x=0
    if not (y%128)==0:
        add_y = 128-(y%128)
    else:
        add_y=0
    if not (z%32)==0:
        add_z = 32-(z%32)
    else:
        add_z=0
    x = (x+add_x+128)//128
    x+=(x-1)
    y = (y+add_y+128)//128
    y+=(y-1)
    z = (z+add_z+32)//32
    z+=(z-1)

    P = P[:,32:96,32:96,8:24,:]

    try:
        P = np.reshape(P, [x,y,z,64,64,16,2])
        P_p = np.zeros([x*64, y*64, z*16, 2])
    except:
        P = np.reshape(P, [x,y,z,64,64,16,1])
        P_p = np.zeros([x*64, y*64, z*16,1])

    for i in range(x):
        for j in range(y):
            for k in range(z):
                P_p[i*64:(i+1)*64,
                    j*64:(j+1)*64,
                    k*16:(k+1)*16,:] = P[i,j,k,:,:,:,:]
    P_p = P_p[32:I.shape[0]+32, 32:I.shape[1]+32, 8:I.shape[2]+8, :]
    return P_p
