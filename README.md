# 3D-confocal-microscopy
Organization of organoid models imaged in 3D with confocal microscopy.

## Installation
### Virtual Environment
1) Clone this repository
2) Install Anaconda <https://docs.anaconda.com/anaconda/install/>
2) Clone the Tensorflow 2.0 GPU anaconda virtual environment with the following bash script

```
conda env create -f anaconda.yml
conda activate tf_gpu
```

### File Structure
* Project Folder
  * data
    * records -> TF Records of training data
  * src -> src folder for python and shell scripts
    * logs -> output folder to tensorboard logs
  * saved-models -> output folder for saved models
    * unet -> sub directory for models trained using unet weight map
    * pot -> sub directory for models trained using potential-field weight map

## Data
Training data can be found at <https://osf.io/g9xv8/> and contains:
* 3D Training images (matlab file)
  * Imaged with confocal microscopy
  * Image scale of 1 cubic-micron per voxel
* 3D Training masks (matlab file)
  * Binary Label Mask
  * Weight map used by loss function (unet definition)
  * Weight map used by loss function (potential-field definition)
  
### Weight Map
The weight map is used as a channel input to the model and encodes the pixel-wise weights to be used by the loss function.
This weight map is precomputed for two definition: (1) UNet; (2) Potential-field [1]

## Train the model
First, download the data and place the tfrecords into the 'records' subfolder under the 'data' directory.

Then, use the python script 'main.py' to train the model from scratch.  
Use of the script is demonstrated in 'train.sh' where hyperparameters are set to be: unet field loss function, batch=2, epochs=10, alpha=10.
These are passed to the script via the command line.
Hyperparameters and usage help can be found with:

```
python main.py -h
```

Pretrained models wegihts are also available at <https://osf.io/g9xv8/>.

### References
[1] Khoshdeli, M., Winkelmaier, G., & Parvin, B. (2019). Deep fusion of contextual and object-based representations for delineation of multiple nuclear phenotypes. Bioinformatics, 35(22), 4860-4861.
