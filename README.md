# UCLA_CS168_W18
Detecting Brain Lesions using 3D CNN Techniques

          
         
        
```
First of all, the image from the dataset is required to be preprocessed to fit the
both of the 3D CNN models. In regard to 3D unet, the main issue is to correct the 
bias before the training to prevent the supervising algorithm in the model from 
generalizing beyond the training set by using ANTs N4BiasFieldCorrection. For the 
DeepMedic model, the default configuration on scans is 200x200x200 and in order to 
fit the model, the format of the images should be NIFTI. Also, it is required that 
the number of voxel per dimension should be the same for each image. Due to 
DeepMedic’s characteristic, the background of each images should be set to zero 
for training. 
```
```
The first 3D CNN model we choose is referencing from the 3D unet. The model is build
from the keras library from python, which provides many useful class to construct 
the 3D unet model. The model is first applied with two types of levels of convolution 
blocks, the max pooling and up-convolution which both are the classes provided the 
keras library. Each levels is based on the depth and number of the base filter. Even 
though a deeper depth could ensure the model to be precise, it is important to choose 
a appropriate depth size and number of base filter for our model since the more of 
the depth or number of base filter, the more memory that the training will take 
since the 3D unet is appending the multiple filters for each layer based on the 
depth parameter and initial base of the filters, i.e. a inappropriate depth or 
number of base filter could lead exceeding memory allocation issues. For us, the 
parameter is set to be 32-base-filters and the depth to be 4. Finally, by applying
Conv3D from Keras, we build the final convolutional neural network for our dataset. 
```

```
The second 3D CNN  model we choose is referencing from the 3D CRF model with
application of the residue connection, specifically the DeepMedic model. The 
DeepMedic model is mainly constructed through the Theano library provided from
the Python. According to the paper, DeepMedic is a double pathway architecture 
for multi-scale processing. To implement, the feature map per layer is set to
be [30,40,40,50] and dimension of kernel per layer is set to be [[5,5,5],[5,5,5],[5,5,5]]. 
It is important to set the appropriate batch size while we create our model. 
Even though the large batch size allows us to process the training simultaneously 
on GPU which allows faster processing, such process would introduce a large 
amount of computation and requires extra amount of memory location which could 
potentially introduce crush of the program. After setting up the model’s 
configuration, we then begin to build our actual model by make the normal 
pathway of the CNN and then apply subsampled pathway of CNN which creates 
essentially a upsampling layer. Then the algorithm will be concatenated 
these two cnn pathway and make the final fully connected layer. 
```
```
reference:
DeepMedic source code: https://github.com/Kamnitsask/deepmedic
3DUnet source code: https://github.com/ellisdg/3DUnetCNN
```
