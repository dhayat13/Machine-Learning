# SKIN CANCER CLASSIFICATION DEEP LEARNING WITH CNN MODEL + Streamlit
Skin cancer is the most common human malignancy, is primarily diagnosed visually, beginning with an initial clinical screening and followed potentially by dermoscopic analysis, a biopsy and histopathological examination. Automated classification of skin lesions using images is a challenging task owing to the fine-grained variability in the appearance of skin lesions.

This the HAM10000 ("Human Against Machine with 10000 training images") dataset.It consists of 10015 dermatoscopicimages which are released as a training set for academic machine learning purposes and are publiclyavailable through the ISIC archive. This benchmark dataset can be used for machine learning and for comparisons with human experts.

It has 7 different classes of skin cancer which are listed below :
1. Melanocytic nevi
2. Melanoma
3. Benign keratosis-like lesions
4. Basal cell carcinoma
5. Actinic keratoses
6. Vascular lesions
7. Dermatofibroma

In this kernel I will try to detect 7 different classes of skin cancer using Convolution Neural Network with keras tensorflow in backend and then analyse the result to see how the model can be useful in practical scenario.
We will move step by step process to classify 7 classes of cancer.


# Step 1 : Install Kaggle Extension and Download Dataset
in this step i will install extension Kaggle on google colab because i need to download the dataset from Kaggle datasets.
```python
! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
Downloading skin-cancer-mnist-ham10000.zip to /content
100% 5.20G/5.20G [01:49<00:00, 20.7MB/s]
100% 5.20G/5.20G [01:49<00:00, 50.9MB/s]
! mkdir skin_cancer
! unzip skin-cancer-mnist-ham10000.zip -d skin_cancer
Streaming output truncated to the last 5000 lines.
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029326.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029327.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029328.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029329.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029330.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029331.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029332.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029333.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029334.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029335.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029336.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029337.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029338.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029339.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029340.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029341.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029342.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029343.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029344.jpg 
  
!rm -r /content/skin_cancer/ham10000_images_part_1
!rm -r /content/skin_cancer/ham10000_images_part_2
```


# Step 2 : Importing Essential Libraries
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
```


# Step 3 : Making Dictionary of images and labels
In this step I have made the image path dictionary by joining the folder path from base directory base_skin_dir and merge the images in jpg format from both the folders HAM10000_images_part1.zip and HAM10000_images_part2.zip
```python
dataset_dir = os.path.join('..', '/content/skin_cancer')

# Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(dataset_dir, '*', '*.jpg'))}

# This dictionary is useful for displaying more human-friendly labels later on

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
```


# Step 3 : Reading & Processing data
In this step we have read the csv by joining the path of image folder which is the base folder where all the images are placed named base_skin_dir. After that we made some new columns which is easily understood for later reference such as we have made column path which contains the image_id, cell_type which contains the short name of lesion type and at last we have made the categorical column cell_type_idx in which we have categorize the lesion type in to codes from 0 to 6

```python
skin_df = pd.read_csv(os.path.join(dataset_dir, 'HAM10000_metadata.csv'))

# Creating New Columns for better readability

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
# Now lets see the sample of tile_df to look on newly made columns
skin_df.head()
```


# Step 4 : Reading & Processing data
In this step we have read the csv by joining the path of image folder which is the base folder where all the images are placed named base_skin_dir. After that we made some new columns which is easily understood for later reference such as we have made column path which contains the image_id, cell_type which contains the short name of lesion type and at last we have made the categorical column cell_type_idx in which we have categorize the lesion type in to codes from 0 to 6

skin_df = pd.read_csv(os.path.join(dataset_dir, 'HAM10000_metadata.csv'))

- Creating New Columns for better readability
```python
skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
# Now lets see the sample of tile_df to look on newly made columns
skin_df.head()

lesion_id	image_id	dx	dx_type	age	sex	localization	path	cell_type	cell_type_idx
HAM_0000118	ISIC_0027419	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2
HAM_0000118	ISIC_0025030	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2
HAM_0002730	ISIC_0026769	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2
HAM_0002730	ISIC_0025661	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2
HAM_0001466	ISIC_0031633	bkl	histo	75.0	male	ear	/content/skin_cancer/HAM10000_images_part_2/IS...	Benign keratosis-like lesions	2
```


# Step 5 : Data Cleaning
In this step we check for Missing values and datatype of each field
```python
skin_df.isnull().sum()
lesion_id         0
image_id          0
dx                0
dx_type           0
age              57
sex               0
localization      0
path              0
cell_type         0
cell_type_idx     0
dtype: int64

skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
skin_df.isnull().sum()
lesion_id        0
image_id         0
dx               0
dx_type          0
age              0
sex              0
localization     0
path             0
cell_type        0
cell_type_idx    0
dtype: int64

skin_df.dtypes
lesion_id         object
image_id          object
dx                object
dx_type           object
age              float64
sex               object
localization      object
path              object
cell_type         object
cell_type_idx       int8
dtype: object
```


# Step 6 : EDA
In this we will explore different features of the dataset , their distrubtions and actual counts

Plot to see distribution of 7 different classes of cell type
```python
fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
```
![cell type plot](https://user-images.githubusercontent.com/72849717/96358211-dbe02600-112e-11eb-862a-c1e195939769.png)

Its seems from the above plot that in this dataset cell type Melanecytic nevi has very large number of instances in comparison to other cell types

Plotting of Technical Validation field (ground truth) which is dx_type to see the distribution of its 4 categories which are listed below :
1. Histopathology(Histo): Histopathologic diagnoses of excised lesions have been performed by specialized dermatopathologists.
2. Confocal: Reflectance confocal microscopy is an in-vivo imaging technique with a resolution at near-cellular level , and some facial benign with a grey-world assumption of all training-set images in Lab-color space before and after manual histogram changes.
3. Follow-up: If nevi monitored by digital dermatoscopy did not show any changes during 3 follow-up visits or 1.5 years biologists accepted this as evidence of biologic benignity. Only nevi, but no other benign diagnoses were labeled with this type of ground-truth because dermatologists usually do not monitor dermatofibromas, seborrheic keratoses, or vascular lesions.
4. Consensus: For typical benign cases without histopathology or followup biologists provide an expert-consensus rating of authors PT and HK. They applied the consensus label only if both authors independently gave the same unequivocal benign diagnosis. Lesions with this type of groundtruth were usually photographed for educational reasons and did not need further follow-up or biopsy for confirmation.

```python
skin_df['dx_type'].value_counts().plot(kind='bar')
```
![dx_type_plot](https://user-images.githubusercontent.com/72849717/96358227-0f22b500-112f-11eb-9901-1026b4df151c.png)

Plotting the distribution of localization field
```python
skin_df['localization'].value_counts().plot(kind='bar')
```
![localization_plot](https://user-images.githubusercontent.com/72849717/96359128-3b433380-1139-11eb-9171-83df8b68fea8.png)

It seems back , lower extremity,trunk and upper extremity are heavily compromised regions of skin cancer

Now, check the distribution of Age
```python
skin_df['age'].hist(bins=40)
```
![age_plot](https://user-images.githubusercontent.com/72849717/96359153-688fe180-1139-11eb-8b68-9432847ce9f6.png)

It seems that there are larger instances of patients having age from 30 to 60
Lets see the distribution of males and females
```python
skin_df['sex'].value_counts().plot(kind='bar')
```
![sex_plot](https://user-images.githubusercontent.com/72849717/96359163-7e9da200-1139-11eb-9f24-7b4552657b4b.png)


# Step 7: Loading and resizing of images
In this step images will be loaded into the column named image from the image path from the image folder. We also resize the images as the original dimension of images are 450 x 600 x3 which TensorFlow can't handle, so that's why we resize it into 100 x 75. As this step resize all the 10015 images dimensions into 100x 75 so be patient it will take some time.
```python
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))

skin_df.head()
lesion_id	image_id	dx	dx_type	age	sex	localization	path	cell_type	cell_type_idx	image
HAM_0000118	ISIC_0027419	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2	[[[190, 153, 194], [192, 154, 196], [191, 153,...
HAM_0000118	ISIC_0025030	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2	[[[23, 13, 22], [24, 14, 24], [25, 14, 28], [3...
HAM_0002730	ISIC_0026769	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2	[[[185, 127, 137], [189, 133, 147], [194, 136,...
HAM_0002730	ISIC_0025661	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2	[[[24, 11, 17], [26, 13, 22], [38, 21, 32], [5...
HAM_0001466	ISIC_0031633	bkl	histo	75.0	male	ear	/content/skin_cancer/HAM10000_images_part_2/IS...	Benign keratosis-like lesions	2	[[[134, 90, 113], [147, 102, 125], [159, 115, ...
```

We will show Sample images of each cancer type
```python
n_samples = 8
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)
```
![samples](https://user-images.githubusercontent.com/72849717/96359201-2ca94c00-113a-11eb-8d5c-cb322c745958.png)

```python
# Checking the image size distribution
skin_df['image'].map(lambda x: x.shape).value_counts()

(75, 100, 3)    10015
Name: image, dtype: int64
```


# Step 8 : Train Test Split
In this step we have splitted the dataset into training and testing set of 80:20 ratio

```python
features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df['cell_type_idx']
```
```python
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234)
```


# Step 9 : Normalization
I choosed to normalize the x_train, x_test by substracting from theor mean values and then dividing by thier standard deviation.
```python
x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
```


# Step 10 : Label Encoding

Labels are 7 different classes of skin cancer types from 0 to 6. We need to encode these lables to one hot vectors
```python
# Perform one-hot encoding on the labels
y_train = to_categorical(y_train_o, num_classes = 7)
y_test = to_categorical(y_test_o, num_classes = 7)
```


# Step 11 : Splitting training and validation split

I choosed to split the train set in two parts : a small fraction (20%) became the validation set which the model is evaluated and the rest (80%) is used to train the model.

```python
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.2, random_state = 2)
```
```python
# Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))
```
# Step 12 : Building CNN Model

I used the Keras Sequential API, where you have just to add one layer at a time, starting from the input.

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=(75, 100, 3)))
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.50))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.summary()
```
```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 75, 100, 32)       896       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 75, 100, 32)       9248      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 37, 50, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 37, 50, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 37, 50, 64)        18496     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 37, 50, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 18, 25, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 18, 25, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 18, 25, 128)       73856     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 18, 25, 128)       147584    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 9, 12, 128)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 9, 12, 128)        0         
_________________________________________________________________
flatten (Flatten)            (None, 13824)             0         
_________________________________________________________________
dense (Dense)                (None, 256)               3539200   
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 7)                 1799      
=================================================================
Total params: 3,828,007
Trainable params: 3,828,007
Non-trainable params: 0
_________________________________________________________________
```


# Step 13: Setting Optimizer and Annealer

Once our layers are added to the model, we need to set up a score function, a loss function and an optimisation algorithm. We define the loss function to measure how poorly our model performs on images with known labels. It is the error rate between the oberved labels and the predicted ones. We use a specific form for categorical classifications (>2 classes) called the "categorical_crossentropy". The most important function is the optimizer. This function will iteratively improve parameters (filters kernel values, weights and bias of neurons ...) in order to minimise the loss. I choosed Adam optimizer because it combines the advantages of two other extensions of stochastic gradient descent. Specifically:

Adaptive Gradient Algorithm (AdaGrad) that maintains a per-parameter learning rate that improves performance on problems with sparse gradients (e.g. natural language and computer vision problems).

Root Mean Square Propagation (RMSProp) that also maintains per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weight (e.g. how quickly it is changing). This means the algorithm does well on online and non-stationary problems (e.g. noisy).

Adam realizes the benefits of both AdaGrad and RMSProp.

Adam is a popular algorithm in the field of deep learning because it achieves good results fast.

The metric function "accuracy" is used is to evaluate the performance our model. This metric function is similar to the loss function, except that the results from the metric evaluation are not used when training the model (only for evaluation).

in this model i try to use SGD Optimizer too, but Adam have a better result than SGD in this case.
```python
#OPTIMIZER
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# optimizer = SGD(lr=0.001, momentum=0.9)
```
```python
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
```
```python
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
```


# Data Augmentation
It is the optional step. In order to avoid overfitting problem, we need to expand artificially our HAM 10000 dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations

Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.

By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.

```python
# AUGMENTATION

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
```
For the data augmentation, i choosed to : Randomly rotate some training images by 10 degrees Randomly Zoom by 10% some training images Randomly shift images horizontally by 10% of the width Randomly shift images vertically by 10% of the height Once our model is ready, we fit the training datase


# Step 14 : Showing Augmentation Result

In this step i try to show the result of data augmentation using 1 sample image
```python 
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
```
```python
def parameter(prm):
  img = load_img('/content/skin_cancer/HAM10000_images_part_1/ISIC_0024306.jpg')
  data = img_to_array(img)
  samples = expand_dims(data, 0)
  datagen2 = (prm)
  it = datagen2.flow(samples, batch_size=1)
  for i in range(9):
    pyplot.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    pyplot.imshow(image)
  pyplot.show()
param=[ImageDataGenerator(width_shift_range=[0.1]),ImageDataGenerator(height_shift_range=0.1),ImageDataGenerator(horizontal_flip=False),ImageDataGenerator(rotation_range=10),ImageDataGenerator(zoom_range=[0.1,0.5])]
name=['Random Horizontal Shift','Random Vertical Shift','Random horizontal Flip','Random Rotation Augmentation','Random Zoom Augmentation']

for i in range(len(name)):
  print(name[i])
  parameter(param[i])
```
Sample Images.

![ISIC_0024306](https://user-images.githubusercontent.com/72849717/96359493-a8f15e80-113d-11eb-8372-a4862257886c.jpg)


![aug1](https://user-images.githubusercontent.com/72849717/96359434-105ade80-113d-11eb-9645-93fd706210ab.png)
![aug2](https://user-images.githubusercontent.com/72849717/96359449-236dae80-113d-11eb-8f16-a0a5433499af.png)
![aug3](https://user-images.githubusercontent.com/72849717/96359453-2ff20700-113d-11eb-87c0-cf687c49a537.png)


# Step 15 : Set Checkpoint and Tensorboard

```python
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime
```
```python
# we save that checkpoint in skincancer.h5 file
filepath='skincancer.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
```
```python
# we save the log for tensordboard in logdir
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks_list.append(TensorBoard(logdir, histogram_freq=1))
```


# Step 16: Fitting the model
In this step finally I fit the model into x_train, y_train. In this step I have choosen batch size of 128 and 100 epochs
```python
# Fit the model
epochs = 100 
batch_size = 128
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_validate,y_validate),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=callbacks_list)
```
```python
WARNING:tensorflow:From <ipython-input-45-e896df7a953c>:7: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
Please use Model.fit, which supports generators.
Epoch 1/100
 1/50 [..............................] - ETA: 0s - loss: 1.8891 - accuracy: 0.2109WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
Instructions for updating:
use `tf.profiler.experimental.stop` instead.
 2/50 [>.............................] - ETA: 3s - loss: 3.3039 - accuracy: 0.4297WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0360s vs `on_train_batch_end` time: 0.1163s). Check your callbacks.
50/50 [==============================] - ETA: 0s - loss: 1.2106 - accuracy: 0.6610
Epoch 00001: val_accuracy improved from -inf to 0.66376, saving model to skincancer.h5
50/50 [==============================] - 13s 264ms/step - loss: 1.2106 - accuracy: 0.6610 - val_loss: 1.0693 - val_accuracy: 0.6638
Epoch 2/100
50/50 [==============================] - ETA: 0s - loss: 0.9749 - accuracy: 0.6680
Epoch 00002: val_accuracy improved from 0.66376 to 0.67561, saving model to skincancer.h5
50/50 [==============================] - 13s 254ms/step - loss: 0.9749 - accuracy: 0.6680 - val_loss: 0.9938 - val_accuracy: 0.6756
Epoch 3/100
50/50 [==============================] - ETA: 0s - loss: 0.9403 - accuracy: 0.6711
Epoch 00003: val_accuracy improved from 0.67561 to 0.68122, saving model to skincancer.h5
50/50 [==============================] - 13s 263ms/step - loss: 0.9403 - accuracy: 0.6711 - val_loss: 0.9174 - val_accuracy: 0.6812
Epoch 4/100
50/50 [==============================] - ETA: 0s - loss: 0.9094 - accuracy: 0.6725
Epoch 00004: val_accuracy improved from 0.68122 to 0.68372, saving model to skincancer.h5
50/50 [==============================] - 13s 260ms/step - loss: 0.9094 - accuracy: 0.6725 - val_loss: 0.8886 - val_accuracy: 0.6837
Epoch 5/100
50/50 [==============================] - ETA: 0s - loss: 0.8779 - accuracy: 0.6806
Epoch 00005: val_accuracy improved from 0.68372 to 0.68497, saving model to skincancer.h5

```
```python

50/50 [==============================] - 13s 257ms/step - loss: 0.4126 - accuracy: 0.8449 - val_loss: 0.6987 - val_accuracy: 0.7742
Epoch 96/100
50/50 [==============================] - ETA: 0s - loss: 0.4553 - accuracy: 0.8282
Epoch 00096: val_accuracy did not improve from 0.78665
50/50 [==============================] - 13s 262ms/step - loss: 0.4553 - accuracy: 0.8282 - val_loss: 0.7150 - val_accuracy: 0.7785
Epoch 97/100
50/50 [==============================] - ETA: 0s - loss: 0.4009 - accuracy: 0.8468
Epoch 00097: val_accuracy improved from 0.78665 to 0.79039, saving model to skincancer.h5
50/50 [==============================] - 13s 261ms/step - loss: 0.4009 - accuracy: 0.8468 - val_loss: 0.6902 - val_accuracy: 0.7904
Epoch 98/100
50/50 [==============================] - ETA: 0s - loss: 0.3806 - accuracy: 0.8569
Epoch 00098: val_accuracy did not improve from 0.79039
50/50 [==============================] - 13s 257ms/step - loss: 0.3806 - accuracy: 0.8569 - val_loss: 0.7072 - val_accuracy: 0.7817
Epoch 99/100
50/50 [==============================] - ETA: 0s - loss: 0.3676 - accuracy: 0.8548
Epoch 00099: val_accuracy did not improve from 0.79039
50/50 [==============================] - 13s 256ms/step - loss: 0.3676 - accuracy: 0.8548 - val_loss: 0.7167 - val_accuracy: 0.7835
Epoch 100/100
50/50 [==============================] - ETA: 0s - loss: 0.4539 - accuracy: 0.8281
Epoch 00100: val_accuracy did not improve from 0.79039
50/50 [==============================] - 13s 258ms/step - loss: 0.4539 - accuracy: 0.8281 - val_loss: 0.7221 - val_accuracy: 0.7792
```


# Step 17: Model Evaluation
In this step we will check the testing accuracy and validation accuracy of our model,plot confusion matrix and also check the missclassified images count of each type
```python
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
model.save("model.h5")
```
```python
63/63 [==============================] - 1s 8ms/step - loss: 0.7120 - accuracy: 0.7778
51/51 [==============================] - 0s 7ms/step - loss: 0.7221 - accuracy: 0.7792
Validation: accuracy = 0.779164  ;  loss_v = 0.722130
Test: accuracy = 0.777833  ;  loss = 0.711978
```
```python
plot_model_history(history)
```
![plot loss](https://user-images.githubusercontent.com/72849717/96359683-cb847700-113f-11eb-8677-0f82a9358c5a.png)


```python
# Function to plot confusion matrix    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(x_validate)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_validate,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

 

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(7)) 
```
![confs matrix](https://user-images.githubusercontent.com/72849717/96359712-11413f80-1140-11eb-8a41-5e125ac5799f.png)

Now, lets see which category has much incorrect predictions
```python
label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
plt.bar(np.arange(7),label_frac_error)
plt.xlabel('True Label')
plt.ylabel('Fraction classified incorrectly')
```
![fractal plot](https://user-images.githubusercontent.com/72849717/96359728-48175580-1140-11eb-8b80-a77c4dc77e3f.png)

# Tensorboard preview
```python
%load_ext tensorboard
%tensorboard --logdir logs
```
![tensorboard](https://user-images.githubusercontent.com/72849717/96359776-e0153f00-1140-11eb-994e-6b0c674b718b.png)

# Display Activation per Layer
```python
model_filename = "model.h5"

model.load_weights(model_filename)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

we made a function for showing that Display activation
```python
from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x_train[10].reshape(1,75,100,3))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap=None)
            activation_index += 1
```
```python
display_activation(activations, 4, 8, 0) # we use index 0 to showing the 1st layer and the size is 32 neuron so we use 4 x 8
```
![layer 1](https://user-images.githubusercontent.com/72849717/96359859-b0b30200-1141-11eb-8a55-b2fc9f770cb3.png)


```python
display_activation(activations, 8, 8, 4) # we use index 4 to showing the 5th layer and the size is 32 neuron so we use 8 x 8
```
![layer 5](https://user-images.githubusercontent.com/72849717/96359900-04bde680-1142-11eb-8e79-d6201d4f9e74.png)

# Conclusion
It seems our model has maximum number of incorrect predictions for Basal cell carcinoma which has code 3, then second most missclassified type is Vascular lesions code 5 then Melanocytic nevi code 0 where as Actinic keratoses code 4 has least misclassified type.

I think still this model is efficient in comparison to detection with human eyes having 79.039% accuracy

Demo on Streamlit [HERE!](http://ca4ea4c81531.ngrok.io)
