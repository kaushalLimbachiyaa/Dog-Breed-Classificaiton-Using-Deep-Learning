### Table of Contents

1. [Project Overview](#projectOverview)
2. [Libraries](#library)
3. [Metrics](#metrics)
4. [Instructions](#instructions)
5. [File Descriptions](#files)
6. [Results and Screenshots](#results)

[//]: # (Image References)

[image1]: ./images/result3dog.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
[image4]: ./images/result1.png "results1"
[image5]: ./images/resultbritt.png "results2"


## Project Overview <a name="projectOverview"></a>

In this project, we will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, our algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  [Here](https://medium.com/@kaushal370/dog-breed-classification-using-deep-learning-9baf34f90f19) is the link to my blog.

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!

## Library <a name="library"></a>

Follwing are the nesesary libraries to run the code.



opencv-python==3.2.0.6

h5py==2.6.0

matplotlib==2.0.0

numpy==1.12.0

scipy==0.18.1

tqdm==4.11.2

keras==2.0.2

scikit-learn==0.18.1

pillow==4.0.0

ipykernel==4.6.1

tensorflow==1.0.0     

### Metrics <a name="metrics"></a>

Because we’re dealing with a multi-classification problem and the data is slightly skewed, the categorical cross-entropy cost function and the accuracy are used as assessment metric. However, the labels must first be in a category format. The target files are a collection of encoded dog labels that are associated with an image in this format. This multi-class log loss penalises the classifier if the projected probability results in a label that differs from the actual label, resulting in higher accuracy. The loss of a perfect classifier is zero, and the accuracy is 100 percent.

## Project Instructions

### Instructions <a name="instructions"></a>

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/kaushalLimbachiyaa/Dog-Breed-Classificaiton-Using-Deep-Learning.git
cd Dog-Breed-Classificaiton-Using-Deep-Learning
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [Resnet50 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

12. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. __Unless requested, do not modify code that has already been included.__


## File Descriptions <a name="files"></a>

When you are ready to submit your project, collect the following files and compress them into a single archive for upload:
- The `dog_app.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- The `extract_bottleneck_features.py` contains functions to process the pretrained models.
- The `haarcascades` folder contains the filters we will use in the human face detector.
- The `images` folder contains images to test the model. 
- The `Test Algorithm` folder contains the images to test the algorithm on.
- The `requirements` contains the necessary dependencies
- The `saved_models` folder contains trained models with best accuracy.



## Results and Screenshots <a name="results"></a>

![results1][image4]


![results2][image5]


[Here](https://medium.com/@kaushal370/dog-breed-classification-using-deep-learning-9baf34f90f19) is my blog for detailed technical analysis.

