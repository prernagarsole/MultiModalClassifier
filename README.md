# Developing an AI application

AI algorithms will be used in an increasing number of commonplace applications. For instance, you might want to incorporate an image classifier into a mobile application. As part of the overall application architecture, you would use a deep learning model that had been trained on hundreds of thousands of images to accomplish this. The use of these kinds of models as standard components of applications will make up a sizable portion of software development in the future.

# Options used :

# TensorFlowLite Inference
<a href="https://colab.research.google.com/drive/1-eVXttddEBeWV0DnJw-XtdOXcA5Lg3BT?usp=share_link" target="_blank">Colab Link</a>

# Working with REST API using TensorFlow
<a href="https://colab.research.google.com/drive/11jRV6DGo3EPlcMM3766Jg5G_gksTESi7?usp=share_link" target="_blank">Colab Link</a>

# IR model in OpenVINO
<a href="https://colab.research.google.com/drive/1VHRkn42CN_hJd9eY4B2UPvGeOzBJ4-99#scrollTo=UzFYM3i4XfUH" target="_blank">Colab Link</a>

# Neural network with tf.keras.layers.Dropout--
<a href="https://colab.research.google.com/drive/1Sapgm791W9aaHrYEk8gjXAwjHmdBfN0r?usp=share_link" target="_blank">Colab Link</a>

# Autotuning--
<a href="https://colab.research.google.com/drive/1Sapgm791W9aaHrYEk8gjXAwjHmdBfN0r?usp=share_link" target="_blank">Colab Link</a>

# Training and Performance Chart for Fashion Dataset
<img width="806" alt="Screen Shot 2022-11-21 at 6 43 23 PM" src="https://user-images.githubusercontent.com/99928364/203208187-78ad8f26-e512-4a4d-8490-4949e2619a80.png">
<img width="614" alt="Screen Shot 2022-11-21 at 9 13 20 PM" src="https://user-images.githubusercontent.com/99928364/203227331-5a9f3508-c72b-4b61-a7d8-c3139dda4e34.png">

# Training and Performance Chart for Flowers Dataset
<img width="518" alt="Screen Shot 2022-11-21 at 6 45 28 PM" src="https://user-images.githubusercontent.com/99928364/203208476-f565f704-4530-4fee-858e-be662f5d7f06.png">

# Building and training the classifier

1. A trained network should be loaded (If you need a starting point, the VGG networks work great and are straightforward to use)

2. Using ReLU activations and dropout, define a new, untrained feed-forward network as a classifier.

3. Backpropagation is used to train the classifier layers using the previously trained network to obtain the features.

4. For the purpose of choosing the ideal hyperparameters, monitor the loss and accuracy on the validation set.

Colab link

<a href="https://colab.research.google.com/drive/1-xUt6xIlrlXxkgOCiYFYK7rkfKFiCjzA?usp=share_link" target="_blank">Colab Link</a>

<img width="1154" alt="Screen Shot 2022-11-21 at 9 12 25 PM" src="https://user-images.githubusercontent.com/99928364/203227220-fba9da7a-5878-4a00-b4bc-7f2fa8096a00.png">
<img width="736" alt="Screen Shot 2022-11-21 at 8 56 46 PM" src="https://user-images.githubusercontent.com/99928364/203225227-0882f532-1be7-4501-98d6-9e2ab03d7b5a.png">


# REST API

Loading image from JSON (Fashion dataset MNIST)

1. Created a new model parameter with the name create simplemodelTest
2. In CNNSimpleModels.py and trained the classification model using myTFDistributedTrainer.py. This would create an output folder inside output/fashion/1. This model is combined with our API for predictions

<img width="271" alt="Screen Shot 2022-11-21 at 8 58 49 PM" src="https://user-images.githubusercontent.com/99928364/203225514-78a2904f-21d6-4713-8a2f-1b5304943172.png">


Prediction of image from JSON (Fashion dataset MNIST)

<img width="782" alt="Screen Shot 2022-11-21 at 8 59 17 PM" src="https://user-images.githubusercontent.com/99928364/203225586-c573277d-a3f0-46c8-bcc6-be09391ad445.png">


# RT Inference changes

Trained with the MNIST model for fashion.

To train the model using the MNIST data, a new model was created (inside CNNsimplemodels.py) and trained.

The class name and other parameters for the MNIST Model were changed as appropriate, and ImageOps was used to transform the RGB to grey scale to create the image array shape (28,28,1). 

The resulting inference model is tested using a sneaker image, and the prediction performed well with greater accuracy.

The output model has been saved in the output/fashion folder and will subsequently be used for lite model conversion and prediction use.

# Changes to TF lite

For embedded and mobile devices where the model needs to be smaller and more precise, lightweight models are used.

This time, the converted model predicted the sneakers as sandals, which is close to the actual prediction.

Export TF lite would take the model saved from the previous step and convert it to a lite model, which is then used to make inferences.



# Worked on Datasets

# Plotting  9 images from fashion training dataset

**COLAB LINK**
<a href="https://colab.research.google.com/drive/11jRV6DGo3EPlcMM3766Jg5G_gksTESi7?usp=share_link" target="_blank">Colab Link</a>


<img width="782" alt="Screen Shot 2022-11-21 at 9 03 15 PM" src="https://user-images.githubusercontent.com/99928364/203226121-5ec59ec6-fcfe-45b6-9eb1-d481c04037f0.png">


# Plotting  9 images from flowers training dataset

**COLAB LINK**
<a href="https://colab.research.google.com/drive/1Sapgm791W9aaHrYEk8gjXAwjHmdBfN0r?usp=share_link" target="_blank">Colab Link</a>


<img width="614" alt="Screen Shot 2022-11-21 at 9 14 19 PM" src="https://user-images.githubusercontent.com/99928364/203227474-db05c40d-8a19-4f70-ad20-90b3dd7f8096.png">


# Changes made to git repo

<a href="https://github.com/prernagarsole/MultiModalClassifier/commit/7306d44e5d5c74044488dc973b33c0f0ebaade08" target="_blank">Code changes Link</a>


# Steps to run

1. Created a new Virtual Environment. With python -m venv, create a virtual environment. 

2. Openvino env, then use openvino env to activate it (on Linux, you might need to run python3) On Windows or Linux, use the source openvino env/bin/activate command.

3. Git clone https://github.com/prernagarsole/MultiModalClassifier/ notebooks to copy the openvino notebooks repository.

4. Make a directory change: Openvino Notebooks CD

5. Use pip install —upgrade pip and pip install -r requirements.txt to install the prerequisites.

6. To allow GPU and CUDA usage on the local machine, installed libraries needed to be used together with setup.py and a few additional libraries.

7. Additional code was added to support the REST APIs.

8. A Google Colab notebook was made to serve the REST API, show predictions from a json file, and display the results.


# Compare the inference performance improvement (e.g., speed improvement over the original TF/Pytorch inference) of your selected feature.

Tensorflow's flow model provided predictions at a rate of no more than 10 frames per second. The trained model was then transferred to the onnx format and then to the openvino runtime. From the aforementioned findings, I was roughly given 130 frames per second with good prediction.

A collection of C++ libraries with C and Python bindings called OpenVINO Runtime offers a standard API for delivering inference solutions on the platform of your choosing. To read an Intermediate Representation (IR), ONNX, or PaddlePaddle model and run it on your chosen devices, use the OpenVINO Runtime API.

Plugin architecture is used by OpenVINO Runtime. Its plugins are software parts that include a full implementation for inference on a specific Intel® hardware component, such as a CPU, GPU, or VPU. Each plugin implements the unified API and offers extra hardware-specific APIs for setting up devices or API compatibility between the underlying plugin backend and OpenVINO Runtime.

# MultiModalClassifier

This is a project repo for multi-modal deep learning classifier with popular models from Tensorflow and Pytorch. The goal of these baseline models is to provide a template to build on and can be a starting point for any new ideas, applications. If you want to learn basics of ML and DL, please refer this repo: https://github.com/lkk688/DeepDataMiningLearning.

# Package setup
Install this project in development mode
```bash
(venv38) MyRepo/MultiModalClassifier$ python setup.py develop
```
After the installation, the package "MultimodalClassifier==0.0.1" is installed in your virtual environment. You can check the import
```bash
>>> import TFClassifier
>>> import TFClassifier.Datasetutil
>>> import TFClassifier.Datasetutil.Visutil
```

If you went to uninstall the package, perform the following step
```bash
(venv38) lkk@cmpeengr276-All-Series:~/Developer/MyRepo/MultiModalClassifier$ python setup.py develop --uninstall
```

# Code organization
* [DatasetTools](./DatasetTools): common tools and code scripts for processing datasets
* [TFClassifier](./TFClassifier): Tensorflow-based classifier
  * [myTFDistributedTrainerv2.py](./TFClassifier/myTFDistributedTrainerv2.py): main training code
  * [myTFInference.py](./TFClassifier/myTFInference.py): main inference code
  * [exportTFlite.py](./TFClassifier/exportTFlite.py): convert form TF model to TFlite
* [TorchClassifier](./TorchClassifier): Pytorch-based classifier
  * [myTorchTrainer.py](./TorchClassifier/myTorchTrainer.py): Pytorch main training code
  * [myTorchEvaluator.py](./TorchClassifier/myTorchEvaluator.py): Pytorch model evaluation code 

# Tensorflow Lite
* Tensorflow lite guide [link](https://www.tensorflow.org/lite/guide)
* [exportTFlite](\TFClassifier\exportTFlite.py) file exports model to TFlite format.
  * testtfliteexport function exports the float format TFlite model
  * tflitequanexport function exports the TFlite model with post-training quantization, the model size can be reduced by
![image](https://user-images.githubusercontent.com/6676586/126202680-e2e53942-7951-418c-a461-99fd88d2c33e.png)
  * The converted quantized model won't be compatible with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU) because the input and output still remain float in order to have the same interface as the original float only model.
* To ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU), we can enforce full integer quantization for all ops including the input and output, add the following code into function tflitequanintexport
```bash
converter_int8.inference_input_type = tf.int8  # or tf.uint8
converter_int8.inference_output_type = tf.int8  # or tf.uint8
```
  * The check of the floating model during inference will show false
```bash
floating_model = input_details[0]['dtype'] == np.float32
```
  * When preparing the image data for the int8 model, we need to conver the uint8 (0-255) image data to int8 (-128-127) via loadimageint function
  
# TensorRT inference
Check this [Colab](https://colab.research.google.com/drive/1aCbuLCWEuEpTVFDxA20xKPFW75FiZgK-?usp=sharing) (require SJSU google account) link to learn TensorRT inference for Tensorflow models.
Check these links for TensorRT inference for Pytorch models: 
* https://github.com/NVIDIA-AI-IOT/torch2trt
* https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
* https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt/
