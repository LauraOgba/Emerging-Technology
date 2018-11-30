# Emerging-Technology
Project 2018

### Overview
The aim of this project is to learn the fundamentals of the python language, which in the long run will help to not only be able to write the language but understand it better. The assignment has helped me in a better understaning of machine learning and the recent technology of Artificial Intellegence.

### Content
This project consists of five components:
1. numpy random notebook: a jupyter notebook explaining the concepts behind and the use of the numpy random package, including plots of the various distributions.
2. Iris dataset notebook: a jupyter notebook explaining the famous iris data set including the difficulty in writing an algorithm to separate the three classes of iris based on the variables in the dataset.
3. MNIST dataset notebook: a jupyter notebook explaining how to read the MNIST dataset efficiently into memory in Python.
4. Digit recognition script: a Python script that takes an image file containing a handwritten digit and identifies the digit using a supervised learning algorithm and the MNIST dataset.
-5. Digit recognition notebook: a jupyter notebook explaining how the above Python script works and discussing its performance.

### Software requirements
- For this project you will need to download the jupyter nootebook software to run the files ending in (.ipynb). Link and tutorial on how to do this is here: http://jupyter.org/.
- Another software you'll need to run the python script code is visua studio code. This is where I developed my code for handwritten data recognition using MNIST dataset. Follow this link on how to download it. Here:https://code.visualstudio.com/.
- Another software would be command console, this where I ran my jupyter notebook on and puched and pulled all my repositories through. Link here: http://cmder.net/

### Running the data recognition script:
Before you can run this python script you'll need to download the following in your command console:
1. Tensorflow it is essential as it's what the MNIST dataset runs on. There's a good tutorial on https://www.tensorflow.org/install/
2. Also python image library used to read each handwritten file, unfortunately that isn't availble anymore so we'll be using Pillow download here: https://pillow.readthedocs.io/en/5.3.x/

### Running
Running the python script consists of the following:
1. create the model file
2. create an image file containing a handwritten number
3. predict the integer

### Create the model file
Ensure to change the directory in the cmd is where all the files are stored for easy access. To run:
1. python create_model_1.py

### Create an image File
You have to create a PNG file that contains a handwritten number. The background has to be white and the number has to be black. Any paint program should be able to do this. Also the image has to be auto cropped so that there is no border around the number.

### Predict the number 
Following step 2 ensure you're in right directory, where the script files are located also. The predict scripts require one argument: the file location of the image file containing the handwritten number. For example when the image file is ‘number1.png’ and is in the same location as the script, run:

python predict_1.py number1.png

This is how the script will be ran. I hope this helps how to maneuver from reading the nootebook files to running the python script.
