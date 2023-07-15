# MiDaS-CV-Depth-Estimation

### main2.py Description

### main2.py Output:


### main.py Description
This Python code uses the MiDaS (Mixed Densely Associated Scale) model to perform depth estimation on live video from a webcam.

The code uses the torch.hub library to download the MiDaS model and its associated transforms pipeline, and then hooks into OpenCV to read live video frames from the webcam.

For each frame, the input is transformed using the MiDaS transforms pipeline, and depth predictions are made using the MiDaS model. The output is then displayed as an image using the matplotlib library.

The code also displays the original video frames in a window using OpenCV's imshow() function. The video stream can be stopped by pressing the 'q' key on the keyboard.

### main.py Output:

![image](https://user-images.githubusercontent.com/87671757/217102950-e287fc77-59b9-40ef-8416-6177a051ca5d.png)
