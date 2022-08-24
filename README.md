# yolov5-onnx-inference
Simple inference script for yolov5.onnx. Reference from detect.py in yolov5. Inference on per image.  
I removed the unused part in detect.py and added every useful fuction/class to only one file.
# Requirements
**Note**:The package version depends on your computer.  
For me:  
Numpy==1.22.3   
Opencv-python==4.5.5  
torch==1.9.0+cu102  
torchvision==0.10.0+cu102  
Onnxruntime-gpu==1.12.1  
# Usage
1.Download my project zip or git clone.  
2.Change the file name in **detect_onnx.py**.(e.g. 'best.onnx','test.png'...)  
3.run **detect_onnx.py**  
4.Then look at the saved picture in your directory. 

