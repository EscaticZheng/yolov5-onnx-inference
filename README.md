# yolov5-onnx-inference
Simple inference script for yolov5.onnx. Reference from detect.py in yolov5. Inference on per image.  
I removed the unused part in detect.py and added every useful fuction/class to only one file.
# Requirements
**Note**:The package version depends on your computer.  
For me:  
numpy==1.22.3   
opencv-python==4.5.5  
torch==1.9.0+cu102  
torchvision==0.10.0+cu102  
onnxruntime-gpu==1.12.1  
**If u use the cpu version, you can pip install torch, torchvision, and onnxruntime**  
# Usage
1.Download my project zip or git clone.  
2.Change the file name in **onnx_gpu_detect.py** or **onnx_cpu_detect.py**.(e.g. 'best.onnx','test.png','names=['nofall','fall']'...)  
3.Run it, then look at the saved picture in your directory. 


