## Quick Start

1. Git clone yolov5 project of u version and install requirements.txt

   ```
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   pip install -r requirements.txt
   pip install coremltools > =4.1 onnx > =1.9.0 scikit-learn==0.19.2
   
   if use gpu
   # pip install onnxruntime-gpu
   ```

2. To export onnx with 640x640, the batch size must be 1,if --train is not added, there will be many redundant layers

   ```
   python export.py --weights yolov5s.pt --img 640 --batch 1 --train   
   ```

3. Onnx simplifer simplified model

   ```
   python -m onnxsim yolov5s.onnx yolov5s-sim.onnx
   ```

4. onnxruntime infer

   ```
   python predict.py
   ```

    

