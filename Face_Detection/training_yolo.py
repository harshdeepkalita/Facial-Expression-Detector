# -*- coding: utf-8 -*-
"""Making the most of your colab subscription

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/notebooks/pro.ipynb
"""

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="8kyoznba5aQQ7KeiXfgq")
project = rf.workspace("mohamed-traore-2ekkp").project("face-detection-mik1i")
version = project.version(18)
dataset = version.download("yolov8")

!pip install ultralytics

import shutil

shutil.move("Face-Detection-18/train","Face-Detection-18/Face-Detection-18/train")
shutil.move("Face-Detection-18/test","Face-Detection-18/Face-Detection-18/test")
shutil.move("Face-Detection-18/valid","Face-Detection-18/Face-Detection-18/valid")

!pip install -U albumentations

!yolo train model=yolov8x.pt data={dataset.location}/data.yaml epochs=100 imgsz=640


