from ultralytics import YOLO

model = YOLO('best-2.pt')


result = model.predict('Untitled.mov',conf=0.2,save=True)

for bbox in result[0].boxes:
    print(bbox)


