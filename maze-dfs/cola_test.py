from ultralytics import YOLO

model = YOLO('/home/sa/turtlebot3_ws/src/maze-dfs/maze-dfs/runs/detect/train3/weights/best.pt')
data = '/home/sa/Downloads/test2.jpg'
results = model(data)
if len(results[0].boxes) != 0:
    for i in range(len(results[0].boxes)):
        print(f'Object detected, confidence: {results[0].boxes.conf[i].cpu().numpy()}')