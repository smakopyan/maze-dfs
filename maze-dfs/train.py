from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="config.yaml",
    epochs=100,          
    batch=4,             
    imgsz=320,           
    workers=0,           
    device='cpu',        
    patience=10,         
    lr0=0.001,          
)
