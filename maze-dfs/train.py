from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="config.yaml",
    epochs=100,           # меньше эпох
    batch=4,             # меньший batch size
    imgsz=320,           # меньшее разрешение
    workers=0,           # 0 воркеров для CPU
    device='cpu',        # явно указываем CPU
    patience=10,         # ранняя остановка
    lr0=0.001,          # меньшая скорость обучения
)
