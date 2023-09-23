from ultralytics import YOLO

model = YOLO("/home/vudangitwork/Documents/FTECH_Reposity/face_mask_detection/runs/detect/train7/weights/best.pt")

results = model.predict(
    source="0",
    show=True,
)