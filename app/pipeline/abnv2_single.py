import cv2
from ultralytics import YOLO

CROP_HEIGHT = 400
CROP_WIDTH = 400

class ABNCropper:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def crop(self, img_path: str):
        img = cv2.imread(img_path)
        if img is None:
            return None

        h, w = img.shape[:2]

        results = self.model(img, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            return None

        # pick largest bounding box
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        x1, y1, x2, y2 = boxes[areas.index(max(areas))]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        y1 = cy
        y2 = min(cy + CROP_HEIGHT, h)

        x1 = max(0, cx - CROP_WIDTH // 2)
        x2 = min(x1 + CROP_WIDTH, w)

        if y2 <= y1 or x2 <= x1:
            return None

        return img[y1:y2, x1:x2]
