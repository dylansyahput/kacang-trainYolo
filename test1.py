from ultralytics import YOLO
# from ultralytics import YOLO
# from PIL import Image
import cv2
model = YOLO('runs/detect/train3/weights/best.pt')
SOURCE='train/images/jerawat01.jpeg'
im2 = cv2.imread(SOURCE)
results = model.predict(
   source=SOURCE,
   conf=0.1
)
# results = model.predict(source=im2, conf=0.4, overlap_mask=True) 
for result in results:
    # print('predictions ->',result.predictions)
    print('->',result.masks)
    print('probs->',result.probs)
    # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

# cv2.imshow('landmarks', im2)
# cv2.waitKey(0)