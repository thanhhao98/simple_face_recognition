import cv2
from PIL import Image
from arcface.identifier import Identifier
from mtcnn_pytorch.detector import MTCNN

identifier = Identifier(theshold=0.24)
detector = MTCNN()
img = Image.open('./vhnh.jpg')
bboxes, _ = detector.detect_faces(img)
cv_img = cv2.imread('./vhnh.jpg')
for bbox in bboxes:
    min_x = int(round(bbox[0]))
    min_y = int(round(bbox[1]))
    max_x = int(round(bbox[2]))
    max_y = int(round(bbox[3]))
    face = cv_img[min_y:max_y, min_x:max_x, :]
    label, sim = identifier.getId(face)
    sim = float(sim)
    cv2.rectangle(cv_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    cv2.putText(
        cv_img,
        f'{label}_{round(sim,2)}',
        (min_x, min_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2)

cv2.imwrite('out.jpg', cv_img)




