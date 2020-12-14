import cv2
from PIL import Image
from arcface.identifier import Identifier
import numpy as np
from tqdm import tqdm
import os

identifier = Identifier(theshold=0.24)
root_path = './VN-celeb'
people = os.listdir(root_path)
imgs_path = []
labels = []
print('INFO: Load test set')
for p in tqdm(people):
    foler_path = os.path.join(root_path, p)
    imgs_name = os.listdir(foler_path)
    imgs_name = sorted(imgs_name, key=lambda x: int(x.split('.')[0]))[5:]
    for img_name in imgs_name:
        imgs_path.append(os.path.join(
            foler_path,
            img_name
        ))
        labels.append(int(p))
numSamples = 1000
result = []
print('INO: Running benchmark')
for i in tqdm(range(numSamples)):
    img = cv2.imread(imgs_path[i])
    label, _ = identifier.getId(img)
    if label != labels[i]:
        result.append(0)
    result.append(1)

print('accuracy: ', np.mean(result))
