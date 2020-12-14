import numpy as np
import torch
import cv2
from arcface.model_irse import IR_50
from tqdm import tqdm

import os


def create_database(root_folder, model, device, n_samples=7):
    people = os.listdir(root_folder)
    features = []
    labels = []
    print("INFO crate database")
    for p in tqdm(people):
        imgs = os.listdir(
            os.path.join(root_folder, p)
        )
        imgs = sorted(imgs, key=lambda x: int(x.split('.')[0]))[:n_samples]
        for img_name in imgs:
            img_path = os.path.join(root_folder, p, img_name)
            img = cv2.imread(img_path)
            features.append(extract_feature(img, model, device))
            labels.append(int(p))
    return features, labels


def load_database(root_folder, model, device, checkpoint=False):
    if checkpoint:
        if not os.path.isfile(checkpoint):
            features, labels = create_database(root_folder, model, device)
            torch.save({
                'features': features,
                'lables': labels
            }, checkpoint)
        else:
            d = torch.load(checkpoint)
            features = d['features']
            labels = d['lables']
    else:
        features, labels = create_database(root_folder, model, device)
    return features, labels


def get_model(device=None):
    input_size = [112, 112]
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone_root = 'arcface/weights/arcface.pth'
    model = IR_50(input_size)
    model.load_state_dict(torch.load(backbone_root))
    model.to(device)
    model.eval()
    return model


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def extract_feature(img, backbone, device, tta=True, rgb=False):
    # resize image to [128, 128]
    resized = cv2.resize(img, (128, 128))

    # center crop image
    a = int((128-112)/2)  # x start
    b = int((128-112)/2+112)  # x end
    c = int((128-112)/2)  # y start
    d = int((128-112)/2+112)  # y end
    ccropped = resized[a:b, c:d]  # center crop the image
    if not rgb:
        ccropped = ccropped[..., ::-1]  # BGR to RGB

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype=np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype=np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)

    with torch.no_grad():
        if tta:
            emb_batch = backbone(
                ccropped.to(device)
            ).cpu() + backbone(
                flipped.to(device)
            ).cpu()
            feature = l2_norm(emb_batch)
        else:
            feature = l2_norm(backbone(ccropped.to(device)).cpu())
    return feature[0]

