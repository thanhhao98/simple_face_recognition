from tqdm import tqdm
import torch
import os
from torch.nn import functional as F
import json
import random
from utils import (
    get_model,
    extract_feature
)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    N = 100000
    distances = [[], []]
    people = os.listdir('../VN-celeb')
    for i in tqdm(range(N)):
        pos = random.randint(0, 1)
        person_1 = random.choice(people)
        person_1_fullpath = os.path.join('../VN-celeb/', str(person_1))
        imgs_1 = os.listdir(person_1_fullpath)
        img_1_raw = random.choice(imgs_1)
        img_1 = os.path.join(person_1_fullpath, img_1_raw)
        if pos:
            img_2_raw = random.choice([j for j in imgs_1 if j != img_1_raw])
            img_2 = os.path.join(person_1_fullpath, img_2_raw)
        else:
            person_2 = random.choice([j for j in people if j != person_1])
            person_2_fullpath = os.path.join('../VN-celeb/', str(person_2))
            img_2_raw = random.choice(os.listdir(person_2_fullpath))
            img_2 = os.path.join(person_2_fullpath, img_2_raw)

        feature_1 = extract_feature(img_1, model, device)
        feature_2 = extract_feature(img_2, model, device)
        distance = F.cosine_similarity(
            feature_1,
            feature_2,
            dim=0
        ).cpu().numpy()
        distances[pos].append(
            (
                img_1,
                img_2,
                float(distance)
            )
        )
    with open(f'result_{N}.json', 'w') as f:
        json.dump(distances, f)
    pos = [i[2] for i in distances[1]]
    neg = [i[2] for i in distances[0]]
    print(min(pos))
    print(max(neg))
