import torch
from torch.nn import functional as F
import numpy as np
from arcface.utils import (
    load_database,
    get_model,
    extract_feature
)


class Identifier:
    def __init__(
            self,
            theshold=.7,
            theshold_compare=.24,
            root_folder='VN-celeb',
            checkpoint='arcface/weights/db.pt'
    ):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.theshold = theshold
        self.theshold_compare = theshold_compare
        self.root_folder = root_folder
        self.model = get_model(self.device)
        self.features, self.labels = load_database(
            root_folder,
            self.model,
            self.device,
            checkpoint
        )

    def getId(self, face, rgb=False):
        feature = extract_feature(
            face,
            self.model,
            self.device,
            rgb=rgb
        )
        result = []
        for p in self.features:
            sim = F.cosine_similarity(
                feature,
                p,
                dim=0
            ).cpu().numpy()
            result.append(sim)

        idx = np.argmax(np.array(result), axis=0)
        if result[idx] < self.theshold:
            return None, result[idx]
        else:
            return self.labels[idx], result[idx]
