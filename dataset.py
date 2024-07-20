from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class FaceSketchDataset(Dataset):
    def __init__(self, root_sketch, root_face, transform=None):
        self.root_sketch = root_sketch
        self.root_face = root_face
        self.transform = transform

        self.sketch_images = os.listdir(root_sketch)
        self.face_images = os.listdir(root_face)
        self.length_dataset = max(len(self.sketch_images), len(self.face_images)) # 1000, 1500
        self.sketch_len = len(self.sketch_images)
        self.face_len = len(self.face_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        sketch_img = self.sketch_images[index % self.sketch_len]
        face_img = self.face_images[index % self.face_len]

        sketch_path = os.path.join(self.root_sketch, sketch_img)
        face_path = os.path.join(self.root_face, face_img)

        sketch_img = np.array(Image.open(sketch_path).convert("RGB"))
        face_img = np.array(Image.open(face_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=sketch_img, image0=face_img)
            sketch_img = augmentations["image"]
            face_img = augmentations["image0"]

        return sketch_img, face_img





