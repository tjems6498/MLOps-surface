import bentoml
import numpy as np
from bentoml.io import Image, NumpyNdarray
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize((0.6948057413101196, 0.6747249960899353, 0.6418852806091309), (0.1313374638557434, 0.12778694927692413, 0.12676562368869781)),
    ToTensorV2(),
])
SURFACE_CLASSES = ['negatieve', 'positive']


surface_clf_runner = bentoml.pytorch.get("surface_classifier:latest").to_runner()

svc = bentoml.Service("surface_convnext2", runners=[surface_clf_runner])

@svc.api(input=Image(), output=NumpyNdarray())
async def classify(imgs):
    # inference preprocess
    imgs = np.array(imgs)
    imgs = transform(image=imgs)['image']
    imgs = imgs.unsqueeze(0)
    print("hey:", imgs.requires_grad)
    result = await surface_clf_runner.async_run(imgs)
    print(np.array([SURFACE_CLASSES[i] for i in torch.argmax(result, dim=1).tolist()]))
    return np.array([SURFACE_CLASSES[i] for i in torch.argmax(result, dim=1).tolist()])

