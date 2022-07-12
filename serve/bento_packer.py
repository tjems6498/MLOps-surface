import argparse
import os
import shutil
from bento_service import SurfaceClassification
from mlflow_model import load_model


def bento_serve(opt):
    surface_classifier_service = SurfaceClassification()
    model = load_model(model_name=opt.model_name, version=opt.model_version)

    surface_classifier_service.pack('pytorch_model', model)

    saved_path = surface_classifier_service.save()

    os.makedirs(os.path.join(opt.data_path, "serve"), exist_ok=True)
    shutil.move(saved_path, os.path.join(opt.data_path, "serve"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    parser.add_argument('--model-name', type=str, help='MLflow model name')
    parser.add_argument('--model-version', type=int, help='MLFlow model version')
    opt = parser.parse_args()

    bento_serve(opt)
