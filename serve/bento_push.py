import argparse
import subprocess
import shlex

from mlflow_model import load_model
import bentoml



def bento_serve(opt):
    model = load_model(model_name=opt.model_name, version=opt.model_version)
    for param in model.parameters():
        param.requires_grad = False

    bentoml.pytorch.save_model("surface_clff", model)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, help='MLflow model name')
    parser.add_argument('--model-version', type=int, help='MLFlow model version')
    parser.add_argument('--api-token', type=str, help='MLFlow model version')
    opt = parser.parse_args()


    bento_serve(opt)
    subprocess.run(["chmod", "+x", "bento_command.sh"])
    subprocess.call(shlex.split(f"./bento_command.sh {opt.api_token} http://116.47.188.227:30080"))
