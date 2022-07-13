import os
import mlflow
from mlflow.tracking import MlflowClient


def load_model(model_name, version):
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    client = MlflowClient("http://mlflow-server-service.mlflow-system.svc:5000")

    # os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://116.47.188.227:31968"
    # client = MlflowClient("http://116.47.188.227:30842")

    filter_string = f"name='{model_name}'"
    results = client.search_model_versions(filter_string)  # 버전별로 따로 나옴


    for res in results:
        if res.version == str(version):
            model_uri = res.source
            break
    reconstructed_model = mlflow.pytorch.load_model(model_uri)
    return reconstructed_model



if __name__ == '__main__':
    import torch
    import pdb
    model = load_model(model_name='Random-Forest', version=1)

    # sample = torch.rand((1, 3, 224, 224))
    # output = model(sample)
    # print(output)
