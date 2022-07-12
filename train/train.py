import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
from mlflow.pytorch import save_model
from mlflow.tracking.client import MlflowClient
from util import seed_everything, MetricMonitor, build_dataset, build_optimizer
import time

class ConvNeXt(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ConvNeXt, self).__init__()
        self.model = timm.create_model('convnext_tiny', pretrained=pretrained)  # timm 라이브러리에서 pretrained model 가져옴
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)


def train_epoch(train_loader, epoch, model, optimizer, criterion, device):
    metric_monitor = MetricMonitor()

    model.train()
    stream = tqdm(train_loader)

    for i, (images, targets) in enumerate(stream, start=1):
        images, targets = images.float().to(device), targets.to(device)

        output = model(images)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(output, dim=1)
        accuracy = round((targets == predicted).sum().item() / targets.shape[0] * 100, 2)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('accuracy', accuracy)

        stream.set_description(
            f"Epoch: {epoch}. Train. {metric_monitor}"
        )

def val_epoch(val_loader, epoch, model, criterion, device):
    metric_monitor = MetricMonitor()

    model.eval()
    stream = tqdm(val_loader)

    with torch.no_grad():
        for i, (images, targets) in enumerate(stream, start=1):
            images, targets = images.float().to(device), targets.to(device)

            output = model(images)
            loss = criterion(output, targets)

            predicted = torch.argmax(output, dim=1)
            accuracy = round((targets == predicted).sum().item() / targets.shape[0] * 100, 2)

            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('accuracy', accuracy)

            stream.set_description(
                f"Epoch: {epoch}. Validation. {metric_monitor}"
            )
    return accuracy


def main(opt, device):
    # read mean std values

    with open(f'{opt.data_path}/mean-std.txt', 'r') as f:
        cc = f.readlines()
        mean_std = list(map(lambda x: x.strip('\n'), cc))


    model = ConvNeXt(num_classes=2, pretrained=True)
    model.to(device)

    train_loader, val_loader, _ = build_dataset(opt.data_path, opt.img_size, opt.batch_size, mean_std)
    optimizer = build_optimizer(model, opt.optimizer, opt.learning_rate)
    criterion = nn.CrossEntropyLoss()

    scheduler = CosineAnnealingLR(optimizer, T_max=10,
                                  eta_min=1e-6,
                                  last_epoch=-1)

    best_accuracy = 0
    start = time.time()
    for epoch in range(1, opt.epoch + 1):
        train_epoch(train_loader, epoch, model, optimizer, criterion, device)
        accuracy = val_epoch(val_loader, epoch, model, criterion, device)
        scheduler.step()

        if accuracy > best_accuracy:
            os.makedirs(f'{opt.data_path}/weight', exist_ok=True)
            torch.save(model.state_dict(), f'{opt.data_path}/weight/best.pth')
            best_accuracy = accuracy
    end = time.time()
    print(f'TRAIN TIME: {end-start:.3f}s')


def upload_model_to_mlflow(opt, device):

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    client = MlflowClient("http://mlflow-server-service.mlflow-system.svc:5000")

    model = ConvNeXt(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(f'{opt.data_path}/weight/best.pth', map_location=device))

    conda_env = {'name': 'mlflow-env', 'channels': ['conda-forge'],
     'dependencies': ['python=3.9.4', 'pip', {'pip': ['mlflow', 'torch==1.8.0', 'cloudpickle==2.0.0']}]}

    save_model(
        pytorch_model=model,
        path=opt.repo_name,
        conda_env=conda_env,
    )

    tags = {"DeepLearning": "surface crack classification"}
    run = client.create_run(experiment_id="2", tags=tags)
    client.log_artifact(run.info.run_id, opt.repo_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    parser.add_argument('--repo-name', type=str, help='model repo name for artifact path')
    parser.add_argument('--epoch', type=int, help='train epoch')
    parser.add_argument('--img-size', type=int, help='resize img size')
    parser.add_argument('--batch-size', type=int, help='batch size')
    parser.add_argument('--learning-rate', type=float, help='learning rate')
    parser.add_argument('--optimizer', type=str, help='optimizer')
    parser.add_argument('--device', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    opt = parser.parse_args()

    if opt.device == 'cpu':
        device = 'cpu'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"DEVICE is {device}")
    seed_everything()
    main(opt, device)
    upload_model_to_mlflow(opt, device)











