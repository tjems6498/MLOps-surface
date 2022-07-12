import kfp
from kfp import dsl
from kfp import onprem
# TODO: dataloader num_workers 값을 주었을때 shm 문제해결
import kubernetes as k8s

def preprocess_op(pvc_name, volume_name, volume_mount_path):

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='tjems6498/surface_pipeline_preprocess:2',
        arguments=['--data-path', volume_mount_path,
                   '--img-size', 224],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def hyp_op(pvc_name, volume_name, volume_mount_path, count, device):

    return dsl.ContainerOp(
        name='Hyperparameter Tuning',
        image='tjems6498/surface_pipeline_hyper:3',
        arguments=['--data-path', volume_mount_path,
                    '--count', count,
                    '--device', device],
    ).set_gpu_limit(4).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)
            ).add_pvolumes({'/dev/shm': dsl.PipelineVolume(volume=k8s.client.V1Volume(
        name="shm",
        empty_dir=k8s.client.V1EmptyDirVolumeSource(medium='Memory', size_limit='256M')))})

def train_op(pvc_name, volume_name, volume_mount_path, repo_name, epoch, img_size, batch_size, learning_rate, optimizer, device):

    return dsl.ContainerOp(
        name='Train Model',
        image='tjems6498/surface_pipeline_train:9',
        arguments=['--data-path', volume_mount_path,
                    '--repo-name', repo_name,
                    '--epoch', epoch,
                    '--img-size', img_size,
                    '--batch-size', batch_size,
                    '--learning-rate', learning_rate,
                    '--optimizer', optimizer,
                    '--device', device]
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)
            ).set_gpu_limit(4).add_pvolumes({'/dev/shm': dsl.PipelineVolume(volume=k8s.client.V1Volume(
        name="shm",
        empty_dir=k8s.client.V1EmptyDirVolumeSource(medium='Memory', size_limit='256M')))})

def test_op(pvc_name, volume_name, volume_mount_path, img_size, batch_size, model_name, model_version, device):

    return dsl.ContainerOp(
        name='Test Model',
        image='tjems6498/surface_pipeline_test:10',
        arguments=['--data-path', volume_mount_path,
                   '--img-size', img_size,
                   '--batch-size', batch_size,
                   '--model-name', model_name,
                   '--model-version', model_version,
                   '--device', device]
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)
            ).set_gpu_limit(4).add_pvolumes({'/dev/shm': dsl.PipelineVolume(volume=k8s.client.V1Volume(
        name="shm",
        empty_dir=k8s.client.V1EmptyDirVolumeSource(medium='Memory', size_limit='256M')))})

def serve_op(pvc_name, volume_name, volume_mount_path, model_name, model_version):

    return dsl.ContainerOp(
        name='Bento packing',
        image='tjems6498/surface_pipeline_serve:19',
        arguments=['--data-path', volume_mount_path,
                   '--model-name', model_name,
                   '--model-version', model_version],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))



@dsl.pipeline(
    name='Surface Crack Pipeline',
    description=''
)
def surface_pipeline(PREPROCESS_yes_no: str,
                    DEVICE: str,
                    MODE_hyp_train_test_serve: str,
                    HYPER_sweep_count: int,
                    TRAIN_repo_name: str,
                    TRAIN_epoch: int,
                    TRAIN_img_size: int,
                    TRAIN_batch_size: int,
                    TRAIN_learning_rate: float,
                    TRAIN_optimizer: str,
                    TEST_model_name: str,
                    TEST_model_version: int,
                    TEST_img_size: int,
                    TEST_batch_size: int,
                    SERVE_model_name: str,
                    SERVE_model_version: int
):

    pvc_name = "workspace-surface"
    volume_name = 'pipeline'
    volume_mount_path = '/home/jeff'

    with dsl.Condition(PREPROCESS_yes_no == 'yes'):
        _preprocess_op = preprocess_op(
            pvc_name = pvc_name, 
            volume_name = volume_name, 
            volume_mount_path = volume_mount_path
        )

    with dsl.Condition(MODE_hyp_train_test_serve == 'hyp'):
        _hyp_op = hyp_op(
            pvc_name = pvc_name, 
            volume_name = volume_name, 
            volume_mount_path = volume_mount_path,
            count = HYPER_sweep_count,
            device = DEVICE,
            ).after(_preprocess_op)

    with dsl.Condition(MODE_hyp_train_test_serve == 'train'):
        _train_op = train_op(
            pvc_name = pvc_name, 
            volume_name = volume_name, 
            volume_mount_path = volume_mount_path,
            repo_name = TRAIN_repo_name,
            epoch = TRAIN_epoch,
            img_size = TRAIN_img_size,
            batch_size = TRAIN_batch_size, 
            learning_rate = TRAIN_learning_rate, 
            optimizer = TRAIN_optimizer, 
            device = DEVICE
        ).after(_preprocess_op)

    with dsl.Condition(MODE_hyp_train_test_serve == 'test'):
        _test_op = test_op(
            pvc_name = pvc_name, 
            volume_name = volume_name, 
            volume_mount_path = volume_mount_path,
            img_size = TEST_img_size,
            batch_size= TEST_batch_size,
            model_name = TEST_model_name,
            model_version= TEST_model_version,
            device = DEVICE
        ).after(_preprocess_op)

    with dsl.Condition(MODE_hyp_train_test_serve == 'serve'):
        _serve_op = serve_op(
            pvc_name = pvc_name, 
            volume_name = volume_name, 
            volume_mount_path = volume_mount_path,
            model_name = SERVE_model_name,
            model_version= SERVE_model_version
        ).after(_test_op)



if __name__ == '__main__':
    kfp.compiler.Compiler().compile(surface_pipeline, './surface.yaml')
