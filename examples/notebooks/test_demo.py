import os
import json
import logging
import xarray
import subprocess
import argparse
from torch import distributed
# from torch.distributed.group import WORLD

from modulus.distributed.manager import DistributedManager

from earth2mip.inference_ensemble import get_model, run_inference, get_initializer
from earth2mip.schema import EnsembleRun

base_config = {
    "ensemble_members": 4,
    "noise_amplitude": 0.01,
    "simulation_length": 8,
    "weather_event": {
        "properties": {
            "name": "Globe",
            "start_time": "2022-07-01 00:00:00",
            "initial_condition_source": "cds"
        },
        "domains": [
            {
                "name": "eu",
                "type": "Window",
                "lat_min": 36,
                "lat_max": 71,
                "lon_min": 0,
                "lon_max": 35,
                "diagnostics": [
                    {
                        "type": "raw",
                        "channels": [
                            "t2m",
                            "tcwv"
                        ]
                    }
                ]
            },
            {
                "name": "eu_cities",
                "type": "MultiPoint",
                "lat": [48.75, 43.50],
                "lon": [2.50, 5.25],
                "diagnostics": [
                    {
                        "type": "raw",
                        "channels": [
                            "t2m",
                            "tcwv"
                        ]
                    }
                ]
            }
        ]
    },
    "output_path": "../outputs/05_demo_script",
    "output_frequency": 1,
    "seed": 12345,
    "use_cuda_graphs": False,
    "ensemble_batch_size": 1,
    "autocast_fp16": False,
    "perturbation_strategy": "correlated",
    "noise_reddening": 2.0
}

def download_model(model: str):
    model_registry = os.environ.get('MODEL_REGISTRY', None)
    assert model_registry != None, \
        print(f'set os variable before running script: $ export MODEL_REGISTRY=/path/to/model/registry')
    os.makedirs(model_registry, exist_ok=True)

    if model == 'dlwp':
        link = 'https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_dlwp_cubesphere/versions/v0.2/files/dlwp_cubesphere.zip'
    elif model == 'fcnv2_sm':
        link = 'https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_fcnv2_sm/versions/v0.2/files/fcnv2_sm.zip'
    else:
        raise ValueError(f'model {model} not implemented, yet.')
    package_name = link.split('/')[-1]

    if not os.path.isdir(os.path.join(model_registry, model)):
        print("Downloading model checkpoint, this may take a bit")
        subprocess.run(['wget', '-nc', '-P', f'{model_registry}', link], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.run(['unzip', '-u', f'{model_registry}/{package_name}', '-d', f'{model_registry}'])
        subprocess.run(['rm', f'{model_registry}/{package_name}'])

    return

def run_demo():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='dlwp', help='dlwp, fcnv2_sm', type=str)
    args = parser.parse_args()

    config: EnsembleRun = EnsembleRun.parse_obj(base_config | {"weather_model": args.model})

    # Set model registry as a local folder
    download_model(args.model)

    # Set up parallel
    DistributedManager.initialize()
    device = DistributedManager().device
    group = distributed.group.WORLD

    # run inference
    print('running ensemble inference')
    logging.info(f"Loading model onto device {device}")
    model = get_model(config.weather_model, device=device)
    logging.info("Constructing initializer data source")
    perturb = get_initializer(model, config, )
    logging.info("Running inference")
    run_inference(model, config, perturb, group)

    return

if __name__ == "__main__":
    run_demo()