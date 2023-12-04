import json
import os
import dotenv
import xarray

dotenv.load_dotenv()

# With the enviroment variables set now we import Earth-2 MIP
from earth2mip import inference_ensemble, registry

print("Fetching model package...")
package = registry.get_model("e2mip://fcnv2_sm")


cds_api = os.path.join(os.path.expanduser("~"), ".cdsapirc")
if not os.path.exists(cds_api):
    uid = input("Enter in CDS UID (e.g. 123456): ")
    key = input("Enter your CDS API key (e.g. 12345678-1234-1234-1234-123456123456): ")
    # Write to config file for CDS library
    with open(cds_api, "w") as f:
        f.write("url: https://cds.climate.copernicus.eu/api/v2\n")
        f.write(f"key: {uid}:{key}\n")


config = {
    "ensemble_members": 2,
    "noise_amplitude": 0.05,
    "simulation_length": 10,
    "weather_event": {
        "properties": {
            "name": "Globe",
            "start_time": "2018-06-01 00:00:00",
            "initial_condition_source": "cds",
        },
        "domains": [
            {
                "name": "global",
                "type": "Window",
                "lat_min": 36,
                "lat_max": 71,
                "lon_min": 0,
                "lon_max": 35,
                "diagnostics": [{"type": "raw", "channels": ["t2m", "u10m"]}],
            }
        ],
    },
    "output_path": "outputs/05_raw_demo",
    "output_frequency": 1,
    "weather_model": "fcnv2_sm",
    "seed": 12345,
    "use_cuda_graphs": False,
    "ensemble_batch_size": 1,
    "autocast_fp16": False,
    "perturbation_strategy": "correlated",
    "noise_reddening": 2.0,
}


config_str = json.dumps(config)
inference_ensemble.main(config_str)