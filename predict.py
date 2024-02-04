### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
###############################################################################
### File source:
### https://github.com/datakami-models/LucidDreamer/blob/main/predict.py
###############################################################################
from typing import List
from pathlib import Path

import os
import shutil
import subprocess
import sys
import yaml

# from train import *
# try:
#     from train import *
# except ImportError:
#     # install the submodules if they are not installed yet
#     subprocess.run(["pip", "install", "./submodules/diff-gaussian-rasterization"], check=True)
#     subprocess.run(["pip", "install", "./submodules/simple-knn/"], check=True)
#     from train import *

###

HUGGINGFACE_HUB_CACHE = Path("/media/data2/mconti/TT3D/checkpoints/LucidDreamer/models")
HUGGINGFACE_HUB_CACHE.mkdir(parents=True, exist_ok=True)

os.environ['HUGGINGFACE_HUB_CACHE'] = str(HUGGINGFACE_HUB_CACHE)
os.environ['HF_HUB_OFFLINE'] = 'true'


class Predictor(BasePredictor):

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        ckpt_url = "https://weights.replicate.delivery/default/luciddreamer/stable-diffusion-2-1-base.tar"

        if not os.path.exists("/src/models/"):
            print("Downloading checkpoint")
            try:
                output = subprocess.check_output(["pget", "-x", ckpt_url, "/src/tmp"])
                os.rename("/src/tmp/", "/src/models/")
            except subprocess.CalledProcessError as e:
                raise e

    def predict(
            self,
            prompt: str = Input(description="Your prompt",),
            init_prompt: str = Input(
                description="Optional Point-E init prompt",
                default=None,
            ),
            neg_prompt: str = Input(
                description="Negative prompt",
                default=None,
            ),
            iterations: int = Input(
                description="Number of iterations",
                default=2000,
                ge=100,
                le=10000,
            ),
            cfg: float = Input(
                description="CFG",
                default=7.5,
            ),
            seed: int = Input(
                description="Seed. Leave blank for a random seed.",
                default=None,
            ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        if neg_prompt is None:
            neg_prompt = ''

        train_path = os.path.join(os.path.dirname(__file__), 'train.py')
        # Use cat_armor as a template.
        template_path = os.path.join(os.path.dirname(__file__), 'configs/cat_armor.yaml')

        output_folder = Path(os.path.join(os.path.dirname(__file__), 'output'))
        if output_folder.exists():
            shutil.rmtree(output_folder)
        os.makedirs(str(output_folder), exist_ok=False)

        config_path = os.path.join(os.path.dirname(__file__), 'output/predict.yaml')

        workspace = 'Replicate'

        output_video_path = os.path.join(
            os.path.dirname(__file__), f"output/{workspace}/videos/{iterations}_iteration/video_rgb_{iterations}.mp4")
        output_proc_path = os.path.join(os.path.dirname(__file__), f"output/{workspace}/process_videos/video_rgb.mp4")

        with open(template_path, 'r') as yml:
            config = yaml.safe_load(yml)

        config['GuidanceParams']['text'] = prompt
        config['GuidanceParams']['negative'] = neg_prompt
        config['GuidanceParams']['noise_seed'] = seed
        config['GuidanceParams']['guidance_scale'] = cfg
        config['ModelParams']['workspace'] = workspace

        if init_prompt is None:
            config['GenerateCamParams']['init_prompt'] = '.'
            config['GenerateCamParams']['init_shape'] = 'sphere'
        else:
            config['GenerateCamParams']['init_prompt'] = init_prompt
            config['GenerateCamParams']['init_shape'] = 'pointe'
        config['OptimizationParams']['iterations'] = iterations

        with open(config_path, 'w') as yml:
            yaml.safe_dump(config, yml, default_flow_style=False)

        try:
            subprocess.check_call([sys.executable, train_path, "--opt", config_path])
        except Exception as e:
            print(str(e))
        return [Path(output_video_path), Path(output_proc_path)]
