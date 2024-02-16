### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, Any, Optional
from pathlib import Path

import argparse
import torch
import yaml
import os
import subprocess
import sys
import warnings
import open3d as o3d

from copy import deepcopy

from tt3d_utils import Utils

###

device = Utils.Cuda.init()

###


def _load_default_config() -> dict:
    path = Path("./configs/default.yaml")

    assert isinstance(path, Path)
    assert path.exists()
    assert path.is_file()

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return deepcopy(config)


def _generate(
    prompt: str,
    out_rootpath: Path,
    train_steps: int,
    use_priors: bool,
    skip_existing: bool,
    prompt_config: Optional[dict] = None,
) -> None:
    assert prompt_config is None or isinstance(prompt_config, dict)

    prompt_enc = Utils.Prompt.encode(prompt)

    tmp_root_path = Path(os.path.join(os.path.dirname(__file__)))
    tmp_output_path = tmp_root_path.joinpath('output')
    tmp_config_filepath = tmp_output_path.joinpath(f"{prompt_enc}.yaml")

    if tmp_config_filepath.exists():
        tmp_config_filepath.unlink()

    #

    if use_priors and prompt_config is None:
        print("")
        warnings.warn(f"Priors are enabled but no priors config was provided for '{prompt}'.")
        print("")
    if not use_priors and prompt_config is not None:
        print("")
        warnings.warn(f"Priors are disabled but a priors config was provided for '{prompt}'.")
        print("")

    assert not (use_priors and prompt_config is None)

    config = _load_default_config()
    config['GuidanceParams']['text'] = prompt
    config['ModelParams']['workspace'] = prompt_enc
    config['OptimizationParams']['iterations'] = train_steps
    # config['GuidanceParams']['negative'] = neg_prompt
    # config['GuidanceParams']['noise_seed'] = seed
    # config['GuidanceParams']['guidance_scale'] = cfg
    config['GenerateCamParams']['init_prompt'] = '.'
    config['GenerateCamParams']['init_shape'] = 'sphere'

    if use_priors and prompt_config is not None:
        if "init_shape" in prompt_config and "init_prompt" in prompt_config:
            config['GenerateCamParams']['init_shape'] = prompt_config["init_shape"]
            config['GenerateCamParams']['init_prompt'] = prompt_config["init_prompt"]

    with open(tmp_config_filepath, "w", encoding="utf-8") as file:
        yaml.dump(config, file)

    #

    # try:
    subprocess.check_call([
        sys.executable,
        "train.py",
        "--opt",
        str(tmp_config_filepath),
        "--test_ratio",
        "1",
        "--save_ratio",
        "1",
    ])
    # except Exception as e:
    #     print(str(e))

    #

    # tmp_export_path = tmp_output_path.joinpath(prompt_enc, "point_cloud", f"iteration_{train_steps}")
    # tmp_ply_filepath = tmp_export_path.joinpath("point_cloud.ply")
    # tmp_obj_filepath = tmp_export_path.joinpath("model.obj")
    # assert tmp_ply_filepath.exists() and tmp_ply_filepath.is_file()
    # pcd = o3d.io.read_point_cloud(str(tmp_ply_filepath))
    # o3d.visualization.draw_plotly([pcd])


###


def main(
    prompt_filepath: Path,
    out_rootpath: Path,
    # batch_size: int,
    train_steps: int,
    use_priors: bool,
    skip_existing: bool,
) -> None:
    assert isinstance(prompt_filepath, Path)
    assert isinstance(out_rootpath, Path)
    # assert isinstance(batch_size, int)
    assert isinstance(train_steps, int)
    assert 0 < train_steps < 10000
    assert isinstance(use_priors, bool)
    assert isinstance(skip_existing, bool)

    if out_rootpath.exists():
        assert out_rootpath.is_dir()
    else:
        out_rootpath.mkdir(parents=True)

    #

    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)
    model_config = Utils.Prompt.extract_model_config_from_prompt_filepath(
        model="luciddreamer",
        prompt_filepath=prompt_filepath,
    )
    model_config = model_config if model_config is not None else {}

    print("")
    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        prompt_enc = Utils.Prompt.encode(prompt)
        prompt_config = model_config.get(prompt_enc, model_config.get("*", None))

        print("")
        print(prompt)
        print(prompt_config)

        try:
            _generate(
                prompt=prompt,
                out_rootpath=out_rootpath,
                train_steps=train_steps,
                use_priors=use_priors,
                skip_existing=skip_existing,
                prompt_config=prompt_config,
            )
        except Exception as e:
            print("")
            print("")
            print("========================================")
            print("Error while running prompt -> ", prompt)
            print(e)
            print("========================================")
            print("")
            print("")
            continue

        print("")
    print("")


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--out-path', type=Path, required=True)
    parser.add_argument("--train-steps", type=int, required=True)
    # parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument("--use-priors", action="store_true", default=True)
    parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(
        prompt_filepath=args.prompt_file,
        out_rootpath=args.out_path,
        # batch_size=args.batch_size,
        train_steps=args.train_steps,
        use_priors=args.use_priors,
        skip_existing=args.skip_existing,
    )
