### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, Any
from pathlib import Path

import argparse
import torch
import yaml
import os
import subprocess
import sys

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
    skip_existing: bool,
) -> None:
    prompt_enc = Utils.Prompt.encode(prompt)

    tmp_root_path = Path(os.path.join(os.path.dirname(__file__)))
    tmp_output_path = tmp_root_path.joinpath('output')
    tmp_config_path = tmp_output_path.joinpath(f"{prompt_enc}.yaml")

    #

    config = _load_default_config()

    config['GuidanceParams']['text'] = prompt
    # config['GuidanceParams']['negative'] = neg_prompt
    # config['GuidanceParams']['noise_seed'] = seed
    # config['GuidanceParams']['guidance_scale'] = cfg
    config['ModelParams']['workspace'] = prompt_enc
    config['OptimizationParams']['iterations'] = train_steps

    with open(tmp_config_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file)

    #

    try:
        subprocess.check_call([sys.executable, "train.py", "--opt", str(tmp_config_path)])
    except Exception as e:
        print(str(e))


###


def main(
    prompt_filepath: Path,
    out_rootpath: Path,
    # batch_size: int,
    train_steps: int,
    skip_existing: bool,
) -> None:
    assert isinstance(prompt_filepath, Path)
    assert isinstance(out_rootpath, Path)
    # assert isinstance(batch_size, int)
    assert isinstance(train_steps, int)
    assert 0 < train_steps < 10000
    assert isinstance(skip_existing, bool)

    if out_rootpath.exists():
        assert out_rootpath.is_dir()
    else:
        out_rootpath.mkdir(parents=True)

    #

    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)

    print("")
    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        print("")
        print(prompt)

        _generate(
            prompt=prompt,
            out_rootpath=out_rootpath,
            train_steps=train_steps,
            skip_existing=skip_existing,
        )

        print("")
    print("")


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--out-path', type=Path, required=True)
    parser.add_argument("--train-steps", type=str, required=True)
    # parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(
        prompt_filepath=args.prompt_file,
        out_rootpath=args.out_path,
        # batch_size=args.batch_size,
        train_steps=args.train_steps,
        skip_existing=args.skip_existing,
    )
