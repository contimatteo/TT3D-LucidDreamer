### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, Any, List, Iterator
from pathlib import Path

import argparse
import torch

from pathlib import Path
from point_e.diffusion.configs import DIFFUSION_CONFIGS
from point_e.diffusion.configs import diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS
from point_e.models.configs import model_from_config
# from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
from point_e.util.pc_to_mesh import marching_cubes_mesh

from utils import Utils
from tt3d_generate import build_sampler

###

T_Prompt = Tuple[str, Path]  ### pylint: disable=invalid-name
# T_Prompts = List[T_Prompt]  ### pylint: disable=invalid-name
T_Prompts = Iterator[T_Prompt]  ### pylint: disable=invalid-name

device = Utils.Cuda.init()

###


def _load_prompts_from_source_path(source_rootpath: Path) -> T_Prompts:
    assert isinstance(source_rootpath, Path)
    assert source_rootpath.exists()
    assert source_rootpath.is_dir()

    experiment_path = Utils.Storage.build_experiment_path(out_rootpath=source_rootpath)

    # for prompt_path in source_path.iterdir():
    for prompt_path in experiment_path.iterdir():
        if prompt_path.is_dir():
            prompt_enc = prompt_path.name
            yield (prompt_enc, prompt_path)


def _convert_pointclouds_to_objs(
    prompt: str,
    source_rootpath: Path,
    pointcloud: PointCloud,
    model: Any,
    skip_existing: bool,
) -> None:
    assert model is not None
    assert isinstance(pointcloud, PointCloud)

    out_ply_filepath = Utils.Storage.build_prompt_mesh_filepath(
        out_rootpath=source_rootpath,
        prompt=prompt,
        assert_exists=False,
        # idx=idx,
        extension="ply",
    )
    out_obj_filepath = Utils.Storage.build_prompt_mesh_filepath(
        out_rootpath=source_rootpath,
        prompt=prompt,
        assert_exists=False,
        # idx=idx,
        extension="obj",
    )

    if skip_existing:
        if out_ply_filepath.exists() and out_obj_filepath.exists():
            print("")
            print("mesh already exists -> ", out_obj_filepath)
            print("")
            return

    out_ply_filepath.parent.mkdir(parents=True, exist_ok=True)
    out_obj_filepath.parent.mkdir(parents=True, exist_ok=True)

    #

    ### produce a mesh (with vertex colors)
    mesh = marching_cubes_mesh(
        pc=pointcloud,
        model=model,
        batch_size=4096,
        # grid_size=32,  # increase to 128 for resolution used in evals
        grid_size=128,
        progress=True,
    )

    with open(out_ply_filepath, 'wb+') as f:
        mesh.write_ply(f)
    with open(out_obj_filepath, 'w+', encoding="utf-8") as f:
        mesh.write_obj(f)


###


def main(source_rootpath: Path, skip_existing: bool) -> None:
    assert isinstance(source_rootpath, Path)
    assert source_rootpath.exists()
    assert source_rootpath.is_dir()
    assert isinstance(skip_existing, bool)

    prompts = _load_prompts_from_source_path(source_rootpath=source_rootpath)

    #

    print("")
    for prompt_enc, _ in prompts:
        prompt = Utils.Prompt.decode(prompt_enc)

        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        print("")
        print(prompt)

        try:
            _convert_pointclouds_to_objs(
                prompt=prompt,
                source_rootpath=source_rootpath,
                skip_existing=skip_existing,
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
    parser.add_argument('--source-path', type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(
        source_rootpath=args.source_path,
        skip_existing=args.skip_existing,
    )
