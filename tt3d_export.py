### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, Iterator
from pathlib import Path

import argparse
import open3d as o3d
# import trimesh
# import numpy as np

from tt3d_utils import Utils

###

T_Prompt = Tuple[str, Path]  ### pylint: disable=invalid-name
T_Prompts = Iterator[T_Prompt]  ### pylint: disable=invalid-name

# device = Utils.Cuda.init()

###


def _load_prompts_from_source_path(source_rootpath: Path) -> T_Prompts:
    assert isinstance(source_rootpath, Path)
    assert source_rootpath.exists()
    assert source_rootpath.is_dir()

    experiment_path = Utils.Storage.build_experiment_path(rootpath=source_rootpath)

    # for prompt_path in source_path.iterdir():
    for prompt_path in experiment_path.iterdir():
        if prompt_path.is_dir():
            prompt_enc = prompt_path.name
            yield (prompt_enc, prompt_path)


def _convert_pointcloud_to_obj(
    prompt: str,
    source_rootpath: Path,
    skip_existing: bool,
) -> None:
    out_obj_filepath = Utils.Storage.build_prompt_mesh_filepath(
        rootpath=source_rootpath,
        prompt=prompt,
        assert_exists=False,
    )

    if out_obj_filepath.exists():
        if skip_existing:
            print("")
            print("mesh already exists -> ", out_obj_filepath)
            print("")
            return
        else:
            out_obj_filepath.unlink()

    out_obj_filepath.parent.mkdir(parents=True, exist_ok=True)

    #

    out_ply_filepath = Utils.Storage.build_prompt_pointcloud_filepath(
        rootpath=source_rootpath,
        prompt=prompt,
        assert_exists=True,
    )

    ### load the point cloud and the colors.
    pcd = o3d.io.read_point_cloud(str(out_ply_filepath), format='xyzrgb')

    assert not pcd.is_empty()
    assert pcd.has_points()
    assert pcd.has_colors()

    ### compute normals
    pcd.estimate_normals()
    ### to obtain a consistent normal orientation
    pcd.orient_normals_towards_camera_location(pcd.get_center())
    ### you might want to flip the normals to make them point outward, not mandatory
    # pcd.normals = o3d.utility.Vector3dVector( - np.asarray(pcd.normals))

    assert pcd.has_normals()

    ### Open3D

    # o3d_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    # o3d_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=1.1, linear_fit=False)
    o3d_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
    assert not o3d_mesh.is_empty()
    assert o3d_mesh.has_vertices()
    assert o3d_mesh.has_vertex_colors()
    o3d.io.write_triangle_mesh(
        str(out_obj_filepath),
        o3d_mesh,
        write_triangle_uvs=True,
        print_progress=True,
    )

    ### Trimesh

    ### INFO: create the mesh with the vertices and faces FROM OPEN3D
    # tri_mesh = trimesh.Trimesh(
    #     np.asarray(o3d_mesh.vertices),
    #     np.asarray(o3d_mesh.triangles),
    #     vertex_normals=np.asarray(o3d_mesh.vertex_normals),
    #     vertex_colors=np.asarray(o3d_mesh.vertex_colors),
    # )
    # trimesh.exchange.export.export_mesh(
    #     tri_mesh,
    #     str(out_obj_filepath),
    #     include_texture=True,
    # )


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
            _convert_pointcloud_to_obj(
                prompt=prompt,
                source_rootpath=source_rootpath,
                skip_existing=skip_existing,
            )
        except Exception as e:  # pylint: disable=broad-except
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
