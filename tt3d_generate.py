### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, Any
from pathlib import Path

import argparse
import torch

from utils import Utils

###

device = Utils.Cuda.init()

###


def main(prompt_filepath: Path, out_rootpath: Path, batch_size: int, skip_existing: bool) -> None:
    assert isinstance(prompt_filepath, Path)
    assert isinstance(out_rootpath, Path)
    assert isinstance(batch_size, int)
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

        # _generate_latents(
        #     prompt=prompt,
        #     out_rootpath=out_rootpath,
        #     sampler=sampler,
        #     skip_existing=skip_existing,
        #     batch_size=batch_size,
        # )
        print("")
    print("")


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--out-path', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(
        prompt_filepath=args.prompt_file,
        out_rootpath=args.out_path,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
    )
