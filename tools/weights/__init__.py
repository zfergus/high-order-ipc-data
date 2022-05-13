from distutils.command.build import build
import sys
import pathlib

build_dir = pathlib.Path(__file__).resolve().parent / "L2_projection" / "build"
if(not build_dir.exists()):
    raise RuntimeError(
        "L2 projection needs to be built first:\ncd tools/weights/L2_projection; mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release; make -j4")

sys.path.append(str(build_dir))

import L2  # noqa
