# PI-NeuFlow

```shell
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install ninja tqdm tensorboard imageio imageio[ffmpeg] tyro pyyaml av pytest scipy pyglet<2 trimesh
cd ./src/cuda_extensions/freqencoder
python -m pip install .
cd ../gridencoder
python -m pip install .
cd ../raymarching
python -m pip install .
cd ../shencoder
python -m pip install .
cd ../ffmlp
python -m pip install .
cd ../../..
```