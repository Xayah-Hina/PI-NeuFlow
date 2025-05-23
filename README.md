# PI-NeuFlow

```shell
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install ninja tqdm tensorboard imageio imageio[ffmpeg] tyro pyyaml
cd ./cuda/freqencoder
python -m pip install .
cd ../gridencoder
python -m pip install .
cd ../raymarching
python -m pip install .
cd ../shencoder
python -m pip install .
cd ../..
```