# cloud-lipreading


### Example notebook: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uDUCD9mDUJ2ghL4XThSidt48-gXKZnbJ)

## Usage
Clone, install dependencies.

`pip install -r requirements.txt`

### Preprocessing
Download GRID dataset with `load_GRID_dataset.sh`.

Use `preprocess.py` script to crop mouth area from videos and save into `.npy` binary files.
```
usage: preprocess.py [-h] [--root ROOT] [--out OUT_ROOT]

Preprocess GRID

optional arguments:
  -h, --help      show this help message and exit
  --root ROOT     Directory where GRID videos are
  --out OUT_ROOT  Directory where numpy arrays are to be cached
```
### Training
Use `train.py` script to train the model. _Training is a bit wonky and untested atm._
```
usage: train.py [-h] [--dataset_path DATASET_PATH] [--aligns_path ALIGNS_PATH]

Train

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Dataset root directory
  --aligns_path ALIGNS_PATH
                        Directory containing all align files
 ```

### Inference
Use `predict.py` script to generate subtitles with a trained model.
```
usage: predict.py [-h] [--video VIDEO_PATH] [--weights WEIGHTS_PATH]
                  [--dict DICT_PATH] [--save]

Predict subtitles

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO_PATH    Video file to predict subs for
  --weights WEIGHTS_PATH
                        Weights of the model
  --dict DICT_PATH      Spellcheck dictionary to use in decoding
  --save                Save video with subtitle overlay
```
Example:
```
python predict.py --video video/test_vid.mp4 --save
```

## Model architecture
![3D conv -> Bi-LSTM -> CTC loss](https://scx2.b-cdn.net/gfx/news/hires/2016/58244eb043a25.jpg)

## How does it work
* Find mouth on frame using MTCNN
* Crop and feed frames to LipNet model
* Get dense predictions
* Decode dense predictions using CTC decoder

![](https://www.electronicproducts.com/uploadedImages/Programming/Software/LipNet_Gif.gif)
