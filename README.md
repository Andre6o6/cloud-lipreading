# cloud-lipreading

Example of MTCNN face detection: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tjm-kv6W-QRLHe9X3JvBl3iMPsP_CFzi)

## Model architecture
![3D conv -> Bi-LSTM -> CTC loss](https://scx2.b-cdn.net/gfx/news/hires/2016/58244eb043a25.jpg)

## How does it work
* Find mouth on frame using MTCNN
* Crop and feed frames to LipNet model
* Get dense predictions
* Decode dense predictions using CTC decoder

![](https://www.electronicproducts.com/uploadedImages/Programming/Software/LipNet_Gif.gif)
