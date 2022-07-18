# Caption-Flickr8K

This is a project aims to automatically generate captions for images in Flickr 8k dataset.

The images will be encoded via VGG16, Resnet50 or InceptionV3 in preprocessing. (You may add more kinds of encoders)
Pre-trained encoder will be downloaded if you perform this kind of encoder first time. 
The encoded image vectors will also be saved to save time in the future.

The generating model is constructed by GRU and LSTM.

### Data source link (register required)
https://www.kaggle.com/datasets/adityajn105/flickr8k/download

In default the code will reach to ```./data/``` to find the data.

Please modify the data path in the head of ```./src/preprocessing.py```

## Usage
- Train with LSTM/GRU
    - Valid Encoder option: vgg, resnet, v3
  ```
  python Fit_LSTM.py [Encoder] train 
  python Fit_GRU.py [Encoder] train
  ```
    
- Predict with LSTM/GRU
  
  ```
  python Fit_LSTM.py [Encoder] predict
  python Fit_GRU.py [Encoder] predict
  ```

