from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3

import tensorflow as tf

from tqdm import tqdm
from collections import Counter
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

vocab_size = 1300
caption_path = "../data/captions.txt"
all_image_path = "../data/Images/"


def create_vocabulary(captions):
    vocal = []
    for each in captions:
        vocal.extend(each.split(" "))
    return vocal


def vocal_counter(captions):
    vocal = create_vocabulary(captions)
    vocal_counter_dict = Counter(vocal)
    print(len(vocal_counter_dict))


def load_data():
    captions_file = open(caption_path)
    all_img_paths = []
    captions = []
    for line in captions_file.readlines()[1:]:
        if "\"" not in line:
            img_name, caption = line.strip().split(",")
        else:
            img_name, caption = line.strip().split(",", maxsplit=1)
            caption = caption.strip("\"")
        img_path = all_image_path + img_name
        all_img_paths.append(img_path)
        captions.append(caption)

    return all_img_paths, captions


class FlickrPreprocessor:
    def __init__(self, encoder="vgg"):
        self.all_img_path, self.captions = load_data()
        self.word_index_dict = dict()
        self.word_index = list()
        self.caption_vector = list()

        self.encoder = encoder
        self.features_dict = {}
        self.image_vector = list()

    def generate_token_vector(self):
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
        tokenizer.fit_on_texts(self.captions)

        tokenizer.word_index['<pad>'] = 0
        # tokenizer.index_word[0] = '<pad>'

        self.word_index_dict = tokenizer.word_index
        self.word_index = list(self.word_index_dict.keys())
        if "<pad>" in self.word_index:
            self.word_index.remove("<pad>")
            self.word_index.insert(0, "<pad>")

        tokenized_text = tokenizer.texts_to_sequences(self.captions)
        max_length = max(map(lambda x: len(x), tokenized_text))
        self.caption_vector = pad_sequences(tokenized_text, padding='post', maxlen=max_length)

    def extract_image_features(self):
        def load_resize_image(img_path):
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize_images(img, size=(299, 299))
            return img, img_path

        filepath = ("../data/%s_features_dict.npy" % self.encoder)
        if os.path.exists(filepath):
            feature_dict_array = np.load(filepath, allow_pickle=True)
            self.features_dict = feature_dict_array.item()
            print("features size:", len(self.features_dict.keys()))

        else:
            tf.enable_eager_execution()

            unique_image_path_tensor = sorted(set(self.all_img_path))
            image_data_set = tf.data.Dataset.from_tensor_slices(unique_image_path_tensor)
            image_data_set = image_data_set.map(load_resize_image).batch(16)

            print(image_data_set)
            if self.encoder == "vgg":
                image_encoder = VGG16(include_top=False,
                                      weights='imagenet',
                                      pooling='avg',
                                      input_shape=(299, 299, 3))
            elif self.encoder == "resnet":
                image_encoder = ResNet50(include_top=False,
                                         weights='imagenet',
                                         pooling='avg',
                                         input_shape=(299, 299, 3))
            elif self.encoder == "v3":
                image_encoder = InceptionV3(include_top=False,
                                            weights='imagenet',
                                            pooling='avg',
                                            input_shape=(299, 299, 3))
            else:
                raise Exception("Invalid image encoder name: %s" % self.encoder)

            # image_encoder.summary()
            self.features_dict = {}
            for image, path in tqdm(image_data_set):
                features = image_encoder(image)
                for batch_features, p in zip(features, path):
                    path_of_feature = p.numpy().decode("utf-8")
                    self.features_dict[path_of_feature] = batch_features.numpy()

            # save features locally to save time
            np.save(filepath, self.features_dict)

    def generate_image_vector(self):
        self.extract_image_features()
        self.image_vector = list(map(lambda p: self.features_dict[p], self.all_img_path))
        self.image_vector = np.asarray(self.image_vector)
        # print(len(image_vector))

    def generate_dataset(self, data_num=1200):
        self.all_img_path = self.all_img_path[:data_num]
        self.captions = self.captions[:data_num]

        self.generate_image_vector()
        self.generate_token_vector()
        x_train = self.image_vector[:1000]
        x_test = self.image_vector[1000:]
        y_train = self.caption_vector[:1000]
        y_test = self.caption_vector[1000:]
        # x_train, x_test, y_train, y_test = train_test_split(self.image_vector, self.caption_vector, test_size=0.2)

        return [x_train, x_test, y_train, y_test]


if __name__ == '__main__':
    # ImgPaths, Captions = load_data()
    # vocal_counter(Captions[:1000])

    MyFlickrPreprocessor = FlickrPreprocessor(encoder="resnet")
    MyFlickrPreprocessor.generate_image_vector()

    # generate_token_vector(Captions)
