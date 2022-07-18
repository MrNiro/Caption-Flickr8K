from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, GRU, add
from tensorflow.keras.optimizers import Adam
from nltk.translate.bleu_score import sentence_bleu

from preprocessing import *
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class GRUCaption:
    def __init__(self, encoder, if_load=True):
        self.flickr_processor = FlickrPreprocessor(encoder=encoder)
        dataset_list = self.flickr_processor.generate_dataset()
        self.x_train, self.x_test, self.y_train, self.y_test = dataset_list

        self.vocab_size = vocab_size
        self.batch_size = 4

        self.model = self.setup_model()
        if if_load:
            self.model.load_weights("model-v3-gru-best.h5")
        self.model.summary()

    def setup_model(self):
        middle_units = 512

        input_1 = Input(shape=(2048, ))
        dropout_1 = Dropout(0.2)(input_1)
        dense_1 = Dense(middle_units, activation='relu')(dropout_1)

        input_2 = Input(shape=(self.y_train.shape[1], ))
        embedding_1 = Embedding(self.vocab_size, middle_units)(input_2)
        dropout_2 = Dropout(0.2)(embedding_1)
        gru_1 = GRU(middle_units)(dropout_2)

        add_1 = add([dense_1, gru_1])
        dense_2 = Dense(middle_units, activation='relu')(add_1)
        dense_3 = Dense(self.vocab_size, activation='softmax')(dense_2)

        return Model(inputs=[input_1, input_2], outputs=dense_3)

    def training(self):
        def data_generator(batch_size):
            while True:
                x1, x2, y = list(), list(), list()      # input_1 & input_2 & label
                max_length = self.y_train.shape[1]
                n = 0
                for idx, caption_token in enumerate(self.y_train):
                    n += 1
                    for j in range(1, max_length):
                        curr_sequence = caption_token[0:j]
                        next_word = caption_token[j]
                        curr_sequence = pad_sequences([curr_sequence], maxlen=max_length, padding='post')[0]
                        one_hot_next_word = to_categorical([next_word], self.vocab_size)[0]
                        x1.append(self.x_train[idx])
                        x2.append(curr_sequence)
                        y.append(one_hot_next_word)
                        if next_word == 0:
                            break
                    if n == batch_size:
                        yield [[np.array(x1), np.array(x2)], np.array(y)]
                        x1, x2, y = list(), list(), list()
                        n = 0

        optimizer = Adam(lr=1e-4, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        generator = data_generator(self.batch_size)
        for i in range(0, 10):
            self.model.fit_generator(generator, epochs=100,
                                     steps_per_epoch=len(self.y_train) // self.batch_size, verbose=2)
            model_file_name = ("model-resnet-gru-0503-256-epoch%d.h5" % (100*(i+1)))
            self.model.save(model_file_name)

    def predicting(self):
        test_image_path = self.flickr_processor.all_img_path[::5]
        test_captions = self.flickr_processor.captions[1000:]

        whole_score = 0
        for i in range(0, 40):
            img_feature = np.array([self.x_test[i*5]])
            in_text = ''
            max_length = self.y_train.shape[1]
            for idx in range(max_length):
                sequence = [self.flickr_processor.word_index_dict[s]
                            for s in in_text.split(" ") if s in self.flickr_processor.word_index]
                sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
                y_pred = self.model.predict([img_feature, np.array(sequence)], verbose=0)
                pred_word_idx = np.argmax(y_pred[0])
                word = self.flickr_processor.word_index[pred_word_idx]
                if word == '<pad>':
                    break
                in_text += ' ' + word

            print("\nTest image %d:" % i)

            print("\tPredicting result: \n\t\t%s" % in_text)
            print("\tCaptions: ")

            # calculating BLEU score
            true_token_set = []
            for cap in test_captions[i*5: i*5+5]:
                true_token_set.append(cap.lower().split(" "))
                print("\t\t%s" % cap)
            score = sentence_bleu(true_token_set, in_text.split())
            print(score)
            whole_score += score

            img = plt.imread(test_image_path[i])
            plt.title(in_text)
            plt.imshow(img)
            plt.show()
            plt.pause(0)
        print("\nAverage score =", whole_score / 40)


if __name__ == '__main__':
    encoder = "resnet"
    if len(sys.argv) > 1:
        encoder = sys.argv[1]

    if len(sys.argv) > 2:
        if sys.argv[2] == "train":
            myCaption = GRUCaption(encoder=encoder, if_load=False)
            myCaption.training()
        elif sys.argv[2] == "predict":
            myCaption = GRUCaption(encoder=encoder, if_load=True)
            myCaption.predicting()
        else:
            Exception("Invalid Option! Please indicate \"train\" or \"predict\" in the command.")
    else:
        Exception("Missing Option! Please indicate \"train\" or \"predict\" in the command.")
