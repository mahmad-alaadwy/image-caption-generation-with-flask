import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import cv2
import pyttsx3

import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


def create_mapping(captions_doc):
    mapping = {}

    for line in captions_doc.split('\n'):
        tokens = line.split(',')
        
        if len(line) < 2:
            continue
            
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        caption = " ".join(caption)
        
        if image_id not in mapping:
            mapping[image_id] = []

        mapping[image_id].append(caption)
    return mapping


def clean(mapping):
	for key, captions in mapping.items():
		for i in range(len(captions)):
             
			caption = captions[i]
               
			caption = caption.lower()
               
			caption = caption.replace('[^A-Za-z]', '')
               
			caption = caption.replace('\s+', ' ')
                
			caption = 'start  ' + " ".join([word for word in caption.split() if len(word)>1]) + '  end'
			captions[i] = caption


def run():
        
    model = VGG16()

    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)


    features = {}
    with open('features.pkl', 'rb') as f:features = pickle.load(f)

    with open('captions.txt', 'r') as f:
        next(f)
        captions_doc = f.read()


    mapping = {}
    mapping = create_mapping(captions_doc)
    clean(mapping)

    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
            

    # tokenizer is used to convert the words to numerical values 
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    vocab_size

    max_length = max(len(caption.split()) for caption in all_captions)

    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90)
    train = image_ids[:split]
    test = image_ids[split:]

    # encoder model
    # making a dense layer with random dropout 0.4 from the original input of shape
    # 4096
    # encoder model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.4)(inputs1)
    # the fe2 represents the images features
    fe2 = Dense(256, activation='relu')(fe1)

    # sequence feature layers
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)


    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    
        # for the five ebochs  model
    import tensorflow as tf

    model =tf.keras.models.load_model('best_model.h5')

    return model,tokenizer,max_length


# mapping = create_mapping()

# clean(mapping)

# # this function seeks to generat a matrix of image ids ,caption as numerical 
# # ,one hot  encoding of the words
# # and will spilit them as ebochs
# def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
  
# 	X1, X2, y = list(), list(), list()
# 	n = 0
# 	while 1:
# 		for key in data_keys:
# 			n += 1
# 			captions = mapping[key]
# 			for caption in captions:
# 				seq = tokenizer.texts_to_sequences([caption])[0]
# 				for i in range(1, len(seq)):
# 					in_seq, out_seq = seq[:i], seq[i]
# 					in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
# 					out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
# 					X1.append(features[key][0])
# 					X2.append(in_seq)
# 					y.append(out_seq)
# 			if n == batch_size:
# 				X1, X2, y = np.array(X1), np.array(X2), np.array(y)
# 				yield [X1, X2], y
# 				X1, X2, y = list(), list(), list()
# 				n = 0
		
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
        	return word
    
def predict_caption(model, image, tokenizer, max_length):
	in_text = 'start'

	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], max_length)
		text = model.predict([image, sequence], verbose=0)
		text = np.argmax(text)
		word = idx_to_word(text, tokenizer)
		if word is None:
			break
		in_text += " " + word
		if word == 'end':
			break

	return in_text


def predict_caption__():
    vgg_model = VGG16() 
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

    image_path = 'checkimg.jpg'

    image1 = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image1)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    feature = vgg_model.predict(image, verbose=0)

    model,tokenizer,max_length=run()
    x=predict_caption(model, feature, tokenizer, max_length)
    captionLast=x.split()
    captionLast.pop(0)
    captionLast.pop()

    captionLast=" ".join(captionLast)
    return captionLast

# Create flask app
flask_app = Flask(__name__)

def cuptuer():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("cupturing the image")
    img_counter = 0
    img_taked=True
    while img_taked:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("cupturing the image", frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "checkimg.jpg"
            cv2.imwrite(img_name, frame)
            break
    cam.release()
    cv2.destroyAllWindows()

def say_text(command,text=" "):
    engine=pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(command)
    engine.say(text)
    engine.runAndWait()

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/whatItIs", methods = ["POST"])
def whatItIs():
    cuptuer()
    text=predict_caption__()
    say_text("the captured image contains a ",text)

    return render_template("index.html")

if __name__ == "__main__":
    flask_app.run(debug=True)