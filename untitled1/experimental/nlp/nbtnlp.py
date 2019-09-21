import pickle
import numpy as np
import tensorflow
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

with open('train_qa.txt', 'rb') as f:
    train_data = pickle.load(f)

with open('test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)

vocab = set()
for story, question, answer in train_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

vocab.add('no')
vocab.add('yes')

vocab_len = len(vocab) + 1
all_data = test_data + train_data
all_story_lens = [len(data[0]) for data in all_data]
max_story_len = (max(all_story_lens))
max_question_len = max([len(data[1]) for data in all_data])

tokenizer = Tokenizer(filters = [])
tokenizer.fit_on_texts(vocab)

train_story_text = []
train_question_text = []
train_answers = []

for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answers.append(answer)

train_story_seq = tokenizer.texts_to_sequences(train_story_text)


def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len,
                      max_question_len=max_question_len):
    # vectorized stories:
    X = []
    # vectorized questions:
    Xq = []
    # vectorized answers:
    Y = []

    for story, question, answer in data:
        # Getting indexes for each word in the story
        x = [word_index[word.lower()] for word in story]
        # Getting indexes for each word in the story
        xq = [word_index[word.lower()] for word in question]
        # For the answers
        y = np.zeros(len(word_index) + 1)  # Index 0 Reserved when padding the sequences
        y[word_index[answer]] = 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)
    # Now we have to pad these sequences:
    return pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y)


inputs_train, questions_train, answers_train = vectorize_stories(train_data)
inputs_test, questions_test, answers_test = vectorize_stories(test_data)

input_sequence = Input((max_story_len,))
question = Input((max_question_len,))

input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_len,output_dim = 64))
input_encoder_m.add(Dropout(0.3))

input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_len,output_dim = max_question_len))
input_encoder_c.add(Dropout(0.3))

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_len,output_dim = 64,input_length=max_question_len))
question_encoder.add(Dropout(0.3))

input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

match = dot([input_encoded_m,question_encoded], axes = (2,2))
match = Activation('softmax')(match)

response = add([match,input_encoded_c])
response = Permute((2,1))(response)

answer = concatenate([response, question_encoded])
answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_len)(answer)
answer = Activation('softmax')(answer)
model = Model([input_sequence,question], answer)
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#model.summary();

history = model.fit([inputs_train,questions_train],answers_train, batch_size = 32, epochs = 200, validation_data = ([inputs_test,questions_test],answers_test))

filename = 'Z_chatbot_100_epochs.h5'
model.save(filename)
import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.figure(figsize=(12,12))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.load_weights('Z_chatbot_100_epochs.h5')
pred_results = model.predict(([inputs_test,questions_test]))
print(pred_results[0])
val_max = np.argmax(pred_results[0])
for key,val in tokenizer.word_index.items():
    if val == val_max:
        k = key
print(k)

my_story = 'Sandra picked up the milk . Mary travelled left . '

my_story.split()
my_question = 'Sandra got the milk ?'
my_question.split()
my_data = [(my_story.split(), my_question.split(),'yes')]
my_story, my_ques, my_ans = vectorize_stories(my_data)
pred_results = model.predict(([my_story,my_ques]))
val_max = np.argmax(pred_results[0])
for key,val in tokenizer.word_index.items():
    if val == val_max:
        k = key
print(k)