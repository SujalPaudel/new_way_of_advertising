import numpy as np
import pandas as pd
import nltk as nl
from numpy import array

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


from subprocess import check_output
print(check_output(["ls", "input"]).decode('utf-8'))

train = pd.read_csv('input/train.csv')
train.head()

train['tokens'] = [nl.word_tokenize(sentences) for sentences in train.text]

words = []
for item in train.tokens:
    words.extend(item)


#stem the sentences
#stemmer is a logic where words such as run, running, runs are changed into one i.e run
stemmer = nl.stem.lancaster.LancasterStemmer() 


words = [stemmer.stem(word) for word in words]
words = set(words) #changes the list into a set 
#type(words)
#set excludes repetition of elements inside it


#create a bag of words
#Apply one hot encoding through all of the datasets
#neural network accepts the label in supervised learning as the one-hot encoded vectors not the entire text

training = []
for index, item in train.iterrows():
    onehot = []
    token_words = [stemmer.stem(word) for word in item['tokens']]
    for w in words:
        onehot.append(1) if w in token_words else onehot.append(0)
        
    training.append([onehot, item['author']])


training_new = np.array(training)
training_new[:, 1]


#onehot encoding to the label

#integer encode

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(training_new[:, 1])

#binary_encode

onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

train_y = onehot_encoded
train_x = list(training_new[:, 0])


#here comes the king, the titan, the rocket scientist, the cool form of uncool before uncool became a thing
#deep neural network from tensorflow highlevel api
#input layer, 2 hidden layers and an output layer 


import tensorflow as tf
import tflearn

#reset underlying graph data

tf.reset_default_graph()

#Build neural network

net = tflearn.input_data(shape=[None,len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

#use softmax as output layer to get the probability of the labels

net = tflearn.fully_connected(net, len(train_y[0]), activation = 'softmax')
net = tflearn.regression(net)

#Define model and setup tensorboard

model = tflearn.DNN(net, tensorboard_dir = 'tflearn_logs')

#start training (apply gradient descent algorithm)

model.fit(train_x, train_y, n_epoch = 10, batch_size = 8, show_metric = True)
model.save('model.tflearn')

test = pd.read_csv('input/test.csv')
test.head()

test['tokens'] = [nl.word_tokenize(sentences) for sentences in test.text]
test.head()

testing = []

for index, item in test.iterrows():
    onehot = []
    token_words = [stemmer.stem(word) for word in item['tokens']]
    for w in words:
        onehot.append(1) if w in token_words else onehot.append(0)
        
    testing.append(onehot)


testing = list(np.array(testing))


predicted = model.predict( X = testing )

result_val = round(pd.DataFrame(predicted), 6)
result_val.columns = ["EAP", "HPL", "HWS"]

result = pd.DataFrame(columns = ['id'])
result['id'] = test['id']

result['EAP'] = result_val['EAP']
result['HPL'] = result_val['HPL']
result['MWS'] = result_val['MWS']

result.head()