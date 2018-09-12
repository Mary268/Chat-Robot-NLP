
import os
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import callbacks
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras import backend as K
from keras.utils import plot_model 
from IPython.display import Image
import seaborn as sns
import matplotlib.pyplot as plt
#from nltk.translate import bleu_score
from BLEU_evaluation import BLEU_EVA
np.random.seed(17)

DIALOGUE_FILE = 'ijcnlp_dailydialog/dialogues_text.txt'
EMB_DIR = 'glove.6B/glove.6B.300d.txt'
MAX_VOCAB_SIZE = 15000
MAX_Q_LEN = 50
MAX_A_LEN = 10
EMBEDDING_DIM = 300
p = 85.0
BATCH_SIZE = 256
LSTM_DIM = 300
N_EPOCHS = 100
Read_EMB = 1
TRAIN_BOT = 0

weights_file = 'saved_models/WordEmbedding_based_150epochs_final.h5'
CANDI_FILE = 'saved_models/bleu_candi_s2s_qas.txt'
REFER_FILE = 'saved_models/bleu_renfer_s2s_qas.txt'
BLEU_FILE = 'saved_models/bleu_scores_word_c_r.txt'

# -----------------------  Data Pre-processing  -------------------------------
with open(DIALOGUE_FILE, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

questions, answers = [], []
qas = []
for dialogue in lines:
    qa = dialogue.split('__eou__')
    for i in range(len(qa)-2):
        qas.append(qa[i])
        if i%2==0: # different Q-A Structure
            continue
        questions.append(qa[i])
        answers.append(qa[i+1])

a_list = questions
q_list = answers
qas_list = qas
print(' Dialogues :{} % of the questions/answers have a length <= {}'
      .format(p, np.percentile([len(q_i) for q_i in qas_list], p)))

filter_list = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
qas_tokenizer = Tokenizer(filters=filter_list)  
qas_tokenizer.fit_on_texts(qas_list)
print("Number of words in Dialugues vocabulary:", len(qas_tokenizer.word_index))

def ids_to_sentence(gen_ids):
    pad = qas_word_index['PAD']
    eos = qas_word_index['EOS']
    #unk = qas_word_index['UNK']
    #gen_ids = [w for w in gen_ids if w not in [pad,eos]]
    return ' '.join([qas_idx_to_word[w] for w in gen_ids if w not in [pad, eos]])

def sentence_to_ids(sent):
    idxs = []
    for w in sent:
        try:
            idx = qas_word_index[w]
        except KeyError:
            # if the words not in |V|, tag as 'UNK' unknow words
            idx = 1
        idxs.append(idx)
    #idxs =  [qas_word_index[w] for w in sent]
    
    return idxs #[qas_word_index[w] for w in sent]

qas_word_index = {}
qas_word_index['PAD'] = 0
qas_word_index['UNK'] = 1
qas_word_index['EOS'] = 2
qas_word_index['SOS'] = 3

for i, word in enumerate(dict(qas_tokenizer.word_index).keys()):
    qas_word_index[word] = i+4 # Move existing indices up by 4 places

X = qas_tokenizer.texts_to_sequences(q_list)
# Replace OOV words with UNK token
# Append EOS to the end of all sentences
for i, seq in enumerate(X):
    seq = [s+3 for s in seq]
    # if the index is larger than MAX_VOCAB_SIZE, then tag as UNK
    if any(t>=MAX_VOCAB_SIZE for t in seq):
        seq = [t if t<MAX_VOCAB_SIZE else qas_word_index['UNK'] for t in seq ]

    seq.append(qas_word_index['EOS'])
    X[i] = seq  
# Padding and truncating sequences
X = pad_sequences(X, padding='post', truncating='post', maxlen=MAX_Q_LEN, value=qas_word_index['PAD'])

Y = qas_tokenizer.texts_to_sequences(a_list)
for i, seq in enumerate(Y):
    seq = [s+3 for s in seq]
    if any(t>=MAX_VOCAB_SIZE for t in seq):
        seq = [t if t<MAX_VOCAB_SIZE else qas_word_index['UNK'] for t in seq ]

    seq.append(qas_word_index['EOS'])
    Y[i] = seq    
Y = pad_sequences(Y, padding='post', truncating='post', maxlen=MAX_A_LEN, value=qas_word_index['PAD'])

# Finalize the dictionaries
qas_word_index = {k: v for k, v in qas_word_index.items() if v < MAX_VOCAB_SIZE} 
qas_idx_to_word = dict((i, word) for word, i in qas_word_index.items()) 
#----------------
# ### Train-Validation-Test Splits
X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=0.05)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

# Load GloVe word embeddings
print("[INFO]: Reading Word Embeddings ...")

if Read_EMB == 1:  
    # Data path
    embeddings = {}
    f = open(EMB_DIR)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings[word] = vector
    f.close()

encoder_embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(qas_word_index), EMBEDDING_DIM)) 

for word, i in qas_word_index.items(): # i=0 is the embedding for the zero padding
    try:
        embeddings_vector = embeddings[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        encoder_embeddings_matrix[i] = embeddings_vector

print('encoder_embeddings_matrix.shape:')
print(encoder_embeddings_matrix.shape)

def batch_generator(X, Y, BATCH_SIZE):  
    while True:
        
        batch_start = 0
        batch_end = BATCH_SIZE

        while batch_start < len(X):
            
            [X_enc, X_dec], Y_dec = prepare_data(X, Y, batch_start, batch_end)
            
            yield ([X_enc, X_dec], Y_dec) 

            batch_start += BATCH_SIZE   
            batch_end += BATCH_SIZE


def prepare_data(X, Y, batch_start, batch_end):
    # Encoder input of certain batch
    X_enc = X[batch_start:batch_end] 
    
    # Decoder input:
    # Set 'SOS' at the beginning of decoder and remove the last element to keep length
    X_dec = np.c_[3 * np.ones([len(Y[batch_start:batch_end])]), Y[batch_start:batch_end, :-1]]
    
    # Decoder output:one hot encodeding for vocabulary (15000): softmax or multinomial logestic regression layer
    Y_dec = np.array([to_categorical(y, num_classes=MAX_VOCAB_SIZE) for y in Y[batch_start:batch_end]])
    return [X_enc, X_dec], Y_dec

def prepare_train_data(X_train, Y_train):
    
    # Encoder input
    X_enc = X_train
    # Decoder input
    X_dec = Y_train
    # decoder_target_data is ahead of decoder_input_data by one timestep: Teacher Forcing
    Y_dec = np.append(np.delete(Y_train,0,1), 
                      np.zeros((len(Y_train),1), dtype='int64'), axis=1)
    # Decoder output - one hot encoded for softmax layer - 1 in |V|
    Y_dec = np.array([to_categorical(y, num_classes=MAX_VOCAB_SIZE) for y in Y_dec])
    
    return [X_enc, X_dec], Y_dec


K.set_learning_phase(1) # 1 for training, 0 for inference time

# Encoder Setup

enc_input = Input(shape=(MAX_Q_LEN, ), name='encoder_input')
enc_emb_look_up = Embedding(input_dim=MAX_VOCAB_SIZE,
                             output_dim=EMBEDDING_DIM,
                             weights = [encoder_embeddings_matrix], 
                             trainable=False, 
                             mask_zero=True,
                             name='encoder_embedding')

enc_emb_text = enc_emb_look_up(enc_input)

# Latent State of Encoder
encoder_lstm = LSTM(LSTM_DIM, return_state=True, name='encoder_lstm', dropout=0.2) 
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb_text)
# The final Latent states:
# state_h with latent dimension == 300
# state_c with dimension == Batch Size
encoder_states = [state_h, state_c] 

# Decoder Setup: None Shape for one-token per step prediction without limitation of capacity
dec_input = Input(shape=(None, ), name='decoder_input') # Specify None instead of MAX_A_LEN

dec_emb_look_up = Embedding(input_dim=MAX_VOCAB_SIZE,
                             output_dim=EMBEDDING_DIM,
                             weights = [encoder_embeddings_matrix], 
                             trainable=False, 
                             mask_zero=True,
                             name='decoder_embedding')

dec_emb_text = dec_emb_look_up(dec_input)

decoder_lstm = LSTM(LSTM_DIM, return_sequences=True, return_state=True, name='decoder_lstm', dropout=0.2)

# Hidden state initialization using `encoder_states` as initial state, Eg: 'SOS'
decoder_outputs, _, _ = decoder_lstm(dec_emb_text,
                                     initial_state=encoder_states)
# Softmax dense/ Multinomial Logistic Regression of word for decoder_outputs 
decoder_dense = Dense(MAX_VOCAB_SIZE, activation='softmax', name='output_layer')
decoder_outputs = decoder_dense(decoder_outputs)

# enc_input: question
# dec_input: answer with 'SOS' tag at the begining
# decoder_outputs: soft max probability of decoded specific word from vocabuary
model = Model([enc_input, dec_input], decoder_outputs)

model.summary()
plot_model(model, to_file='saved_models/Enc_Dec_Glove_BeamSearch_model.png', show_layer_names=True)
Image('Enc_Dec_Glove_BeamSearch_model.png')

# load previous trained model if the file exist
if os.path.isfile(weights_file):
    print('----------Load Previous Trained Model--------')
    model.load_weights(weights_file)

print('-------------- Training -----------------')
# Set optimizer and loss function 
optimizer = keras.optimizers.Adam(lr=0.001)

loss = 'categorical_crossentropy'

filepath="saved_models/seq2seq_train_XY_{epoch:02d}_{val_loss:.4f}.h5"
checkpoint = callbacks.ModelCheckpoint(filepath, 
                                       monitor='val_loss', 
                                       verbose=0, 
                                       save_best_only=False)

callbacks_list = [checkpoint]

model.compile(optimizer=optimizer, loss=loss)

STEPS_PER_EPOCH = len(X_train)//BATCH_SIZE
VAL_STEPS = len(X_val)//BATCH_SIZE

if TRAIN_BOT == 1:
    model.fit_generator(batch_generator(X_train, Y_train, BATCH_SIZE), 
                        steps_per_epoch = STEPS_PER_EPOCH, 
                        epochs = N_EPOCHS,
                        validation_data = batch_generator(X_val, Y_val, BATCH_SIZE), 
                        validation_steps = VAL_STEPS, 
                        callbacks = callbacks_list
                       )
if TRAIN_BOT == 2:
    # Not appropreate: without generator, there will be memory overflow
    [X_enc, X_dec], Y_dec = prepare_train_data(X_train, Y_train)
    [X_enc_val, X_dec_val], Y_dec_val = prepare_train_data(X_val, Y_val)
    
    model.fit([X_enc, X_dec], Y_dec, 
                        #steps_per_epoch = STEPS_PER_EPOCH, 
                        epochs = N_EPOCHS,
                        validation_data = ([X_enc_val, X_dec_val], Y_dec_val), 
                        #validation_steps = VAL_STEPS, 
                        callbacks = callbacks_list
                       )

#---------  Visulization of latent Vector / Attention of Hidden State ---------
output_layer_vs = model.get_layer('output_layer')
# Latent Variable with length of 300 for |V| = 15000
dense_latent = output_layer_vs.get_weights()[0]
dense_latent.shape
# Out[123]: (300, 15000)

#Hidden State with index == 1 with softmax over |V| = 15000
output_layer_vs.get_weights()[0][1]

latent_n = 40
words_n = 40
p_all =  np.zeros((latent_n,words_n), dtype = float) #[[0 for i in range(latent_n)] for j in range(words_n)]
# Get first 10 first latent variable for visualization
for latent in range(latent_n):
    p = []
    p.append(dense_latent[latent][4:4+words_n])
    p_all[latent]=p[0]

ax = sns.heatmap(p_all)
ax.set(xlabel=ids_to_sentence(
        [4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22])+'..',
       ylabel='Latent Index')
plt.title("Latent Vector Visulization")
plt.show()

#-----------------------  Evaluation / Testing --------------------------------
print('-------------- Testing -----------------')
# Decoder target is ahead of decoder input by one timestep:
# Prediction at timestep t is used as input data for timestep (t+1)
#def encoder_decoder_init():
encoder_model = Model(enc_input, encoder_states)
encoder_model.summary()
plot_model(model, to_file='saved_models/Enc_Dec_Glove_BeamSearch_encoder.png', show_layer_names=True)
Image('Enc_Dec_Glove_BeamSearch_encoder.png')

decoder_state_input_h = Input(shape=(LSTM_DIM,), name='decoder_state_input_h')
decoder_state_input_c = Input(shape=(LSTM_DIM,), name='decoder_state_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(inputs=dec_emb_text, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(inputs = [dec_input] + decoder_states_inputs,
                      outputs = [decoder_outputs] + decoder_states
                     )

decoder_model.summary()
plot_model(model, to_file='saved_models/Enc_Dec_Glove_BeamSearch_decoder.png', show_layer_names=True)
Image('Enc_Dec_Glove_BeamSearch_decoder.png')

# Use Pre-trained model to set weights for decoding/prediction
model_weight_list = model.get_weights()

encoder_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9] # Index 1 is the decoder-side embedding matrix weights
encoder_weights = [model_weight_list[idx] for idx in encoder_indices]
encoder_model.set_weights(encoder_weights)

decoder_indices = list(set(range(len(model_weight_list))) - set(encoder_indices))
decoder_weights = [model_weight_list[idx] for idx in decoder_indices]
decoder_model.set_weights(decoder_weights)

# Set testing stage (set0 for inference and set 1 is training)
K.set_learning_phase(0)

#encoder_decoder_init()

# Decoder for answers based on encoded inputs
def decode_test_sequence(test_seq_input):
    ids_to_sentence(test_seq_input[0])
    # Retrieve the encoder output states, h and c
    states_value = encoder_model.predict(test_seq_input)

    # Initialize the decoding process
    time_step = 0
    stop_condition = False
    input_seq_decoder = [qas_word_index['SOS']] # Start with SOS token
    output_seq = [] 

    while not stop_condition:

        output_probs, h, c = decoder_model.predict([np.array([input_seq_decoder]), 
                                                 states_value[0], 
                                                 states_value[1]] )

        predicted_token = np.argmax(output_probs, axis=-1)[0][time_step]
        output_seq.append(predicted_token)

        time_step += 1

        # Stop iteration either when EOS is reached or reach the last timestep
        if (predicted_token == qas_word_index['EOS'] or time_step > MAX_A_LEN - 1): 
                stop_condition = True
        else:
            # Create the inputs for the next iteration
            input_seq_decoder.append(predicted_token)

            states_value[0] = h
            states_value[1] = c
    
    return output_seq

#------------------------ Beam Search Decoder ----------------------------------------
def Beam_Search(test_seq_input):
    ids_to_sentence(test_seq_input[0])
    # Retrieve the encoder output states, h and c
    states_value = encoder_model.predict(test_seq_input)

    # Initialize the decoding process
    time_step = 0
    stop_condition = False
    input_seq_decoder1 = [qas_word_index['SOS']] # Start with SOS token
    input_seq_decoder2 = [qas_word_index['SOS']] 
    
# ---- Beam Search for 1(frist) step: number of candidate words to consider per step: 3;
    #output_seq = [] 
    output_seq1 = [] 
    output_seq2 = []
    p1 = 1
    p2 = 1
    words_p_step = 2
    words_n =[]
    
    output_probs, h_i, c_i = decoder_model.predict([np.array([input_seq_decoder1]), 
                                                 states_value[0], 
                                                 states_value[1]] )
    # Decode words_p_step = 3 tokens with maximum prob for Beam Search
    s = pd.Series(output_probs[0][0])
    words_n.append(pd.Series(s.nlargest(words_p_step)))
    # find maximuum from last dimension: |V| == 15000
    predicted_token_12 = pd.Series(output_probs[0][time_step]).nlargest(2)
    output_seq1.append(predicted_token_12.keys()[0])
    output_seq2.append(predicted_token_12.keys()[1])
    
    p1 = p1 * predicted_token_12.get_values()[0]
    p2 = p2 * predicted_token_12.get_values()[1]
    input_seq_decoder1.append(predicted_token_12.keys()[0])
    input_seq_decoder2.append(predicted_token_12.keys()[1])
    states_value[0] = h_i
    states_value[1] = c_i
# ----- Beam Search for 1(frist) step: number of candidate words to consider per step: 3;
# ----- Beam Search for the first:  
    time_step = 1
    while not stop_condition:
        output_probs, h, c = decoder_model.predict([np.array([input_seq_decoder1]), 
                                                 states_value[0], 
                                                 states_value[1]] )
        predicted_token = np.argmax(output_probs, axis=-1)[0][time_step]
        output_seq1.append(predicted_token)
        # Update the probability of the sentence
        p1 = p1 * np.max(output_probs, axis=-1)[0][time_step]
        time_step += 1
        # Stop iteration either when EOS is reached or when we reach the last timestep
        # Timestep is from 0 to 4. Max allowed is 4
        if (predicted_token == qas_word_index['EOS'] or time_step > MAX_A_LEN - 1): 
                stop_condition = True
        else:
            # Create the inputs for the next iteration
            input_seq_decoder1.append(predicted_token)
            states_value[0] = h
            states_value[1] = c

# ----- Beam Search for the Second:
    states_value[0] = h_i
    states_value[1] = c_i
    time_step = 1
    while not stop_condition:
        output_probs, h, c = decoder_model.predict([np.array([input_seq_decoder2]), 
                                                 states_value[0], 
                                                 states_value[1]] )
        predicted_token = np.argmax(output_probs, axis=-1)[0][time_step]
        output_seq2.append(predicted_token)
        # Update the probability of the sentence
        p2 = p2 * np.max(output_probs, axis=-1)[0][time_step]
        time_step += 1
        # Stop iteration either when EOS is reached or when we reach the last timestep
        # Timestep is from 0 to 4. Max allowed is 4
        if (predicted_token == qas_word_index['EOS'] or time_step > MAX_A_LEN - 1): 
                stop_condition = True
        else:
            # Create the inputs for the next iteration
            input_seq_decoder2.append(predicted_token)
            states_value[0] = h
            states_value[1] = c
    
    print(p1)
    print(p2)
    print(ids_to_sentence(output_seq1))
    print(ids_to_sentence(output_seq2))
    
    # Return the sentence with higher probability
    if p1 >= p2:
        return output_seq1
    else:
        return output_seq2

# -----------------------  Evaluation of Model: BLUE 1-4 grams  ---------------
def BLEU_evaluation(BeamSearch):
    f_candi = open(CANDI_FILE,'w',encoding='utf-8')
    f_refer = open(REFER_FILE,'w',encoding='utf-8')
    f_bleus = open(BLEU_FILE,'w',encoding='utf-8')
    references = []
    candidates = []
    bleu_n_s_list = []
    
    for idx_test in range(len(X_test)-1):
        encoded_question = X_test[idx_test: idx_test + 1]
        if BeamSearch == 0: # Odinary Decoder
            decoded_answer = ids_to_sentence(decode_test_sequence(encoded_question))
        elif BeamSearch == 1: # Beam Search Decoder with two cadidant in the first timestep
            decoded_answer = ids_to_sentence(Beam_Search(encoded_question))
        quest = ids_to_sentence(encoded_question[0])
        refer = ids_to_sentence(Y_test[idx_test]) #Y_test[idx_test].replace("\t", "").replace("\n", "")
        _, bleu_n_s = BLEU_EVA(refer, decoded_answer)
        bleu_n_s_list.append(bleu_n_s)
        if bleu_n_s[1] > 0:
            print('-')
            print('Question:', quest)
            print('Answer:', decoded_answer)
            print('Reference:', refer)
            print('BLEU 1-4 of %d test sample: '%idx_test, end='')
            print(bleu_n_s)
        f_bleus.write('Question :' + quest+'\n'+
                      'Answer   :'+ decoded_answer+'\n'+
                      'Reference:'+ refer+'\n'+
                      'BLEU 1-4 :'+ str(bleu_n_s)+'\n'+
                      '----------\n')
        f_refer.write(refer+'\n')
        f_candi.write(decoded_answer+'\n')
        references.append(refer)
        candidates.append(decoded_answer)
    
    
    bleu_1, bleu_2, bleu_3, bleu_4 = [], [], [] ,[]
    for bleu_n_s in bleu_n_s_list:
        bleu_1.append(bleu_n_s[0])
        bleu_2.append(bleu_n_s[1])
        bleu_3.append(bleu_n_s[2])
        bleu_4.append(bleu_n_s[3])
    
    bleu_n_mean = []
    bleu_n_mean.append(np.mean(bleu_1))
    bleu_n_mean.append(np.mean(bleu_2))
    bleu_n_mean.append(np.mean(bleu_3))
    bleu_n_mean.append(np.mean(bleu_4))
    print('--------OVERALL BLEU 1 - 4 on unseen test data -------')
    print(bleu_n_mean)
    f_bleus.write('--------OVERALL BLEU 1 - 4 on unseen test data -------'+'\n'+
                  str(bleu_n_mean))
    f_candi.close()
    f_refer.close()
    f_bleus.close()
    
    return bleu_n_mean

BLEU_evaluation(0)

#------------------------ Conversation Demo------------------------------------
# For common usages sentence:
GREETING_KEYWORDS = ("hello", "hi", "greetings", "how are you", "what's up",)
GREETING_RESPONSES = ["'hey", "hi", "hello, I am good", "I am good, Thanks you"]
def check_for_greeting(sentence):
    #for word in sentence.split(' '):
    if sentence.lower() in GREETING_KEYWORDS:
        print(np.random.choice(GREETING_RESPONSES))
        return True
    else:
        return False

def chat():
    question = 'None'
    decoded_answer = ''
    # Chat with you until you type: 'bye'
    while question < 'bye' or question > 'bye':
        question = input('You: ')
        if len(question) == 0:
            print('if you want to quit, type \'bye\'')
            continue
        elif question == 'bye':
            print ('Chatbot: ' + 'bye bye')
            break
        
        if(check_for_greeting(question)):
            continue
        
        # Encode question you want to chat with
        question = 'what are we going for dinner ?'
        tokens = question.split(' ')
        ids = sentence_to_ids(tokens)
        #print(ids)
        #print(ids_to_sentence([i for i in ids]))
        
        convs = qas_tokenizer.texts_to_sequences([question])
        # Replace Unknow words with UNK token
        # Append EOS to the end of all sentences
        for i, seq in enumerate(convs):
            if any(t>=MAX_VOCAB_SIZE for t in seq):
                seq = [t if t<MAX_VOCAB_SIZE else qas_word_index['UNK'] for t in seq ]
            #seq.append(qas_word_index['EOS'])
            convs[i] = seq
        # Padding and truncating sequences
        convs = pad_sequences(convs, padding='post', truncating='post', maxlen=MAX_Q_LEN, value=qas_word_index['PAD'])
        
        # Decode answer that the Chatbot will reply
        print(ids_to_sentence(decode_test_sequence(convs[0:1]) ))

chat()


print('---------- model ------------')
#model.summary()
plot_model(model, to_file='pic/Enc_Dec_Glove_BeamSearch_model.png', show_layer_names=True)
Image('pic/Enc_Dec_Glove_BeamSearch_model.png')

print('---------- encoder_model ------------')
#encoder_model.summary()
plot_model(encoder_model, to_file='pic/Enc_Dec_Glove_BeamSearch_encoder.png', show_layer_names=True)
Image('pic/Enc_Dec_Glove_BeamSearch_encoder.png')

print('---------- decoder_model ------------')
#decoder_model.summary()
plot_model(decoder_model, to_file='pic/Enc_Dec_Glove_BeamSearch_decoder.png', show_layer_names=True)
Image('pic/Enc_Dec_Glove_BeamSearch_decoder.png')

