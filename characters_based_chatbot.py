from __future__ import print_function
from keras.models import Model #, load_model
from keras.layers import Input, LSTM, Dense
import numpy as np
import os
#from matplotlib import pyplot
from keras.utils import plot_model 
from IPython.display import Image
from sklearn.model_selection import train_test_split
from keras import callbacks, optimizers
from BLEU_evaluation import BLEU_EVA

DIALOGUE_FILE = 'ijcnlp_dailydialog/dialogues_text.txt'
Q_A_FILE = 'ijcnlp_dailydialog/dialogues_text_Q_and_A.txt'
saved_model = 'saved_models/Character_based_200epochs_0.8223loss_final.h5'
saved_encoder_model = 'saved_models/characters_encoder.model'
saved_decoder_model = 'saved_models/characters_decoder.model'
CANDI_FILE = 'saved_models/bleu_candi_charcter.txt'
REFER_FILE = 'saved_models/bleu_renfer_character.txt'
BLEU_FILE = 'saved_models/bleu_scores_character_c_r.txt'

# Traing config
batch_size = 128  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding and decoding space.
max_words = 60
train_bot = 0

# -----------------------  Data Pre-processing  -------------------------------
with open(DIALOGUE_FILE, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

questions, answers = [], []
for dialogue in lines:
    qa = dialogue.split('__eou__')
    for i in range(len(qa)-2):
        if i%2==0: # different Q-A Structure
            continue
        if max(len(qa[i]),len(qa[i+1])) > max_words: # 10 words per q and a
            qa[i] = qa[i][1:max_words]
            qa[i+1] = qa[i+1][1:max_words]
        questions.append(qa[i])
        answers.append(qa[i+1])

    
q_lens = [len(txt) for txt in questions]
a_lens = [len(txt) for txt in answers]

q_lens_sorted = sorted(q_lens)
a_lens_sorted = sorted(a_lens)

#Visualize distribution of len of Q and A
#pyplot.xticks([0,14,29,49,69,89,109,129,149,166])
#pyplot.hist(q_lens_sorted[0:len(q_lens)-8000 ])

## store Q and A in one file
with open(Q_A_FILE,'w',encoding='utf-8') as f:
    for i in range(len(questions)):
        f.write(questions[i] + '\t' + answers[i] + '\n')
    f.write('' + '\t' + questions[0] + '\n')
    f.write(answers[len(answers)-1] + '\t' + '' + '\n')
    f.close()

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(Q_A_FILE, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[0: len(lines)-2]:
#for line in lines[2000: 22000]:
    input_text, target_text = line.split('\t')
    # We use "tab" ato separate Q and A
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# Dict: from char => index
input_char_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
output_char_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

# -----------------------  Training/Testing Split  ----------------------------
input_texts, input_texts_test, target_texts, target_texts_test = train_test_split(input_texts, target_texts, test_size=0.05)
#X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

# -----------------------  Define Characters Vocabulary/Dictionary  -----------
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='int32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='int32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='int32')

encoder_input_data_test = np.zeros(
    (len(input_texts_test), max_encoder_seq_length, num_encoder_tokens),
    dtype='int32')
decoder_input_data_test = np.zeros(
    (len(input_texts_test), max_decoder_seq_length, num_decoder_tokens),
    dtype='int32')
decoder_target_data_test = np.zeros(
    (len(input_texts_test), max_decoder_seq_length, num_decoder_tokens),
    dtype='int32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_char_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, output_char_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, output_char_index[char]] = 1.

for i, (input_text, target_text) in enumerate(zip(input_texts_test, target_texts_test)):
    for t, char in enumerate(input_text):
        encoder_input_data_test[i, t, input_char_index[char]] = 1.
    for t, char in enumerate(target_text):
        # Decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data_test[i, t, output_char_index[char]] = 1.
        if t > 0:
            decoder_target_data_test[i, t - 1, output_char_index[char]] = 1.

print('Size of training:', len(input_texts))
print('Size of Testing:', len(input_texts_test))
print('Number of characters of inputs/encoder:', num_encoder_tokens)
print('Number of characters of outputs/decoder:', num_decoder_tokens)
print('Max length of inputs/encoder (characters):', max_encoder_seq_length)
print('Max length outputs/decoder (characters):', max_decoder_seq_length)

# -----------------------  Define Seq2seq Model  ------------------------------
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define seq2seq model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

filepath="saved_models/character_based_200over_{epoch:02d}_{val_loss:.4f}.h5"
checkpoint = callbacks.ModelCheckpoint(filepath, 
                                       monitor='val_loss', 
                                       verbose=0, 
                                       save_best_only=False)
callbacks_list = [checkpoint]
# Training model
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy')

if train_bot == 1:
    if os.path.isfile(saved_model):
        print('load previous trained model')
        model.load_weights(saved_model)
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2
                  ,callbacks=callbacks_list
                  )
        model.save(saved_model+'final')
else:
    if os.path.isfile(saved_model):
        model.load_weights(saved_model)
    else:
        print('ERROR!')

# -----------------------  Define encoder for Inference  ----------------------
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.save(saved_encoder_model)
# -----------------------  Define decoder for Inference  ----------------------
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
decoder_model.save(saved_decoder_model)

# -----------------------  Model Structure  -----------------------------------
print('---------- Training Model -------')
model.summary()
print('---------- Encoder Model --------')
encoder_model.summary()
print('---------- Decoder Model --------')
decoder_model.summary()

# Dict: from index => char
input_index_char = dict(
    (i, char) for char, i in input_char_index.items())
output_index_char = dict(
    (i, char) for char, i in output_char_index.items())

def decode_answer(encoded_question):
    # states_value: initial or internal state of decoder with [h,c] = [256,256] latent dimension
    states_value = encoder_model.predict(encoded_question)
    target_sequence = np.zeros((1, 1, num_decoder_tokens))
    # Set Begin of Sentence (BOS)/the start character:'\t' with one-hot encoding
    target_sequence[0, 0, output_char_index['\t']] = 1.

    stop = False
    decoded_answer = ''
    while not stop:
        output_tokens, h, c = decoder_model.predict(
            [target_sequence] + states_value)

        # Append next char
        char_index = np.argmax(output_tokens[0, -1, :])
        char = output_index_char[char_index]
        decoded_answer += char

        # Stop encode when hit max length or find stop character: '\n'
        if (char == '\n' or
           len(decoded_answer) > max_decoder_seq_length):
            stop = True

        # Update the target sequence (of length 1).
        target_sequence = np.zeros((1, 1, num_decoder_tokens))
        target_sequence[0, 0, char_index] = 1.
        
        states_value = [h, c]

    return decoded_answer

# -----------------------  Evaluation of Model: BLUE 1-4 grams  ---------------
f_candi = open(CANDI_FILE,'w',encoding='utf-8')
f_refer = open(REFER_FILE,'w',encoding='utf-8')
f_bleus = open(BLEU_FILE,'w',encoding='utf-8')
references = []
candidates = []
bleu_n_s_list = []

for idx_test in range(len(encoder_input_data_test)-1):
    encoded_question = encoder_input_data_test[idx_test: idx_test + 1]
    decoded_answer = decode_answer(encoded_question)
    refer = target_texts[idx_test].replace("\t", "").replace("\n", "")
    _, bleu_n_s = BLEU_EVA(refer, decoded_answer)
    bleu_n_s_list.append(bleu_n_s)
    if bleu_n_s[2] > 0:
        print('-')
        print('Question:', input_texts_test[idx_test])
        print('Answer:', decoded_answer)
        print('Reference:', target_texts_test[idx_test])
        print('BLEU 1-4 of %d test sample: '%idx_test, end='')
        print(bleu_n_s)
    f_bleus.write('Question :' + input_texts_test[idx_test]+'\n'+
                  'Answer   :'+ decoded_answer+#'\n'+
                  'Reference:'+ refer+'\n'+
                  'BLEU 1-4 on '+str(idx_test)+' test sample::'+ str(bleu_n_s)+'\n'+
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
f_bleus.write('--------OVERALL BLEU 1 - 4 on unseen test data -------'+'\n')
f_bleus.write(str(bleu_n_mean))

f_candi.close()
f_refer.close()
f_bleus.close()

# -----------------------  Conversation  --------------------------------------
# For common usages sentence:
GREETING_KEYWORDS = ("hello", "hi", "how are you?", "how are you", "what's up",)
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
        encoded_question = np.zeros(
        (len(question), max_encoder_seq_length, num_encoder_tokens),
        dtype='int32')
        for i, input_text in enumerate(zip(question)):
            for t, char in enumerate(input_text):
                encoded_question[i, t, input_char_index[char]] = 1.
    
        # Decode answer that the Chatbot will reply
        decoded_answer = decode_answer(encoded_question)
        print ('Chatbot: ' + decoded_answer)

chat()

print('---------- Training Model -------')
plot_model(model, to_file='pic/character_based_chatbot_model.png', show_layer_names=True)
Image('pic/character_based_chatbot_model.png')
print('---------- Encoder Model --------')
plot_model(encoder_model, to_file='pic/character_based_chatbot_encoder.png', show_layer_names=True)
Image('pic/character_based_chatbot_encoder.png')
print('---------- Decoder Model --------')
plot_model(decoder_model, to_file='pic/character_based_chatbot_decoder.png', show_layer_names=True)
Image('pic/character_based_chatbot_decoder.png')
