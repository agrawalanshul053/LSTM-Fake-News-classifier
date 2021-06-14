import numpy as np
from flask import Flask, request, jsonify, render_template
import torch
# import torchtext.data import 
# from torchtext.data import Field
import torch.optim as optim
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
app = Flask(__name__)
device = 'cpu'
text_field  = pickle.load(open('TextField.pkl', 'rb'))
path='F:\Downloads\Project under Jivnesh pr\machine-learning-deployment-2\lstm\Model\model.pt'

class LSTM(nn.Module):                             # Inheriting class nn.Module
                                                   # LSTM is child class/sub class and nn.Module is parent/ super class
    def __init__(self, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out
# model.eval()
def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


model = LSTM().to('cpu')
optimizer = optim.Adam(model.parameters(), lr=0.001)
load_checkpoint(path, model, optimizer)


import spacy
nlp = spacy.load('en')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():


    def trim_string(x):
        first_n_words = 200
        x = x.split(maxsplit=first_n_words)
        x = ' '.join(x[:first_n_words])

        return x    

    y_pred = []
    str_features = [str(x) for x in request.form.values()]
    titletext_str = str_features[0]+'. '+str_features[1]
    titletext_str = trim_string(titletext_str)
    tokenized = [tok.text for tok in nlp.tokenizer(titletext_str)] #tokenize the sentence 
    # print(tokenized)
    # print("Type of text_field is :", type(text_field))
    indexed = [text_field.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    # print(text_field.vocab.stoi['.'])
    # print(indexed)
    # print(type(indexed))
    length = len(indexed)                                      #compute no. of words
    titletext = torch.LongTensor(indexed).to(device)              #convert to tensor
    # print("titltext is :", titletext)
    # print("Type of titletext is :", type(titletext))
    
    titletext = titletext.unsqueeze(1).T                            #reshape in form of batch,no. of words
    # print("Unsqeezed titltext is :", titletext)
    # print("Unsqeezed Type of titletext is :", type(titletext))
    length_tensor = torch.LongTensor([length])
    # print("length_tensor is :",length_tensor)
    threshold=0.5
    model.eval()
    with torch.no_grad():
        titletext = titletext.to(device)
        length_tensor = length_tensor.to(device)
        output = model(titletext, length_tensor)

        output = (output > threshold).int()
        y_pred = output.tolist()[0] # series.tolist() converts series into list
        if y_pred == 1:
            pred = "FAKE"
        elif y_pred == 0:
            pred = "REAL"
    
    return render_template('index.html', prediction_text='News will be {}'.format(pred))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)