import random
import string
import numpy
import openpyxl
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModel
import xlrd
defaultencoding = 'utf-8'
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional
import nltk

# pretrained_model = "PubMedBERT-base-uncased-abstract-fulltext"
pretrained_model = "scibert_scivocab_uncased"


tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModel.from_pretrained(pretrained_model)

dataset1 = xlrd.open_workbook(r'Dataset1.xlsx')
positive_sheet = dataset1.sheet_by_index(0)
negative_sheet = dataset1.sheet_by_index(1)

positive_sentences_array = []
for num in range(1,positive_sheet.nrows):
    positive_sentences_array.append(positive_sheet.cell(num, 0).value)

negative_sentences_array = []
for num in range(1,positive_sheet.nrows):
    negative_sentences_array.append(negative_sheet.cell(num,0).value)

# CWLE embedding
positive_sentences_embedding_array=[]
negative_sentences_embedding_array=[]

for sentence in positive_sentences_array:
    for c in string.punctuation:
        sentence=sentence.replace(c,'')
    words_array=nltk.word_tokenize(sentence)
    sentence_embedding=[]
    for word in words_array:
        training_inputs = tokenizer(word, return_tensors="pt")
        training_outputs = model(**training_inputs)[1]
        training_embedding = list(training_outputs.detach().numpy())[0]
        sentence_embedding.append(training_embedding)
    positive_sentences_embedding_array.append(sentence_embedding)

for sentence in negative_sentences_array:
    for c in string.punctuation:
        sentence = sentence.replace(c,'')
    words_array = nltk.word_tokenize(sentence)
    sentence_embedding=[]
    for word in words_array:
        training_inputs = tokenizer(word, return_tensors="pt")
        training_outputs = model(**training_inputs)[1]
        training_embedding = list(training_outputs.detach().numpy())[0]
        sentence_embedding.append(training_embedding)
    negative_sentences_embedding_array.append(sentence_embedding)

training_data=[]
val_data=[]
for num in range(0,len(positive_sentences_embedding_array)):
    if num < len(positive_sentences_embedding_array)*7/10:
        training_data.append([torch.tensor(numpy.array(positive_sentences_embedding_array[num])),1])
    else:
        val_data.append([torch.tensor(numpy.array(positive_sentences_embedding_array[num])),1])

for num in range(0,len(negative_sentences_embedding_array)):
    if num < len(negative_sentences_embedding_array)*7/10:
        training_data.append([torch.tensor(numpy.array(negative_sentences_embedding_array[num])), 0])
    else:
        val_data.append([torch.tensor(numpy.array(negative_sentences_embedding_array[num])), 0])

random.shuffle(training_data)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size=768
num_layers=3
hidden_size=300
num_classes=2
learning_rate=0.001
num_epochs=35

# Create LDNN model
class RNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)

        # Forward Prop
        out,_=self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])

        return out

# Initialize network
model = RNN(input_size,hidden_size,num_layers,num_classes).to(device)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# Training
for epoch in range(num_epochs):
    loss_run = 0.0
    for item in training_data:
        data=item[0]
        targets=item[1]

        model.train()

        data = data.unsqueeze(0).to(device)
        targets=torch.tensor(targets).unsqueeze(0).to(device)

        # forward
        scores=model(data)

        loss=criterion(scores,targets)
        loss_run = loss_run + float(loss)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # gradient descent or adam step
        optimizer.step()
    # print("epoch:" + str(epoch) + ";loss:" + str(loss_run / len(training_data)))


# check accuracy on training & test to see how good our model
def check_accuracy(loader,model):
    num_correct=0
    num_samples=0

    model.eval()

    with torch.no_grad():
        TP = 0.00
        FP = 0.00
        FN = 0.00
        TN = 0.00
        for x,y in loader:
            x = x.unsqueeze(0).to(device)
            y = torch.tensor(y).to(device=device)
            scores = model(x)

            score, predictions = scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)

            if predictions == 1 and y == 1:
                TP = TP + 1
            elif predictions == 1 and y == 0:
                FP = FP + 1
            elif predictions == 0 and y == 1:
                FN = FN + 1
            elif predictions == 0 and y == 0:
                TN = TN + 1
            print ("sample label:"+str(y)+";"+"predicted label:"+str(precisions))

        print ("TP:"+str(TP))
        print ("FP:"+str(FP))
        print ("TN:"+str(TN))
        print ("FN:"+str(FN))
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        accuracy=(TP+TN)/(TP+FP+FN+TN)
        F1=2*precision*recall/(precision+recall)
        specificity=TN/(TN+FP)
            
        print ("precision:"+str(precision))
        print ("recall:"+str(recall))
        print ("accuracy:"+str(accuracy))
        print ("F1:"+str(F1))
        print ("Specificity:"+str(specificity))

check_accuracy(val_data, model)
