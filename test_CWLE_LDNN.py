import random
import string
import numpy
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModel
import xlrd
defaultencoding = 'utf-8'
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional
import nltk

pretrained_model = "PubMedBERT-base-uncased-abstract-fulltext"
# pretrained_model = "scibert_scivocab_uncased"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModel.from_pretrained(pretrained_model)

dataset1 = xlrd.open_workbook(r'Dataset1.xlsx')
positive_sheet = dataset1.sheet_by_index(0)
negative_sheet = dataset1.sheet_by_index(1)

positive_sentences_array = []
for num in range(1,positive_sheet.nrows):
    positive_sentences_array.append(positive_sheet.cell(num, 0).value)

negative_sentences_array = []
for num in range(1,negative_sheet.nrows):
    negative_sentences_array.append(negative_sheet.cell(num,0).value)
while len(negative_sentences_array)!=len(positive_sentences_array):
    random_number = random.randint(0,len(negative_sentences_array))
    negative_sentences_array.remove(negative_sentences_array[random_number])

test_data=xlrd.open_workbook(r'Dataset2.xlsx')
test_positive_sheet = test_data.sheet_by_index(0)
test_negative_sheet = test_data.sheet_by_index(1)

test_positive_sentences=[]
for num in range(1,test_positive_sheet.nrows):
    test_positive_sentences.append(test_positive_sheet.cell(num,0).value)
test_negative_sentences=[]
for num in range(1,test_negative_sheet.nrows):
    test_negative_sentences.append(test_negative_sheet.cell(num,0).value)
while len(test_negative_sentences)!=len(test_positive_sentences):
    random_number=random.randint(0,len(test_negative_sentences))
    test_negative_sentences.remove(test_negative_sentences[random_number])

# CWLE embedding
X=[]
Y=[]

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
    X.append(sentence_embedding)
    Y.append(1)

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
    X.append(sentence_embedding)
    Y.append(0)

test_X=[]
test_Y=[]
for sentence in test_positive_sentences:
    for c in string.punctuation:
        sentence = sentence.replace(c,'')
    words_array = nltk.word_tokenize(sentence)
    sentence_embedding=[]
    for word in words_array:
        training_inputs = tokenizer(word, return_tensors="pt")
        training_outputs = model(**training_inputs)[1]
        training_embedding = list(training_outputs.detach().numpy())[0]
        sentence_embedding.append(training_embedding)
    test_X.append(sentence_embedding)
    test_Y.append(1)

for sentence in test_negative_sentences:
    for c in string.punctuation:
        sentence = sentence.replace(c,'')
    words_array = nltk.word_tokenize(sentence)
    sentence_embedding=[]
    for word in words_array:
        training_inputs = tokenizer(word, return_tensors="pt")
        training_outputs = model(**training_inputs)[1]
        training_embedding = list(training_outputs.detach().numpy())[0]
        sentence_embedding.append(training_embedding)
    test_X.append(sentence_embedding)
    test_Y.append(0)

training_data=[]
test_data=[]
for num in range(0,len(X)):
    training_data.append([torch.tensor(numpy.array(X[num])),torch.tensor(numpy.array(Y[num]))])

for num in range(0,len(test_X)):
    test_data.append([torch.tensor(numpy.array(test_X[num])), torch.tensor(numpy.array(test_Y[num]))])


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

# Create a LSTM
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

# Train Network
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


# check accuracy on test to see how good our model
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

        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        accuracy=(TP+TN)/(TP+FP+FN+TN)
        F1=2*precision*recall/(precision+recall)
        Specificity=TN/(TN+FP)
        print ("TP:"+str(TP))
        print ("FP:"+str(FP))
        print ("TN:"+str(TN))
        print ("FN:"+str(FN))
        print ("precision:"+str(precision))
        print ("recall:"+str(recall))
        print ("accuracy:"+str(accuracy))
        print ("F1:"+str(F1))
        print ("Specificity:"+str(Specificity))

check_accuracy(test_data,model)
