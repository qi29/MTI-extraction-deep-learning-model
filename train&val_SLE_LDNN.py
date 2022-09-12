import string
import openpyxl
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModel
from sklearn.decomposition import PCA
import xlrd
defaultencoding = 'utf-8'
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

pretrained_model = "PubMedBERT-base-uncased-abstract-fulltext"
# pretrained_model = "scibert_scivocab_uncased"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModel.from_pretrained(pretrained_model)

dataset1 = xlrd.open_workbook(r'Dataset1.xlsx')
positive_sheet = dataset1.sheet_by_index(0)
negative_sheet = dataset1.sheet_by_index(1)

positive_sentences_array = []

# for num in range(1,10):
for num in range(1,positive_sheet.nrows):
    positive_sentences_array.append(positive_sheet.cell(num, 0).value)

negative_sentences_array = []
for num in range(1,positive_sheet.nrows):
# for num in range(1,10):
    negative_sentences_array.append(negative_sheet.cell(num,0).value)


# SLE embedding
positive_sentences_embedding_array=[]
negative_sentences_embedding_array=[]

for sentence in positive_sentences_array:
    training_inputs = tokenizer(sentence, return_tensors="pt")
    training_outputs = model(**training_inputs)[1]
    training_embedding = list(training_outputs.detach().numpy())[0]
    positive_sentences_embedding_array.append(training_embedding)

for sentence in negative_sentences_array:
    training_inputs = tokenizer(sentence, return_tensors="pt")
    training_outputs = model(**training_inputs)[1]
    training_embedding = list(training_outputs.detach().numpy())[0]
    negative_sentences_embedding_array.append(training_embedding)


X=[]
Y=[]
val_X=[]
val_Y=[]

for num in range(0,len(positive_sentences_embedding_array)):
    if num < len(positive_sentences_embedding_array)*7/10:
        X.append(positive_sentences_embedding_array[num])
        Y.append(1)
    else:
        val_X.append(positive_sentences_embedding_array[num])
        val_Y.append(1)

for num in range(0,len(negative_sentences_embedding_array)):
    if num < len(negative_sentences_embedding_array)*7/10:
        X.append(negative_sentences_embedding_array[num])
        Y.append(0)
    else:
        val_X.append(negative_sentences_embedding_array[num])
        val_Y.append(0)


# data loader
class TensorDataset(Dataset):
    def __init__(self, data_tensor,target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __getitem__(self,index):
        return self.data_tensor[index],self.target_tensor[index]
    def __len__(self):
        return self.data_tensor.size(0)

training_data_tensor = torch.tensor(X, dtype=torch.float32)
training_target_tensor = Y
training_tensor_dataset = TensorDataset(training_data_tensor,training_target_tensor)
training_tensor_dataloader = DataLoader(training_tensor_dataset,shuffle=True)

val_data_tensor = torch.tensor(val_X, dtype=torch.float32)
val_target_tensor = val_Y
val_tensor_dataset = TensorDataset(val_data_tensor,val_target_tensor)
val_tensor_dataloader = DataLoader(val_tensor_dataset)


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size=768
num_layers=3
hidden_size=300
num_classes=2
learning_rate=0.001
num_epochs=100

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
for epoch in range(1,num_epochs+1):
    loss_run=0.0
    for batch_idx, (data,targets) in enumerate(training_tensor_dataloader):

        data = data.to(device=device).unsqueeze(0)
        targets=targets.to(device=device)

        model.train()
        # forward
        scores=model(data)
        loss=criterion(scores,targets)
        loss_run=loss_run+float(loss)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # gradient descent or adam step
        optimizer.step()

    # print("epoch:" + str(epoch) + ";loss:" + str(loss_run / len(training_tensor_dataloader)))


# check accuracy on training & val to see how good our model
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

            x = x.to(device=device).unsqueeze(0)
            y = y.to(device=device)
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
        print ("F1:" + str(F1))
        print ("accuracy:"+str(accuracy))
        print ("specificity:"+str(specificity))
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')


check_accuracy(val_tensor_dataloader, model)