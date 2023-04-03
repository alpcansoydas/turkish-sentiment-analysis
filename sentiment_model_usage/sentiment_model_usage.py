import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
import warnings
from transformers import logging
logging.set_verbosity_error()

# Ignore warnings
warnings.filterwarnings("ignore")

# Import pretrained tokenizer and pretrained model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-128k-uncased",return_dict=False)

# Define Neural Network
class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,3)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)

      # apply softmax activation
      x = self.softmax(x)

      return x
  
# Pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# Load weights of the model
path = 'saved_weights_10k.pt'
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

# Define predict function (2=>POSITIVE 1=>NEGATIVE 0=>NEUTRAL)
def predict_sentiment(text):
    
  tokenized = tokenizer.encode_plus(
    text,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
    )

  input_ids = tokenized['input_ids']
  attention_mask = tokenized['attention_mask']

  seq = torch.tensor(input_ids)
  mask = torch.tensor(attention_mask)
  seq = seq.unsqueeze(0)
  mask = mask.unsqueeze(0)
  preds = model(seq, mask)
  preds = preds.detach().cpu().numpy()
  result = np.argmax(preds, axis=1)
  preds = torch.tensor(preds)
  probabilities = nn.functional.softmax(preds)

  '''return {'POSITIVE':float(probabilities[0][2]),
          'NEGATIVE':float(probabilities[0][1]),
          'NEUTRAL':float(probabilities[0][0])}
  '''
  
  if result == 2:
    return {'Label':'POSITIVE', 'Ratio':float(probabilities[0][2])}
  elif result == 1:
    return {'Label':'NEGATIVE', 'Ratio':float(probabilities[0][1])}
  else:
    return {'Label':'NEUTRAL', 'Ratio':float(probabilities[0][0])}
  
# Predict the sentiment score of given text
text = input('Sentiment analizi için metin giriniz:')
print(predict_sentiment(text))
