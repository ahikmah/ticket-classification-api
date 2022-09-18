from django.shortcuts import render
from django.conf import settings
from .apps import TicketClassificationConfig

# Create your views here.
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import os 
import re
import torch
import pickle
import torch.nn.functional as F
from keras.preprocessing.sequence import pad_sequences
from .mapping import *

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text    = re.sub(pattern, ' ', x)
    return text

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        x = re.sub('[0-9]{1}', '', x)
    return x

def replace_norms(text):
    pattern = re.compile(r'\b(' + '|'.join(mapping.keys()) + r')\b')
    return pattern.sub(lambda x: mapping[x.group()], text)
    
with open(os.path.join(settings.MODELS, "encoder.pickle"), 'rb') as handle:
    le = pickle.load(handle)
    
with open(os.path.join(settings.MODELS, "tokenizer.pickle"), 'rb') as handle:
    tokenizer = pickle.load(handle)

class get_ticket_class(APIView):

    def post(self, request):
        if request.method == 'POST':
          data = request.data
          
          # get complaint data from body request
          complaint_raw =  data['complaint']

          # lower the text
          complaint = complaint_raw.lower()

          # Clean the text
          complaint =  clean_text(complaint)

          # Clean numbers
          complaint =  clean_numbers(complaint)

          # Clean Contractions
          complaint = replace_norms(complaint)

          # Tokenize
          complaint = tokenizer.texts_to_sequences([complaint])

          # pad
          complaint = pad_sequences(complaint, maxlen=150)

          # create dataset
          complaint = torch.tensor(complaint, dtype=torch.long)

          lstm_model = TicketClassificationConfig.model
          pred = lstm_model(complaint).detach()
          pred = F.softmax(pred).cpu().numpy()

          pred = pred.argmax(axis=1)

          pred = le.classes_[pred]         

          response = {"complaint": complaint_raw, "category": pred[0]}
            
          # returning JSON response
          return Response(response, status=200)