import sys
sys.path.append('/home/kousik/ABC_OTHERS/campusx/nlp_movie_data/experiment_with_PyTorch (copy)/src')
from model import LSTM2Classify

import streamlit as st
import torch
import pickle 
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("spacy")
vocab = pickle.load(open('artifacts/vocabulary.pkl','rb'))

st.set_page_config(page_title="Tweeter Sentiment Analysis")
st.title("Sentiment Analysis of Tweet")

loaded_model = LSTM2Classify()
loaded_model.load_state_dict(torch.load('artifacts/best_model.pth', map_location=torch.device('cpu')))


# write a function to predict
def sentiment_analysis(text,model=loaded_model):
  text_input_ids = torch.tensor(vocab(tokenizer(text))).unsqueeze(dim=0)
  output = model(text_input_ids)
  sentiment_id = torch.argmax(output,dim=1).item()
  if sentiment_id==0:
    return 'Irrelevant'
  elif sentiment_id==1:
    return 'Negative'
  elif sentiment_id==2:
    return 'Neutral'
  elif sentiment_id==3:
    return 'Positive'


input_review = st.text_area("Enter any text or tweet to know the sentiment")

if st.button('Predict'):
  resp = sentiment_analysis(input_review)
  st.header(resp)
    