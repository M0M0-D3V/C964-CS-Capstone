import asyncio
import os
import re

import httpx
import joblib
import nltk
import numpy as np
import pandas as pd
from quart import (Quart, jsonify, redirect, render_template, request, session,
                   url_for)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams

app = Quart(__name__)
app.secret_key = 'secret'

@app.route("/")
async def hello():
    return await render_template('index.html')
  
@app.route('/api', methods=['POST'])
async def predict():
  if request.method == 'POST':
    # Get data from form inputs
    form = await request.form
    data = []
    data.append(form.get('title'))
    data.append(form.get('location'))
    data.append(form.get('department'))
    data.append(form.get('salary'))
    data.append(form.get('company-profile'))
    data.append(form.get('company'))
    data.append(form.get('description'))
    data.append(form.get('requirements'))
    data.append(form.get('benefits'))
    data.append(form.get('telecommute'))
    data.append(form.get('has_logo'))
    data.append(form.get('has_questions'))
    data.append(form.get('employment_type'))
    data.append(form.get('required_experience'))
    data.append(form.get('required_education'))
    data.append(form.get('industry'))
    
    concat_posting = ' '.join(str(i) for i in data)
    
    result = await preprocess_and_predict(concat_posting)
    print("****************************************************************")
    print(result)
    
    if result == 0:
      result = 'Fake'
    elif result == 1:
      result = 'Real'
    
    # Store the result into session
    session['result'] = result
    
    return redirect(url_for('success'))
  
  return await render_template('index.html')

@app.route('/success')
async def success():
  # Get the result from session
  result = session.get('result', 'No result')
  return await render_template('result.html', result=result)

async def preprocess_and_predict(data):
  model = joblib.load('model.pkl')
  vectorizer = joblib.load('vectorizer.pkl')
  # preprocess data the same as trained model
  job_posting = preprocess_text(data)
  # apply tokens
  job_posting_tokens = word_tokenize(job_posting)
  # apply posting sentences
  job_posting_sentences = sent_tokenize(job_posting)
  # apply pos tagged
  job_posting_pos_tagged = [[pos_tagging(sentence) for sentence in x] for x in job_posting_sentences]
  # apply ngrams
  n = 2
  job_posting_ngrams = [generate_ngrams(x, n) for x in job_posting_tokens]
  
  # Flatten the lists and convert all elements to strings
  flat_tokens = ' '.join([str(item) for sublist in job_posting_tokens for item in sublist])
  flat_sentences = ' '.join([str(item) for sublist in job_posting_sentences for item in sublist])
  flat_pos_tagged = ' '.join([str(item) for sublist in job_posting_pos_tagged for item in sublist])
  flat_ngrams = ' '.join([str(item) for sublist in job_posting_ngrams for item in sublist])

  # Now join everything together
  all_text = ' '.join([job_posting, flat_tokens, flat_sentences, flat_pos_tagged, flat_ngrams])
  print("***************all_text************************")
  print(all_text)
  
  # Transfor the new data using the same vectorizer object
  X_new_vec = vectorizer.transform([all_text])
  print("***************X_new_vec************************")
  print(X_new_vec)
  
  prediction = model.predict(X_new_vec)
  prediction = prediction[0]
  prediction = prediction.astype(int)
  return prediction

def preprocess_text(text):
  # convert to lowercasae
  text = text.lower()
  # Remove all URLs
  text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
  # Remove special characters
  text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
  # Remove punctuation
  text = re.sub(r'[^\w\s]', '', text)
  # Remove digits
  text = re.sub(r'\d', '', text)
  # Remove stop words
  stop_words = set(stopwords.words('english'))
  words = [word for word in text.split() if word.lower() not in stop_words]
  text = ' '.join(words)
  return text

@app.route('/api/images')
def get_image_filenames():
  image_folder = os.path.join(app.static_folder, 'images')
  image_filenames = os.listdir(image_folder)
  return jsonify(image_filenames)

def pos_tagging(sentence):
  tokens = word_tokenize(sentence)
  tagged_tokens = pos_tag(tokens)
  return tagged_tokens

def generate_ngrams(tokens, n):
  return list(ngrams(tokens, n))
  
if __name__ == "__main__":
  app.run(debug=True, port=4000)