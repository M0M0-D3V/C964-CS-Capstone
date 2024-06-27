import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from flask import (Flask, jsonify, redirect, render_template, request, session,
                   url_for)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams

app = Flask(__name__)
app.secret_key = 'secret'

with open('model.pkl', 'rb') as m:
  model = pickle.load(m)

with open ('vectorizer.pkl', 'rb') as v:
  vectorizer = pickle.load(v)

@app.route("/")
def hello():
  return render_template('index.html')
  
@app.route('/jobcheck')
def jobcheck():
  return render_template('jobform.html')
  
@app.route('/api', methods=['POST'])
def predict():
  if request.method == 'POST':
    # Get data from form inputs
    data = []
    if request.form.get('title') != None:
      data.append(request.form.get('title'))
    if request.form.get('location') != None:
      data.append(request.form.get('location'))
    if request.form.get('department') != None:
      data.append(request.form.get('department'))
    if request.form.get('salary') != None:
      data.append(request.form.get('salary'))
    if request.form.get('company') != None:
      data.append(request.form.get('company-profile'))
    if request.form.get('company') != None:
      data.append(request.form.get('company'))
    if request.form.get('description') != None:
      data.append(request.form.get('description'))
    if request.form.get('requirements') != None:
      data.append(request.form.get('requirements'))
    if request.form.get('benefits') != None:
      data.append(request.form.get('benefits'))
    if request.form.get('telecommute') != None:
      data.append(request.form.get('telecommute'))
    if request.form.get('has_logo') != None:
      data.append(request.form.get('has_logo'))
    if request.form.get('has_questions') != None:
      data.append(request.form.get('has_questions'))
    if request.form.get('employment_type') != None:
      data.append(request.form.get('employment_type'))
    if request.form.get('required_experience') != None:
      data.append(request.form.get('required_experience'))
    if request.form.get('required_education') != None:
      data.append(request.form.get('required_education'))
    if request.form.get('industry') != None:
      data.append(request.form.get('industry'))
    
    concat_posting = ' '.join(str(i) for i in data)
    
    # preprocess data the same as trained model
    job_posting = preprocess_text(concat_posting)
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
    result = prediction[0]
    print("**************************result**************************************")
    print(result)
    # change type to int
    result = result.astype(int)
    
    if result == 0:
      result = 'Real'
    else:
      result = 'Fake'
    
    # Store the result into session
    session['result'] = result
    
    return redirect(url_for('success'))
  
  return render_template('index.html')

@app.route('/success')
def success():
  # Get the result from session
  result = session.get('result', 'No result')
  return render_template('result.html', result=result)

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