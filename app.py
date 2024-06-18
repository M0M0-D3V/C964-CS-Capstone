import pickle
import re

import nltk
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def hello():
    return render_template('index.html')
  
@app.route('/api', methods=['POST'])
def predict():
  if request.method == 'POST':
    # Get data from form inputs
    data = []
    data.append(request.form.get('title'))
    data.append(request.form.get('location'))
    data.append(request.form.get('department'))
    data.append(request.form.get('salary'))
    data.append(request.form.get('company-profile'))
    data.append(request.form.get('company'))
    data.append(request.form.get('description'))
    data.append(request.form.get('requirements'))
    data.append(request.form.get('benefits'))
    data.append(request.form.get('telecommute'))
    data.append(request.form.get('has_logo'))
    data.append(request.form.get('has_questions'))
    data.append(request.form.get('employment_type'))
    data.append(request.form.get('required_experience'))
    data.append(request.form.get('required_education'))
    data.append(request.form.get('industry'))
    
    concat_posting = ' '.join(str(i) for i in data)
    
    # preprocess data the same as trained model
    job_posting = preprocess_text(concat_posting)
    print("**************jobposting******************")
    print(job_posting)
    
    # fraudulent?
    fraudulent = ''
    
    # apply tokens
    job_posting_tokens = word_tokenize(job_posting)
    print("**************tokens******************")
    print(job_posting_tokens)
    
    # apply posting sentences
    job_posting_sentences = sent_tokenize(job_posting)
    print("**************sentences******************")
    print(job_posting_sentences)
    
    # apply pos tagged
    job_posting_pos_tagged = [[pos_tagging(sentence) for sentence in x] for x in job_posting_sentences]
    
    # apply ngrams
    n = 2
    job_posting_ngrams = [generate_ngrams(x, n) for x in job_posting_tokens]
    print("****************ngrams****************")
    print(job_posting_ngrams)
    
    df = pd.DataFrame(columns=['job_posting', 'fraudulent', 'job_posting_tokens', 'job_posting_sentences', 'job_posting_pos_tagged', 'job_posting_ngrams'])
    
    new_data = {'job_posting': [job_posting], 'fraudulent': [fraudulent], 'job_posting_tokens': [job_posting_tokens], 'job_posting_sentences': [job_posting_sentences], 'job_posting_pos_tagged': [job_posting_pos_tagged], 'job_posting_ngrams': [job_posting_ngrams]}
    
    new_data_df = pd.DataFrame(new_data, index=[0])
    # Convert the dictionary to a DataFrame
    df = pd.concat([df, new_data_df], ignore_index=True)
    
    print("****************df****************")
    print(df)
    
    # make prediction
    prediction = model.predict(df)
    output = prediction[0]
    print(output)
    return redirect(url_for('success'), value=output)
  
  # Render the result in a new template
  # return str(output)
  return render_template('index.html')

def success(value):
  print(value)
  return render_template('result.html', value=value)

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

def pos_tagging(sentence):
  tokens = word_tokenize(sentence)
  tagged_tokens = pos_tag(tokens)
  return tagged_tokens

def generate_ngrams(tokens, n):
  return list(ngrams(tokens, n))
  
if __name__ == "__main__":
  app.run(debug=True)