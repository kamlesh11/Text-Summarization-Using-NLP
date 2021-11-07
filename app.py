from flask import Flask, render_template, request
import numpy as np
import os
from newspaper import fulltext
import requests
from summarizer import Summarizer,TransformerSummarizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch

app = Flask(__name__)
@app.route("/", methods=['GET'])
def home():
    options = ['Bert','XLNET','GPT-2','T5-Transformer']
    return render_template('index.html',options=options)

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        article_url = request.form['url']
        model_name = request.form['options']
        article = fulltext(requests.get(article_url).text)
        if model_name=='Bert':
            model = Summarizer()
            summary = ''.join(model(article, min_length=60))
        elif model_name=='XLNET':
            model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
            summary = ''.join(model(article, min_length=60))
        elif model_name=='GPT-2':
            model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
            summary = ''.join(model(article, min_length=60))
        else:
            model = T5ForConditionalGeneration.from_pretrained('t5-small')
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            device = torch.device('cpu')
            preprocess_text = article.strip().replace("\n","")
            t5_prepared_Text = "summarize: "+preprocess_text

            tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    
            summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=500,
                                    max_length=2000,
                                    early_stopping=True)

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        f = open("Web-app/templates/summary.txt", "w")
        f.write(summary)
        f.close()
        return render_template('sec.html', pred_output=summary)#, user_image="../static/user uploaded/"+filename)

if __name__ == "__main__":
    app.run(threaded=True)