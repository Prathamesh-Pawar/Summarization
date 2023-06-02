"""
This file generate a small sample of the data we use to train, test and validate our models it create a json with topics summary and the 4 differnt pre-processing techniques we use on the articles.
"""
# Get the Wikipedia page for each keyword
import wikipedia
import json
from wikipedia.exceptions import WikipediaException

import spacy

# Download and install the language model
nlp = spacy.load('en_core_web_sm')

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
# Set the language to English
wikipedia.set_lang("en")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json

def getdata():
    keywords = ['P vs NP', 'Navier–Stokes existence and smoothness', 'Birch and Swinnerton-Dyer conjecture', 'Twin prime conjecture', 'Riemann hypothesis', 'Yang-Mills existence and mass gap', 'Poincaré conjecture']
    data = []
    for keyword in keywords:
        if len(data) >= 10:
            break
        try:
            pages = wikipedia.search(keyword, results=600)
            for page in pages:
                try:
                    if len(data) != 0:
                        with open('5run.json', 'r') as f:
                            data = json.load(f)

                    summary = wikipedia.summary(page)
                    content = wikipedia.page(page).content
                    # Add the article data to the list
                    data.append({
                        "topic": page,
                        "summary": summary,
                        "content": content
                    })
                    with open('5run.json', 'w') as f:
                        json.dump(data, f)
                    if len(data) >= 10:
                        break
                except wikipedia.exceptions.DisambiguationError as e:
                    # If the page is a disambiguation page, skip it
                    continue 
        except wikipedia.exceptions.PageError as e:
            # If no pages are found for the keyword, skip it
            continue
        except KeyError:

            continue

def Traditional_approach(text):
    '''
    Input:
        review: a string containing a review.
    Output:
        review_cleaned: a processed review. 

    '''
    lst = re.findall('http://\S+|https://\S+', text)
    for i in lst:
        text = text.replace(i,'')
    text = text.translate(str.maketrans('','',string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = nlp(text)
    tokens = [token.text for token in word_tokens ]
    text_cleaned = []
    for w in tokens:
        if w not in stop_words:
            text_cleaned.append(w)
    filtered_text = [word for word in text_cleaned if '\n' not in word]
    filtered_text = [word for word in filtered_text if ' ' not in word]
    return ' '.join(filtered_text)

def section_creator(article):
# Download article from wikipedia
    section_content = ''

    # Split article into sections by headers
    sections = {}
    lines = article.split('\n')
    current_section = None
    for line in lines:
        if line.startswith('='):
            if current_section is not None:
                sections[current_section] = section_content.strip()
            current_section = line.strip('= ')
            section_content = ''
        else:
            section_content += line + '\n'
    if current_section is not None:
        sections[current_section] = section_content.strip()
        for head in sections.keys():
            sections[head] = re.sub(r'\s{1,}', ' ', sections[head]).replace('\n', '')

    return sections

nltk.download('punkt')
def sentence_segmentation(content):
    sentences = nltk.sent_tokenize(content)
    return sentences

exclude = ["See also",
"References",
"External links",
"Notes",
"Sources",
"Further reading",
"Bibliography",
"Production",
"Abstracting and indexing",
"Examples",
"Citations",
"Nomenclature",
"Evolution",
"Uses"]
def exclusion(content):
    new_content = {}
    for title, value in content.items():
        if title not in exclude:
            new_content[title] = value
    return new_content

def custom_approach(content):
    con_dict = section_creator(content)
    con_dict = exclusion(con_dict)
    total = ""
    for key in con_dict.keys():
        total += con_dict[key]
    return total

def combined_approach(content):
    return Traditional_approach(custom_approach(content))

def tfidf_content(content):
    # Load the input document and split it into sentences
    if len(nltk.word_tokenize(content)) < 1500:
        return content
    else:
        token = len(nltk.word_tokenize(content))
        sentences = sentence_segmentation(content)
        tfidf = TfidfVectorizer().fit_transform(sentences).toarray()
        x = 1
        para = ""
        while token > 1500 :
            N = len(sentences) - x 
            top_indices = np.argsort(tfidf.sum(axis=1))[::-1][:N]
            # Concatenate the selected sentences into a single input sequence
            para =  ' '.join([sentences[i] for i in top_indices])
            token = len(nltk.word_tokenize(para))
            x+=1
        return para

def new_trad_approach(content):
    seg_content = sentence_segmentation(content)
    trad_content = []
    for c in seg_content:
        trad_content.append(Traditional_approach(c))
    content = '. '.join(trad_content)
    content = content + '.'
    return content

def new_combined_approach(content):
    return new_trad_approach(tfidf_content(custom_approach(content)))


if __name__ == "__main__":
    getdata()
    with open('5run.json','r+') as file:
        data = json.load(file)
        for i in range(len(data)):
                data[i]['content'] = data[i]['content'].replace(data[i]['summary'],'') 
                data[i]['content_traditional'] = new_trad_approach(tfidf_content(data[i]['content']))
                data[i]['custom_approach'] = tfidf_content(custom_approach(data[i]['content']))
                data[i]['combined_approach'] = new_combined_approach(data[i]['content'])
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()