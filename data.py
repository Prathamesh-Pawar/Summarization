
# For data collection we make a list of topics with complex scientific background and make use of the wikipedia package to download and store
# the tiles, content and summary in a json file. Here is a code snipit showing how.


import string
import re

import wikipedia
import json
from wikipedia.exceptions import WikipediaException
# Set the language to English
wikipedia.set_lang("en")


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

import spacy
nlp = spacy.load('en_core_web_sm')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import re





keywords = ['Hilbert\'s fifth problem', 'P vs NP', 'Navier–Stokes existence and smoothness', 'Birch and Swinnerton-Dyer conjecture', 'Twin prime conjecture']
titles = []
i = 0
for keyword in keywords:
    if i > 1000:
        break
    try:
        pages = wikipedia.search(keyword, results=600)
        for page in pages:
            if page not in titles:
                try:
                    with open('10krun.json', 'r') as f:
                        data = json.load(f)

                    summary = wikipedia.summary(page)
                    content = wikipedia.page(page).content
                    # Add the article data to the list
                    data.append({
                        "topic": page,
                        "summary": summary,
                        "content": content
                    })
                    with open('10krun.json', 'w') as f:
                        json.dump(data, f)
                        if len(data) > 1000:
                            i = 1000
                            break
                except wikipedia.exceptions.DisambiguationError as e:
                    # If the page is a disambiguation page, skip it
                    continue 
    except wikipedia.exceptions.PageError as e:
        # If no pages are found for the keyword, skip it
        continue
    except KeyError:

        continue

# Pre-processing:
# We utilize four distinct preprocessing approaches, including the Traditional, Custom, Raw, and Combined approaches. Each approach has its unique
# strengths and benefits, allowing us to tailor our data processing for better evaluation and achieve higher accuracy rates in our models.


# Traditional Approach
# We utilize a range of traditional preprocessing techniques, including stopword removal, punctuation filtering, and tokenization.
# To perform these tasks, we rely on the Spacy library, which is renowned for its effectiveness in handling scientific text and
# related terminology.

# Moreover, we employ the Term Frequency-Inverse Document Frequency (TFIDF) method to reduce the size of the document to 1500 tokens,
# which is essential for the model's optimal performance. Following this, we perform sentence segmentation and rank each sentence's
# importance using the TFIDF score. Finally, we only retain the highest ranking sentences that fit under the 1500 token limit,
# ensuring that only the most relevant and informative content is included in the analysis.



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

def sentence_segmentation(content):
    sentences = nltk.sent_tokenize(content)
    return sentences



# Custon Approach:
# In this approach we get rid of non-informational sections of the content like refernces, notes or symbols in some cases. before running TFIDF on it.


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

# Combined Approach
# In this is apart of the abliation stuy where we use both traditional and custum approach of pre-processing on this content
# for a better comparision and to see if it can give a higher accuracy.

def combined_approach(content):
    return Traditional_approach(custom_approach(content))

