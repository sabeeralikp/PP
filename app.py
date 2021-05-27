# app.py
from flask import Flask, request, Response
app = Flask(__name__)

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

data = pd.read_csv('dataset.csv') 

# We want to remove these from the psosts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
  
unique_type_list = [x.lower() for x in unique_type_list]

b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

cntizer = CountVectorizer(analyzer="word", 
                             max_features=1500, 
                             tokenizer=None,    
                             preprocessor=None, 
                             stop_words=None,  
                             max_df=0.7,
                             min_df=0.1) 

lemmatiser = WordNetLemmatizer()

cachedStopWords = stopwords.words("english")

def translate_personality(personality):
    return [b_Pers[l] for l in personality]

def translate_back(personality):
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s

def pre_process_data(data, remove_stop_words=True, remove_mbti_profiles=True):
    list_personality = []
    list_posts = []
    len_data = len(data)
    i=0
    
    for row in data.iterrows():
        i+=1
        
            

        ##### Remove and clean comments
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
            
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t,"")

        type_labelized = translate_personality(row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(temp)

    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality

list_posts, list_personality  = pre_process_data(data, remove_stop_words=True)



cntizer = CountVectorizer(analyzer="word", 
                             max_features=1500, 
                             tokenizer=None,    
                             preprocessor=None, 
                             stop_words=None,  
                             max_df=0.7,
                             min_df=0.1) 


X_cnt = cntizer.fit_transform(list_posts)


tfizer = TfidfTransformer()

X_tfidf =  tfizer.fit_transform(X_cnt).toarray()

feature_names = list(enumerate(cntizer.get_feature_names()))

type_indicators = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) – Sensing (S)", 
                   "FT: Feeling (F) - Thinking (T)", "JP: Judging (J) – Perceiving (P)"  ]

def give_rec(title):

# A few few tweets and blog post
    my_posts  = title
# The type is just a dummy so that the data prep fucntion can be reused
    mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

    my_posts, dummy  = pre_process_data(mydata, remove_stop_words=True)

    my_X_cnt = cntizer.transform(my_posts)
    my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()


    result = {}
    
    # Let's train type indicator individually
    for l in range(len(type_indicators)):

        file_name = type_indicators[l].replace('/', '').replace(':','_') + ".pkl"
        xgb_model_loaded = pickle.load(open(file_name, "rb"))
        y_pred = xgb_model_loaded.predict(my_X_tfidf)
        score = xgb_model_loaded.predict_proba(my_X_tfidf)

        result[type_indicators[l][0:2]] = {
            'prediction': type_indicators[l][1:2] if y_pred[0] else type_indicators[l][0:1],
            'score': score.max()
        }
        

    return result

@app.route('/getrecs/', methods=['GET'])
def respond():
    # Retrieve the title from url parameter
    title = request.args.get("title", None)

    # For debugging
    print(f"got title {title}")

    response = [{}]

    if not title:
        response["ERROR"] = "no title found, please send a title."
    elif str(title).isdigit():
        response["ERROR"] = "title can't be numeric."
    else:
        response = give_rec(title)
        if response.empty:
            response = {'ERROR' : "no title found, please send a title."}
        
            
        
    # Return the response in json format
    return Response(json.dumps(response), mimetype='application/json')

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)