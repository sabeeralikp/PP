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
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
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
    my_posts  = "No, I can't draw on my own nails (haha). Those were done by professionals on my nails. And yes, those are all gel.  You mean those you posted were done by yourself on your own nails? Awesome!|||Probably the Electronic Screen Syndrome. With the advent of technology and social media, we all suffer from overstimulation on a daily basis. I'm guilty as well. In the past, I can be happy just...|||I love nail arts too! These are some of mine:  718282 718290 718298 718306 718314|||This is the first time I'm hearing this - about menstruation and church. Thanks for sharing but yeah, it's crazy. I thought only Taoists have such a belief.|||Dear very bad person,  Not trying to get in between your arguments but I'm copying down that sentence for future use. :tongue:|||*Speaking from a bomb shelter*  So which Christian values do you still hold? I agree that people can still be good in the absence of religion and people also do bad in the name of religion but the...|||I never really thought about my childhood experiences until I was much older when my brothers and I talked about the past and we all agreed that we had suffered physical and verbal abuse. It wasn't...|||My ISFJ friend almost always instantly shuts down and becomes upset whenever someone disagrees with her POV or the way she does things. She can't seem to understand why a certain behaviour, though...|||These two questions seem unrelated.   If people know my real thoughts, they may or may not like me better, it depends on whether they like my thoughts to begin with. I have bad thoughts but some...|||BS？I don't BS.|||I think INFJs sometimes need to be kicked out of our comfort zones, for the sake of our emotional wellbeing.    -- I can relate to this and it actually helps when friends pull me out of my rut. I...|||Haha... sorry if I offended anyone. I don't mean for this to be a dirty jokes thread. I've amended the original.  That was a funny one you shared, btw. I recalled a similar situation which...|||LOL, can I pay that in instalments?|||Yes, and this is a double edged sword. I think mental health awareness and education are important, and what's more important is people need to know where to get help. On the flip side, I'm also...|||Haha, hopefully? :proud:|||I'm not calling out recent posts specifically, just the general overall. Some posts may have been made several years ago but their remarks still stand.|||I'm creating this as a stress-reliever as I think all of us can use a good laugh once in awhile... :tongue:    Perhaps I'll start off with this funny conversation between two friends who were...|||I have a thinking... that many INFJs think they suffer from trauma but a lot times it's actually just us torturing ourselves in our minds because we're so good at that.     I talk it out with...|||So many stereotypical statements here. The way you guys describe the girly girls are like a princess. Pls label them as princess instead.|||Mental health is such a in-thing nowadays a lot of people are slapping labels on themselves to look cool. I'm not saying all of you are doing that... I'm just saying don't do that.|||Researchers are always coming up with new studies and BBC is hardly a reliable source. Even if you identify with a few symptoms on the checklist, it doesn't necessarily make you a psychopath. It's...|||I think this sucks and you have my sympathy. Personally, I hate it when a guy dumps me for another gal and then tells me he wishes me happiness when he's the exact source of my unhappiness. Don't buy...|||EveJ, I agree with the INTJ who says other people's expectations are other people's problems. I'm quite sick of meeting people's expectations simply because they feel entitled. The truth is, when the...|||Ashton Vern This may help you further - https://thoughtcatalog.com/heidi-priebe/2015/07/how-to-recognize-each-myers-briggs-personality-type-in-real-life/|||Only once. I confessed to him. He loves me back. We were together for 7yrs, then we broke up. :crying:  Unbreak my heart, pls.|||fabi, OMG, I finally found someone who also love these morbid subjects! :blushed:  I don't know if this is caused by Ni-Ti and if other INFJs also generally love such topics because my INFJ friend...|||Green, orange, white, yellow or red?|||I must say that this is by far the first religious thread I've read that didn't erupt into a war (lol) :joyous: so I feel safe posting here. I grew up as an atheist and later became a believer of God...|||Above ground or underground?|||What I mean by realistic is when we use Ti to analyse a situation. When I get into high Ti mode, I can really zoom in on the facts and question everything that's happening - happening in real life,...|||xtctr Have you done a MBTI test or you just think you're an INFJ because you feel lonely? Because you said you didn't feel this way up until this year...  So why is it that you suddenly feel...|||For I know the plans I have for you, declares the Lord,  plans to prosper you and not to harm you,  plans to give you hope and a future.  - Jeremiah 29:11|||I don't think trauma makes an INFJ or all INFJs must have experienced trauma. This is like saying other types don't experience traumas and it sounds rather snobbish.  When we experience trauma, we...|||You're right that INFJs can be very idealistic, but those with a strong Ti can also be very realistic as well, especially if we go into Ti overdrive.     After reading this, I still don't think...|||Hi Ashton Vern I doubt your friends are INFJs. See my point-by-point below...  -------------------------------------- Characteristics of Friend A: - very friendly and caring. She cares for her...|||Transition. Anxiety. Prayers.|||https://www.youtube.com/watch?v=tuunqfdz388|||IDontKnowMe  When did you have your first relationship (<20, 20-25, 26-30, >30)? <20  What do you look for in a partner?I like guys with boyish good looks, :wink: with a good heart and good...|||I want people to understand me but I won't bother telling them my type because most of the people around me don't seem too bothered with MBTI. :dry:|||Comfortably 2 hrs, maximum 3 hrs and then the full army of Ni barges in and I'm not paying attention anymore. Depends on how interesting the conversation is... shut down can happen earlier.|||Gotterdammerung  My brother had night terrors when he was young. He would wake up EVERY NIGHT at the EXACT SAME TIME and cried about someone trying to catch him. It was traumatising just watching...|||There was a period I had dreams of ghosts and when I woke up in paralysis mode I saw them in my room/outside my window. Another time I dream of a ghost in my room and then my spirit self saw myself...|||Yes, I also read that everyone dreams. At one time I did quite a bit of research on dreams because I was dreaming so much and some were rather supernatural.     Lucid dreaming is interesting but...|||More or less... but what I don't fit is some people think INFJs will never go to a club or anywhere noisy. This is untrue of me. I love music festivals, concerts, dancing to my favourite songs in a...|||How do you lucid dream on purpose?|||I like all the badass quotes:  712170  712178  712186  712194|||It is said that the INFJ minds don't switch off. I believe mine doesn't switch off even when I sleep which is why I'm such a prolific dreamer. My dreams are also often vivid, prophetic or sometimes...|||Emotional connection to me is to be able to relate our feelings (sadness, anger, joy, love) with one another. Sometimes I can feel an instant emotional connection with a friend over a particular...|||Not very good at doing romantic things... perhaps the most romantic thing I've done is folding a bottle of wishing stars for my boyfriend?|||Are you collecting MBTI types? LOL  Perhaps you can learn to spot them:  1. The one sitting in a crowded room quietly but observing everyone like a hawk (or creep). :ninja: 2. The one who looks..."

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