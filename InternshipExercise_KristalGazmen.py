'''
@author: Kristal Gazmen
'''


'''
*************************************************************************************************************************
    1/ Can you write a python script (or use another language of your choice) to 
        (a) parse the following  json data stream, 
        (b) create an LDA or another model of your choice for the review data and 
        (c) convert the reviews into feature vectors using the model that you built. 
    Command-line driven scripts are fine (e.g., with input and output filenames)
    Note:                                      
        We are not looking for perfectly executable code and will not run the script. 
        We put more emphasis on the quality of the models built and the understandability/layout of the code. 
*************************************************************************************************************************        
'''

'''
--------------------------------------------------------------------------------------------------------------------------
    This is for 1(a).
    json.dumps, json.loads, and nested for statements were used to parse the data
--------------------------------------------------------------------------------------------------------------------------
'''
print('This is part 1(a)')
import json
import os
import string
import pandas as pd

'''
Step 1: to be able to create/read the json data stream, edit the review data as follows:
        (i) add a single closing quotation mark at the end of the 2nd recommendation for Donna Summer to prevent an error; and
        (ii) add 'employee' to form a nested dictionary, so that 'json.dumps' can be used
'''

review_data = json.dumps(
                          {
                            'employee': 
                              [
                                {
                                    'name': 'Donna Summer',
                                    'updated': '2016-02-26T09:09:37',
                                    'recommendations': [
                                        'I worked with Donna for 5 years and can highly recommend her as an experienced, knowledgeable. She is very hardworking designer with a meticulous eye for detail and creativity',
                                        'I have worked with Donna for a number of years and have always found her to be extremely professional and a real expert in her field. ',
                                        'Donna is always on the ball, highly organised and a real pleasure to work with.'
                                        ]
                                },
                                {
                                    'name': 'Justin Bieber',
                                    'updated': '2016-02-26T09:09:37',
                                    'recommendations': [
                                        'Justin is a consummate professional and a pleasure to work with at all times!'
                                        ]
                                }
                              ]
                          }
                        )
#https://blog.eduonix.com/web-programming-tutorials/learn-different-json-data-types-format/

'''
Step 2: use 'json.loads' to load the review data
'''

json_loaded = json.loads(review_data)

'''
Step 3: Assign the loaded json's data to an empty array and dictionary by:
    (i) creating an empty array called 'parsed_data'
    (ii) creating an empty dictionary called 'parsed_details' and assigning the loaded json values to the empty dictionary
    (iii) creating an empty array called 'recommendations'
    (iv) nesting a 'for' statement to call each feedback provided and assigning each loaded feedback to the empty recommendations array
    (v) appending the details at the end of each 'for' statement to the parsed_data array
'''

parsed_data = []

for item in json_loaded['employee']:
    parsed_details = {"name":None, "updated":None, "recommendations":None}
    parsed_details['name'] = item['name']
    parsed_details['updated'] = item['updated']
    parsed_details['recommendations'] = []
        
    for feedback in item['recommendations']:
        parsed_details['recommendations'].append(feedback)
        
    parsed_data.append(parsed_details)

#https://stackoverflow.com/questions/47060035/python-parse-json-array   

'''
Step 4: Test if the data has been parsed successfully by calling the recommendations for Donna Summer
'''
print(parsed_data[0]['recommendations'])

'''
--------------------------------------------------------------------------------------------------------------------------
    This is an alternative solution to 1(a).
    Package: os, json
    
    In this solution 'employee,' was not added to create a nested dictionary for the json data stream.
    However, a .json file was created and read, while json.dumps wasn't used.
    Once the json has been loaded, parsing and assigning the values to an empty array are the same as the previous solution.
    
    Risk: replacing single quotation with double quotes, especially if double quotes were used inside the feedback provided
--------------------------------------------------------------------------------------------------------------------------
'''

#https://stackoverflow.com/questions/27907633/multiple-json-objects-in-one-file-extract-by-python/27907893
print('\n\nThis is an alternative solution to 1(a)')

'''
Step 1: Update the file path and read the file
'''

script_path = os.path.abspath(__file__) 
script_dir = os.path.split(script_path)[0]
rel_path = '../../../../Desktop/Kristal/1. Personal/e. CVs/Chosen/Review_Data.json'
abs_file_path = os.path.join(script_dir, rel_path)

review_data2=open(abs_file_path).read()

#https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python

'''
Step 2: Replace the single quotation with double quotation marks, so that json.loads can be used
'''

review_data2 = review_data2.replace('\'','\"')
json_array = json.loads(review_data2)

'''
Step 3: Assign the loaded json's data to an empty array and dictionary by:
    (i) creating an empty array called 'parsed_data'
    (ii) creating an empty dictionary called 'parsed_details' and assigning the loaded json values to the empty dictionary
    (iii) creating an empty array called 'recommendations'
    (iv) nesting a 'for' statement to call each feedback provided and assigning each loaded feedback to the empty recommendations array
    (v) appending the details at the end of each 'for' statement to the parsed_data array
'''
  
parsed_data = []

for item in json_array:
    parsed_details = {"name":None, "updated":None, "recommendations":None}
    parsed_details['name'] = item['name']
    parsed_details['updated'] = item['updated']
    parsed_details['recommendations'] = []
    
    for feedback in item['recommendations']:
        parsed_details['recommendations'].append(feedback)
    
    parsed_data.append(parsed_details)
#https://stackoverflow.com/questions/47060035/python-parse-json-array

'''
Step 4: Test if the data has been parsed successfully by calling the recommendations for Justin
'''
    
print(parsed_data[1]['recommendations'])

'''
Step 5: For future use, create a dataframe for name and recommendations
'''
#https://stackoverflow.com/questions/41168558/python-how-to-convert-json-file-to-dataframe
id_no = 0
tom_array = []

for item in json_array:
    tom_dict = {"id":None, "title": None, "text":None}
    tom_dict['id'] = id_no
    tom_dict['title'] = item['name']
    for feedback in item['recommendations']:
        tom_dict['text'] = feedback
        tom_array.append(tom_dict)
+id_no 
tom_df = pd.DataFrame(tom_array)
tom_df = tom_df[['id','title','text']]

'''
--------------------------------------------------------------------------------------------------------------------------
    This is for 1(b): create an LDA or another model of your choice for the review data.
    
    Packages: nltk, tom_lib, gensim, lda, stop_words, and pandas
        (i) 'nltk' and 'stop_words' to pre-process the data;
        (ii) 'tom_lib' with 'lda' to identify the optimal number of topics;
        (ii) 'gensim' to create the LDA model; and
        (iv) 'pandas' to create and manage the data frame
--------------------------------------------------------------------------------------------------------------------------
'''
print('\n\nThis is for 1(b)')

import pandas as pd

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

from stop_words import get_stop_words

import gensim
from gensim import corpora, models, similarities

import tom_lib
from tom_lib.visualization import visualization
from tom_lib.nlp.topic_model import LatentDirichletAllocation,\
    NonNegativeMatrixFactorization
from tom_lib.visualization.visualization import Visualization
from tom_lib.structure.corpus import Corpus

import lda

from bokeh.io import show, output_notebook
from bokeh.plotting import figure

import numpy as np
np.random.seed(0)
#https://rpubs.com/barberje/LDA
#https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
#https://pypi.org/project/tom_lib/
#https://pythonprogramminglanguage.com/bag-of-words/

'''
Step 1: Consolidate feedback for all employees under a single list called 'feedback'
'''
feedback = parsed_data[0]['recommendations'] + parsed_data[1]['recommendations']

'''
Step 2: Based on the packages imported, set-up the functions needed for the pre-processing of data
'''
tokenizer = RegexpTokenizer(r'\w+')
stop_ref = get_stop_words('en')
punc_ref = set(string.punctuation)
lemma = WordNetLemmatizer()
stemmer = SnowballStemmer('english') #stemmer = PorterStemmer() is an alternative

'''
Step 3: Create an empty array for pre-processed data and pre-process the data by:
    (i) tokenizing or splitting the string into individual words or texts
    (ii) removing stop words, such as articles
    (iii) removing punctuations
    (iv) stemming the words into their root form
Note: convert strings to lower case, so that they can be processed more accurately
'''
def preprocess(feedback):
    processed_feedback = []
    
    if len(feedback) > 1: #there is more than one feedback
        for item in feedback:   
            feedback_lc = item.lower()
            
            word_i = tokenizer.tokenize(feedback_lc)
            word_ii = [item for item in word_i if not item in stop_ref]
            word_iii = [item for item in word_ii if item not in punc_ref]
            word_iv = [lemma.lemmatize(item) for item in word_iii]
            word_v = [stemmer.stem(item) for item in word_iv]
            
            processed_feedback.append(word_v)
            
    else: #there is only one review. prevents from splitting into letters.
        feedback_lc = str(feedback).lower()
        
        word_i = tokenizer.tokenize(feedback_lc)
        word_ii = [item for item in word_i if not item in stop_ref]
        word_iii = [item for item in word_ii if item not in punc_ref]
        word_iv = [lemma.lemmatize(item) for item in word_iii]
        word_v = [stemmer.stem(item) for item in word_iv]
        
        processed_feedback.append(word_v)
        
    return processed_feedback

processed_feedback = preprocess(feedback)
'''
Step 4: Create dictionary and corpus through BOW
'''
dictionary = corpora.Dictionary(processed_feedback)
print(dictionary)
    
corpus = [dictionary.doc2bow(item) for item in processed_feedback]
print(corpus)
    

'''
Step 5: To create the LDA model, we need to determine the number of topics first.
tom_lib can be used to determine the best count. However, input data must be prepared
to fit the required package format. For this, we refer to the 'tom_df' created in 1(a)
'''
#required tom_lib format for the input: (already created in step 1)
#id    title    text
#1    Document 1's title    This is the full content of document 1.
#2    Document 2's title    This is the full content of document 2.
#etc.
#https://github.com/AdrienGuille/TOM/blob/master/README.md

#use the tom_df from 1(a)
tom_df.to_csv('tom_df.csv', sep='\t', index=False, encoding='utf-8') #, sep='\t'

tom_lib_corpus = Corpus(source_file_path='tom_df.csv', 
                vectorization='tfidf', 
                n_gram=1,
                max_relative_frequency=0.8, 
                min_absolute_frequency=2)

topic_model = LatentDirichletAllocation(tom_lib_corpus)

output_notebook()

#we have 2 as the minimum because we want the documents to be clustered and not fall under a single group
#we have 4 as the maximum because we assume that each document can belong to a unique topic, and anything more
#than that is too mayn for the sample that we have

p = figure(plot_height=250)
p.line(range(2, 4), topic_model.arun_metric(min_num_topics=2, max_num_topics=4, iterations=1), line_width=2)
show(p)

#the output shows that two is the optimal number of topic.
#https://github.com/AdrienGuille/TOM/blob/388c71ef0da7190740f19e5e8a838df95521a06e/TOM.ipynb

'''
Step 6: Create an LDA model for two topics
'''

ldamodel_2 = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=20)
print(ldamodel_2)

'''
--------------------------------------------------------------------------------------------------------------------------
    This is for 1(c).
    To convert the reviews to feature vectors, 
--------------------------------------------------------------------------------------------------------------------------
'''
print('\n\nThis is for 1(c): convert the reviews into feature vectors using the model that you built. ')

'''
Step 1: View the top words from the LDA model built in 1(b)
'''
print('\nThis shows the top words (based on probability) per topic cluster')
print(ldamodel_2.print_topics(num_topics=2, num_words=3))

'''
Step 2: Apply the LDA model to the corpus to see how the reviews are clustered together
'''
corpus_on_LDA = ldamodel_2[corpus]
print('\nThis shows the reviews that fall under each topic cluster')
cluster1 = [j for i,j in zip(corpus_on_LDA,feedback) if i[0][1] > 0.80]
cluster2 = [j for i,j in zip(corpus_on_LDA,feedback) if i[1][1] > 0.80]
#threshold has been set at 80% in this case, but can be tested if reasonable https://stackoverflow.com/questions/20984841/topic-distribution-how-do-we-see-which-document-belong-to-which-topic-after-doi
print(cluster1) #['I worked with Donna for 5 years and can highly recommend her as an experienced, knowledgeable. She is very hardworking designer with a meticulous eye for detail and creativity', 'I have worked with Donna for a number of years and have always found her to be extremely professional and a real expert in her field. ']
print(cluster2) #['Donna is always on the ball, highly organised and a real pleasure to work with.', 'Justin is a consummate professional and a pleasure to work with at all times!']

'''
Step 3: Apply the LDA model to the corpus to convert the reviews into feature vectors stored in 'feature_data'
'''
all_topics = ldamodel_2.get_document_topics(corpus, per_word_topics=True)

i=0
feature_data = []

for review_topics, word_topics, phi_values in all_topics:
    feature_details = {'Review':None, 'Document Topics':None, 
                       'Word Topics':None,'Phi Values':None}
    feature_details['Review'] = feedback[i]
    feature_details['Document Topics'] = review_topics
    feature_details['Word Topics'] = word_topics
    feature_details['Phi Values'] = phi_values
    feature_data.append(feature_details)
    i = i + 1

print(feature_data)
#https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/topic_methods.ipynb


'''
*************************************************************************************************************************
    2/ Now, the team would like to search the reviews. Given a free-text string, called search_review, 
    can you write another python script to compare the free-text string to all the review feature vectors in the database. 
    The team would like to rank the comparisons and display the top three on the screen (python print).
*************************************************************************************************************************
'''
#http://web.archive.org/web/20081224234350/http://www.dcs.shef.ac.uk/~sam/stringmetrics.html
print('\n\nThis is for 2/. Rank the comparisons and display the top 3')

'''
Step 1: Create a string variable called 'search_review' based on the dictionary
'''
search_review = []
given_data = {'search_review': 'Excellent professional attitude that works across the whole company and is a pleasure to work with'}
search_review.append(given_data['search_review'])
print(search_review)

'''
Step 2: Pre-process search_review and convert to a vector
'''

def vector(review):
    processed_data = preprocess(review) #function defined in 1(b)
    dictionary_data = corpora.Dictionary(processed_data)    
    corpus_data = [dictionary_data.doc2bow(item) for item in processed_data]
    vec_data = ldamodel_2[corpus_data]
    return vec_data

search_vec = vector(search_review)
print(search_vec)

'''
Step 3: Create a similarity matrix
'''
print(corpus_on_LDA) #corpus is based on all 4 reviews from review_data. refer to 1(b) and (c)
index = similarities.MatrixSimilarity(ldamodel_2[corpus]) 
sims = index[search_vec] 
print(list(enumerate(sims)))

'''
Step 4: Sort and print top 3 reviews
'''
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)

print(sims[0:2])
#https://radimrehurek.com/gensim/tut3.html