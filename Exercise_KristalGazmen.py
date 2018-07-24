'''
Created on 17 Jul 2018

@author: Kristal Gazmen
'''
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from tom_lib.visualization import visualization
from tom_lib.nlp.topic_model import LatentDirichletAllocation,\
    NonNegativeMatrixFactorization
from tom_lib.visualization.visualization import Visualization

'''
    1/ Can you write a python script (or use another language of your choice) to 
        (a) parse the following  json data stream, 
        (b) create an LDA or another model of your choice for the review data and 
        (c) convert the reviews into feature vectors using the model that you built. 
    Command-line driven scripts are fine (e.g., with input and output filenames)
    Note:                                      
        We are not looking for perfectly executable code and will not run the script. 
        We put more emphasis on the quality of the models built and the understandability/layout of the code. 
'''

'''
*************************************************************************************************************************
    This is for 1(a).
    json.dumps, json.loads, and nested for statements were used to parse the data
*************************************************************************************************************************
'''

import json
import os
import string

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
Step 2: use 'json.loads' to load the review data;
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
*************************************************************************************************************************
    This is an alternative solution to 1(a).
    Package: os, json
    
    In this solution 'employee,' was not added to create a nested dictionary for the json data stream.
    However, a .json file was created and read, while json.dumps wasn't used.
    Once the json has been loaded, parsing and assigning the values to an empty array are the same as the previous solution.
    
    Risk: replacing single quotation with double quotes, especially if double quotes were used inside the feedback provided
*************************************************************************************************************************
'''

#https://stackoverflow.com/questions/27907633/multiple-json-objects-in-one-file-extract-by-python/27907893

'''
Step 1: Update the file path and read the file
'''


script_path = os.path.abspath(__file__) 
script_dir = os.path.split(script_path)[0]
rel_path = '../../../../Desktop/Kristal/1. Personal/e. CVs/Connected/Review_Data.json'
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
*************************************************************************************************************************
    This is for 1(b).
    json.dumps, json.loads, and nested for statements were used to parse the data
*************************************************************************************************************************
'''

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models
import gensim
import tom_lib  #part of the rquirements for tom_lib
import lda

#https://rpubs.com/barberje/LDA
#https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
#https://pypi.org/project/tom_lib/
#https://pythonprogramminglanguage.com/bag-of-words/
'''
Step 1: View feedbacks as list
'''

feedback = parsed_data[0]['recommendations'] + parsed_data[1]['recommendations']
print(feedback)


'''
Step 2:

'''
tokenizer = RegexpTokenizer(r'\w+')
stop_ref = get_stop_words('en')
punc_ref = set(string.punctuation)
lemma = WordNetLemmatizer()
#stemmer = PorterStemmer()
stemmer = SnowballStemmer('english')

'''
Step 3: Create an empty array for pre-processed data
'''
processed_feedback = []
'''
Step 4: Cleanse the data by performing the following:
    (i) tokenizing the string into words
    (ii) removing stop words, such as articles
    (iii) stemming the words to their root form
'''

for item in feedback:   
    feedback_lc = item.lower()
    
    word_i = tokenizer.tokenize(feedback_lc)
    word_ii = [item for item in word_i if not item in stop_ref]
    word_iii = [item for item in word_ii if item not in punc_ref]
    word_iv = [lemma.lemmatize(item) for item in word_iii]
    word_v = [stemmer.stem(item) for item in word_iv]

    processed_feedback.append(word_v)

dictionary = corpora.Dictionary(processed_feedback)
print(dictionary)

corpus = [dictionary.doc2bow(item) for item in processed_feedback]
print(corpus)
#######
'''
id    title    text
1    Document 1's title    This is the full content of document 1.
2    Document 2's title    This is the full content of document 2.
etc.
https://github.com/AdrienGuille/TOM/blob/master/README.md
'''
corpus = Corpus(processed_feedback, language='english',vectorization='tfidf',
                n_gram=1, max_relative_frequency=0.8, min_absolute_frequency=4)
#
topic_model = LatentDirichletAllocation(corpus)
topic_model.infer_topics(num_topics=3, algorithm='gibbs')

viz = Visualization(topic_model)

viz.plot_greene_metric(min_num_topics=2,
                       max_num_topics=6,
                       tao=2, step=1,
                       top_n_words=2)
viz.plot_arun_metric(min_num_topics=2,
                     max_num_topics=6,
                     iterations=1)
viz.plot_brunet_metric(min_num_topics=2,
                       max_num_topics=6,
                       iterations=1)

######
ldamodel_2 = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=20)
ldamodel_3 = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
ldamodel_4 = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word=dictionary, passes=20)
ldamodel_5 = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)
ldamodel_6 = gensim.models.ldamodel.LdaModel(corpus, num_topics=6, id2word=dictionary, passes=20)

print(ldamodel_2)


'''
Print results
'''

print(ldamodel_4.print_topics(num_topics=4, num_words=5))
'''
*************************************************************************************************************************
    This is for 1(c).
    To convert ther reviews to feature vectors, 
*************************************************************************************************************************
'''
#https://stackoverflow.com/questions/20984841/topic-distribution-how-do-we-see-which-document-belong-to-which-topic-after-doi

'''
2/ Now, the team would like to search the reviews. Given a free-text string, called search_review, 
can you write another python script to compare the free-text string to all the review feature vectors in the database. 
The team would like to rank the comparisons and display the top three on the screen (python print).
'''

given_data = {'search_review': 'Excellent professional attitude that works across the whole company and is a pleasure to work with'}

search_review = given_data['search_review']
print(search_review)
