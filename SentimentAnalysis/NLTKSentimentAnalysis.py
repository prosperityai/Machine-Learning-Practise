import pandas as pd
import csv
from nltk import sent_tokenize, word_tokenize

positive=[]
negative=[]
over_looking_keys = ['Defined', 'Source', 'Entry']
output_file_name = 'output_sentiment_analysis.csv'

with open('review_dictionary.txt') as fin:
    reader = csv.DictReader(fin,delimiter='\t')
    for i,line in enumerate(reader):
        if line['Negativ']=='Negativ':
            if line['Entry'].find('#')== -1:
                negative.append(line['Entry'].lower())
            if line['Entry'].find('#')!= -1:
                negative.append(line['Entry'].lower()[:line['Entry'].index('#')])
        if line['Positiv']=='Positiv':
            if line['Entry'].find('#')== -1:
                positive.append(line['Entry'].lower())
            if line['Entry'].find('#')!= -1:
                positive.append(line['Entry'].lower()[:line['Entry'].index('#')])
fin.close()

pvocabulary=sorted(list(set(positive)))
nvocabulary=sorted(list(set(negative)))

review = pd.read_csv('review_music_out.csv')
review['p_word_count'] = 0
review['n_word_count'] = 0
review['s_indicator'] = 0
review_index = 0

# Creating word list
def getWordList(text,word_proc=lambda x:x):
    word_list=[]
    for sent in sent_tokenize(text):
        for word in word_tokenize(sent):
            word_list.append(word)
    return word_list


positive_count_list = []
negative_count_list = []
lsenti_list = []

# Creating a count for positive and negative words
for text in review['reviewText']:
    vocabulary = getWordList(text, lambda x: x.lower())
    positive_count = 0
    negative_count = 0
    for pword in pvocabulary:
        positive_count += vocabulary.count(pword)
    for nword in nvocabulary:
        negative_count += vocabulary.count(nword)

    positive_count_list.append(positive_count)
    negative_count_list.append(negative_count)
    lsenti_list.append(positive_count - negative_count)

    review_index += 1
    # DataFrame and Store to external file
    review['p_word_count'] = pd.Series(positive_count_list)
    review['n_word_count'] = pd.Series(negative_count_list)
    review['s_indicator'] = pd.Series(lsenti_list)
    
    train = review[:2000]
    test = review[2000:]
    review.to_csv(output_file_name)