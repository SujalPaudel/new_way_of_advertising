import json
from nltk import word_tokenize
from collections import Counter
from nltk.stem import PorterStemmer

file_de_path = "/home/velar/Documents/python2.0/raju.json"
json_file = open(file_de_path, "r", encoding="UTF-8")
json_load = json.load(json_file)
stemmer = PorterStemmer()


def get_the_url():
	all_tokens = []
	for surf_history in json_load:
 
		tokens = word_tokenize(surf_history['URL'].encode('utf-8').lower().decode('utf-8')) 
				
		all_tokens = all_tokens + tokens

		all_tokens_stemmed = []

	for token in all_tokens:
		stemmed_token = stemmer.stem(token)
		all_tokens_stemmed.append(stemmed_token)

	frequencies = Counter(all_tokens_stemmed)

	for token,count in frequencies.most_common(50):
		print (token,count)

	

get_the_url()
