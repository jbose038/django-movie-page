# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import warnings
warnings.filterwarnings('ignore')

from django.shortcuts import render, redirect, get_object_or_404
from .forms import CommentForm
from .models import Comment
from django.conf import settings
from django.http import HttpResponse
DATA_DIR = settings.BASE_DIR+'/recom/data/'

from .by_vec import movie_recommendation_vec
from .by_keyword import movie_recommendation
from .by_clustering import movie_recommendation_cluster

# spacing correction
import pycrfsuite
from .crf import correcter

# keras model
import tensorflow as tf
import re
from keras.models import load_model
graph1, m1 = tf.get_default_graph(), load_model(DATA_DIR+'charcnn_model.hdf5')
graph2, m2 = tf.get_default_graph(), load_model(DATA_DIR+'bi-lstm_word_model.hdf5')

# data
import numpy as np
import pandas as pd
data = pd.read_csv(DATA_DIR+'movies04293.csv', encoding='utf-8')

def get_original_size_img(data):
	if type(data) is str: # single
		return data.split('?')[0]
	else:
		return data.apply(lambda x: x.split('?')[0])

# Create your views here.
def index(request):
	return render(request, 'index.html')

def base(request):
	return render(request, 'base.html')

def home(request):
	return render(request, 'home.html')

def about(request):
	return render(request, 'about.html')

def search(request):
	if request.method == 'POST':
		search_word = str(request.POST.get('title_name'))
		msg = ''
		if len(search_word) == 0:
			msg = '영화 제목이 없습니다.'
		else:
			result_idx = data[(data['title'].str.replace(' ','').str.contains(search_word)) | (data['title'].str.contains(search_word)) |\
							   data['main_act'].str.replace(' ','').str.contains(search_word) | data['main_act'].str.contains(search_word)].index
							   #data['supp_act'].str.replace(' ','').str.contains(search_word) | data['supp_act'].str.contains(search_word)].index
			if result_idx.shape[0] == 0:
				msg = '검색 결과가 없습니다.'
			else:
				result = data.loc[result_idx,['img_url','title']]
				result['img_url'] = get_original_size_img(result['img_url'])
				
				result = [(img,title,idx) for (img,title),idx in zip(result.values,result_idx)]
				context = {'result': result,
				'search_word': search_word
				}
				return render(request, 'search.html', context)
		return render(request, 'search.html', {'msg':msg})
	return render(request, 'search.html')

def detail(request, movie_idx):
	# 영화 데이터
	movie_info = data.iloc[movie_idx]
	movie_info['img_url'] = get_original_size_img(movie_info['img_url'])
	if '|' in str(movie_info['genre']):
		movie_info['genre'] = movie_info['genre'].replace('|', ', ')
	if '|' in str(movie_info['main_act']):
		movie_info['main_act'] = movie_info['main_act'].replace('|', ', ')
	if '|' in str(movie_info['supp_act']):
		movie_info['supp_act'] = movie_info['supp_act'].replace('|', ', ')
	movie_info['page_idx'] = movie_idx

	# 페이지에 필요한 요소들
	result = dict([(col,data) for col,data in zip(data.columns,movie_info.values)])
	comments = Comment.objects.filter(movie_idx=movie_idx).order_by('-comment_date')
	comment_form = CommentForm()

	comments_pos = comments.filter(comment_sentiment=1)
	comments_neg = comments.filter(comment_sentiment=0)	
	context = {
	'movie': result,
	'comments_pos': comments_pos,
	'comments_neg': comments_neg,
	'comment_form': comment_form
	}

	# leave comment
	if request.method == 'POST':
		comment_text = request.POST.get('comment_area')
		sentiment = comments_sentiment([comment_text])[0]
		print(comment_text,'의 감성분석은 ',sentiment)
		comment_form = Comment(movie_idx=movie_idx,
		comment_textfield=comment_text,
		comment_sentiment=sentiment,
		comment_thumbnail_url=settings.MEDIA_URL+'homer_head.jpg')

		comment_form.save()

		return render(request, 'detail.html', context)
	else:
		return render(request, 'detail.html', context)

def recommend(request):
	if request.method == 'POST':
		search_name = str(request.POST.get('title_name'))
		radio_val = (request.POST.get('type'))
		msg = ''
		print('radio_val : ',radio_val)

		if len(search_name) == 0:
			msg = '영화 제목이 없습니다'
		elif radio_val is None:
			msg = '추천 방식을 선택해 주세요.'
		else:
			result_idx = data[(data['title'].str.replace(' ','').str.contains(search_name)) | (data['title'].str.contains(search_name))].index

			if result_idx.shape[0] == 0:
				msg = '검색 결과가 없습니다.'
			else:
				result = data.loc[result_idx,['img_url','title']]
				result['img_url'] = get_original_size_img(result['img_url'])
				result = [(img,title,idx) for (img,title),idx in zip(result.values,result_idx)]
				
				context = {
				'result':result,
				'radio_val': radio_val
				}
				return render(request, 'recommend.html',context)

		return render(request, 'recommend.html', {'msg':msg})
	else:
		return render(request, 'recommend.html')

def recomm_result(request, movie_idx, radio_val):
	print('radio_val : ',radio_val)
	movie_title = data.iloc[movie_idx].title

	if radio_val == 'k':
		recomm = movie_recommendation(data=data, verbose=1)
	elif radio_val == 'v':
		recomm = movie_recommendation_vec(vec_type='tfidf', data=data, verbose=1)
	else:# radio_val == 'c'
		recomm = movie_recommendation_cluster(data=data, verbose=1)
	movies = recomm.getMovies(movie_title)
	print(movies.title)

	result = [(get_original_size_img(img),title,int(idx)) for (img,title),idx in zip(movies[['img_url','title']].values, movies['key_0'])]

	radio_mapping = {'k':'키워드 기반', 'v':'벡터 기반', 'c':'클러스터링 기반'}
	radio_val = radio_mapping.get(radio_val)

	print(result)
	context = {
	'src_movie':movie_title,
	'result' : result,
	'radio_val' :radio_val
	}
	del recomm
	return render(request, 'recomm_result.html', context)

def auto_spacing(request):
	#print('comment from ajax :',request.GET['comment_ajax'])
	comment_corrected = correcter(request.GET['comment_ajax'])
	
	return HttpResponse(comment_corrected)


def sent_to_chars(sentences):
	BASE_CODE, CHO, JUNG = 44032, 588, 28
	BEGIN_OF_JONG, END_OF_JONG = 12593, 12622
	BEGIN_OF_JUNG, END_OF_JUNG = 12623, 12643

	CONSONANT = [chr(i) for i in range(ord('ㄱ'), ord('ㅎ')+1)]

	CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
	JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
	JONGSUNG_LIST = ['#', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

	def syllable_decomp(sent):
		sent_list = list(sent)

		result = []
		for syl in sent_list:
			if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*',syl) is not None:
				if ord(syl) < BASE_CODE:
					char_code = ord(syl)
					if char_code <= END_OF_JONG:
						char = int(char_code - BASE_OF_JONG)
						result.append(CONSONANT[char])
					else:
						char = int(char_code - BEGIN_OF_JUNG)
						result.append(JUNGSUNG_LIST[char])
				else:
					char_code = ord(syl) - BASE_CODE
					char1 = int(char_code / CHO)
					result.append(CHOSUNG_LIST[char1])

					char2 = int((char_code - (CHO*char1)) / JUNG)
					result.append(JUNGSUNG_LIST[char2])

					char3 = int((char_code - (CHO*char1) - (JUNG*char2)))
					result.append(JONGSUNG_LIST[char3])
			else:
				result.append(syl)
		return ''.join(result)

	max_len = 512
	char_dict = ''.join(sorted(''.join(set(CHOSUNG_LIST+JUNGSUNG_LIST+JONGSUNG_LIST)))) + '0123456789 .!?:,\'%-\(\)/$|&;[]"'

	unknown_label = 'UNK'
	chars = [unknown_label]
	for c in char_dict:
		chars.append(c)

	n_chars = len(char_dict)
	char2idx = dict((c,i) for i,c in enumerate(chars))
	idx2char = dict((i,c) for i,c in enumerate(chars))

	arr_s = np.zeros((len(sentences),max_len))
	for i,sent in enumerate(sentences):
		sent_decomp = syllable_decomp(sent)

		for j,char in enumerate(sent_decomp):
			if char in char_dict:
				arr_s[i, (max_len-len(sent_decomp)+j)] = char2idx[char]
			else:
				arr_s[i, (max_len-len(sent_decomp)+j)] = char2idx[unknown_label]
	arr_s = arr_s.reshape(-1,1,max_len)
	return arr_s

def sent_to_sequence(sentence):
	import konlpy
	from keras.preprocessing.sequence import pad_sequences
	import pickle

	max_len = 70
	okt = konlpy.tag.Okt()
	with open(DATA_DIR+'tokenizer.pickle','rb') as tok:
		tokenizer = pickle.load(tok)

	seq = tokenizer.texts_to_sequences(okt.morphs(sentence, stem=True))
	seq = [lst[0] for lst in seq if lst]

	return pad_sequences([seq], maxlen=70)

def comments_sentiment(sentence):
	print('Comment Input :',sentence)
	sents_m1 = sent_to_chars(sentence)
	sents_m2 = sent_to_sequence(sentence[0])
	
	global graph1, graph2, m1, m2
	#graph1, m1 = tf.get_default_graph(), load_model(settings.BASE_DIR+'/recom/data/charcnn_model.hdf5')
	with graph1.as_default():
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			m1.load_weights(DATA_DIR+'charcnn_weights.hdf5')
			preds1 = m1.predict(sents_m1)

	#graph2, m2 = tf.get_default_graph(), load_model(settings.BASE_DIR+'/recom/data/bi-lstm_char_model.hdf5')
	with graph2.as_default():
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			m2.load_weights(DATA_DIR+'bi-lstm_word_weights.hdf5')
			preds2 = m2.predict(sents_m2)

	print(preds1, preds2)
	return list(map(int, (((preds1+preds2) / 2) > 0.45)))

