import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from django.conf import settings

class movie_recommendation_vec:
    def __init__(self, **kargs):
        self.topn = kargs.get('topn',10)
        self.vec_type = kargs.get('vec_type','tfidf')
        self.df = kargs.get('data', pd.read_csv(settings.BASE_DIR+'/recom/data/movies04293.csv'))
        self.a, self.b, self.c = kargs.get('a', 0.6), kargs.get('b', 0.2), kargs.get('c', 0.2)
        self.vote_thres = kargs.get('vote_thres', 100)
        self.verbose = kargs.get('verbose', 1)
        
        self.cvec = CountVectorizer(min_df=0, ngram_range=(1,2))
        self.stops = []
        with open(settings.BASE_DIR+'/recom/data/total_stopwords', encoding='utf-8') as f:
            self.stops.append(f.readline()[:-2])
        
        if self.vec_type == 'tfidf':
            self.vec = TfidfVectorizer(analyzer='word', ngram_range=(1,3),stop_words=self.stops,
                       min_df=3, max_df=0.95, max_features=10000)
        elif self.vec_type == 'count':
            self.vec = CountVectorizer(analyzer='word',ngram_range=(1,1), stop_words=self.stops,
                      min_df=10, max_df=0.95, max_features=10000)
        else:
            raise ValueError('vec_type is not in [tfidf, count]')
            
        if self.verbose == 1:
            print('-'*35)
            print('# Parameters')
            print('      a, b, c        : {0}, {1}, {2}'.format(self.a, self.b, self.c))
            print('vote count threshold :', self.vote_thres)
            print('vec_type :',self.vec_type.capitalize())
            print('weighted_sum = vec_sim_scaled*{0}(a) + genre_scaled*{1}(b) + wvote_scaled*{2}(c)'.format(self.a, self.b, self.c))
            print('-'*35)
            
    def search_title(self, title_name):
        return self.df[self.df['title'].str.contains(title_name)].title

    def genre_cos_sim(self, title_idx, genre_vec):
        genre_sims = []
        for idx,vec in enumerate(genre_vec):
            if idx != title_idx:
                genre_sims.append((idx, cosine_similarity([genre_vec[title_idx]], [vec])[0,0]))
        return genre_sims
    
    def genre_similarity(self, title_idx):
        genre_literal = self.df['genre'].apply(lambda x: x.replace('|',' '))
        genre = self.cvec.fit_transform(genre_literal).toarray()
        genre_sims = self.genre_cos_sim(title_idx, genre)

        return np.array(genre_sims)
        
    def raw_to_vec(self, data_preprocess):
        return self.vec.fit_transform(data_preprocess)
    
    def cos_sim(self, data_preprocess, title_idx):
        cos_sims = []
        data_transformed = self.raw_to_vec(data_preprocess)
        
        for df_idx, src_idx in zip(self.df.index, range(data_transformed.shape[0])):
            if df_idx != title_idx:
                cos_sims.append((df_idx, cosine_similarity(data_transformed[title_idx].reshape(1,-1),
                                                          data_transformed[src_idx].reshape(1,-1))))
        return cos_sims
    
    def similar_vec_movies(self, title_idx):
        idx_sims = np.array(self.cos_sim(self.df['plot_preprocessed_kkma'], title_idx))
        sims_scaled = MinMaxScaler().fit_transform(idx_sims[:,1].reshape(-1,1))
        idx_sims[:,1] = sims_scaled.reshape(-1)

        idx_sims = np.array(sorted(idx_sims, key=lambda x: x[1], reverse=True))
        result_df = self.df.loc[idx_sims[:,0]]
        result_df['vec_sim'] = idx_sims[:,1]
        
        return result_df[result_df['vote_count'] > self.vote_thres]
    
    def result_by_weights(self, dataf):
        dataf['weighted_sum'] = dataf['vec_sim_scaled']*self.a+ dataf['genre_scaled']*self.b + dataf['wvote_scaled']*self.c
        
        return dataf.sort_values('weighted_sum', ascending=False)
    
    def getMovies(self, title):
        # no title result
        try:title_idx = self.df[self.df['title']== title].index.values[0]
        except:
            raise ValueError('There is no such title name in data. Search with "search_title" function')

        # get movies
        result = self.similar_vec_movies(title_idx)
        
        # IMDB's weighted_vote
        def weighted_vote_average(record):
            v, r = record['vote_count'], record['rating']
            return (v/(v+m))*r + (m/(m+v))*c
        c = result['rating'].mean()
        m = result['vote_count'].quantile(.6)
        result['weighted_vote'] = result.apply(weighted_vote_average,axis=1)
        
        # merge with genre
        genre_sim = self.genre_similarity(title_idx)
        result_with_genre = pd.merge(result, pd.Series(genre_sim[:,1], name='genre_sim'), left_on=result.index, right_on=genre_sim[:,0],)
        
        # minmax scale
        result_with_genre['vec_sim_scaled'] = MinMaxScaler().fit_transform(result_with_genre['vec_sim'].values.reshape(-1,1))
        result_with_genre['wvote_scaled'] = MinMaxScaler().fit_transform(result_with_genre['weighted_vote'].values.reshape(-1,1))
        result_with_genre['genre_scaled'] = MinMaxScaler().fit_transform(result_with_genre['genre_sim'].values.reshape(-1,1))

        # (optional)remove data genre score is 0
        no_genre_score_idx = result_with_genre[result_with_genre['genre_sim'] == 0].index
        result_with_genre.drop(no_genre_score_idx, inplace=True)
        
        result_with_genre = self.result_by_weights(result_with_genre)
        return result_with_genre.head(self.topn)
