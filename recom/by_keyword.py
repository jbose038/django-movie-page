import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import ast
from django.conf import settings

class movie_recommendation:
    def __init__(self, **kargs):
        self.topn = kargs.get('topn', 10)
        self.vote_thres = kargs.get('vote_thres', 100)
        self.df = kargs.get('data', pd.read_csv(settings.BASE_DIR+'/recom/data/movies04293.csv'))
        self.a, self.b, self.c = kargs.get('a',0.8), kargs.get('b',0.1), kargs.get('c',0.1)
        self.verbose = kargs.get('verbose', 1)
        
        self.cvec = CountVectorizer(min_df=0, ngram_range=(1,2))
        self.w2v = word2vec.Word2Vec.load(settings.BASE_DIR+'/recom/data/w2v_movie_plot.model')
        self.scaler = MinMaxScaler()
        
        if self.verbose == 1:
            print('-'*35)
            print('# Parameters')
            print('      a, b, c        : {0}, {1}, {2}'.format(self.a, self.b, self.c))
            print('vote count threshold :', self.vote_thres)
            print('weighted_sum = keywords*{0}(a) + genre*{1}(b) + weighted vote*{2}(c)'.format(self.a, self.b, self.c))
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

    def cos_sim(self, corp1, corp2):
        vec1, vec2 = [], []
        for word1, word2 in zip(corp1,corp2):
            vec1.append(self.w2v[word1])
            vec2.append(self.w2v[word2])

        vec1, vec2 = np.array(vec1).mean(axis=0), np.array(vec2).mean(axis=0)
        return np.inner(vec1,vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))

    def similar_keywords_movies(self, title_idx):
        self.df['keywords_literal'] = self.df['keywords'].apply(lambda x: ' '.join(ast.literal_eval(x)))
        keywords_sims = []
        for idx in self.df.index:
            keywords_src = ast.literal_eval(self.df.loc[title_idx,'keywords'])
            keywords_tgt = ast.literal_eval(self.df.loc[idx,'keywords'])
            keywords_sims.append((idx,self.cos_sim(keywords_src, keywords_tgt)))

        df_with_ksim = self.df.copy()
        df_with_ksim['keywords_sim'] = np.array(keywords_sims)[:,1]
        df_with_ksim = df_with_ksim[df_with_ksim['vote_count'] > self.vote_thres]
        
        return df_with_ksim.sort_values('keywords_sim',ascending=False)[1:]

    def result_by_weights(self, dataf):
        dataf['weighted_sum'] = dataf['keywords_sim_scaled']*self.a + dataf['genre_scaled']*self.b + dataf['wvote_scaled']*self.c
        
        return dataf.sort_values('weighted_sum', ascending=False)
    
    def getMovies(self, title):
        # no title result
        try: title_idx = self.df[self.df['title']== title].index.values[0]
        except:
            raise ValueError('There is no such title name. Search with "search_title" function')
        
        # get movies
        result = self.similar_keywords_movies(title_idx)

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
        result_with_genre['keywords_sim_scaled'] = MinMaxScaler().fit_transform(result_with_genre['keywords_sim'].values.reshape(-1,1))
        result_with_genre['wvote_scaled'] = MinMaxScaler().fit_transform(result_with_genre['weighted_vote'].values.reshape(-1,1))
        result_with_genre['genre_scaled'] = MinMaxScaler().fit_transform(result_with_genre['genre_sim'].values.reshape(-1,1))
        
        # (optional)remove data genre score is 0
        no_genre_score_idx = result_with_genre[result_with_genre['genre_sim'] == 0].index
        result_with_genre.drop(no_genre_score_idx, inplace=True)
        
        result_with_genre = self.result_by_weights(result_with_genre)

        return result_with_genre.head(self.topn)
