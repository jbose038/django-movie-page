import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition.truncated_svd import TruncatedSVD
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from django.conf import settings

class movie_recommendation_cluster:
    def __init__(self, **kargs):
        self.topn = kargs.get('topn', 10)
        self.df = kargs.get('data', pd.read_csv(settings.BASE_DIR+'/recom/data/movies04293.csv'))
        self.a, self.b, self.c = kargs.get('a',0.8), kargs.get('b',0.1), kargs.get('c',0.1)
        self.n_clusters = kargs.get('n_clusters',30)# kmeans
        self.n_components = kargs.get('n_components', 500)# svd
        self.vote_thres = kargs.get('vote_thres',100)# vote_count
        self.verbose = kargs.get('verbose', 1)
        self.re_cluster = kargs.get('re_cluster', 1)# kmeans
        self.batch_size = kargs.get('batch_size', 2000)
        self.max_iter = kargs.get('max_iter', 500)
        
        self.cvec = CountVectorizer(min_df=0, ngram_range=(1,2))
        self.stops = []
        with open(settings.BASE_DIR+'/recom/data/total_stopwords', encoding='utf-8') as f:
            self.stops.append(f.readline()[:-2])
        
        if self.verbose == 1:
            print('-'*35)
            print('# Parameters')
            print('      a, b, c        : {0}, {1}, {2}'.format(self.a, self.b, self.c))
            print('vote count threshold :', self.vote_thres)
            print("n_components of SVD  :", self.n_components)
            print("n_clusters of KMeans :", self.n_clusters)
            print('batch_size of Kmeans :', self.batch_size)
            print('max_iter of Kmeans   :', self.max_iter)
            print('weighted_sum = dist_scaled*{0}(a) + genre_scaled*{1}(b) + wvote_scaled*{2}(c)'.format(self.a, self.b, self.c))
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
        
    def raw_to_tfidf(self, data_preprocess):
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,3),stop_words=self.stops,
                                     min_df=3, max_df=0.95, max_features=10000)
        return tfidf.fit_transform(data_preprocess)
    def tfidf_to_svd(self, data_tfidf):
        svd = TruncatedSVD(n_components=self.n_components, n_iter=10)
        return svd.fit_transform(data_tfidf)
    
    def similar_cluster_movies(self, title_idx):
        do_cluster, loop_cnt = True, 0
        
        # data preprocessing
        data_tfidf = self.raw_to_tfidf(list(map(str, self.df['plot_preprocessed_kkma'].values)))
        data_svd = self.tfidf_to_svd(data_tfidf)
        
        # K-means clustering
        print('Clustering...')
        while do_cluster:
            kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size,
                                     max_iter=self.max_iter, verbose=0 ,n_init=3)

            vote_over_thres_idx = self.df[self.df['vote_count'] > self.vote_thres].index
            data_svd_idx = np.array([(idx,val) for idx,val in zip(self.df.index,data_svd)])
            data_svd_to_km = [val for idx,val in data_svd_idx if idx in vote_over_thres_idx]
            data_svd_dict = dict([(idx,val) for idx,val in filter(lambda x: x[0] in vote_over_thres_idx, data_svd_idx)])
            
            # (optional)avoid biggest cluster
            km = kmeans.fit(data_svd_to_km)
            km_dict = dict([(df_idx,label_) for df_idx,label_ in zip(vote_over_thres_idx,km.labels_)])
            km_cluster = list(filter(lambda x: km_dict.get(x) == km_dict.get(title_idx), km_dict.keys()))

            clusters = [0]*self.n_clusters
            for label_ in km.labels_:
                clusters[label_] += 1

            clusters_idx = np.array(clusters).argsort()
            bad_clusters = clusters_idx[-3:]
            
            if self.re_cluster:            
                if km_dict.get(title_idx) not in bad_clusters:
                    do_cluster=False
                elif loop_cnt >= 20:
                    print('Loop count exceeded')
                    do_cluster=False
                else:
                    del kmeans
                    loop_cnt += 1
                    print('Re-clustering...(%d)'%(loop_cnt))
                    
            else:
                do_cluster = False

        if self.verbose == 1:
            print('-'*35)
            print('# K-means clustering distribution')
            for i,size in enumerate(clusters):
                postfix = '<==' if i == km_dict.get(title_idx) else ''
                print('cluster #%3d : %4d items %s'%(i,size,postfix))
            print('-'*35)

        closest = []
        for i in km_cluster:
            if i != title_idx:
                closest.append((i,euclidean(data_svd_dict.get(title_idx), data_svd_dict.get(i))))

        return np.array(closest), self.df.loc[np.array(sorted(closest, key=lambda x: x[1]))[:,0]]

    def result_by_weights(self, dataf):
        dataf['weighted_sum'] = dataf['dist_scaled']*self.a + dataf['genre_scaled']*self.b + dataf['wvote_scaled']*self.c
        
        return dataf.sort_values('weighted_sum', ascending=False)

            
    def getMovies(self, title):
        # no title result
        try: title_idx = self.df[self.df['title']== title].index.values[0]
        except:
            raise ValueError('There is no such title name. Search with "search_title" function')
        
        # get movies in same cluster
        dist, result = self.similar_cluster_movies(title_idx)
        
        # merge with distance
        result = pd.merge(result, pd.Series(dist[:,1], name='dist'), left_on=result.index, right_on=dist[:,0])
        result.rename(columns={'key_0':'idx'}, inplace=True)
        
        # IMDB's weighted_vote
        def weighted_vote_average(record):
            v, r = record['vote_count'], record['rating']
            return (v/(v+m))*r + (m/(m+v))*c
        c = result['rating'].mean()
        m = result['vote_count'].quantile(.6)
        result['weighted_vote'] = result.apply(weighted_vote_average,axis=1)
        
        # merge with genre
        genre_sim = self.genre_similarity(title_idx)
        result_with_genre = pd.merge(result, pd.Series(genre_sim[:,1], name='genre_sim'), left_on=result.idx, right_on=genre_sim[:,0],)
        
        # minmax scale
        result_with_genre['wvote_scaled'] = MinMaxScaler().fit_transform(result_with_genre['weighted_vote'].values.reshape(-1,1))
        result_with_genre['genre_scaled'] = MinMaxScaler().fit_transform(result_with_genre['genre_sim'].values.reshape(-1,1))
        result_with_genre['dist_scaled'] = MinMaxScaler().fit_transform(result_with_genre['dist'].max() - result_with_genre['dist'].values.reshape(-1,1))
        
        # (optional)remove data with 0 genre score
        no_genre_score_idx = result_with_genre[result_with_genre['genre_sim'] == 0].index
        result_with_genre.drop(no_genre_score_idx, inplace=True)
        
        result_with_genre = self.result_by_weights(result_with_genre)
        return result_with_genre.head(self.topn)
