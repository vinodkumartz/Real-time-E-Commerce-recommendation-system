import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix_content = None
        self.train_data = None

    def fit(self, train_data):
        self.train_data = train_data
        self.tfidf_matrix_content = self.tfidf_vectorizer.fit_transform(train_data['Tags'])

    def recommend(self, item_name, top_n=10):
        if item_name not in self.train_data['Name'].values:
            return pd.DataFrame()

        item_index = self.train_data[self.train_data['Name'] == item_name].index[0]
        cosine_similarities_content = cosine_similarity(self.tfidf_matrix_content, self.tfidf_matrix_content)
        similar_items = list(enumerate(cosine_similarities_content[item_index]))
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
        top_similar_items = similar_items[1:top_n+1]
        recommended_item_indices = [x[0] for x in top_similar_items]
        recommended_items_details = self.train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
        
        return recommended_items_details





import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity = None
        self.train_data = None

    def fit(self, train_data):
        self.train_data = train_data
        self.user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
        self.user_similarity = cosine_similarity(self.user_item_matrix)

    def recommend(self, target_user_id, top_n=10):
        # Ensure that user_item_matrix.index is an Index object
        if not isinstance(self.user_item_matrix.index, pd.Index):
                 raise TypeError("user_item_matrix.index is not a valid Index type")

    # Check if target_user_id is in the index
        if target_user_id not in self.user_item_matrix.index:
                  raise ValueError(f"User ID {target_user_id} not found in user-item matrix.")

        target_user_index = self.user_item_matrix.index.get_loc(target_user_id)
        user_similarities = self.user_similarity[target_user_index]
        similar_users_indices = user_similarities.argsort()[::-1][1:]

        recommended_items = []

        for user_index in similar_users_indices:
            rated_by_similar_user = self.user_item_matrix.iloc[user_index]
            not_rated_by_target_user = (rated_by_similar_user == 0) & (self.user_item_matrix.iloc[target_user_index] == 0)
            recommended_items.extend(self.user_item_matrix.columns[not_rated_by_target_user][:top_n])

        recommended_items_details = self.train_data[self.train_data['ProdID'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
        
        return recommended_items_details.head(top_n)
    



import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self):
        self.content_tfidf_vectorizer = None
        self.content_tfidf_matrix = None
        self.user_item_matrix = None
        self.user_similarity = None
        self.train_data = None

    def fit(self, train_data):
        self.train_data = train_data

        # Content-based part: Train TF-IDF vectorizer on item descriptions
        self.content_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.content_tfidf_matrix = self.content_tfidf_vectorizer.fit_transform(train_data['Tags'])

        # Collaborative part: Create user-item matrix and calculate similarity
        self.user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
        self.user_similarity = cosine_similarity(self.user_item_matrix)

    def recommend(self, target_user_id, item_name, top_n=10):
        # Content-based recommendations
        content_based_rec = self._content_based_recommendations(item_name, top_n)
        
        # Collaborative filtering recommendations
        collaborative_filtering_rec = self._collaborative_filtering_recommendations(target_user_id, top_n)
        
        # Combine and deduplicate the recommendations
        hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates().reset_index(drop=True)
        return hybrid_rec.head(top_n)

    def _content_based_recommendations(self, item_name, top_n):
        if item_name not in self.train_data['Name'].values:
            return pd.DataFrame()

        item_index = self.train_data[self.train_data['Name'] == item_name].index[0]
        cosine_similarities_content = cosine_similarity(self.content_tfidf_matrix, self.content_tfidf_matrix)
        similar_items = list(enumerate(cosine_similarities_content[item_index]))
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
        top_similar_items = similar_items[1:top_n+1]
        recommended_item_indices = [x[0] for x in top_similar_items]
        return self.train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    def _collaborative_filtering_recommendations(self, target_user_id, top_n):
        if self.user_item_matrix is None or self.user_similarity is None:
            raise ValueError("Model has not been trained. Call 'fit' with training data first.")
        
        if target_user_id not in self.user_item_matrix.index:
            return pd.DataFrame()

        target_user_index = self.user_item_matrix.index.get_loc(target_user_id)
        user_similarities = self.user_similarity[target_user_index]
        similar_users_indices = user_similarities.argsort()[::-1][1:]

        recommended_items = []

        for user_index in similar_users_indices:
            rated_by_similar_user = self.user_item_matrix.iloc[user_index]
            not_rated_by_target_user = (rated_by_similar_user == 0) & (self.user_item_matrix.iloc[target_user_index] == 0)
            recommended_items.extend(self.user_item_matrix.columns[not_rated_by_target_user][:top_n])

        return self.train_data[self.train_data['ProdID'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]








