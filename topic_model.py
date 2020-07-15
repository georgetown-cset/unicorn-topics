from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
import pandas as pd
import argparse
import os

class TopicModel:

    def __init__(self, num_features, num_topics, top_words, top_documents):
        self.num_features = num_features
        self.num_topics = num_topics
        self.top_words = top_words
        self.top_documents = top_documents
        self.documents = []
        self.df = None
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=self.num_features, stop_words='english')
        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=self.num_features, stop_words='english')

    def get_data(self):
        if os.path.exists("data/intermediate/documents.pkl"):
            self.df = pd.read_pickle("data/intermediate/documents.pkl")
        else:
            query = "SELECT id, title, CONCAT(title, '. ', abstract) as texts FROM " \
                "project_unicorn.coauthors_dimensions_publications_with_abstracts"
            self.df = pd.read_gbq(query, project_id='gcp-cset-projects')
            pd.to_pickle(self.df, "data/intermediate/documents.pkl")
        self.documents = [value[2] for value in self.df.iloc[0:].values]

    def fit_nmf_model(self):
        tfidf = self.tfidf_vectorizer.fit_transform(self.documents)
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names()
        nmf_model = NMF(n_components=self.num_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
        nmf_W = nmf_model.transform(tfidf)
        nmf_H = nmf_model.components_
        self.display_topics(nmf_H, nmf_W, tfidf_feature_names)

    def fit_lda_model(self):
        tf = self.tf_vectorizer.fit_transform(self.documents)
        tf_feature_names = self.tf_vectorizer.get_feature_names()
        lda_model = LatentDirichletAllocation(n_components=self.num_topics, max_iter=5, learning_method='online',
                                              learning_offset=50., random_state=0).fit(tf)
        lda_W = lda_model.transform(tf)
        lda_H = lda_model.components_
        self.display_topics(lda_H, lda_W, tf_feature_names)

    def display_topics(self, H, W, feature_names):
        for topic_idx, topic in enumerate(H):
            print(f"Topic {topic_idx}:")
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-self.top_words - 1:-1]]))
            top_doc_indices = np.argsort(W[:, topic_idx])[::-1][0:self.top_documents]
            for doc_index in top_doc_indices:
                print(self.df.loc[self.df.index[self.df["texts"] == self.documents[doc_index]][0], "title"])
            print("--------------")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_topics", type=int,
                        help="The number of topics for the model")
    parser.add_argument("-w", "--top_words", type=int, help="How many words to print out from each topic",
                        required=False, default=10)
    parser.add_argument("-d", "--top_documents", type=int, help="How many document titles to print out from each topic",
                        required=False, default=3)
    parser.add_argument("-f", "--num_features", type=int,
                        help="The number of features to use in the model", required=False, default=1000)
    args = parser.parse_args()
    if not args.num_topics:
        parser.print_help()
    model = TopicModel(args.num_features, args.num_topics, args.top_words, args.top_documents)
    print("Getting data")
    model.get_data()
    print("Fitting NMF Model")
    model.fit_nmf_model()
    print("-----------------")
    print("Fitting LDA Model")
    model.fit_lda_model()


if __name__ == "__main__":
    main()