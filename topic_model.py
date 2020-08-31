from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import argparse
import os
import re
import spacy
import scispacy
import pickle
from collections import Counter, defaultdict

nlp = spacy.load("en_core_sci_lg", disable=['ner'])


def tokenizer_lemmatizer(text):
    """
    Uses spacy to tokenize and lemmatize a given string
    :param text: String to tokenize
    :return: list of tokens
    """
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    tokens = tokenizer(text)
    # nlp.Defaults.stop_words.add("model")
    lemma_list = [i.lemma_ for i in tokens if not i.is_stop]
    return lemma_list


def preprocessor(text):
    """
    Removes punctuation and whitespace from a token list.
    :param text: Either the token list or the individual token (recursive function)
    :return: Either the processed token or the full list of tokens
    """
    if isinstance(text, str):
        text = re.sub('<[^>]*>', '', text)
        text = re.sub('[\W]+', '', text.lower())
        return text
    if isinstance(text, list):
        return_list = [preprocessor(i) for i in text if preprocessor(i)]
        return return_list


def sentence_join(sentence_list):
    """
    Turns a list of tokens into a string of tokens separated by a space.
    This is done because LDA takes in strings as input, not lists.
    :param sentence_list: token list
    :return: string of tokens
    """
    return " ".join(sentence_list)


def pipelinize(function, active=True):
    """
    Turns preprocessor into a pipeline that runs all the provided preprocessing functions in order
    :param function: Function to be put in pipeline
    :param active: If function is active
    :return:
    """
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            return [function(i) for i in list_or_series]
        else:  # if it's not active, just pass it right back
            return list_or_series

    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active': active})


class TopicModel:

    def __init__(self, num_features, num_topics, top_words, top_documents):
        """
        Initializes the topic model
        :param num_features: Number of model features for training
        :param num_topics: Number of topics to create
        :param top_words: Number of words to print out from each topic
        :param top_documents: Number of document titles to print out from each topic
        """
        self.num_features = num_features
        self.num_topics = num_topics
        self.top_words = top_words
        self.top_documents = top_documents
        self.documents = []
        self.df = None
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=self.num_features,
                                                stop_words='english')
        self.tf_vectorizer = CountVectorizer(max_df=0.8, min_df=2, max_features=self.num_features, stop_words='english')
        # self.tf_vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english', max_features = self.num_features,
                                             # ngram_range=(1, 2))
        self.vocab = None
        self.documents_map = {}
        self.document_topics = {}
        self.document_max = defaultdict(float)

    def get_data(self):
        """
        Acquire data to run the model with.
        Data either comes from BQ or is loaded from pickle if it's been previously pulled.
        :return:
        """
        if os.path.exists("data/intermediate/documents.pkl"):
            self.df = pd.read_pickle("data/intermediate/documents.pkl")
        else:
            query = """SELECT id, title, CONCAT(title, '. ', abstract) as texts, coauthors, `year` FROM 
                project_unicorn.coauthors_dimensions_publications_with_abstracts"""
            self.df = pd.read_gbq(query, project_id='gcp-cset-projects')
            pd.to_pickle(self.df, "data/intermediate/documents.pkl")

    def preprocess_data(self):
        """
        Run the preprocessing functions. If the preprocessing functions
        have already been run, load preprocessed data from pickle.
        :return:
        """
        if os.path.exists("data/intermediate/preprocessed_abstracts.pkl"):
            with open("data/intermediate/preprocessed_abstracts.pkl", "rb") as file_in:
                self.documents = pickle.load(file_in)
        else:
            self.documents = [value[2] for value in self.df.iloc[0:].values]
            estimators = [('tokenizer', pipelinize(tokenizer_lemmatizer)), ('preprocessor', pipelinize(preprocessor)),
                          ('sentence_join', pipelinize(sentence_join))]
            pipe = Pipeline(estimators)
            self.documents = pipe.transform(self.documents)
            with open("data/intermediate/preprocessed_abstracts.pkl", "wb") as file_out:
                pickle.dump(self.documents, file_out)
        assert len(self.documents) == len(self.df)
        # Create map from preprocessed document string to document id
        for i, doc in enumerate(self.documents):
            self.documents_map[doc] = self.df.iloc[i, 0]  # get the ids
        print(f"Total Documents: {len(self.documents)}")

    def fit_nmf_model(self, run_number):
        """
        Fit the NMF model. Run if --nmf flag set.
        Once model fit, divide documents by topic, display topic results, and save topics by year.
        :return:
        """
        tfidf = self.tfidf_vectorizer.fit_transform(self.documents)
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names()
        nmf_model = NMF(n_components=self.num_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
        nmf_W = nmf_model.transform(tfidf)
        nmf_H = nmf_model.components_
        self.divide_documents_by_topic(nmf_H, nmf_W)
        topics_by_year = self.display_topics(nmf_H, nmf_W, tfidf_feature_names)
        if not os.path.exists(f"data/intermediate/nmf_t_{self.num_topics}_r_{run_number}"):
            os.mkdir(f"data/intermediate/nmf_t_{self.num_topics}_r_{run_number}")
        with open(f"data/intermediate/nmf_t_{self.num_topics}_r_{run_number}/topics_by_year.pkl", "wb") as file_out:
            pickle.dump(topics_by_year, file_out)

    def fit_lda_model(self, run_number):
        """
        Fit the LDA model. Run if --nmf flag not set.
        Once model fit, divide documents by topic, display topic results, and save topics by year.
        :return:
        """
        tf = self.tf_vectorizer.fit_transform(self.documents)
        self.vocab = self.tf_vectorizer.get_feature_names()
        print(f"Total Vocab: {len(self.vocab)}")
        tf_feature_names = self.tf_vectorizer.get_feature_names()
        lda_model = LatentDirichletAllocation(n_components=self.num_topics, max_iter=150, learning_method='online',
                                              learning_offset=50., random_state=0).fit(tf)
        lda_W = lda_model.transform(tf)
        lda_H = lda_model.components_
        if not os.path.exists(f"data/intermediate/t_{self.num_topics}_r_{run_number}"):
            os.mkdir(f"data/intermediate/t_{self.num_topics}_r_{run_number}")
        with open(f"data/intermediate/t_{self.num_topics}_r_{run_number}/lda_H.pkl", "wb") as file_out:
            pickle.dump(lda_H, file_out)
        with open(f"data/intermediate/t_{self.num_topics}_r_{run_number}/lda_W.pkl", "wb") as file_out:
            pickle.dump(lda_W, file_out)
        self.divide_documents_by_topic(lda_H, lda_W)
        topics_by_year, top_topics_by_org = self.display_topics(lda_H, lda_W, tf_feature_names)
        with open(f"data/intermediate/t_{self.num_topics}_r_{run_number}/topics_by_year.pkl", "wb") as file_out:
            pickle.dump(topics_by_year, file_out)
        with open(f"data/intermediate/t_{self.num_topics}_r_{run_number}/top_topics_by_org.pkl", "wb") as file_out:
            pickle.dump(top_topics_by_org, file_out)

    def divide_documents_by_topic(self, H, W):
        """
        Divide the documents into topics. This is done by, for each document,
        selecting the topic for which they have the highest weighting.
        This means all documents are assigned some topic.
        :param H: transformed LDA data
        :param W: Variational parameters for topic word distribution
        :return:
        """
        for topic_idx, topic in enumerate(H):
            doc_indices = np.argsort(W[:, topic_idx])[::-1]
            doc_values = np.sort(W[:, topic_idx])[::-1]
            for i, doc_index in enumerate(doc_indices):
                doc_id = self.documents_map[self.documents[doc_index]]
                if doc_values[i] > self.document_max[doc_id]:
                    self.document_max[doc_id] = doc_values[i]
                    self.document_topics[doc_id] = topic_idx

    def display_topics(self, H, W, feature_names):
        """
        Print out, for each topic:
        a) top_words words per topic
        b) top_documents titles per topic
        c) The number of papers assigned to the topic
        d) The paper counts by year
        e) top_documents titles per topic
        f) The number of papers assigned to the topic by each of the topic 6 tech companies of interest
        :param H: transformed LDA data
        :param W: Variational parameters for topic word distribution
        :param feature_names: The names of the features (this provides top words)
        :return: Paper counts by year for all topicsz
        """
        topics_by_year = {}
        top_topics_by_org = defaultdict(Counter)
        for topic_idx, topic in enumerate(H):
            print(f"Topic {topic_idx}:")
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-self.top_words - 1:-1]]))
            top_doc_indices = np.argsort(W[:, topic_idx])[::-1][0:self.top_documents]
            top_doc_probabilities = np.sort(W[:, topic_idx])[::-1]
            for doc_index in top_doc_indices:
                doc_id = self.documents_map[self.documents[doc_index]]
                print(self.df.loc[self.df.index[self.df["id"] == doc_id][0], "title"])
            for prob in top_doc_probabilities[:10]:
                print(prob)
            doc_indices = np.argsort(W[:, topic_idx])[::-1]
            coauthors_count = Counter()
            year_count = Counter()
            papers_count = 0
            for doc_index in doc_indices:
                doc_id = self.documents_map[self.documents[doc_index]]
                # if this document is assigned to this topic
                if self.document_topics[doc_id] == topic_idx:
                    papers_count += 1
                    coauthors = self.df.loc[self.df.index[self.df["id"] == doc_id][0], "coauthors"].split("; ")
                    year = self.df.loc[self.df.index[self.df["id"] == doc_id][0], "year"]
                    coauthors_count.update(coauthors)
                    year_count[int(year)] += 1
            for coauthor in coauthors_count.keys():
                top_topics_by_org[coauthor][topic_idx] += coauthors_count[coauthor]
            print(f"Total papers in topic: {papers_count}")
            topics_by_year[topic_idx] = sorted(list(year_count.items()))
            print(f"Paper counts by year: {topics_by_year[topic_idx]}")
            for coauthor in coauthors_count.most_common(5):
                print(coauthor[0], coauthor[1])  # print coauthor and its count in topic
            print("~~~~~")
            companies = ["Google", "Amazon", "Apple", "Facebook", "IBM", "Microsoft"]
            for company in companies:
                if company in coauthors_count.keys():
                    print(company, coauthors_count[company])
            print("--------------")
        return topics_by_year, top_topics_by_org


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_topics", type=int,
                        help="The number of topics for the model")
    parser.add_argument("run_number", type=int,
                        help="The run number to reference this model by. An integer.")
    parser.add_argument("-w", "--top_words", type=int, help="How many words to print out from each topic",
                        required=False, default=10)
    parser.add_argument("-d", "--top_documents", type=int, help="How many document titles to print out from each topic",
                        required=False, default=0)
    parser.add_argument("-f", "--num_features", type=int,
                        help="The number of features to use in the model", required=False, default=1000)
    parser.add_argument("-n", "--nmf", action="store_true", help="Set this flag to use NMF instead of LDA")
    args = parser.parse_args()
    if not args.num_topics or not args.run_number:
        parser.print_help()
    model = TopicModel(args.num_features, args.num_topics, args.top_words, args.top_documents)
    print("Getting data")
    model.get_data()
    print("Preprocessing data")
    model.preprocess_data()
    print("-----------------")
    if args.nmf:
        print("Fitting NMF Model")
        model.fit_nmf_model(args.run_number)
    else:
        print("Fitting LDA Model")
        model.fit_lda_model(args.run_number)


if __name__ == "__main__":
    main()
