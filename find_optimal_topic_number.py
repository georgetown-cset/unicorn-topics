import numpy as np
import pandas as pd
import argparse
import os
import re
import spacy
import scispacy
import pickle
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from collections import Counter, defaultdict
import logging

nlp = spacy.load("en_core_sci_lg", disable=['ner'])
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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


def make_bigrams(documents, bigram_mod):
    return [bigram_mod[document.split(" ")] for document in documents]


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

    def __init__(self, min_topics, max_topics, step):
        """
        Initializes the topic model
        :param num_features: Number of model features for training
        :param num_topics: Number of topics to create
        :param top_words: Number of words to print out from each topic
        :param top_documents: Number of document titles to print out from each topic
        """
        # self.num_features = num_features
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.step = step
        self.documents = []
        self.df = None
        # self.tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=self.num_features,
        #                                         stop_words='english')
        # self.tf_vectorizer = CountVectorizer(max_df=0.8, min_df=2, max_features=self.num_features, stop_words='english')
        # self.tf_vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english', max_features = self.num_features,
        # ngram_range=(1, 2))
        self.vocab = None
        self.documents_map = {}
        self.document_topics = None
        # self.document_max = defaultdict(float)
        self.id_to_word = None

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
            query = """SELECT id, orig_text, processed_text FROM `gcp-cset-projects.tmp.test_unicorn_preproc_output`"""
            preprocessed = pd.read_gbq(query, project_id='gcp-cset-projects')
            self.documents = [value[2] for value in preprocessed.iloc[0:].values]
            # estimators = [('tokenizer', pipelinize(tokenizer_lemmatizer)), ('preprocessor', pipelinize(preprocessor))]
            # ('sentence_join', pipelinize(sentence_join))]
            # pipe = Pipeline(estimators)
            bigram = gensim.models.Phrases(self.documents, min_count=5, threshold=100)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            # self.documents = pipe.transform(self.documents)
            # with open("data/intermediate/partial_abstracts.pkl", "wb") as file_out:
            #     pickle.dump(self.documents, file_out)
            self.documents = make_bigrams(self.documents, bigram_mod)
            # Delete the superfluous _ that get added by the bigram maker
            self.documents = [[word for word in text if word != "_"] for text in self.documents]
            print(self.documents[:10])
            with open("data/intermediate/preprocessed_abstracts.pkl", "wb") as file_out:
                pickle.dump(self.documents, file_out)
        assert len(self.documents) == len(self.df)
        # Create map from preprocessed document string to document id
        # for i, doc in enumerate(self.documents):
        #     self.documents_map[doc] = self.df.iloc[i, 0]  # get the ids
        print(f"Total Documents: {len(self.documents)}")

    def fit_lda_model(self):
        self.id2word = corpora.Dictionary(self.documents)
        self.id2word.filter_extremes(no_below=20, no_above=0.5)
        corpus = [self.id2word.doc2bow(text) for text in self.documents]
        coherence_c_v = []
        coherence_u_mass = []
        print("Fitting models")
        for num_topics in range(self.min_topics, self.max_topics, self.step):
            lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=self.id2word, num_topics=num_topics,
                                                   random_state=100, chunksize=100, passes=20,
                                                   per_word_topics=True, minimum_probability=0)
            if not os.path.exists(f"data/intermediate/optimal_testing"):
                os.mkdir(f"data/intermediate/optimal_testing")
            with open(f"data/intermediate/optimal_testing/lda_model_{num_topics}_topics.pkl", "wb") as file_out:
                pickle.dump(lda_model, file_out)
            coherence_model_lda = CoherenceModel(model=lda_model, texts=self.documents, dictionary=self.id2word,
                                             coherence='c_v')
            coherence = coherence_model_lda.get_coherence()
            print(f"Topic {num_topics} coherence: {coherence}")
            coherence_c_v.append(coherence)
            coherence_model_lda = CoherenceModel(model=lda_model, texts=self.documents, dictionary=self.id2word,
                                             coherence='u_mass')
            coherence_u_mass.append(coherence_model_lda.get_coherence())
        return coherence_c_v, coherence_u_mass

    def plot_coherence(self, coherence_values, filename):
        x = range(self.min_topics, self.max_topics, self.step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence Score")
        plt.legend(("coherence_values"), loc="best")
        plt.savefig(filename)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--min_topics", type=int, default=40, required=False,
                        help="The minimum number of topics for the model")
    parser.add_argument("-x", "--max_topics", type=int, default=90, required=False,
                        help="The maximum number of topics for the model")
    parser.add_argument("-s", "--topics_step", type=int, default=5, required=False,
                        help="The number of topics for the model")
    args = parser.parse_args()
    model = TopicModel(args.min_topics, args.max_topics, args.topics_step)
    print("Getting data")
    model.get_data()
    print("Preprocessing data")
    model.preprocess_data()
    print("-----------------")
    print("Fitting LDA Model")
    coherence_c_v, coherence_u_mass = model.fit_lda_model()
    model.plot_coherence(coherence_u_mass, "coherence_u_mass.png")
    model.plot_coherence(coherence_c_v, "coherence_c_v.png")


if __name__ == "__main__":
    main()