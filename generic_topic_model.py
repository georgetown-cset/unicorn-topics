import pandas as pd
import os
import spacy
import pickle
import gensim
import logging

nlp = spacy.load("en_core_sci_lg", disable=['ner'])
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def make_bigrams(documents, bigram_mod):
    return [bigram_mod[document.split(" ")] for document in documents]


class TopicModel:

    def __init__(self):
        """
        Initializes the topic model
        """
        self.documents = []
        self.df = None
        self.vocab = None
        self.documents_map = {}
        self.document_topics = None
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
            # We pull the initial cleaned data from BigQuery. This data is created by preprocessor/preprocess_text.py
            query = """SELECT id, orig_text, processed_text FROM `gcp-cset-projects.tmp.test_unicorn_preproc_output`"""
            preprocessed = pd.read_gbq(query, project_id='gcp-cset-projects')
            # Now add bigrams
            self.documents = [value[2] for value in preprocessed.iloc[0:].values]
            bigram = gensim.models.Phrases(self.documents, min_count=5, threshold=100)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            self.documents = make_bigrams(self.documents, bigram_mod)
            # Delete the superfluous _ that get added by the bigram maker
            self.documents = [[word for word in text if word != "_"] for text in self.documents]
            with open("data/intermediate/preprocessed_abstracts.pkl", "wb") as file_out:
                pickle.dump(self.documents, file_out)
        assert len(self.documents) == len(self.df)
        print(f"Total Documents: {len(self.documents)}")
