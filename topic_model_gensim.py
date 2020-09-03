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

    def __init__(self, num_topics, top_words, top_documents):
        """
        Initializes the topic model
        :param num_features: Number of model features for training
        :param num_topics: Number of topics to create
        :param top_words: Number of words to print out from each topic
        :param top_documents: Number of document titles to print out from each topic
        """
        # self.num_features = num_features
        self.num_topics = num_topics
        self.top_words = top_words
        self.top_documents = top_documents
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

    def fit_lda_model(self, run_number):
        self.id2word = corpora.Dictionary(self.documents)
        self.id2word.filter_extremes(no_below=20, no_above=0.5)
        corpus = [self.id2word.doc2bow(text) for text in self.documents]
        if os.path.exists(f"data/intermediate/t_{self.num_topics}_r_{run_number}/lda_model.pkl"):
            with open(f"data/intermediate/t_{self.num_topics}_r_{run_number}/lda_model.pkl", "rb") as file_in:
                lda_model = pickle.load(file_in)
        else:
            lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=self.id2word, num_topics=self.num_topics,
                                                   random_state=100, chunksize=100, passes=20,
                                                   per_word_topics=True, minimum_probability=0)
            if not os.path.exists(f"data/intermediate/t_{self.num_topics}_r_{run_number}"):
                os.mkdir(f"data/intermediate/t_{self.num_topics}_r_{run_number}")
            with open(f"data/intermediate/t_{self.num_topics}_r_{run_number}/lda_model.pkl", "wb") as file_out:
                pickle.dump(lda_model, file_out)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=self.documents, dictionary=self.id2word,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print(f"Coherence Score c_v: {coherence_lda}")
        coherence_model_lda = CoherenceModel(model=lda_model, texts=self.documents, dictionary=self.id2word,
                                             coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        print(f"Coherence Score u_mass: {coherence_lda}")
        top_topics = lda_model.top_topics(corpus, topn=self.top_words)
        print("Dividing documents by topic")
        if not os.path.exists(f"data/intermediate/t_{self.num_topics}_r_{run_number}/document_topics.pkl"):
            self.divide_documents_by_topic(lda_model, corpus, run_number)
        else:
            with open(f"data/intermediate/t_{self.num_topics}_r_{run_number}/document_topics.pkl", "rb") as file_in:
                self.document_topics = pickle.load(file_in)
        print("Displaying topics")
        topics_by_year, top_topics_by_org = self.display_topics(top_topics)
        with open(f"data/intermediate/t_{self.num_topics}_r_{run_number}/topics_by_year.pkl", "wb") as file_out:
            pickle.dump(topics_by_year, file_out)
        with open(f"data/intermediate/t_{self.num_topics}_r_{run_number}/top_topics_by_org.pkl", "wb") as file_out:
            pickle.dump(top_topics_by_org, file_out)

    def divide_documents_by_topic(self, lda_model, corpus, run_number):
        self.document_topics = pd.DataFrame()
        for i, row_list in enumerate(lda_model[corpus]):
            row = row_list[0] if lda_model.per_word_topics else row_list
            # Sort by second item in the list
            row = sorted(row, key=lambda x: x[1], reverse=True)
            for j, (topic_number, prop_topic) in enumerate(row):
                if j == 0:  # dominant topic
                    word_props = lda_model.show_topic(topic_number)
                    topic_keywords = ", ".join([word for word, prop in word_props])
                    self.document_topics = self.document_topics.append(
                        pd.Series([int(topic_number), round(prop_topic, 4),
                                   topic_keywords]), ignore_index=True)
                else:
                    break  # not dominant topic
        self.document_topics.columns = ['Dominant_Topic', 'Percentage_Contribution', 'Topic_Keywords']
        # Add original titles back in:
        contents = pd.Series(self.df["title"])
        self.document_topics = pd.concat([self.document_topics, contents], axis=1)
        # add doc ids back in
        ids = pd.Series(self.df["id"])
        self.document_topics = pd.concat([self.document_topics, ids], axis=1)
        with open(f"data/intermediate/t_{self.num_topics}_r_{run_number}/document_topics.pkl", "wb") as file_out:
            pickle.dump(self.document_topics, file_out)

    def display_topics(self, top_topics):
        # topic coherence follows the list of topics for every topic in top_topics
        average_topic_coherence = sum([t[1] for t in top_topics]) / self.num_topics
        topics_by_year = {}
        top_topics_by_org = defaultdict(Counter)
        print(f"Average u_mass topic coherence: {average_topic_coherence}")
        for topic_number, topic in enumerate(top_topics):
            print(f"Topic {topic_number}, topic coherence {topic[1]}")
            # First val in top_topics is a list of topics of (probablility, word)
            print(" ".join([probability_word_pair[1] for probability_word_pair in topic[0]]))
            topic_papers = self.document_topics[self.document_topics["Dominant_Topic"] == topic_number]
            # print(topic_papers["title"].head(self.top_documents))
            topic_papers.reset_index()
            for row in range(self.top_documents):
                print(topic_papers["title"][row])
            coauthors_count = Counter()
            year_count = Counter()
            for row in range(len(topic_papers)):
                doc_id = topic_papers["id"][row]
                coauthors = self.df.loc[self.df.index[self.df["id"] == doc_id][0], "coauthors"].split("; ")
                year = self.df.loc[self.df.index[self.df["id"] == doc_id][0], "year"]
                coauthors_count.update(coauthors)
                year_count[int(year)] += 1
            for coauthor in coauthors_count.keys():
                top_topics_by_org[coauthor][topic_number] += coauthors_count[coauthor]
            print(f"Total papers in topic: {len(topic_papers)}")
            topics_by_year[topic_number] = sorted(list(year_count.items()))
            print(f"Paper counts by year: {topics_by_year[topic_number]}")
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
    # parser.add_argument("-f", "--num_features", type=int,
    #                     help="The number of features to use in the model", required=False, default=1000)
    args = parser.parse_args()
    if not args.num_topics or not args.run_number:
        parser.print_help()
    model = TopicModel(args.num_topics, args.top_words, args.top_documents)
    print("Getting data")
    model.get_data()
    print("Preprocessing data")
    model.preprocess_data()
    print("-----------------")
    print("Fitting LDA Model")
    model.fit_lda_model(args.run_number)


if __name__ == "__main__":
    main()
