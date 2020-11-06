import pandas as pd
import argparse
import os
import spacy
import pickle
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from collections import Counter, defaultdict
import logging
from generic_topic_model import intermediate_path, TopicModel

nlp = spacy.load("en_core_sci_lg", disable=['ner'])
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class PrimaryTopicModel(TopicModel):

    def __init__(self, num_topics, top_words, top_documents):
        """
        Initializes the topic model
        :param num_topics: Number of topics to create
        :param top_words: Number of words to print out from each topic
        :param top_documents: N umber of document titles to print out from each topic
        """
        super().__init__()
        self.num_topics = num_topics
        self.top_words = top_words
        self.top_documents = top_documents

    def get_data(self):
        """
        Acquire data to run the model with.
        Data either comes from BQ or is loaded from pickle if it's been previously pulled.
        :return:
        """
        super(PrimaryTopicModel, self).get_data()

    def preprocess_data(self):
        """
        Run the preprocessing functions. If the preprocessing functions
        have already been run, load preprocessed data from pickle.
        :return:
        """
        super(PrimaryTopicModel, self).preprocess_data()

    def fit_lda_model(self, run_number, model):
        """
        Fit an LDA model to our dictionary, find the top topics and the coherence.
        :param run_number: The current run of the model, for storage purposes
        :return:
        """
        self.id2word = corpora.Dictionary(self.documents)
        self.id2word.filter_extremes(no_below=20, no_above=0.5)
        corpus = [self.id2word.doc2bow(text) for text in self.documents]
        # We're dumping this so we can calculate perplexity
        with open("data/intermediate/corpus.pkl", "wb") as file_out:
            pickle.dump(corpus, file_out)
        if model != "" and os.path.exists(model):
            with open(model, "rb") as file_in:
                lda_model = pickle.load(file_in)
        else:
            lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=self.id2word, num_topics=self.num_topics,
                                                   random_state=100, chunksize=100, passes=40,
                                                   per_word_topics=True, minimum_probability=0)
            if not os.path.exists(os.path.join(intermediate_path, f"t_{self.num_topics}_r_{run_number}")):
                os.mkdir(os.path.join(intermediate_path, f"t_{self.num_topics}_r_{run_number}"))
            with open(os.path.join(intermediate_path, f"t_{self.num_topics}_r_{run_number}/lda_model.pkl"), "wb")\
                    as file_out:
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
        if not os.path.exists(os.path.join(intermediate_path,
                                           f"t_{self.num_topics}_r_{run_number}/document_topics.pkl")):
            self.divide_documents_by_topic(lda_model, corpus, run_number)
        else:
            with open(os.path.join(intermediate_path, f"t_{self.num_topics}_r_{run_number}/document_topics.pkl"), "rb")\
                    as file_in:
                self.document_topics = pickle.load(file_in)
        print("Displaying topics")
        topics_by_year, top_topics_by_org = self.display_topics(top_topics)
        with open(os.path.join(intermediate_path, f"t_{self.num_topics}_r_{run_number}/topics_by_year.pkl"), "wb")\
                as file_out:
            pickle.dump(topics_by_year, file_out)
        with open(os.path.join(intermediate_path, f"t_{self.num_topics}_r_{run_number}/top_topics_by_org.pkl"), "wb")\
                as file_out:
            pickle.dump(top_topics_by_org, file_out)

    def divide_documents_by_topic(self, lda_model, corpus, run_number):
        """
        Divide the documents used as model input into topics based on the dominant topic they were classified as.
        :param lda_model: The LDA model that was built
        :param corpus: The corpus underlying the model
        :param run_number: The current run number, for storage
        :return:
        """
        self.document_topics = pd.DataFrame()
        for i, row_list in enumerate(lda_model[corpus]):
            # One of the options for the model is to select per_word_topics which means model computes a list of most
            # likely topics for each word sorted in descending order
            # If we do that then we only want the most likely one!
            row = row_list[0] if lda_model.per_word_topics else row_list
            # Sort by second item in the list
            # This is the proportion of that topic that the document was assigned
            row = sorted(row, key=lambda x: x[1], reverse=True)
            dominant_topic = row[0]
            topic_number, prop_topic = dominant_topic[0], dominant_topic[1]
            word_props = lda_model.show_topic(topic_number)
            topic_keywords = ", ".join([word for word, prop in word_props])
            self.document_topics = self.document_topics.append(pd.Series([int(topic_number), round(prop_topic, 4),
                           topic_keywords]), ignore_index=True)
        self.document_topics.columns = ['Dominant_Topic', 'Percentage_Contribution', 'Topic_Keywords']
        # Add original titles back in:
        contents = pd.Series(self.df["title"])
        self.document_topics = pd.concat([self.document_topics, contents], axis=1)
        # add doc ids back in
        ids = pd.Series(self.df["id"])
        self.document_topics = pd.concat([self.document_topics, ids], axis=1)
        if not os.path.exists(os.path.join(intermediate_path, f"t_{self.num_topics}_r_{run_number}")):
            os.mkdir(os.path.join(intermediate_path, f"t_{self.num_topics}_r_{run_number}"))
        try:
            with open(os.path.join(intermediate_path, f"t_{self.num_topics}_r_{run_number}/document_topics.pkl"), "wb")\
                    as file_out:
                pickle.dump(self.document_topics, file_out)
        except FileNotFoundError:
            pass

    def display_topics(self, top_topics):
        """
        Display the topics created by the LDA model.
        :param top_topics: The top topics object from the model.
        :return:
        """
        # In gensim, topic coherence follows the list of topics for every topic in top_topics
        average_topic_coherence = sum([t[1] for t in top_topics]) / self.num_topics
        topics_by_year = {}
        top_topics_by_org = defaultdict(Counter)
        print(f"Average u_mass topic coherence: {average_topic_coherence}")
        for topic_number, topic in enumerate(top_topics):
            print(f"Topic {topic_number}, topic coherence {topic[1]}")
            # First val in top_topics is a list of topics of (probablility, word)
            print(" ".join([probability_word_pair[1] for probability_word_pair in topic[0]]))
            topic_papers = self.document_topics[self.document_topics["Dominant_Topic"] == topic_number]
            topic_papers = topic_papers.sort_values(by="Percentage_Contribution", ascending=False)
            topic_papers = topic_papers.reset_index()
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
            # We care especially about these companies; print additional info on them
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
                        required=False, default=3)
    parser.add_argument("-m", "--model", type=str, help="Pickle file of saved LDA model.", required=False, default="")
    args = parser.parse_args()
    if not args.num_topics or not args.run_number:
        parser.print_help()
    model = PrimaryTopicModel(args.num_topics, args.top_words, args.top_documents)
    print("Getting data")
    model.get_data()
    print("Preprocessing data")
    model.preprocess_data()
    print("-----------------")
    print("Fitting LDA Model")
    model.fit_lda_model(args.run_number, args.model)


if __name__ == "__main__":
    main()
