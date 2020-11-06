import argparse
import os
import spacy
import pickle
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import logging
from generic_topic_model import TopicModel

nlp = spacy.load("en_core_sci_lg", disable=['ner'])
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TopicOptimizer(TopicModel):

    def __init__(self, min_topics, max_topics, step):
        """
        Initializes the topic model
        :param num_features: Number of model features for training
        :param num_topics: Number of topics to create
        :param top_words: Number of words to print out from each topic
        :param top_documents: Number of document titles to print out from each topic
        """
        super(TopicOptimizer, self).__init__()
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.step = step

    def fit_lda_model(self):
        """
        Fit a variety of topic models for different numbers of topics. The numbers used will be determined by
        a range (min and max) and a step. Find topic coherence for each model.
        :return:
        """
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
        plt.legend("coherence_values", loc="best")
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
    model = TopicOptimizer(args.min_topics, args.max_topics, args.topics_step)
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
