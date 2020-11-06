import numpy as np
import pandas as pd
import argparse
import os
import spacy
import pickle
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import logging
import copy
from generic_topic_model import TopicModel

nlp = spacy.load("en_core_sci_lg", disable=['ner'])
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class TuneHyperParameters(TopicModel):

    def __init__(self, topics_to_test):
        """
        Initializes the topic model
        :param topics_to_test A list of the different topic quantities to test.
        """
        super().__init__()
        self.topics_to_test = topics_to_test

    def fit_lda_model(self):
        """
        Fit a variety of LDA models based on a range of options for the alpha and beta hyperparameters.
        Also allow multiple topic values within the model, and test on a complete and 75% corpus for validity.
        Find topic coherence for each model.
        :return:
        """
        self.id2word = corpora.Dictionary(self.documents)
        self.id2word.filter_extremes(no_below=20, no_above=0.5)
        corpus = [self.id2word.doc2bow(text) for text in self.documents]
        alpha = list(np.arange(0.1, 1, 0.3))
        alpha.append("symmetric")
        beta = copy.deepcopy(alpha)
        alpha.append("asymmetric")
        corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(len(corpus) * 0.75)), corpus]
        corpus_titles = ["75% corpus", "100% corpus"]
        model_results = {"Validation_set": [], "Topics": [], "Alpha": [], "Beta": [], "Coherence": []}
        print("Fitting models")
        for i, corpus_set in enumerate(corpus_sets):
            for num_topics in self.topics_to_test:
                for a in alpha:
                    for b in beta:
                        lda_model = gensim.models.LdaMulticore(corpus=corpus_set, id2word=self.id2word, alpha=a,
                                                               random_state=100, chunksize=100, passes=20,
                                                               num_topics=num_topics,
                                                               per_word_topics=True, minimum_probability=0, eta=b)
                        if i == 1:  # we only want to save the model if it's a model on the whole corpus
                            if not os.path.exists(f"data/intermediate/hyperparameter_testing"):
                                os.mkdir(f"data/intermediate/hyperparameter_testing")
                            with open(f"data/intermediate/hyperparameter_testing/lda_{num_topics}_"
                                      f"topics{a}_alpha_{b}_eta.pkl", "wb") as file_out:
                                pickle.dump(lda_model, file_out)
                        coherence_model_lda = CoherenceModel(model=lda_model, texts=self.documents,
                                                             dictionary=self.id2word, coherence='c_v')
                        coherence = coherence_model_lda.get_coherence()
                        print(f"Topic {num_topics}, alpha {a} eta {b} corpus {corpus_titles[i]} coherence: {coherence}")
                        model_results['Validation_set'].append(corpus_titles[i])
                        model_results['Topics'].append(num_topics)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(coherence)
        pd.DataFrame(model_results).to_csv("hyperparamter_tuning_results.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topics", metavar="T", type=int, nargs="+",
                        help="The topic values you want to test. List as many as you want, but each additional will"
                             "significantly slow down testing time.")
    args = parser.parse_args()
    if not args.topics:
        parser.print_help()
    model = TuneHyperParameters(args.topics)
    print("Getting data")
    model.get_data()
    print("Preprocessing data")
    model.preprocess_data()
    print("-----------------")
    print("Fitting LDA Model")
    model.fit_lda_model()


if __name__ == "__main__":
    main()
