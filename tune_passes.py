import pandas as pd
import argparse
import os
import spacy
import pickle
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import logging
from generic_topic_model import TopicModel

nlp = spacy.load("en_core_sci_lg", disable=['ner'])
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def make_bigrams(documents, bigram_mod):
    return [bigram_mod[document.split(" ")] for document in documents]

class PassTuner(TopicModel):

    def __init__(self, topics_to_test):
        """
        Initializes the topic model
        :param num_features: Number of model features for training
        :param num_topics: Number of topics to create
        :param top_words: Number of words to print out from each topic
        :param top_documents: Number of document titles to print out from each topic
        """
        super().__init__()
        self.topics_to_test = topics_to_test

    def fit_lda_model(self):
        """
        Fit a variety of LDA models based on a range of passes and chunksize options.
        Use provided options of number of topics, and test with both a 75% corpus and a full corpus for validity.
        Find topic coherence of each model.
        :return:
        """
        self.id2word = corpora.Dictionary(self.documents)
        self.id2word.filter_extremes(no_below=20, no_above=0.5)
        corpus = [self.id2word.doc2bow(text) for text in self.documents]
        # To select parameters for chunksize, I started with a wide range
        # (initial chunksize parameters were [10, 100, 500, 1000, 5000]).
        # Initial results showed very small chunk sizes were quite bad, and any chunksize much over 100 was also awful
        # so I narrowed dramatically to tune more carefully.
        # For passes, I began by using a relatively low number of passes before starting to tune (20) and then
        # noticed model quality increase when this number went up, trying 30 and then 40. The number of passes
        # drastically slows down model speed, so I started by only increasing in increments of 10 to see whether
        # improvement continued. As coherence eventually plateaued and then got worse with more passes, I did not try
        # oass numbers above 80.
        passes = [40, 50, 60, 70, 80]
        chunksize = [50, 75, 100]
        corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(len(corpus)*0.75)), corpus]
        corpus_titles = ["75% corpus", "100% corpus"]
        model_results = {"Validation_set": [], "Topics": [], "Passes": [], "Chunksize": [], "Coherence": []}
        print("Fitting models")
        for i, corpus_set in enumerate(corpus_sets):
            for num_topics in self.topics_to_test:
                for p in passes:
                    for c in chunksize:
                        lda_model = gensim.models.LdaMulticore(corpus=corpus_set, id2word=self.id2word,
                                                               alpha="symmetric", random_state=100, chunksize=c,
                                                               passes=p, num_topics=num_topics, per_word_topics=True,
                                                               minimum_probability=0, eta=0.7)
                        if i == 1: # we only want to save the model if it's a model on the whole corpus
                            if not os.path.exists(f"data/intermediate/passes_testing"):
                                os.mkdir(f"data/intermediate/passes_testing")
                            with open(f"data/intermediate/passes_testing/lda_{num_topics}_"
                                      f"topics{p}_passes_{c}_chunksize.pkl", "wb") as file_out:
                                pickle.dump(lda_model, file_out)
                        coherence_model_lda = CoherenceModel(model=lda_model, texts=self.documents,
                                             dictionary=self.id2word, coherence='c_v')
                        coherence = coherence_model_lda.get_coherence()
                        print(f"Topic {num_topics}, passes {p} chunksize {c} corpus {corpus_titles[i]} "
                              f"coherence: {coherence}")
                        model_results['Validation_set'].append(corpus_titles[i])
                        model_results['Topics'].append(num_topics)
                        model_results['Passes'].append(p)
                        model_results['Chunksize'].append(c)
                        model_results['Coherence'].append(coherence)
        pd.DataFrame(model_results).to_csv("passes_tuning_results.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topics", metavar="T", type=int, nargs="+",
                        help="The topic values you want to test. List as many as you want, but each additional will"
                             "significantly slow down testing time.")
    args = parser.parse_args()
    model = PassTuner(args.topics)
    print("Getting data")
    model.get_data()
    print("Preprocessing data")
    model.preprocess_data()
    print("-----------------")
    print("Fitting LDA Model")
    model.fit_lda_model()


if __name__ == "__main__":
    main()
