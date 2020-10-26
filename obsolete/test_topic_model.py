import unittest
import topic_model
from collections import defaultdict
import pandas as pd
import os


class TestTopicModel(unittest.TestCase):

    def test_tokenizer_lemmatizer(self):
        lemma_list = topic_model.tokenizer_lemmatizer("All dogs are the best dogs.")
        self.assertEqual(lemma_list, ['dog', 'well', 'dog', '.'])

    def test_preprocessor(self):
        preprocessed = topic_model.preprocessor(['dog', 'well', 'dog', '.'])
        self.assertEqual(preprocessed, ['dog', 'well', 'dog'])

    def test_sentence_join(self):
        joined = topic_model.sentence_join(['dog', 'well', 'dog'])
        self.assertEqual(joined, "dog well dog")

    def test_pipelinize(self):
        estimators = [('tokenizer', topic_model.pipelinize(topic_model.tokenizer_lemmatizer)),
                      ('preprocessor', topic_model.pipelinize(topic_model.preprocessor)),
                      ('sentence_join', topic_model.pipelinize(topic_model.sentence_join))]
        pipe = topic_model.Pipeline(estimators)
        sentence = pipe.transform(["All dogs are the best dogs."])
        self.assertEqual(sentence, ["dog well dog"])

    def test_init_topic_model(self):
        # Check whether we're running from the subdirectory or not
        if "obsolete" in os.getcwd():
            model = topic_model.TopicModel(1000, 3, 2, 3, "../data/intermediate")
        else:
            model = topic_model.TopicModel(1000, 3, 2, 3, "data/intermediate")
        self.assertEqual(model.num_features, 1000)
        self.assertEqual(model.num_topics, 3)
        self.assertEqual(model.top_words, 2)
        self.assertEqual(model.top_documents, 3)
        self.assertEqual(model.documents, [])
        self.assertEqual(model.df, None)
        self.assertEqual(model.documents_map, {})
        self.assertEqual(model.document_topics, {})
        self.assertEqual(model.document_max, defaultdict(float))

    def test_get_data(self):
        if "obsolete" in os.getcwd():
            model = topic_model.TopicModel(1000, 3, 2, 3, "../data/intermediate")
        else:
            model = topic_model.TopicModel(1000, 3, 2, 3, "data/intermediate")
        model.get_data()
        self.assertIsNotNone(model.df)
        self.assertIsInstance(model.df, pd.core.frame.DataFrame)

if __name__ == '__main__':
    unittest.main()
