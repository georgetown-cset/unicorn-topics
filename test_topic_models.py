import unittest
from topic_model_gensim import PrimaryTopicModel
from find_optimal_topic_number import TopicOptimizer
from tune_hyperparameters import TuneHyperParameters
from tune_passes import PassTuner

class TestTopicModels(unittest.TestCase):

    def test_init_primary(self):
        model = PrimaryTopicModel(60, 10, 3)
        self.generic_init_equals(model)
        self.assertEqual(model.num_topics, 60)
        self.assertEqual(model.top_words, 10)
        self.assertEqual(model.top_documents, 3)

    def test_init_find_optimal_topic_number(self):
        model = TopicOptimizer(40, 90, 5)
        self.generic_init_equals(model)
        self.assertEqual(model.min_topics, 40)
        self.assertEqual(model.max_topics, 90)
        self.assertEqual(model. step, 5)

    def test_init_tune_hyperparameters(self):
        model = TuneHyperParameters([40, 60])
        self.generic_init_equals(model)
        self.assertEqual(model.topics_to_test, [40, 60])

    def test_init_tune_passes(self):
        model = PassTuner([40, 60])
        self.generic_init_equals(model)
        self.assertEqual(model.topics_to_test, [40, 60])

    def generic_init_equals(self, model):
        self.assertEqual(model.documents, [])
        self.assertIsNone(model.df)
        self.assertIsNone(model.vocab)
        self.assertEqual(model.documents_map, {})
        self.assertIsNone(model.document_topics)
        self.assertIsNone(model.id_to_word)

    def test_get_data(self):
        model = PrimaryTopicModel(60, 10, 3)
        model.get_data()
        self.assertIsNotNone(model.df, PrimaryTopicModel)
        # Every id should be unique
        self.assertEqual(model.df.id.nunique(), len(model.df))







if __name__ == '__main__':
    unittest.main()
