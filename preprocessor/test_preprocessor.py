import unittest
from preprocess_text import PreprocessorDoFn
from typing import Generator


class TestPreprocessor(unittest.TestCase):

    def test_start_bundle(self):
        pre = PreprocessorDoFn()
        pre.start_bundle()
        self.assertIsNotNone(pre.tokenizer)

    def test_tokenizer_lemmatizer(self):
        pre = PreprocessorDoFn()
        pre.start_bundle()
        lemma_list = pre.tokenizer_lemmatizer("All dogs are the best dogs.")
        self.assertEqual(lemma_list, ['dog', 'well', 'dog', '.'])

    def test_clean(self):
        pre = PreprocessorDoFn()
        preprocessed = pre.clean(['dog', 'well', 'dog', '.'])
        self.assertEqual(list(preprocessed), ["dog", "well", "dog"])

    def test_process(self):
        pre = PreprocessorDoFn()
        pre.start_bundle()
        preprocessed = pre.process({"id": 36747, "text": "All dogs are the best dogs."})
        self.assertEqual(list(preprocessed), [{'error_msg': None,
                                               'id': 36747,
                                               'orig_text': 'All dogs are the best dogs.',
                                               'processed_text': 'dog well dog',
                                               'success': True}])


if __name__ == '__main__':
    unittest.main()
