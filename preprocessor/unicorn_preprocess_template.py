import apache_beam as beam
import json
import re

from apache_beam.options.pipeline_options import PipelineOptions
from typing import Generator


class PreprocessorDoFn(beam.DoFn):
    def __init__(self):
        self.tokenizer = None

    def start_bundle(self):
        if self.tokenizer is None:
            # import must be done here instead of in init to avoid serialization issues
            import spacy
            nlp = spacy.load("en_core_sci_lg", disable=['ner'])
            self.tokenizer = nlp.Defaults.create_tokenizer(nlp)

    def tokenizer_lemmatizer(self, text: str) -> list:
        """
        Uses spacy to tokenize and lemmatize a given string
        :param text: String to tokenize
        :return: list of tokens
        """
        tokens = self.tokenizer(text)
        # nlp.Defaults.stop_words.add("model")
        lemma_list = [i.lemma_ for i in tokens if not i.is_stop]
        return lemma_list

    @staticmethod
    def clean(tokens: list) -> Generator[str, None, None]:
        """
        Removes punctuation and whitespace from a token list.
        :param tokens: The token list
        :return: A generator of non-empty cleaned tokens
        """
        for text in tokens:
            text = re.sub('<[^>]*>', '', text)
            text = re.sub('\W+', '', text.lower())
            if len(text) > 0:
                yield text

    def process(self, record: str) -> Generator[str, None, None]:
        """
        Tokenize and clean record text
        :param record: BQ row containing id and text
        :return: dict
        """
        try:
            js = json.loads(record)
            tokens = self.tokenizer_lemmatizer(js["text"])
            return_list = self.clean(tokens)
            preprocessed_text = " ".join(return_list)
            yield json.dumps({
                "id": js["id"],
                "orig_text": js["text"],
                "processed_text": preprocessed_text,
                "success": True,
                "error_msg": None
            })
        except Exception as e:
            yield json.dumps({
                "id": js["id"],
                "orig_text": js["text"],
                "processed_text": None,
                "success": False,
                "error_msg": str(e)
            })

class PreprocessorOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument(
            "--input_text_prefix",
            help=("gcs path of data to be processed (e.g. gs://my_bucket/my_subdir/my_file_prefix*. "
                  "Must be jsonl containing two columns, id and text"))
        parser.add_value_provider_argument(
            "--output_text_prefix",
            help="prefix of gcs path where processed data should go")


def run_pipeline(pipeline_options):
    preprocessor_options = pipeline_options.view_as(PreprocessorOptions)
    with beam.Pipeline(options=pipeline_options) as p:
        (p | "Read from Text" >> beam.io.ReadFromText(preprocessor_options.input_text_prefix)
            | "Preprocessor" >> beam.ParDo(PreprocessorDoFn())
            | "Write to JSON" >> beam.io.WriteToText(preprocessor_options.output_text_prefix, file_name_suffix=".jsonl"))

if __name__ == "__main__":
    pipeline_options = PreprocessorOptions()
    run_pipeline(pipeline_options)

