import apache_beam as beam
import argparse
import re

from apache_beam.options.pipeline_options import PipelineOptions
#from apache_beam.options.value_provider import StaticValueProvider


class PreprocessorDoFn(beam.DoFn):
    def __init__(self):
        self.tokenizer = None

    def start_bundle(self):
        if self.tokenizer is None:
            import spacy
            nlp = spacy.load("en_core_sci_lg", disable=['ner'])
            self.tokenizer = nlp.Defaults.create_tokenizer(nlp)

    def tokenizer_lemmatizer(self, text):
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
    def clean(tokens):
        for text in tokens:
            text = re.sub('<[^>]*>', '', text)
            text = re.sub('\W+', '', text.lower())
            if len(text) > 0:
                yield text

    def process(self, record):
        """
        Removes punctuation and whitespace from a token list.
        :param text: Either the token list or the individual token (recursive function)
        :return: Either the processed token or the full list of tokens
        """
        try:
            tokens = self.tokenizer_lemmatizer(record["text"])
            return_list = self.clean(tokens)
            preprocessed_text = " ".join(return_list)
            yield {
                "id": record["id"],
                "orig_text": record["text"],
                "processed_text": preprocessed_text,
                "success": True,
                "error_msg": None
            }
        except Exception as e:
            yield {
                "id": record["id"],
                "orig_text": record["text"],
                "processed_text": None,
                "success": False,
                "error_msg": str(e)
            }


def run_pipeline(input_table, output_table, pipeline_args):
    output_table_schema = {
        "fields": [{
            "name": "id", "type": "STRING", "mode": "REQUIRED"
        }, {
            "name": "orig_text", "type": "STRING", "mode": "REQUIRED"
        }, {
            "name": "processed_text", "type": "STRING", "mode": "NULLABLE"
        }, {
            "name": "success", "type": "BOOLEAN", "mode": "REQUIRED"
        }, {
            "name": "error_msg", "type": "STRING", "mode": "NULLABLE"
        }]
    }
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        (p | "Read from BQ" >> beam.io.Read(beam.io.BigQuerySource(input_table))
            | "Preprocessor" >> beam.ParDo(PreprocessorDoFn())
            | "Write to BQ" >> beam.io.WriteToBigQuery(output_table, schema = output_table_schema,
                                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_table",
                        help="bq table containing data to be processed. Must contain two columns, id and text",
                        required=True)
    parser.add_argument("--output_table",
                        help=("bq table where processed data should go. If this table already exists, "
                              "this pipeline will overwrite its contents."), required=True)
    args, pipeline_args = parser.parse_known_args()

    run_pipeline(args.input_table, args.output_table, pipeline_args)
