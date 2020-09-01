### unicorn-topics preprocessing

This directory contains a beam script to do text preprocessing. It reads text from an input BQ table
and writes the text to an output BQ table. The input table must have two columns, id and text. The
output table will have five columns:

- `id` - document id
- `orig_text` - the original text from the input table
- `processed_text` - the processed text (null if processing failed)
- `success` - a boolean, true if processing failed
- `error_msg` - a string containing the exception raised if processing failed, otherwise null if processing succeeded

To run on the DataflowRunner, do:

```bash
python3 preprocess_text.py --input_table tmp.test_unicorn_preproc_input \
--output_table tmp.test_unicorn_preproc_output --project gcp-cset-projects \
--runner DataflowRunner --disk_size_gb 30 --job_name test-unicorn-preproc1 \
--region us-east1 --temp_location gs://cset-dataflow-test/example-tmps/ --setup_file ./setup.py
```

