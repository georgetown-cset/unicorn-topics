### unicorn-topics preprocessing

This directory contains a beam script to do text preprocessing on the contents of an input BQ table, and
then to write the preprocessed text to an output BQ table. The input table must have two columns, id and text. The
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

This directory also contains an example of a Beam template script. Unfortunately, the python beam SDK does not 
allow the use of templated input/output BQ locations at the moment, so this script only works on input/output 
gcs locations. Nevertheless, using the template allows users to execute the beam pipeline on input/output
gcs locations of their choosing without having to install any local dependencies or even clone this repository.

The template is associated with a metadata file. If you need to update the metadata file, do

```bash
gsutil cp unicorn_preprocess_metadata gs://cset-dataflow-templates/templates/
```

To update the template on GCS, you must run this command (don't just gsutil cp the script):

```bash
python unicorn_preprocess_template.py --project gcp-cset-projects --runner DataflowRunner \
--temp_location gs://cset-dataflow-test/example-tmps/ \
--template_location  gs://cset-dataflow-templates/templates/unicorn_preprocess --setup_file ./setup.py
```

You can then run a dataflow job using this template in the UI, or from the command line like this:

```bash
gcloud dataflow jobs run yet-another-unicorn-preproc-test \
--gcs-location gs://cset-dataflow-templates/templates/unicorn_preprocess \
--parameters input_text_prefix=gs://jtm-tmp1/test_unicorn_preproc_input_sm/out,output_text_prefix=gs://jtm-tmp1/test_unicorn_preproc_output_sm/yetanotheroutputprefix --region us-east1
```

