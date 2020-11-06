# Documentation

## Building Dataset

The dataset used as a source for the LDA documents in the model is `project_unicorn.coauthors_dimensions_publications_with_abstracts`

To build up this dataset, we run the following queries, in order:

1.) [creating_grid_ai_pubs.sql](sql/creating_grid_ai_pubs.sql)
creates `project_unicorn.grid_ai_pubs_052920`

2.) [creating_top_organizations.sql](sql/creating_top_organizations.sql)
 creates `project_unicorn.top_organizations`

3.) [selecting_ai_dimensions_publication_ids_top_organizations.sql](sql/selecting_ai_dimensions_publication_ids_top_organizations.sql)
creates `project_unicorn.dimensions_publication_ids_top_organizations_052920`

4.) [selecting_abstracts.sql](sql/selecting_abstracts.sql)
creates `project_unicorn.dimensions_publications_with_abstracts_top_organizations_052920`

5.) [creating_coauthors_publication_table.sql](sql/creating_coauthors_publication_table.sql)
creates `project_unicorn.coauthors_dimensions_publications_with_abstracts`

## Setting Up Workspace

To set up your workspace to enable running this code:

1.) Make a new virtualenv:
 
 ```bash
python3 -m venv venv
source venv/bin/activate
```

2.) Set up spacy and your other requirements:

```
pip3 install spacy==2.3.2
pip3 install scispacy==0.2.5
pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_lg-0.2.5.tar.gz
pip3 install -r requirements.txt
```

3.) Enable access to BigQuery.

`export GOOGLE_APPLICATION_CREDENTIALS=<path to your credentials>` (a service account json).

You should have at least BQ reader permissions.

4.) If you wish to use preloaded data (pickle files) for the datasets
or the preprocessed data, put that data into `data/intermediate`. The dataset file
should be named `documents.pkl` and the preprocessed data file should be
named `preprocessed_abstracts.pkl`.

5.) Preloaded data (which will be created by `topic_model_gensim.py`) should also be used to run `plot_topic_by_year.py`, 
`top_five_topics_py`, and `calculate_perplexity.py`. This data should
also be stored in `data/intermediate`.

---

## Running Code

### topic_model_gensim.py

Run `topic_model_gensim.py` as follows:

`python3 topic_model_gensim.py num_topics run_number [-w TOP_WORDS] [-d TOP_DOCUMENTS] [-m MODEL]`

The current defaults are:
- 10 top words
- 3 top documents

The current recommended number of topics is 60.

### find_optimal_topic_number.py

Run `find_optimal_topic_number.py` as follows:

`python3 find_optimal_topic_number.py min_topics max_topics topics_step`

The current defaults are:
- Minimum 40 topics
- Maximum 90 topics
- A step of 5

### tune_hyperparameters.py

Run `tune_hyperparameters.py` as follows:

`python3 tune_hyperparameters.py topics [topics ...]`

Where topics is one or more topics sizes. For example:

`python3 tune_hyperparameters.py 40 60`

### tune_passes.py

Run `tune_passes.py` as follows:

`python3 tune_passes.py TOPICS [TOPICS ...]`

Where topics is one or more topics sizes. For example:

`python3 tune_passes.py 40 60`

### calculate_perplexity.py

Run `calculate_perplexity.py` as follows:

`python3 calculate_perplexity.py lda_model corpus`

The model and corpus will both be produced in [data/intermediate](data/intermediate)
after running `topic_model_gensim.py`. The corpus will be in the main directory, while the 
model will exist in a subdirectory defined by the topic number and run
number you provided for the run. So for example, if you ran `topic_model_gensim.py`
with 60 topics with a run number of 1, you would run as follows:

`python3 calculate_perplexity.py data/intermediate/t_60_r_1/lda_model.pkl data/intermediate/corpus.pkl`

### plot_topic_by_year.py

Run `plot_topic_by_year.py` as follows:

`python3 plot_topic_by_year.py filename TOPIC_NUMBERS [TOPIC_NUMBERS ...] [-i]`

The topic numbers are whichever topics numbers you which to be plotted by year;
as many as 10 topic numbers can be plotted. By default, data from 2020 is disincluded,
as there is not a full year of data so a decline is expected; if you
wish to include 2020, the -i flag allows you to undo this.

The data for plotting will be written out by `topic_model_gensim.py`
into the `data/intermediate` subdirectory defined by the number of topics
and run number you selected. An example run is below:

`python3 plot_topic_by_year.py data/intermediate/t_60_r_1/topics_by_year.pkl 2 5 16 31 42`

### top_five_topics.py

Run `top_five_topics.py` as follows:

`python3 top_five_topics.py filename [-n num_topics]`

The data for plotting will be written out by `topic_model_gensim.py`
into the `data/intermediate` subdirectory defined by the number of topics
and run number you selected. An example run is below:

`python3 top_five_topics.py data/intermediate/t_60_r_1/top_topics_by_org.pkl -n 3`

---

## Other Notes

* There are unit tests available. These will all begin with `test_`

* `generic_topic_model.py` is a parent class; the code is not intended to
be run directly, only inherited.