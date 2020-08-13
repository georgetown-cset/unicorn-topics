# Overview

The goal of this project is to gain an understanding how the research agendas of different industrial AI labs are evolving over time, as shown by their publications, and
to compare the agendas of these labs to the agendas of top academic institutions publishing in AI.

This repository contains the following:

1.) Code to build an LDA- or NMF-based topic model of scientific
literature data from 100 universities with top publishing totals
in AI, as well as the "big six" US tech companies (Google, Apple,
Amazon, Facebook, Microsoft, and IBM).

2.) Code to create visualizations of the resulting topic models.

3.) SQL queries to create the dataset of scientific literature used as input to the topic model.

# Building Dataset

The ultimate dataset used in this project is `project_unicorn.coauthors_dimensions_publications_with_abstracts`

To build up this dataset, we run the following queries, in order:

1.) [creating_top_organizations.sql](sql/creating_top_organizations.sql)

2.) [selecting_ai_dimensions_publication_ids_top_organizations.sql](sql/selecting_ai_dimensions_publication_ids_top_organizations.sql)

3.) [selecting_abstracts.sql](sql/selecting_abstracts.sql)

4.) [creating_coauthors_publication_table.sql](sql/creating_coauthors_publication_table.sql)

# Building Topic Model

To run this code:

1.) Make a new virtualenv:
 
 ```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2.) `export GOOGLE_APPLICATION_CREDENTIALS=<path to your credentials>` - a service account json.
You should have at least BQ reader permissions

3.) If you wish to use preloaded data (pickle files) for the datasets
or the preprocessed data, put that data into `data/intermediate`. The dataset file
should be named `documents.okl` and the preprocessed data file should be
named `preprocessed_abstracts.pkl`.

4.) Preloaded data should also be used to run `plot_topic_by_year.py`. This data should
also be stored in `data/intermediate` and should be named `topics_by_year.pkl`

Run topic model as follows:

`python3 topic_model.py num_topics [-w TOP_WORDS] [-d TOP_DOCUMENTS] [-f NUM_FEATURES] [-n|--nmf]`

The current defaults are:
- 10 top words
- 0 top documents
- 1000 features
- LDA rather than NMF

The current recommended number of topics is 80.

So a sample run with current defaults but 15 top words would look like the following:

`python3 topic_model.py 80 -w 15`