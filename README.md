# Overview

The goal of this project is to gain an understanding how the research agendas of different industrial AI labs are evolving over time, as shown by their publications, and
to compare the agendas of these labs to the agendas of top academic institutions publishing in AI.

This repository contains the following:

1.) Code to build an LDA-based topic model of scientific
literature data from 100 universities with top publishing totals
in AI, as well as the "big six" US tech companies (Google, Apple,
Amazon, Facebook, Microsoft, and IBM). The primary model code is
[topic_model_gensim.py](topic_model_gensim.py) and the foundational code for all the model
runs is in [generic_topic_model.py](generic_topic_model.py).

2.) Code to optimize this model, tuning for the best number of topics ([find_optimal_topic_number.py](find_optimal_topic_number.py)),
hyperparameters ([tune_hyperparameters.py](tune_hyperparameters.py)), and number of passes ([tune_passes.py](tune_passes.py)). This code also includes
a separate file to calculate model perplexity ([calculate_perplexity.py](calculate_perplexity.py)).

3.) Code to create visualizations ([plot_topics_by_year.py](plot_topics_by_year.py)) and drill-downs ([top_five_topics.py](top_five_topics.py)) of the resulting topic models.

4.) SQL queries to create the dataset of scientific literature used as input to the topic model, found in [sql](sql).

5.) Code to run the preprocessing of the data used in the model, found in [preprocessor](preprocessor). See
[README](preprocessor/README.md) contained within for details.

6.) Obsolete code used to create and analyze a previous version of the model, found in [obsolete](obsolete). See
[README](obsolete/README.md).

For details on how to build the dataset and run the code contained within, please see [technical_documentation.md](technical_documentation.md)

## Building Topic Model

Run topic model as follows:

`python3 topic_model.py num_topics run_number data_directory [-w TOP_WORDS] [-d TOP_DOCUMENTS] [-f NUM_FEATURES] [-n|--nmf]`

The current defaults are:
- 10 top words
- 0 top documents
- 1000 features
- LDA rather than NMF

The current recommended number of topics is 60.

So a sample run with current defaults but 15 top words would look like the following:

`python3 topic_model.py 60 1 data/intermediate -w 15`