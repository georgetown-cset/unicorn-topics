# Overview

The files contained within this directory are no longer in use in this
project, and are included only for archiving/completeness purposes.
They were part of the initial analysis, but after further work it was
concluded that the primary code was more valuable for producing a viable
model.

In particular, the `sklearn` package, used here for building the LDA model,
shows no benefit over `gensim`, the package we eventually used, which
does have benefits over `sklearn`: ease of incorporating bigrams, ease of
calculating perplexity and topic coherence, and built-in topic distance metrics.
As a result, choosing `gensim` as a base library to build with made more sense.

We also considered NMF as a baseline in this version, and concluded it was inferior to LDA.

Additionally, while we considered using word and document ratios to evaluate
our model, we ended up concluding that perplexity and topic coherence were better
metrics for our purposes, and served to more accurately produce topics
that both represented the underlying data and were highly human-interpretable.

# Running Code

Should you wish to run this code anyway, it can be done as follows:

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

3.) `export GOOGLE_APPLICATION_CREDENTIALS=<path to your credentials>` - a service account json.
You should have at least BQ reader permissions

4.) If you wish to use preloaded data (pickle files) for the datasets
or the preprocessed data, put that data into `data/intermediate`. The dataset file
should be named `documents.pkl` and the preprocessed data file should be
named `preprocessed_abstracts.pkl`.

5.) Preloaded data should also be used to run `plot_topic_by_year.py`. This data should
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