import argparse
import pickle


def calculate_perplexity(lda_model_name, corpus_name):
    """
    Calculates the perplexity of a previously saved LDA model.
    :param lda_model_name: The name of the pickled LDA model file.
    :param corpus_name: The name of the pickled corpus file used to build the LDA model.
    :return:
    """
    with open(corpus_name, "rb") as file_in:
        corpus = pickle.load(file_in)
    with open(lda_model_name, "rb") as file_in:
        lda_model = pickle.load(file_in)
    perplexity = lda_model.log_perplexity(corpus)
    return perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lda_model", type=str,
                        help="The pkl filename containing the lda model you want to calculate perplexity of.")
    parser.add_argument("corpus", type=str,
                        help="The pkl filename of the corpus.")
    args = parser.parse_args()
    perplexity = calculate_perplexity(args.lda_model, args.corpus)
    print(f"Perplexity: {perplexity}")


if __name__ == "__main__":
    main()