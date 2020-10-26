import pandas as pd
import argparse
import os
import pickle
import numpy as np


def calculate_document_ratio(H, W, topic_count, num_documents):
    """
    Calculating the document ratio, based on metrics in the paper
    "Latent Dirichlet Allocation: Stability and Applications toStudies of User-Generated Content"
    (Koltkov, Koltsova, and Nikolenko).
    :param H: Model transform returned for LDA
    :param W: Model components returned for LDA
    :param topic_count: Number of topics
    :param num_documents: Number of documents in the model
    :return:
    """
    theta_count = 0
    for topic_idx, topic in enumerate(H):
        top_doc_probabilities = np.sort(W[:, topic_idx])[::-1]
        for probability in top_doc_probabilities:
            if probability <= 0.5:
                break
            theta_count += 1
    ratio = theta_count/(topic_count * num_documents)
    return ratio


def calculate_word_ratio(H, topic_count, vocab_length):
    """
    Calculating the word ratio, based on metrics in the paper
    "Latent Dirichlet Allocation: Stability and Applications toStudies of User-Generated Content"
    (Koltkov, Koltsova, and Nikolenko).
    :param H: Model transform returned for LDA
    :param topic_count: Number of topics
    :param vocab_length: Number of vocab words included in the model
    :return:
    """
    phi_count = 0
    for topic_idx, topic in enumerate(H):
        top_word_probabilities = np.sort(topic)[::-1]
        for probability in top_word_probabilities:
            if probability <= 0.5:
                break
            phi_count += 1
    ratio = phi_count/(topic_count * vocab_length)
    return ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h_filename", type=str,
                        help="The name of the pickle file containing H features of model")
    parser.add_argument("w_filename", type=str,
                        help="The name of the pickle file containing W features of model")
    parser.add_argument("vocab_length", type=int,
                        help="The number of words in the model vocab.")
    parser.add_argument("num_documents", type=int,
                        help="The number of documents used in the model.")
    parser.add_argument("-t", "--topic_count", type=int, default=80, required=False,
                        help="The number of topics used in the model.")
    args = parser.parse_args()
    if not args.h_filename or not args.w_filename or not args.vocab_length or not args.num_documents:
        parser.print_help()
    if os.path.exists(args.h_filename):
        with open(args.h_filename, "rb") as file_in:
            H = pickle.load(file_in)
    else:
        print("File provided for H features does not exist.")
    if os.path.exists(args.w_filename):
        with open(args.w_filename, "rb") as file_in:
            W = pickle.load(file_in)
    else:
        print("File provided for W features does not exist.")
    print(f"Document ratio: {calculate_document_ratio(H, W, args.topic_count, args.num_documents)}")
    print(f"Word ratio: {calculate_word_ratio(H, args.topic_count, args.vocab_length)}")


if __name__ == "__main__":
    main()