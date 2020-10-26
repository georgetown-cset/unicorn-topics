import argparse
import pickle
import os


def print_topics_by_org(filename, num_topics):
    """
    Print the topics with the most papers in them for each organization (and the number of papers).
    :param filename: Filename containing top topics by org information.
    :param num_topics: The number of topics to print per org. Defaults to 5.
    :return:
    """
    if os.path.exists(filename):
        with open(filename, "rb") as file_in:
            top_topics_by_org = pickle.load(file_in)
    for organization in top_topics_by_org.keys():
        print(f"Organization: {organization}")
        for topic, count in top_topics_by_org[organization].most_common(num_topics):
            print(f"Topic {topic}: {count} papers")
        print("---------------------")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str,
                        help="The name of the pickle file containing topics by organization data.")
    parser.add_argument("-n", "--num-topics", type=int, default=5, required=False,
                        help="The number of topics to print per organization. Defaults to 5.")
    args = parser.parse_args()
    if not args.filename:
        parser.print_help()
    print_topics_by_org(args.filename, args.num_topics)


if __name__ == "__main__":
    main()