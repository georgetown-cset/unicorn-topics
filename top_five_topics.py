import argparse
import pickle
import os


def print_topics_by_org(filename, num_topics):
    if os.path.exists("data/intermediate/top_topics_by_org.pkl"):
        with open("data/intermediate/top_topics_by_org.pkl", "rb") as file_in:
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
                        help="The number of topics to print per organization")
    args = parser.parse_args()
    if not args.filename:
        parser.print_help()
    print_topics_by_org(args.filename, args.num_topics)


if __name__ == "__main__":
    main()