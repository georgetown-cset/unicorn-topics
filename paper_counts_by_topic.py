import pickle
import argparse
import csv
import os
from collections import OrderedDict

# Maps from topic orderings by topic coherence to topic
# orderings by size (as presented in the paper
topic_mapper = {7: 1,
                0: 2,
                3: 3,
                1: 4,
                20: 5,
                11: 6,
                5: 7,
                2: 8,
                8: 9,
                6: 10,
                23: 11,
                37: 12,
                35: 13,
                24: 14,
                13: 15,
                25: 16,
                15: 17,
                21: 18,
                28: 19,
                19: 20,
                41: 21,
                42: 22,
                14: 23,
                10: 24,
                30: 25,
                45: 26,
                27: 27,
                31: 28,
                12: 29,
                34: 30,
                36: 31,
                22: 32,
                43: 33,
                46: 34,
                4: 35,
                17: 36,
                26: 37,
                29: 38,
                44: 39,
                16: 40,
                47: 41,
                9: 42,
                33: 43,
                18: 44,
                39: 45,
                38: 46,
                40: 47,
                32: 48,
                50: 49,
                48: 50,
                49: 51,
                51: 51,
                53: 52,
                54: 54,
                56: 55,
                52: 56,
                57: 57,
                55: 58,
                59: 59,
                58: 60}


def get_total_counts(num_topics, document_topics_filename):
    """
    Gets the total counts of papers where a given topic is
    dominant for each topic
    :param num_topics: The number of topics in the model
    :param document_topics_filename: The prebuilt document_topics structure
    :return:
    """
    if os.path.exists(document_topics_filename):
        with open(document_topics_filename, "rb") as file_in:
            document_topics = pickle.load(file_in)
    else:
        return None
    base_row = OrderedDict({"organization": "all"})
    row = {}
    for topic_number in range(num_topics):
        topic_papers = document_topics[document_topics["Dominant_Topic"] == topic_number]
        total_papers_in_topic = len(topic_papers)
        corrected_topic_number = topic_mapper[topic_number]
        row.update({f"topic_{corrected_topic_number}": total_papers_in_topic})
    # We want to order the topics by their number, so we sort by the number even
    # though the actual label is a string
    sorted_row = OrderedDict(sorted(row.items(), key=lambda x: int(x[0].split("_")[1])))
    # The ordering can't include the organization label so we add that last
    base_row.update(sorted_row)
    return base_row


def get_organization_counts(num_topics, top_topics_by_org_filename, rows):
    """
    Getting the counts of which organizations have published papers that fit
    into which topic (again using the "dominant topic" metric).
    :param num_topics: Number of topics in our model
    :param top_topics_by_org_filename: The prebuilt top_topics_by_org structure
    :param rows: Our first row of all topics, to populate
    :return:
    """
    if os.path.exists(top_topics_by_org_filename):
        with open(top_topics_by_org_filename, "rb") as file_in:
            top_topics_by_org = pickle.load(file_in)
    for organization in top_topics_by_org.keys():
        base_row = OrderedDict({"organization": organization})
        row = {}
        for topic, count in top_topics_by_org[organization].most_common(num_topics):
            corrected_topic_number = topic_mapper[topic]
            row.update({f"topic_{corrected_topic_number}": count})
        sorted_row = OrderedDict(sorted(row.items(), key=lambda x: int(x[0].split("_")[1])))
        base_row.update(sorted_row)
        rows.append(base_row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("document_topics", type=str,
                        help="The name of the pickle file containing document_topics data.")
    parser.add_argument("top_topics_by_org", type=str,
                        help="The name of the pickle file containing topics by organization data.")
    parser.add_argument("output_csv", type=str, help="The name of the output csv.")
    parser.add_argument("-n", "--num_topics", type=int, default=60, required=False,
                        help="The number of topics in the topic model. Defaults to 60.")
    args = parser.parse_args()
    if not args.document_topics or not args.top_topics_by_org or not args.output_csv:
        parser.print_help()
    rows = [get_total_counts(args.num_topics, args.document_topics)]
    rows = get_organization_counts(args.num_topics, args.top_topics_by_org, rows)
    # Building the fieldnames
    fieldnames = ["organization"]
    fieldnames.extend([f"topic_{i}" for i in range(1, args.num_topics + 1)])
    # Writing out the CSV
    with open(args.output_csv, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()