import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import pickle


def plot_topic_by_year(topics_by_year, topics, skip_2020=True):
    """
    Plots changes in counts of papers in topic by year.
    :param topics_by_year: Dict containing counts of papers in topic by year.
    :param topics: The topics to chart. Maximum 10 allowed.
    :param skip_2020: Whether to skip 2020 in the chart. Defaults to yes since 2020 is incomplete.
    :return:
    """
    colors = ["#0B1F41", "#003DA6", "#7C336F", "#B53A6D", "#3C8786", "#7AC4A5", "#F17F4C", "#15AFD0", "#839DC5", "#E5BF21"]
    if len(topics) > 10:
      print ("Too many topics")
      return
    for i, topic in enumerate(topics):
        data = topics_by_year[topic]
        # years are first element in tuple, counts are second
        x_axis = [i[0] for i in data]
        y_axis = [i[1] for i in data]
        if skip_2020 and x_axis[-1] == 2020:
            x_axis = x_axis[:-1]
            y_axis = y_axis[:-1]
        plt.plot(x_axis, y_axis, colors[i])
    plt.grid()
    title_obj = plt.title(f"Paper Counts of Topics {topics}")
    ax = plt.gca()
    plt.setp(title_obj, color="#31353A")
    x_label = plt.xlabel('Year')
    plt.setp(x_label, color="#63676B")
    plt.setp(plt.getp(ax, 'xticklabels'), color="#63676B")
    y_label = plt.ylabel('Number of Papers')
    plt.setp(y_label, color="#63676B")
    plt.setp(plt.getp(ax, 'yticklabels'), color="#63676B")
    plt.legend(topics)
    plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str,
                        help="The name of the file containing data to plot.")
    parser.add_argument("topic_numbers", type=int, nargs="+",
                        help="The topic numbers you wish to plot.")
    parser.add_argument("-i", "--include_2020", action="store_false",
                        help="Include this flag to plot 2020 even though data for the year is incomplete.")
    args = parser.parse_args()
    if not args.filename or not args.topic_numbers:
        parser.print_help()
    with open(args.filename, "rb") as file_in:
        topic_data = pickle.load(file_in)
    plot_topic_by_year(topic_data, args.topic_numbers, args.include_2020)


if __name__ == "__main__":
    main()
