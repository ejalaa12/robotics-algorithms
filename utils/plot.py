import matplotlib.pyplot as plt


def identify_time_axis(time, array):
    l, c = array.shape
    if time == c:
        return 0


def plot_data(time, data):

    fig, ax = plt.subplots()
    for i in range(data.shape[0]):
        ax.plot(time, data[i])

