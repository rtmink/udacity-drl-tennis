import matplotlib.pyplot as plt
import time
import numpy as np

'''
A function to plot the scores obtained from the training fuction.
'''
def plot_scores(scores, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.title(title)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Score")
    plt.show()

'''
A helper function to calculate elapsed time during training.
'''
def get_minutes_and_seconds_from_start_time(start_time):
    elapsed_seconds = int(time.time() - start_time)
    minutes = elapsed_seconds // 60
    seconds = elapsed_seconds - (minutes * 60)
    return minutes, seconds