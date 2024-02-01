# import numpy for stats calculation
import numpy as np


# data set
# python list
x = [10, 20, 53, 64, 36, 72, 51, 32, 66, 72, 84, 92]


def calculate_mean_1(x):
    sum_x = sum(x)
    mean_x = sum_x / len(x)
    print(f"mean = {mean_x}")


# calculate_mean_1(x)

def calculate_mean_2(x):
    # convert the list into an numpy array
    array = np.array(x)
    print(array)
    print(f"mean = {array.mean()}")

    # use mean function and pass the list without conversion
    mean = np.mean(x)
    print(f"mean = {mean}")


calculate_mean_2(x)


