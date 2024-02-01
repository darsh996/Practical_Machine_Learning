# mode

def calculate_mode(x):
    dictionary = {}

    # get the count of every value
    for value in x:
        if dictionary.get(value) is None:
            dictionary[value] = 1
        else:
            dictionary[value] += 1

    # find the max count
    mode_count = 0
    for key in dictionary.keys():
        count = dictionary[key]
        if count >= mode_count:
            mode_count = count

    print(dictionary)

    # find single or multi mode
    mode = []
    for key in dictionary.keys():
        if dictionary.get(key) == mode_count:
            mode.append(key)

    print(f"mode = {mode} repeated {mode_count} times")
    print()


calculate_mode([10, 20, 30, 10, 40, 20, 10, 10])
calculate_mode([10, 20, 30, 10, 40, 20, 20, 10])
calculate_mode([10, 20, 30, 10, 20, 30])