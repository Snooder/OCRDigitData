import inspect
import sys
import numpy
import math

computed_statistics = []
prior_dist = []
attributes = []
data_amnt = 0

'''
Raise a "not defined" exception as a reminder
'''
def _raise_not_defined():
    print "Method not implemented: %s" % inspect.stack()[1][3]
    sys.exit(1)


'''
Extract 'basic' features, i.e., whether a pixel is background or
forground (part of the digit)
'''
def extract_basic_features(digit_data, width, height):
    features=[]
    for x in range(0, width):
        for y in range(0, height):
            if digit_data[x][y] == 0:
                features.append(0)
            else:
                features.append(1)
    return features

'''
Extract advanced features that you will come up with
'''
def extract_advanced_features(digit_data, width, height):
    features=[]
    for x in range(0, width):
        for y in range(0, height):
            if digit_data[x][y] == 0:
                features.append(0)
            elif digit_data[x][y] == 1:
                features.append(1)
            else:
                features.append(2)
    return features

'''
Extract the final features that you would like to use
'''
def extract_final_features(digit_data, width, height):
    features=[]
    for x in range(0, width):
        for y in range(0, height):
            if digit_data[x][y] == 0:
                features.append(0)
            elif digit_data[x][y] == 1:
                features.append(1)
            else:
                features.append(2)
    return features

'''
Compute the parameters including the prior and and all the P(x_i|y). Note
that the features to be used must be computed using the passed in method
feature_extractor, which takes in a single digit data along with the width
and height of the image. For example, the method extract_basic_features
defined above is a function than be passed in as a feature_extractor
implementation.

The percentage parameter controls what percentage of the example data
should be used for training.
'''
def compute_statistics(data, label, width, height, feature_extractor, percentage=100.0):
    global computed_statistics
    global prior_dist
    global attributes
    global data_amnt

    data_amnt = int((percentage/100) * len(data))
    prior_dist = [0]*10

    # finds the occurrence of each digit using the label array
    for x in range(0, data_amnt):
        value = label[x]
        prior_dist[value] += 1

    computed_statistics = numpy.zeros((width*height, 10))
    attributes = numpy.zeros((3, 10))

    if feature_extractor == extract_basic_features:
        for x in range(0, data_amnt):
            features = feature_extractor(data[x], width, height)
            number = label[x]
            for y in range(0, width*height):
                computed_statistics[y][number] += features[y]

    elif feature_extractor == extract_advanced_features or feature_extractor == extract_final_features:
        leftmost_pixel = width-1
        rightmost_pixel = 0
        topmost_pixel = height-1
        bottommost_pixel = 0
        for x in range(0, data_amnt):
            features = feature_extractor(data[x], width, height)
            number = label[x]
            x_loc = 0
            y_loc = 0
            for y in range(0, width*height):
                if features[y] == 2:
                    computed_statistics[y][number] += 1
                    attributes[0][number] += 1
                else:
                    computed_statistics[y][number] += features[y]
                if features[y] == 1 or features[y] == 2:
                    x_loc = y%width
                    y_loc = y/width
                    if x_loc < leftmost_pixel:
                        leftmost_pixel = x_loc
                    if x_loc > rightmost_pixel:
                        rightmost_pixel = x_loc
                    if y_loc < topmost_pixel:
                        topmost_pixel = y_loc
                    if y_loc > bottommost_pixel:
                        bottommost_pixel = y_loc
            attributes[1][number] += (rightmost_pixel - leftmost_pixel + 1)
            attributes[2][number] += (bottommost_pixel - topmost_pixel + 1)

        for x in range(0, 3):
            for y in range(0, 10):
                attributes[x][y] = float(attributes[x][y])/prior_dist[y]
                if x == 0:
                    attributes[x][y] *= 0.7
                elif x == 1:
                    attributes[x][y] *= 0.1
                elif x == 2:
                    attributes[x][y] *= 0.2


'''
For the given features for a single digit image, compute the class
'''
def compute_class(features):
    predicted = -1
    max_arr = [0] * 10
    smoothing = 1
    sum = 0

    for x in range(0,10):
        for y in range(0, len(features)):
            if features[y] == 1:
                numerator = computed_statistics[y][x] + smoothing
                denominator = prior_dist[x] + smoothing
                sum += math.log10(numerator/denominator)
            else:
                numerator = (prior_dist[x]-computed_statistics[y][x]) + smoothing
                denominator = prior_dist[x] + smoothing
                sum += math.log10(numerator/denominator)

        val = float(prior_dist[x])/data_amnt
        sum += math.log10(val)
        max_arr[x] = sum
        sum = 0

    predicted = numpy.argmax(max_arr)
    return predicted


def compute_class_extended(features, width, height):
    predicted = -1
    max_arr = [0]*10
    smoothing = 1
    sum_attributes = [0]*10
    sum = 0

    for x in range(0, 10):
        for y in range(0, 3):
            sum_attributes[x] += attributes[y][x]

    counter = 0
    leftmost_pixel = width-1
    rightmost_pixel = 0
    topmost_pixel = height-1
    bottommost_pixel = 0
    for x in range(0,10):
        x_loc = 0
        y_loc = 0
        for y in range(0, len(features)):
            if features[y] == 1 or features[y] == 2:
                if features[y] == 2:
                    counter += 1
                numerator = computed_statistics[y][x] + 0.1/1000
                denominator = prior_dist[x] + 0.1/1000
                sum += math.log10(numerator/denominator)
                x_loc = y%width
                y_loc = y/width
                if x_loc < leftmost_pixel:
                    leftmost_pixel = x_loc
                if x_loc > rightmost_pixel:
                    rightmost_pixel = x_loc
                if y_loc < topmost_pixel:
                    topmost_pixel = y_loc
                if y_loc > bottommost_pixel:
                    bottommost_pixel = y_loc
            else:
                numerator = (prior_dist[x]-computed_statistics[y][x]) + 0.1/1000
                denominator = prior_dist[x] + 0.1/1000
                sum += math.log10(numerator/denominator)

        val = float(prior_dist[x])/data_amnt
        sum += math.log10(val)
        max_arr[x] = sum
        sum = 0

    predicted = numpy.argmax(max_arr)
    return predicted

'''
Compute joint probaility for all the classes and make predictions for a list
of data
'''
def classify(data, width, height, feature_extractor):
    predicted=[]
    for x in range(0, len(data)):
        features = feature_extractor(data[x], width, height)
        if feature_extractor == extract_basic_features:
            predicted_class = compute_class(features)
        elif feature_extractor == extract_advanced_features or feature_extractor == extract_final_features:
            predicted_class = compute_class_extended(features, width, height)
        predicted.append(predicted_class)
    return predicted
