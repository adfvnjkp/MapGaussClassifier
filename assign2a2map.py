from pylab import *
import numpy as np

"""
@Author Tuanzhang Li 
tl7587@rit.edu
MAP classifier
"""

"""
To calculate the mu for the MAP classifier
"""


def mu(inputs):
    x0 = inputs[:100, 0]
    x1 = inputs[100:, 0]
    y0 = inputs[:100, 1]
    y1 = inputs[100:, 1]
    mu_x0 = 0
    mu_x1 = 0
    mu_y0 = 0
    mu_y1 = 0
    for i in range(len(x0)):
        mu_x0 += x0[i]
        mu_x1 += x1[i]
        mu_y0 += y0[i]
        mu_y1 += y1[i]
    mu_x0 /= 100
    mu_x1 /= 100
    mu_y1 /= 100
    mu_y0 /= 100
    return [mu_x0, mu_y0], [mu_x1, mu_y1], x0, x1, y0, y1


"""
To calculate the covariance
"""


def cov(a, b, mean_vector):
    total = 0
    for i in range(len(a)):
        total += ((a[i] - mean_vector[0]) * (b[i] - mean_vector[1]))
    total /= (len(a) - 1)
    return total


"""
To calculate the covariance matrix for the MAP classifier
"""


def cov_matrix(inputs):
    [mu_x0, mu_y0], [mu_x1, mu_y1], x0, x1, y0, y1 = mu(inputs)
    matrix_0 = [[0, 0], [0, 0]]
    matrix_1 = [[0, 0], [0, 0]]
    matrix_0[0][0] = cov(x0, x0, [mu_x0, mu_x0])
    matrix_0[0][1] = cov(x0, y0, [mu_x0, mu_y0])
    matrix_0[1][0] = matrix_0[0][1]
    matrix_0[1][1] = cov(y0, y0, [mu_y0, mu_y0])

    matrix_1[0][0] = cov(x1, x1, [mu_x1, mu_x1])
    matrix_1[0][1] = cov(x1, y1, [mu_x1, mu_y1])
    matrix_1[1][0] = matrix_1[0][1]
    matrix_1[1][1] = cov(y1, y1, [mu_y1, mu_y1])

    return matrix_0, matrix_1, [mu_x0, mu_y0], [mu_x1, mu_y1], x0, x1, y0, y1


"""
MAP classifier achieved by discriminant function for the normal density
"""


def map_classifier(x, mu_i, cov_i):
    subtraction_result = np.array([x[0] - mu_i[0], x[1] - mu_i[1]])
    return -0.5 * np.dot(np.dot(subtraction_result.T, np.linalg.inv(cov_i)), subtraction_result) - 0.5 * np.log(
        np.linalg.det(cov_i)) + np.log(0.5)


"""
To plot the background, decision boundary, and the given points of the input matrix
"""


def plot_graph(inputs, mu_0, cov_0, mu_1, cov_1):
    plt.figure()
    frame = plt.gca()
    # To set the boundary
    x_range = [-3, 5]
    y_range = [-3, 3]
    frame.set_xlim(x_range)
    frame.set_ylim(y_range)
    # To plot the background
    points = []
    x0_list = []
    x1_list = []
    y0_list = []
    y1_list = []
    for x in np.arange(x_range[0], x_range[1], 0.1):
        for y in np.arange(y_range[0], y_range[1], 0.1):
            if map_classifier(np.array([x, y]).T, mu_0, cov_0) > map_classifier(np.array([x, y]).T, mu_1, cov_1):
                x0_list.append(x)
                y0_list.append(y)
                points.append([x, y, 0])
            else:
                x1_list.append(x)
                y1_list.append(y)
                points.append([x, y, 1])
    plt.scatter(x0_list, y0_list, color='c', marker='.', label='1', s=0.1)
    plt.scatter(x1_list, y1_list, color='orange', marker='.', label='1', s=0.1)
    # To plot the decision boundary
    line_x = []
    line_y = []
    count = 1
    while count < 4800:
        last = points[count - 1]
        current = points[count]
        if last[-1] != current[-1]:
            border_x = (current[0] + last[0]) / 2
            border_y = (current[1] + last[1]) / 2
            if not (border_y < 0 and border_x < 2):
                line_x.append(border_x)
                line_y.append(border_y)
        count += 1
    plt.scatter(line_x, line_y, marker='.', color='black', s=10)
    # To plot the given points of the input matrix
    ele_x1 = []
    ele_y1 = []
    ele_x2 = []
    ele_y2 = []
    for row in inputs:
        if row[2] == 0:
            ele_x1.append(row[0])
            ele_y1.append(row[1])
        else:
            ele_x2.append(row[0])
            ele_y2.append(row[1])
    plt.scatter(ele_x1, ele_y1, color='c', marker='o', label='1')
    plt.scatter(ele_x2, ele_y2, color='orange', marker='o', label='2')
    plt.show()


"""
To calculate the confusion matrix
"""


def confusion_matrix(cov_0, cov_1, mu_0, mu_1, inputs):
    prediction0_0, prediction0_1, prediction1_0, prediction1_1 = 0, 0, 0, 0
    for x, y, label in zip(inputs[:, 0], inputs[:, 1], inputs[:, 2]):
        predict_for_0 = map_classifier([x, y], mu_0, cov_0)
        predict_for_1 = map_classifier([x, y], mu_1, cov_1)
        if predict_for_0 >= predict_for_1:
            predict = 0
            if predict == label:
                prediction0_0 += 1
            else:
                prediction1_0 += 1
        else:
            predict = 1
            if predict == label:
                prediction1_1 += 1
            else:
                prediction0_1 += 1
    return prediction0_0, prediction0_1, prediction1_0, prediction1_1


"""
To calculate the classification rate
"""


def classification_rate(prediction0_0, prediction0_1, prediction1_0, prediction1_1):
    total = prediction0_0 + prediction0_1 + prediction1_0 + prediction1_1
    return (prediction0_0 + prediction1_1) / total * 100


def main():
    data = np.load('/Users/lituanzhang/Downloads/data.npy')
    cov_0, cov_1, mu_0, mu_1, x0, x1, y0, y1 = cov_matrix(data)
    prediction0_0, prediction0_1, prediction1_0, prediction1_1 = confusion_matrix(cov_0, cov_1, mu_0, mu_1, data)
    print("      Classification rate is " + str(
        classification_rate(prediction0_0, prediction0_1, prediction1_0, prediction1_1)) + "%")
    print("      The confusion matrix is :")
    print("            Predicted       Predicted")
    print("            0               1")
    print("   0        " + str(prediction0_0) + "              " + str(prediction0_1) + "")
    print("   1        " + str(prediction1_0) + "              " + str(prediction1_1) + "")
    print()
    plot_graph(data, mu_0, cov_0, mu_1, cov_1)
    plt.savefig("a2map.png")


if __name__ == "__main__":
    main()
