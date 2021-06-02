from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

"""
restructured for assignment#02 question#02 
"""


def read_data_from_file(file_address):
    # read from csv line by line
    csv_data = np.ndarray(shape=(94, 3))
    with open(file_address) as file:
        data_lines = csv.reader(file, delimiter=',', quotechar='|')
        i = 0
        for line in data_lines:
            csv_data.put(indices=i, values=float(line[0]))
            csv_data.put(indices=i + 1, values=(float(line[1])))
            csv_data.put(indices=i + 2, values=(int(line[2])))
            i = i + 3
    # collecting given train data
    class1, class2, class3, class4 = [], [], [], []
    for line in csv_data:
        if line[2] == 1:
            class1.append([line[0], line[1]])
        elif line[2] == 2:
            class2.append([line[0], line[1]])
        elif line[2] == 3:
            class3.append([line[0], line[1]])
        elif line[2] == 4:
            class4.append([line[0], line[1]])

    return csv_data, class1, class2, class3, class4


def fill_covariance(xyc, mean_xy):
    filled_matrix = [[0, 0], [0, 0]]
    for xy in xyc:
        for i in range(2):
            for j in range(2):
                filled_matrix[i][j] += (xy[i] - mean_xy[i]) * (xy[j] - mean_xy[j])
    filled_matrix = [[filled_value / (len(xyc) - 1) for filled_value in filled_row] for filled_row in filled_matrix]
    return filled_matrix


def get_mean_covariance(data_input, question_number):
    # slice from csv file, as source file had separate each class apart
    xyc1 = data_input[:20, :2].tolist()
    xyc2 = data_input[20:48, :2].tolist()
    xyc3 = data_input[48:75, :2].tolist()
    xyc4 = data_input[75:, :2].tolist()
    mean_class = []
    cov_class = []
    for xyc in [xyc1, xyc2, xyc3, xyc4]:
        __x_ = __y_ = 0
        for xy in xyc:
            __x_ += xy[0]
            __y_ += xy[1]
        mean_class.append([__x_ / (len(xyc) - 1), __y_ / (len(xyc) -1)])

    for xyc, mean_xy in zip([xyc1, xyc2, xyc3, xyc4], mean_class):
        cov_class.append(fill_covariance(xyc, mean_xy))

    # priors and re-do prior
    if question_number == "b":
        priors = [20 / 94 - ((47 - 19) / 94) / 3, 28 / 94 - ((47 - 19) / 94) / 3, 27 / 94 - ((47 - 19) / 94) / 3, 47 / 94]
        if sum(priors) >= 1:
            print("prior changed for b question.. ")
    else:
        priors = [20 / 94, 28 / 94, 27 / 94, 19 / 94]

    return priors, mean_class, cov_class


def drawTool(class1, class2, class3, class4, boundary_line, draw_background):
    # define different marker for different
    if draw_background:
        marker_shape = ["." for _ in range(4)]
        marker_size = 1
    else:
        marker_shape = ["s", "o", "x", "d"]
        marker_size = 10
    # draw by class
    plt.scatter(class1[:, 0], class1[:, 1], marker=marker_shape[0], c="blue", s=marker_size)
    plt.scatter(class2[:, 0], class2[:, 1], marker=marker_shape[1], c="orange", s=marker_size)
    plt.scatter(class3[:, 0], class3[:, 1], marker=marker_shape[2], c="red", s=marker_size)
    plt.scatter(class4[:, 0], class4[:, 1], marker=marker_shape[3], c="green", s=marker_size)
    # draw boundary if boundary collection exist
    if boundary_line is not None:
        plt.scatter(boundary_line[:, 0], boundary_line[:, 1], marker="o", c="black", s=marker_size + 1)


def create_background():
    gap = 0.006
    x_list = []
    y_list = []
    point_list = []
    # computing for each point in the grid
    for new_x in np.arange(0.1, 0.8, gap):
        x_list.append(new_x)
    for new_y in np.arange(0, 1, (1 / len(x_list))):
        y_list.append(new_y)
    for __x_ in x_list:
        for __y_ in y_list:
            point_list.append([__x_, __y_])
    return np.array(point_list)


def get_cost_function(i, j, question_number):
    # cost matrix
    given_matrix = [[-0.20, 0.07, 0.07, 0.07], [0.07, -0.15, 0.07, 0.07], [0.07, 0.07, -0.05, 0.07], [0.03, 0.03, 0.03, 0.03]]
    # uniform cost
    uniform_matrix = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
    if question_number == "a" or question_number == "b":
        cost_matrix = given_matrix
    else:
        cost_matrix = uniform_matrix
    return cost_matrix[i][j]


def get_map_classifier(point_xy, prior_i, mu_i, cov_i, question_number):
    subtraction_result = np.array([point_xy[0] - mu_i[0], point_xy[1] - mu_i[1]])
    return 1 / (2 * math.pi * np.linalg.det(cov_i) ** (1/2)) \
           * math.exp(-0.5 * np.dot(np.dot(subtraction_result, np.linalg.inv(cov_i)), subtraction_result)) * prior_i


def classifyTool(point_list, prior_list, mean_list, cov_list, question_number):
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    last_class = 0
    decision_boundary = []
    classify_result = []
    for __xy_ in point_list:
        aggregate_risks = []
        for minima_class in range(4):
            new_risk = 0
            for map_class in range(4):
                new_risk += get_map_classifier(__xy_, prior_list[map_class], mean_list[map_class], cov_list[map_class], question_number) \
                            * get_cost_function(minima_class, map_class, question_number)
            aggregate_risks.append(new_risk)
        minima_class = aggregate_risks.index(min(aggregate_risks)) + 1
        classify_result.append(minima_class)
        # if current point classified different from last point
        if last_class != minima_class:
            decision_boundary.append(__xy_)
            # update last point class
            last_class = minima_class

        # pour point according to class belongs to
        if minima_class == 1:
            class1.append(__xy_)
        elif minima_class == 2:
            class2.append(__xy_)
        elif minima_class == 3:
            class3.append(__xy_)
        else:
            class4.append(__xy_)
    return classify_result, np.array(decision_boundary), np.array(class1), np.array(class2), np.array(class3), np.array(class4)


def confusionTool(res_classification, raw_classification, question_number):
    class1 = [0 for _ in range(4)]
    class2 = [0 for _ in range(4)]
    class3 = [0 for _ in range(4)]
    class4 = [0 for _ in range(4)]
    res_Eq_raw = 0
    # raw as row, res as col
    for res, raw in zip(res_classification, raw_classification):
        raw = int(raw)
        if res == 1:
            class1[raw - 1] += 1
        elif res == 2:
            class2[raw - 1] += 1
        elif res == 3:
            class3[raw - 1] += 1
        else:
            class4[raw - 1] += 1
        if res == raw:
            res_Eq_raw += 1

    print("====  Confusion Matrix of Graph (" + question_number + ") is:  =====")
    print("Prediction  ", "  Bolts ", "  Nuts ", " Rings ", " Scraps ")
    print("       Bolts", "     " + str(class1[0]), "     " + str(class1[1]), "     " + str(class1[2]), "     " + str(class1[3]))
    print("        Nuts", "     " + str(class2[0]), "     " + str(class2[1]), "     " + str(class2[2]), "     " + str(class2[3]))
    print("       Rings", "     " + str(class3[0]), "     " + str(class3[1]), "     " + str(class3[2]), "     " + str(class3[3]))
    print("      Scraps", "     " + str(class4[0]), "     " + str(class4[1]), "     " + str(class4[2]), "     " + str(class4[3]))
    print("  *Correction Rate is: " + str(res_Eq_raw / len(res_classification) * 100) + "%.")
    print("")


def main():
    # read data. CSV and this python file should be in the same folder
    data_input, raw1, raw2, raw3, raw4 = read_data_from_file(
        "nuts_bolts.csv")
    # run 3 question within same main structure
    for question_number in ["a", "b", "c"]:
        # set figure size
        plt.figure()
        # get cov, mean, prior
        priors, mean_class, cov_class = get_mean_covariance(data_input, question_number)
        # classify given point
        classified_csv, _, class1, class2, class3, class4 = classifyTool(raw1 + raw2 + raw3 + raw4, priors, mean_class, cov_class, question_number)
        # draw csv point
        drawTool(class1, class2, class3, class4, None, False)
        # create confusion table
        confusionTool(classified_csv, data_input[:, 2:], question_number)
        # classify background
        _, decision_boundary, class1, class2, class3, class4 = classifyTool(create_background(), priors, mean_class, cov_class, question_number)
        # draw background
        drawTool(class1, class2, class3, class4, decision_boundary, True)
        # save final picture, reset current plot graph for next round of draws
        plt.savefig("Bayesian Classifier Graph (" + question_number + ").png")
        plt.clf()


if __name__ == "__main__":
    main()
