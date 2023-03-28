"""
This is a boilerplate pipeline
generated using Kedro 0.18.6
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import qr


from mdutils.mdutils import MdUtils
from mdutils import Html


# def GramSchmidt(A):
#     '''
#     Gram-Schmidt method of QR decomposition A=QR
#
#     Inputs
#     ======
#     A : float
#         (m,n) matrix
#
#     Returns
#     =======
#     Q : float
#         Orthogonal matrix.
#     R : float
#         Upper triangular matrix.
#
#     '''
#
#     # get the dimensions of A
#     m, n = A.shape
#
#     # U and E have the same shape
#     U = np.zeros(A.shape, dtype='float64')
#     E = np.zeros(A.shape, dtype='float64')
#
#     # calculate the orthogonal vectors (and unit vectors)
#     for i in range(0, n):
#         U[:, i] = A[:, i]
#         for j in range(0, i):
#             U[:, i] -= np.sum(A[:, i] * E[:, j]) * E[:, j]
#         E[:, i] = U[:, i] / np.linalg.norm(U[:, i])
#
#     # E is actually Q!!
#     Q = E
#
#     # calculate R
#     R = np.dot(Q.T, A)
#
#     # make sure it's upper-triangular!
#     R = np.triu(R)
#
#     return Q, R
#
#
# def QRLeastSq(x, y, deg):
#     '''
#     Use QR decomposition to perform least squares regression.
#
#     Inputs
#     ======
#     x : float
#         x-coordinates
#     y : float
#         y-coordinates
#     deg : int
#         Degree of polynomial to fit
#
#     Returns
#     =======
#     beta : float
#         Array of polynomial coefficients in order from highest degree
#         to the lowest (this is for compatibility with numpy.poly1d)
#     '''
#
#     # get the X matrix, with one column for each degree (deg)
#     X = np.zeros((x.size, deg + 1), dtype='float64')
#     X[:, 0] = 1.0
#     for i in range(1, deg + 1):
#         X[:, i] = X[:, i - 1] * x
#
#     # turn y into a matrix
#     Y = np.array([y]).T
#
#     # Get Q and R using the Gram-Schmidt process
#     Q, R = GramSchmidt(X)
#
#     # time to solve Rβ = Q^T Y
#     # where β contains the polynomial coefficients
#     qty = np.dot(Q.T, Y)
#
#     # solve set of equations for β
#     beta = np.linalg.solve(R, qty)
#
#     # return in reverse order like numpy.polyfit does
#     return beta.flatten()[::-1]


def read_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> pd.DataFrame:

    data['Datetime'] = pd.to_datetime(data['Datetime'])
    mask = (data['Datetime'] > np.datetime64('2019-05')) & (data['Datetime'] < np.datetime64('2019-06'))
    data = data.loc[mask]
    data = data[['Datetime', 'Humidity']]
    return data


def compute(data: pd.DataFrame, np_: int, mp_: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x = data['Datetime'].to_numpy().astype('datetime64[m]').astype('int')
    y = data['Humidity'].to_numpy()
    beta_1 = np.ndarray
    beta_2 = np.ndarray
    avg_min = float("inf")
    avg_sq_diff_min = float("inf")
    date = np.datetime64
    ptr = 0
    for i in range(int(len(x)/np_)*mp_, int(len(x)/np_)*(mp_ + 1)):
        if i < 2:  # can't evaluate polynome by 1 point!
            continue
        split_x = np.split(x, [i])
        split_y = np.split(y, [i])
        x_l = split_x[0]
        y_l = split_y[0]
        beta_l = qr.QRLeastSq(x_l, y_l, 1)
        p_l = np.poly1d(beta_l)
        yp_l = p_l(x_l)
        diff_l = yp_l - y_l

        x_r = split_x[1]
        y_r = split_y[1]
        beta_r = qr.QRLeastSq(x_r, y_r, 1)
        p_r = np.poly1d(beta_r)
        yp_r = p_r(x_r)
        diff_r = yp_r - y_r

        diff = np.concatenate((diff_l, diff_r), axis=0)
        average = np.sum(np.abs(diff), 0) / len(diff)
        diff = diff - average
        sq_diff = np.square(diff)
        avg_sq_diff = np.sqrt(np.sum(sq_diff, 0) / (len(sq_diff) - 1))

        if avg_sq_diff < avg_sq_diff_min:
            avg_sq_diff_min = avg_sq_diff
            avg_min = average
            beta_1 = beta_l
            beta_2 = beta_r
            date = x[i].astype('datetime64[m]')
            ptr = i

    split_x = np.split(x, [ptr])
    x_l = split_x[0]
    x_r = split_x[1]
    p_l = np.poly1d(beta_1)
    yp_l = p_l(x_l)
    p_r = np.poly1d(beta_2)
    yp_r = p_r(x_r)
    plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots(figsize=[16, 8])

    lines = []
    styles = ['-', '--', '-.', ':']

    x = x.astype('datetime64[m]')
    x_l = x_l.astype('datetime64[m]')
    x_r = x_r.astype('datetime64[m]')
    lines += ax.plot(x, y, '-', color='blue')

    lines += ax.plot(x_l, yp_l, '-', color='green')

    lines += ax.plot(x_r, yp_r, '-', color='red')

    ax.scatter([x[ptr]], [(yp_l[-1] + yp_r[0])/2], color="red", s=20)  # plotting single point

    ax.grid('on')
    ax.set(xlabel='Дата', ylabel='Давление')

    # specify the lines and labels of the first legend
    ax.legend(lines[:9], ['Выборка',
                          'Тренд слева',
                          'Тренд справа'],
              loc='center right', frameon=True, labelspacing=1, fontsize=16)
    fig.savefig('data/final_report/' + str(mp_) + '.png', format='png')

    return beta_1, beta_2, avg_sq_diff_min, avg_min, date

def chose_and_write(b1: np.ndarray, b2: np.ndarray, b3: np.ndarray, b4: np.ndarray, avg_sq_diff1: float, avg_sq_diff2: float, avg1: float, avg2: float, point1: np.datetime64, point2: np.datetime64):
    md_file = MdUtils(file_name='data/final_report/finalReport', title='Final Report')
    if avg_sq_diff1 > avg_sq_diff2:
        md_file.write('Уравнение левой прямой: ' + 'y = ' + str(b1[0]) + 'x + (' + str(b1[1]) + ')  \n')
        md_file.write('Уравнение правой прямой: ' + 'y = ' + str(b2[0]) + 'x + (' + str(b2[1]) + ')  \n')
        md_file.write('Дата изменения тренда: ' + point1.astype(str) + '  \n')
        md_file.write('Среднее отклонение от модели: ' + str(avg1) + '  \n')
        md_file.write('Среднеквадратичное отклонение от модели: ' + str(avg_sq_diff1) + '  \n')
        md_file.write('## trend change illustration: ')
        md_file.new_line(md_file.new_reference_image(text='trend change illustration', path=os.getcwd() + '/data/final_report/0.png', reference_tag='im'))
    else:
        md_file.write('Уравнение левой прямой: ' + 'y = ' + str(b3[0]) + 'x + (' + str(b3[1]) + ')  \n')
        md_file.write('Уравнение правой прямой: ' + 'y = ' + str(b4[0]) + 'x + (' + str(b4[1]) + ')  \n')
        md_file.write('Дата изменения тренда: ' + point2.astype(str) + '  \n')
        md_file.write('Среднее отклонение от модели: ' + str(avg2) + '  \n')
        md_file.write('Среднеквадратичное отклонение от модели: ' + str(avg_sq_diff2) + '  \n')
        md_file.write('## trend change illustration: ')
        md_file.new_line(
        md_file.new_reference_image(text='trend change illustration', path=os.getcwd() + '/data/final_report/1.png',
                                        reference_tag='im'))
    md_file.create_md_file()


def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    data_train = data.sample(
        frac=parameters["train_fraction"], random_state=parameters["random_state"]
    )
    data_test = data.drop(data_train.index)

    X_train = data_train.drop(columns=parameters["target_column"])
    X_test = data_test.drop(columns=parameters["target_column"])
    y_train = data_train[parameters["target_column"]]
    y_test = data_test[parameters["target_column"]]

    return X_train, X_test, y_train, y_test


def make_predictions(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> pd.Series:
    """Uses 1-nearest neighbour classifier to create predictions.

    Args:
        X_train: Training data of features.
        y_train: Training data for target.
        X_test: Test data for features.

    Returns:
        y_pred: Prediction of the target variable.
    """

    X_train_numpy = X_train.to_numpy()
    X_test_numpy = X_test.to_numpy()

    squared_distances = np.sum(
        (X_train_numpy[:, None, :] - X_test_numpy[None, :, :]) ** 2, axis=-1
    )
    nearest_neighbour = squared_distances.argmin(axis=0)
    y_pred = y_train.iloc[nearest_neighbour]
    y_pred.index = X_test.index

    return y_pred


def report_accuracy(y_pred: pd.Series, y_test: pd.Series):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """
    accuracy = (y_pred == y_test).sum() / len(y_test)
    logger = logging.getLogger(__name__)
    logger.info("Model has accuracy of %.3f on test data.", accuracy)
