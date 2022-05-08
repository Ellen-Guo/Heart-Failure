# Machine Learning | Final Project
import csv
import numpy as np
import matplotlib.pyplot as plt


# extracts data file information
def read_data(name):
    # open file in read mode
    filename = open(name, 'r')
    # creating dictreader object
    file = csv.DictReader(filename)
    # empty lists for data features
    age = []
    anaemia = []
    creatinine_phosphokinase = []
    diabetes = []
    ejection_fraction = []
    high_blood_pressure = []
    platelets = []
    serum_creatinine = []
    serum_sodium = []
    sex = []
    smoking = []
    time = []
    death_event = []
    # iterate over datafile and append values
    for col in file:
        age.append(col['age'])
        anaemia.append(col['anaemia'])
        creatinine_phosphokinase.append(col['creatinine_phosphokinase'])
        diabetes.append(col['diabetes'])
        ejection_fraction.append(col['ejection_fraction'])
        high_blood_pressure.append(col['high_blood_pressure'])
        platelets.append(col['platelets'])
        serum_creatinine.append(col['serum_creatinine'])
        serum_sodium.append(col['serum_sodium'])
        sex.append(col['sex'])
        smoking.append(col['smoking'])
        time.append(col['time'])
        death_event.append(col['DEATH_EVENT'])
    # convert string lists to int lists; make each list an array (299 x 1)
    age = np.array(list(map(float, age))).reshape(-1, 1)
    anaemia = np.array(list(map(int, anaemia))).reshape(-1, 1)
    creatinine_phosphokinase = np.array(list(map(int, creatinine_phosphokinase))).reshape(-1, 1)
    diabetes = np.array(list(map(int, diabetes))).reshape(-1, 1)
    ejection_fraction = np.array(list(map(int, ejection_fraction))).reshape(-1, 1)
    high_blood_pressure = np.array(list(map(int, high_blood_pressure))).reshape(-1, 1)
    platelets = np.array(list(map(float, platelets))).reshape(-1, 1)
    serum_creatinine = np.array(list(map(float, serum_creatinine))).reshape(-1, 1)
    serum_sodium = np.array(list(map(int, serum_sodium))).reshape(-1, 1)
    sex = np.array(list(map(int, sex))).reshape(-1, 1)
    smoking = np.array(list(map(int, smoking))).reshape(-1, 1)
    time = np.array(list(map(int, time))).reshape(-1, 1)
    death_event = np.array(list(map(int, death_event))).reshape(-1, 1)
    # make a feature vector for each patient
    # all features (299 x 12 array)
    all_features = np.concatenate((age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                                   high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex,
                                   smoking, time), axis=1)
    # two major features (299 x 2 array)
    two_features = np.concatenate((ejection_fraction, serum_creatinine, time), axis=1)
    # target of predictions
    labels = np.array(death_event)
    return all_features, two_features, labels


# function that splits the data into test, training, and validation sets
def split_data(data, index, n):
    data_split = (data[index[0:n], :], data[index[n:], :])
    return data_split


# apply basis to feature matrix; return phi(x)
def identity_basis(array):
    # apply identity basis
    r, _ = np.shape(array)
    array_basis = np.append(np.ones((1, r)), array.transpose(), axis=0)
    return array_basis


# cross validation for finding lambda
def cross_validation(train, train_target, validate, validate_target, delta_lambda):
    lambda_arr = []
    error_arr = []
    stop = False
    lambda_k = 0
    k = 0
    r, c = np.shape(train)
    prev_theta = np.zeros((r, 1))
    prev_err_val = np.inf
    err_train_k, theta_k = IRLS(train, train_target, lambda_k)
    err_val = obj_fun(len(validate_target), theta_k, validate, validate_target, lambda_k)
    while not stop:
        # save previous validation error, lambda, and theta
        lambda_arr.append(lambda_k)
        error_arr.append(np.abs(err_val))
        prev_theta = np.copy(theta_k)
        prev_err_val = np.copy(err_val)
        # train and update
        lambda_k += delta_lambda
        err_train, theta_k = IRLS(train, train_target, lambda_k)
        err_val = obj_fun(len(validate_target), theta_k, validate, validate_target, lambda_k)
        k += 1
        stop = (k > 500) | (np.abs(err_val) > np.abs(prev_err_val))
    return prev_theta, lambda_arr, error_arr


def visual(label1, x1, y1, label2, x2, y2):
    plt.figure()
    plt.plot(x1, y1, 'bo')
    plt.plot(x2, y2, 'ro')
    plt.xlabel('lambda')
    plt.ylabel('error value')
    plt.legend([label1, label2])
    plt.show()


# Logistic Regression Methods
def sigmoid(theta, phi):
    x = np.matmul(theta.transpose(), phi)
    sig = 1 / (1 + np.exp(-1 * x))
    return sig


# gradient calculation; regularized by lambda
def grad_obj(theta, t, feature, lambda_k):
    pred = sigmoid(theta, feature)
    grad = np.matmul(feature, (pred.transpose() - t)) + lambda_k*theta
    return grad


# Hessian calculation; regularized by lambda
def hessian(R, feature, lambda_k):
    H = np.matmul(feature, np.matmul(R, feature.transpose())) + lambda_k
    return H


# Error calculation; Regularized Objective Function
def obj_fun(N, theta, feature, target, lambda_k):
    err = 0
    for i in range(N):
        pred = sigmoid(theta, feature[:, i])
        if pred == 0:
            pred += 1 * (10**-12)
        elif pred == 1:
            pred -= 1 * (10**-12)
        err += ((target[i] * np.log(pred)) + ((1 - target[i]) * np.log(1 - pred)))**2
    err = (-1/2) * err + ((lambda_k/2) * np.linalg.norm(theta)**2)
    return err


# Diagonal matrix for Hessian Calculation
def r_matrix(N, theta, feature):
    R = np.zeros((N, N))
    for i in range(N):
        pred = sigmoid(theta, feature[:, i])
        R[i][i] = pred * (1-pred)
    return R


# Iterative Re-weighted Least Squares (IRLS) Function: Training
def IRLS(train, train_tar, lambda_k):
    stop = False
    k = 0
    eps = 10**-3
    r, c = np.shape(train)
    theta = np.zeros((r, 1))
    R = r_matrix(c, theta, train)
    while not stop:
        H = hessian(R, train, lambda_k)
        grad = grad_obj(theta, train_tar, train, lambda_k)
        try:
            theta = theta - np.matmul(np.linalg.inv(H), grad)
        except:
            theta = theta - np.matmul(np.linalg.pinv(H), grad)
        R = r_matrix(c, theta, train)
        k += 1
        stop = (k > 700) | (np.linalg.norm(grad_obj(theta, train_tar, train, lambda_k), ord=1) < eps)
    err = obj_fun(c, theta, train, train_tar, lambda_k)
    return err, theta


# Final Error Calculations
def final_error(feature, target, theta):
    error = 0
    r, c = np.shape(feature)
    for i in range(c):
        predict = sigmoid(theta, feature[:, i])
        if predict < 0.5:
            if target[i] != 0:
                error += 1
        else:
            if target[i] != 1:
                error += 1
    return error/c


def main():
    # get features and label from data set
    all_features, two_features, labels = read_data('heart_failure_clinical_records_dataset.csv')

    iterations = 100
    # gather training, validation & test errors for each iteration
    t1_err = []
    v1_err = []
    test1_err = []
    t2_err = []
    v2_err = []
    test2_err = []

    # for plotting and visualization
    lambda1 = 0
    error1 = []
    lambda2 = 0
    error2 = []

    for i in range(iterations):
        # split data 2/3 training, 1/3 testing; random index shuffling
        n = len(labels)
        index = np.arange(n)
        # np.random.seed(128)
        np.random.shuffle(index)
        train_features, test_features = split_data(all_features, index, int(n * 2 / 3))
        train_targets, test_targets = split_data(labels, index, int(n * 2 / 3))
        # -------------------------- Multi-Feature Training ---------------------------------------------------
        # split training into validation (1/3) and train (2/3)
        n_train = len(train_features)
        train_index = np.arange(n_train)
        np.random.shuffle(train_index)
        train, vaildation = split_data(train_features, train_index, int(n_train * 2 / 3))
        train_tar, vaildation_tar = split_data(train_targets, train_index, int(n_train * 2 / 3))

        # apply basis
        phi_train = identity_basis(train)
        phi_validation = identity_basis(vaildation)
        phi_test = identity_basis(test_features)

        # Regularized Logistic Regression w/ cross validation
        theta1, lambda1, error1 = cross_validation(phi_train, train_tar, phi_validation, vaildation_tar, 10 ** -3)
        # print('Multi-Feature Iterantions: ', len(error1))

        # theta parameters
        # np.set_printoptions(threshold=np.inf)
        # print('Multi-Feature Parameters: \n', theta1)

        # Training Set Error
        train_error = final_error(phi_train, train_tar, theta1)
        validation_error = final_error(phi_validation, vaildation_tar, theta1)
        t1_err.append(train_error)
        v1_err.append(validation_error)
        # print('Training Error: ', train_error)
        # print('Validation Error: ', validation_error)
        # Testing Set Error
        test_error = final_error(phi_test, test_targets, theta1)
        test1_err.append(test_error)
        # print('Testing Error: ', test_error)

        # --------------------------------- Two-Feature Training w/ check-in time ---------------------------------------
        train_features2, test_features2 = split_data(two_features, index, int(n * 2 / 3))
        train_targets2, test_targets2 = split_data(labels, index, int(n * 2 / 3))
        train2, vaildation2 = split_data(train_features2, train_index, int(n_train * 2 / 3))
        train_tar2, vaildation_tar2 = split_data(train_targets2, train_index, int(n_train * 2 / 3))

        # apply identity basis
        phi_train2 = identity_basis(train2)
        phi_validation2 = identity_basis(vaildation2)
        phi_test2 = identity_basis(test_features2)

        # Regularized Logistic Regression w/ cross validation
        theta2, lambda2, error2 = cross_validation(phi_train2, train_tar2, phi_validation2, vaildation_tar2, 10 ** -3)
        # print('Two-Feature Iterantions: ', len(error2))

        # theta parameters
        # np.set_printoptions(threshold=np.inf)
        # print('Two-Feature Parameters: \n', theta2)

        # Training Set Error
        train_error2 = final_error(phi_train2, train_tar2, theta2)
        validation_error2 = final_error(phi_validation2, vaildation_tar2, theta2)
        t2_err.append(train_error2)
        v2_err.append(validation_error2)
        # print('Training Error: ', train_error2)
        # print('Validation Error: ', validation_error2)
        # Testing Set Error
        test_error2 = final_error(phi_test2, test_targets2, theta2)
        test2_err.append(test_error2)
        # print('Test Error: ', test_error2)
        print("Iteration: ", i)

    # Average Result After 100 iterations
    # All-features
    print('All-Feature Average Errors')
    print('Training Error: ', np.average(t1_err))
    print('Validation Error: ', np.average(v1_err))
    print('Testing Error: ', np.average(test1_err))
    # Two-features & check-in time
    print('Two-Feature Average Errors')
    print('Training Error: ', np.average(t2_err))
    print('Validation Error: ', np.average(v2_err))
    print('Testing Error: ', np.average(test2_err))
    # plot error & lambda for visualization -- last iteration
    visual('All-Feature', lambda1, error1, 'Two-Feature', lambda2, error2)


# Run main function
if __name__ == '__main__':
    main()
