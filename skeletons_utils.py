import sys
import numpy as np
import skeletons_utils


def missing_to_nan(data):
    if data.shape[1]!=52 and data.shape[1]!=55: 
        print("the data does not have 52 or 55 collumns as it should")
        sys.exit()
    else:
        x_index_list = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52]
        y_index_list = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53]
        score_index_list = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54]

        new_data = np.copy(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if j in score_index_list and np.isclose(data[i,j], 0, atol=0.1) and j!=0:
                    new_data[i, j-1]=np.NaN
                    new_data[i, j-2]=np.NaN
    return new_data


def center_openpose(data):
    if data.shape[1]!=55: 
        print("the data does not have 55 collumns as it should")
        sys.exit()
    else:
        center_x=(data[:,4]*data[:,6] + data[:,7]*data[:,9] + data[:,16]*data[:,18])/(data[:,6]+data[:,9]+data[:,18])
        center_y=(data[:,5]*data[:,6] + data[:,8]*data[:,9] + data[:,17]*data[:,18])/(data[:,6]+data[:,9]+data[:,18])
        x_index_list = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52]
        y_index_list = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53]

        centered_data = np.copy(data)
        for i in range(data.shape[1]):
            if i in x_index_list:
                centered_data[:, i]= centered_data[:, i] - center_x
            elif i in y_index_list:
                centered_data[:, i]= centered_data[:, i] - center_y
            else: continue
    return centered_data

def scale_openpose(data):
    if data.shape[1]!=55: 
        print("the data does not have 55 collumns as it should")
        sys.exit()
    else:
        x_index_list = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52]
        y_index_list = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53]

        scaled_data = np.copy(data)
        for i in range(data.shape[1]):
            if i in x_index_list:
                scaled_data[:, i]= scaled_data[:, i] / np.nanstd(data[:, i])
            elif i in y_index_list:
                scaled_data[:, i]= scaled_data[:, i] / np.nanstd(data[:, i])
            else: continue
    return scaled_data

    import sys

def center_otherpose(data):
    if data.shape[1]!=52: 
        print("the data does not have 52 collumns as it should")
        sys.exit()
    else:
        center_x=(data[:,16]*data[:,18] + data[:,19]*data[:,21] )/(data[:,18]+data[:,21])
        center_y=(data[:,17]*data[:,18] + data[:,20]*data[:,21] )/(data[:,18]+data[:,21])
        print(center_x, center_y)
        x_index_list = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52]
        y_index_list = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53]

        centered_data = np.copy(data)
        for i in range(data.shape[1]):
            if i in x_index_list:
                centered_data[:, i]= centered_data[:, i] - center_x
            elif i in y_index_list:
                centered_data[:, i]= centered_data[:, i] - center_y
            else: continue
    return centered_data

def scale_otherpose(data):
    if data.shape[1]!=52: 
        print("the data does not have 52 collumns as it should")
        sys.exit()
    else:
        x_index_list = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52]
        y_index_list = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53]

        scaled_data = np.copy(data)
        for i in range(data.shape[1]):
            if i in x_index_list:
                scaled_data[:, i]= scaled_data[:, i] / np.nanstd(data[:, i])
            elif i in y_index_list:
                scaled_data[:, i]= scaled_data[:, i] / np.nanstd(data[:, i])
            else: continue
    return scaled_data

def cum_explained_variance(singular_values):
    sing = np.asarray(singular_values)
    explained_variance=np.zeros(sing.shape[0])
    sum = np.sum(sing)
    for i in range(sing.shape[0]):
        explained_variance[i]=sing[i] /sum  
    cum_variance=np.copy(explained_variance)
    for i in range(1, sing.shape[0], 1):
        cum_variance[i] = cum_variance[i] + cum_variance[i-1]
    return cum_variance

"""
def inverse_centering_and_scaling(data, center_x, center_y, stds_x, stds_y):
    new_data = np.copy(data)
    for i in range(data.shape[1]):
        if i%2==0:
            new_data[:, i]=new_data[:, i]*stds_x[i] + center_x
        if i%2==1:
            new_data[:, i]=new_data[:, i]*stds_y[i] + center_y
        
    return new_data
"""