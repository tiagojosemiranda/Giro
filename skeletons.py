import sys
import numpy as np
import h2o
from h2o.estimators import H2OGeneralizedLowRankEstimator
import skeletons_utils
from sklearn.decomposition import TruncatedSVD
from collections import Counter

def value_imputation(data):
    input = skeletons_utils.missing_to_nan(data)

    if data.shape[1]==55:
        print("Openpose skeleton data")
        #center and scale the data
        centered_data = skeletons_utils.center_openpose(input)
        scaled_data = skeletons_utils.scale_openpose(centered_data)
        #get only coordinates 
        coordinates_index_list = [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 25,26, 28,29, 31,32, 34,35, 37,38, 40,41, 43,44, 46,47, 49,50, 52,53]
    elif data.shape[1]==52:
        print("Otherpose skeleton data")
        centered_data = skeletons_utils.center_otherpose(input)
        scaled_data = skeletons_utils.scale_otherpose(centered_data)

        #get only coordinates 
        coordinates_index_list = [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 25,26, 28,29, 31,32, 34,35, 37,38, 40,41, 43,44, 46,47, 49,50]

    else:
        print("Data does not have 52 or 55 collumns as it should")
        sys.exit()

    output = scaled_data[:, coordinates_index_list]
    print("data shape after scaling", output.shape)

    output = input
    imputed_data = GLRM_imputation(output)

    return imputed_data


def GLRM_imputation(input):
    # Initialize and connect to H2O
    h2o.init()
    # Load your dataset into H2O
    data = h2o.H2OFrame(input)
    
    # Split the dataset into training and testing sets
    train, test = data.split_frame(ratios=[0.8], seed=1234)
    # Identify the column indices with missing values
    missing_cols = [col for col in data.columns if data[col].isna().any()]
    print("Missing colllumns shape", len(missing_cols))
    print("Non-missing colllumns", list(set(data.columns) - set(missing_cols)))

    # Define and train the GLRM model to impute missing values
    glrm_model = H2OGeneralizedLowRankEstimator(k=20,
                                            loss="Quadratic",
                                            regularization_x="L1",
                                            regularization_y="L1",
                                            max_iterations=100, 
                                            recover_svd=True)
    glrm_model.train(training_frame=train, x=missing_cols)
    # glrm_model._model_json['output']
    # glrm_model._model_json['output']['singular_vals']
    # print(skeletons_utils.cum_explained_variance(glrm_model._model_json['output']['singular_vals']))
    # glrm_model._model_json['output']['eigenvectors']
    # glrm_model._model_json['output']['archetypes']

    # Impute missing values in the original dataset
    imputed_data = glrm_model.predict(data)

    return imputed_data.as_data_frame().to_numpy()

def remove_outliers(data, rank, u, s, v):
    residuals = data - u[:, :rank] @ np.diag(s[:rank])  @ v[:, :rank].T
    norms = np.linalg.norm(residuals, axis=1)
    indexes = np.where(norms> np.average(norms) + 3*np.std(norms))[0]
    
    a = data[norms <= np.average(norms) + 3*np.std(norms)]
    print('Percentage of removed outliers = ', np.round(((data.shape[0]-a.shape[0])/data.shape[0] *100), 3) , '%')
    return a, indexes

def readd_index(orig_data, outliers_idx,  compressed_data, giroslow_flag):
    idx = np.delete(orig_data[:,0], outliers_idx)
    if giroslow_flag==True: idx += 1 #syncronize frames with skeleton data for giroslow dataset
    return np.insert(compressed_data, 0, idx, axis=1)

def reduce_dimensions(data, rank):
    svd = TruncatedSVD(n_components=rank)
    data_reduced = svd.fit_transform(data)
    #print(svd.explained_variance_ratio_)
    #print(svd.explained_variance_ratio_.sum())
    #print(svd.singular_values_)
    return data_reduced

def rank_detection(s, cumulative_percentage=0.9):
    cumulative_variance = skeletons_utils.cum_explained_variance(s)
    rank = np.argmax(cumulative_variance >= cumulative_percentage) + 1 #numpy return the index, since in python it starts at 0 we add 1 
    return rank

def number_skeletons_per_frame(data): #counting how many skeletons per frame
    frame_numbers = data[:,0].astype(int)
    counts = Counter(frame_numbers)

    return np.array(list(counts.keys())), np.array(list(counts.values()))
