import sys
import matplotlib.pyplot as plt
#from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
#from sklearn.decomposition import PCA
# Import the sklearn function
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler #StandardScaler, RobustScaler, 
from pygor.strf.clustering.clustering_funcs import cols_like, df_GMM, apply_clusters, prep_input_df, df_pca
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline#, make_pipeline
import pandas as pd
import pygor.filehandling
import pygor.strf
import copy

clustering_params = ["ampl", "area", "peak", "cent"]
clust_params_regex = '|'.join(clustering_params)

"""
TODO Break the following into functions and make 1 functions that calls all:
"""
def run_clustering(clust_df):
    def clust_pipeline(input_df, cluster_params = clustering_params, *args):
        # Run clustering on PCA result
        _df_clust = df_GMM(input_df)
        # Apply output by writing clusters to input data
        df_output = apply_clusters(input_df.copy(), _df_clust)
        return df_output
    ## Step 0: Remove all-zero enteries from data
    pruned_df = prep_input_df(clust_df, scaler = None).reset_index()
    ## Step 1: Scale data
    # Initialise transfomers 
    standard_maxabs_transformer  = Pipeline(steps=[("maxabs", MaxAbsScaler())])
    standard_minmax_transformer  = Pipeline(steps=[("minmax", MinMaxScaler())])
    sparse_transformer  = Pipeline(steps=[('maxsabs', MaxAbsScaler()),])
    # Initialise preprocessor job
    """
    NB! Here the order is VERY IMPORTANT! --> Need to automate
    
    Issue is that depending on the clusterimg_params, the preprocessor column
    transformer *should* update... At least for ease of use...
    """
    preprocessor = ColumnTransformer(
            remainder='drop', #passthough features not listed
            transformers=[
                ('ampl',   standard_maxabs_transformer, cols_like(["ampl"], pruned_df)),
                ('area', standard_minmax_transformer , cols_like(["area"], pruned_df)),
                ("peak, cent", sparse_transformer, cols_like(["peak", "cent"], pruned_df)),
            ])
    # Fit transform and write result to DataFrame 
    output_df = pd.DataFrame(preprocessor.fit_transform(pruned_df.copy()), columns=cols_like(clustering_params, pruned_df))
    # output_df = pruned_df[]
    # Check that we are happy with outcome
    pruned_df.filter(regex= clust_params_regex).plot(kind = "box", figsize = (10, 3))
    # plt.ylim(-15, 15)
    plt.axhline(0, c = "red", alpha = 0.25)
    plt.xticks(rotation=90);
    output_df.plot(kind = "box", figsize = (10, 3))
    for j in [0, -1, 1]:
        plt.axhline(j, c = "red", alpha = 0.25)
    plt.xticks(rotation=90);

    ## Step 1: Split the data into groups depending on cat_pol (can be optional step)
    # Add cat_pol and wavelength pol back into output for filtering (reset_index is crucial here for correct merging of indeces)
    split_df = pygor.strf.analyse.split_df_by(output_df.join(pruned_df.filter(like = "pol"), how = "inner", validate = "one_to_one"), "cat_pol")
    ## Step 2: Apply PCA to each split independently
    pca_results = {key : df_pca(split_df[key].filter(regex=clust_params_regex), whiten = False) for key in split_df.keys()}
    pca_dict = {key : pca_results[key][0] for key in split_df.keys()}
    pca_df_dict = {key : pca_results[key][1] for key in split_df.keys()}
    # ## Step 3: Apply clustering to each split
    clust_dict = {i : clust_pipeline(pca_df_dict[i].filter(like = "PC")) for i in split_df.keys() if i != "empty"}
    ## Ste 3.5: Rename cluster IDs to prep for merging
    for i in clust_dict.keys():
        if clust_dict[i]["cluster_id"].dtype != "object":
            split_df[i]["cluster_id"] = f'{i}' + '_' + clust_dict[i]["cluster_id"].astype(str)
            clust_dict[i]["cluster_id"] = f'{i}' + '_' + clust_dict[i]["cluster_id"].astype(str)
    ## Step 4: Merge results
    ### To PCA DF
    merged_pca_df = pd.concat(clust_dict[i] for i in clust_dict.keys())
    merged_pca_df["cluster_id"] = merged_pca_df["cluster_id"].astype('category')
    merged_pca_df["cluster"] = merged_pca_df.cluster_id.cat.codes
    merged_pca_df["cat_pol"] = pruned_df["cat_pol"]
    # merged_pca_df["pca_obj"] = pca_results

    ### To scaled DF
    merged_stats_df = pd.concat(split_df[i] for i in clust_dict.keys())
    merged_stats_df = merged_stats_df[merged_stats_df.columns.drop(list(merged_stats_df.filter(like='pol_')))]
    merged_stats_df["ipl_depths"] = pruned_df["ipl_depths"]
    merged_stats_df["cluster_id"] = merged_pca_df["cluster_id"].astype('category')
    merged_stats_df["cluster"] = merged_pca_df.cluster_id.cat.codes
    ### And write that to an unprocessed copy
    org_stats_df = copy.copy(clust_df)
    org_stats_df = prep_input_df(org_stats_df, scaler=None, select_cols=clustering_params).reset_index()
    org_stats_df["cluster_id"] = merged_pca_df["cluster_id"].astype('category')
    org_stats_df["cluster"] = merged_pca_df.cluster_id.cat.codes
    org_stats_df["cat_pol"] = pruned_df["cat_pol"]
    org_stats_df["ipl_depths"] = pruned_df["ipl_depths"]
    org_stats_df["roi"] = pruned_df["roi"]
    org_stats_df["strf_obj"] = pruned_df["strf_obj"]
    ## Finally, define outputs 
    return merged_pca_df, merged_stats_df, org_stats_df, pca_dict

# if __name__ == "__main__":
#     data_argument = globals().get(sys.argv[1])
#     print(type(data_argument))
#     run_clustering(data_argument)


# script.py
# import pandas as pd
# import sys

# def main(data):
#     # Your script logic using the DataFrame 'data'
#     print(data.head())

# if __name__ == "__main__":
#     # Check if the script is being run directly
#     if len(sys.argv) > 1:
#         # Retrieve the DataFrame from the global namespace
#         data_argument = globals().get(sys.argv[1], None)

#         if isinstance(data_argument, pd.DataFrame):
#             # Call the main function with the DataFrame as an argument
#             main(data_argument)
#         else:
#             print("Error: Invalid DataFrame argument.")
#     else:
#         print("Error: Please provide a DataFrame argument.")

