"""

https://openclassrooms.com/en/courses/5869986-perform-an-exploratory-data-analysis/6177861-analyze-the-results-of-a-k-means-clustering 

"""


# Library of Functions for the OpenClassrooms Multivariate Exploratory Data Analysis Course
# Kudos: https://github.com/OpenClassrooms-Student-Center/Multivariate-Exploratory-Analysis/blob/master/functions.py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import copy
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
import seaborn as sns
from sklearn.cluster import KMeans, OPTICS, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA
# Import the sklearn function
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
#from collections import Iterable
import warnings
#from collections.abs import Iterable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline#, make_pipeline
import pygor.filehandling as filehandling
import pygor.strf.analyse 

palette = sns.color_palette("bright", 10)

def elbow_plot(df, max_it = 50):
    # Assuming df is your DataFrame
    # Specify the range of k values you want to try
    k_values = range(1, max_it)
    # Initialize an empty list to store the sum of squared distances (inertia) for each k
    inertia = []
    # Perform k-means clustering for each k value
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init = "auto")
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)
    # Create an elbow plot using Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(5, 4))
    sns.lineplot(x=k_values, y=inertia, marker='o')
    plt.title('Elbow Plot for K-Means Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances (Inertia)')
    plt.show()

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """Display correlation circles, one for each factorial plane"""

    # For each factorial plane
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # Initialise the matplotlib figure
            fig, ax = plt.subplots(figsize=(10,10))

            # Determine the limits of the chart
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # Add arrows
            # If there are more than 30 arrows, we do not display the triangle at the end
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (see the doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # Display variable names
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # Display circle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # Define the limits of the chart
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # Display grid lines
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Correlation Circle (PC{} and PC{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, pca, axis_ranks, labels = None, alpha=1, clust_labels=None, cmap = "viridis"):
    '''Display a scatter plot on a factorial plane, one for each factorial plane'''

    # For each factorial plane
    for d1,d2 in axis_ranks: 
            # Initialise the matplotlib figure      
            fig = plt.figure(figsize=(7,6))
            colormap = matplotlib.cm.get_cmap(cmap, len(np.unique(labels)) + 1)
            colormap = colormap(range(np.max(labels) + 1))
            # colormap = colormap(range(0,255))
            # Display the points
            if clust_labels is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                clust_labels = np.array(clust_labels)
                for n, value in enumerate(np.unique(clust_labels)):
                    selected = np.where(clust_labels == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, color = colormap[n], label=value)
                plt.legend()
            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
            # Display grid lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            plt.title("Projection of points (on PC{} and PC{})".format(d1+1, d2+1))
            #plt.show(block=False)
   
def display_scree_plot(pca):
    '''Display a scree plot for the pca'''

    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Number of principal components")
    plt.ylabel("Percentage explained variance")
    plt.title("Scree plot")
    plt.show(block=False)

def append_class(df, class_name, feature, thresholds, names):
    '''Append a new class feature named 'class_name' based on a threshold split of 'feature'.  Threshold values are in 'thresholds' and class names are in 'names'.'''
    
    n = pd.cut(df[feature], bins = thresholds, labels=names)
    df[class_name] = n

def plot_dendrogram(Z, names, figsize=(10,25)):
    '''Plot a dendrogram to illustrate hierarchical clustering'''

    plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    #plt.show()

def addAlpha(colour, alpha):
    '''Add an alpha to the RGB colour'''
    
    return (colour[0],colour[1],colour[2],alpha)

def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster==i])
    
    # Create the plot
    fig = plt.figure(figsize=(12, 15))
    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):    
        plt.subplot(num_clusters, 1, i+1)
        for j,c in enumerate(cluster_points): 
            if i!= j:
                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])
        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.5)])

        # Stagger the axes
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)        


def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)    




## Kills metadata
def cols_like(input_list, df):
    """
    TODO "curr_path" bug likely originates here, but I really dont understand why
    """
    final_list = []
    #Prevent overwriting
    df = df.copy()
    for term in input_list:
        curr_list = [i for i in df.columns if term in i]
        final_list.extend(curr_list)
    return final_list

def remove_nondata(input_df, to_drop = ['date', 'path', 'filename', 'curr_path', 'strf_keys', 'cell_id', 'size']):
    input_df = input_df.copy()
    actual_drop = [i for i in to_drop if i in input_df.columns]
    input_df = input_df.loc[(input_df!=0).any(axis=1)].drop(actual_drop, axis = 1)
    return input_df

def remove_nonnumeric(input_df, numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']):
    input_df = input_df.copy()
    input_df = input_df.select_dtypes(include=numerics)
    return input_df

def remove_missing_vals(input_df, ignore_cols = ["ipl_depths"], numerical_only = True, return_numerical_only = False):
    # Prevent overwriting 
    input_df = input_df.copy()
    # Numerical or not 
    if numerical_only == True:
        purge_df_numerical = remove_nonnumeric(input_df)
    # say which columns to ignore
    if ignore_cols != None:
        purge_df = purge_df_numerical.drop([i for i in ignore_cols if i in input_df.columns], axis = 1) # the list comprehension
        #just makes sure that if columns are not present the script does not error out:) 
    else:
        purge_df = input_df.copy()
    col = list(purge_df.columns)[:8]
    drop_list = purge_df.index[purge_df[col].eq(0).all(axis=1)].to_list()
    if return_numerical_only == True:
        non_empty_df = purge_df.drop(drop_list, axis = 0)
        # Remember to add back in the ignored columns so we get ones that were numerical
        non_cleaned_input_drop = input_df.drop(drop_list, axis = 0)
        # For each location
        order = list(purge_df.columns)
        for ignored_col, i in zip(ignore_cols, [purge_df_numerical.columns.get_loc(i) for i in ["ipl_depths"]]):
            order.insert(i, ignored_col)
        for fetch_col in order:
            non_empty_df[fetch_col] = non_cleaned_input_drop[fetch_col]
        non_empty_df = non_empty_df[order]
    else:
        non_empty_df = input_df.drop(drop_list, axis = 0)
    return non_empty_df

def prep_input_df(input_df, select_cols = None, scaler = StandardScaler(), nan_replace = 0, remove_missing = True, drop_redundant = True, ignore_scale = None):
    # Can specify columns, otherwise use everything available
    if select_cols == None:
        select_cols = list(input_df.columns)
        if "curr_path" in select_cols:
            select_cols.remove("curr_path")
        """
        TODO Fix the above, its a temporary solution and I do not understand where the problem comes from 
        Basically it solves a bug where "curr_path" within the input_df seemingly gets duplicated recurisvely,
        making it impossible to acces becasue the shape of df["curr_path"] goes from being (n, ) to (n, 2)
        """
    # Prevent overwriting 
    input_df = input_df.copy()
    if "roi" in input_df.columns:
        input_df.drop("roi", axis = 1)
    # Drop irrelevant cols (lazy by searching any match, meaning more
    # specific search criteria will yield less data)
    input_df = input_df[cols_like(select_cols, input_df)]
    # Remove missing values (but crucially keep index the same)
    if remove_missing == True: # need a better way of automatiing the below
        input_df = remove_missing_vals(input_df)
    # Kill nans 
    input_df = input_df.fillna(nan_replace)
    # Drop columns where all is the same 
    # if drop_redundant == True:
    #     nunique = input_df.nunique()
    #     cols_to_drop = nunique[nunique == 1].index
    #     input_df = input_df.drop(cols_to_drop, axis=1)
    # Can scale, otherwise just return the cleaned DF
    if scaler != None:
        # If specified, ignore these columns in the scaling process
        if ignore_scale != None:
            if isinstance(ignore_scale, Iterable) is False:  # noqa: F821
                raise AttributeError("param 'ignore_scale' must be iterable")
            # Determine which are actually present 
            ignore_scale = [i for i in ignore_scale if i in input_df.columns]
            # Extract temporarily the parts of DF that are to be ignored (for inserting later)
            temp_extract_cols = input_df[ignore_scale].reset_index(drop = True)
            # For user-friendliness lets keep track of the indeces and insert them back in their original order
            ignored_indeces = [input_df.columns.get_loc(c) for c in ignore_scale if c in input_df]
            input_df = input_df.drop(ignore_scale, axis = 1)
        try:
            input_df = pd.DataFrame(scaler.fit_transform(input_df), columns = input_df.columns)
        except TypeError:
            warnings.warn("TypeError excepted in scaling")
            numerics = ['int', 'float', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            input_df = input_df.select_dtypes(include=numerics)
            input_df = pd.DataFrame(scaler.fit_transform(input_df), columns = input_df.columns)
        if ignore_scale != None: # add ignored cols back in their original order:)
            for column_index, column_name in zip(ignored_indeces, ignore_scale):
                input_df.insert(column_index, column_name, temp_extract_cols[column_name].values)# = temp_extract_cols
    return input_df

def elbow_plot(df, max_it = 50):
    # Assuming df is your DataFrame
    # Specify the range of k values you want to try
    k_values = range(1, max_it)
    # Initialize an empty list to store the sum of squared distances (inertia) for each k
    inertia = []
    # Perform k-means clustering for each k value
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init = "auto")
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)
    # Create an elbow plot using Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(5, 4))
    sns.lineplot(x=k_values, y=inertia, marker='o')
    plt.title('Elbow Plot for K-Means Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances (Inertia)')
    plt.show()

def df_kmeans(input_df, n_clusters = 15, random_state=0, n_init="auto", apply_cluster_id = True):
    input_df = input_df.copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    kmeans.fit(input_df)
    clusters = kmeans.predict(input_df)
    output_df = pd.DataFrame(input_df, index=input_df.index, columns=input_df.columns)
    output_df["cluster_id"] = clusters
    # output_df["curr_path"] = load_df["curr_path"]
    # if apply_cluster_id == True:
    #     input_df["cluster_id"] = clusters
    return output_df

def df_BayesGMM(input_df, n_components=30, random_state=0, max_iter = 1000, covariance_type="full", apply_cluster_id = True):
    input_df = input_df.copy()
    gm = BayesianGaussianMixture(n_components=n_components, random_state=random_state, 
        max_iter = max_iter, covariance_type=covariance_type).fit(input_df)
    clusters = gm.predict(input_df)
    # output_df["cluster_id"] = clusters
    output_df = pd.DataFrame(input_df, index=input_df.index, columns=input_df.columns)
    output_df["cluster_id"] = clusters
    # if apply_cluster_id == True:
    #     input_df["cluster_id"] = clusters
    return output_df

def df_GMM(input_df, n_components="auto", random_state=200, max_iter = 25000, max_comp = 8, covariance_type="full", apply_cluster_id = True):
    np.random.seed(random_state)
    input_df = input_df.copy()
    if n_components == "auto":
        # If there are more components than samples, we need an exception 
        if input_df.shape[-1] < max_comp:
            max_comp = input_df.shape[-1] + 1
        n_components = range(1, max_comp)
        covariance_type = ['spherical', 'tied', 'diag', 'full']
        score=[]
        for cov in covariance_type:
            for n_comp in n_components:
                gmm=GaussianMixture(n_components=n_comp,covariance_type=cov)
                gmm.fit(input_df)
                score.append((cov,n_comp ,gmm.aic(input_df)))
        score = np.array(score)
        max_index_vals = score[np.where(score[:, 2].astype("float") == np.min(score[:, 2].astype("float")))][0]
        # print(max_index_vals)
        covariance_type = max_index_vals[0]
        n_components = max_index_vals[1].astype("int")
        smallest_bic = max_index_vals[2]
        print("Automatic number determined as AIC =",smallest_bic, "landing on covariance_type =", covariance_type, "with ", n_components, "n_components")
    gm = GaussianMixture(n_components=n_components, random_state=random_state, 
        max_iter = max_iter, covariance_type="full").fit(input_df)
    clusters = gm.predict(input_df)
    # output_df["cluster_id"] = clusters
    output_df = pd.DataFrame(input_df, index=input_df.index, columns=input_df.columns)
    output_df["cluster_id"] = clusters
    # if apply_cluster_id == True:
    #     input_df["cluster_id"] = clusters
    return output_df

def df_AggHierarchy(input_df, n_components=None, dist_thresh = 12.5, apply_cluster_id = True):
    input_df = input_df.copy()
    # print("distnaces:", distances_)s
    ag_model = AgglomerativeClustering(n_clusters = None, distance_threshold=dist_thresh, compute_distances = True).fit(input_df)
    # print(np.sum(ag_model.distances_))
    input_df["cluster_id"] = ag_model.labels_
    # if apply_cluster_id == True:
    #     input_df["cluster_id"] = clusters
    return input_df    

def apply_clusters(input_df, cluster_id_df, inplace = False):
    if cluster_id_df.shape[0] != input_df.shape[0]:
        raise AttributeError("Index mismatch")
    if "cluster_id" not in list(cluster_id_df.columns):
        raise AttributeError("Column ['cluster_id'] missing from clutser_id_df")
    if inplace == False:
        input_df = input_df.copy()
        input_df["cluster_id"] = cluster_id_df["cluster_id"].values
        return input_df
    else:
        input_df["cluster_id"] = cluster_id_df["cluster_id"].values

def df_pca(input_df, n_comps = "auto", whiten = False):
    if n_comps == "auto":
        # Maximaise availible dimesnions always
        n_samples = input_df.shape[0]
        n_measures = input_df.shape[-1]
        if n_samples > n_measures:
            n_comps = n_measures
        else:
            n_comps = n_samples
    pca = PCA(n_components=n_comps, whiten = whiten)
    pca.fit(input_df)
    reduced = pca.transform(input_df)
    # print(df_pca(pruned_df).to_numpy() == reduced)
    return pca, pd.DataFrame(reduced, index = input_df.index, columns = [f"PC{i}" for i in range(len(pca.explained_variance_))])

def get_cluster_cols(cluster_id, col_str_list, df):
    search_cols = cols_like(col_str_list, df) + ["cluster"]
    df_query = df[search_cols].query(f"cluster == {cluster_id}")
    return df_query.drop("cluster", axis = 1)

# def plot_df_stats(describe_df, print_stat = False):
#     means =  describe_df.loc["mean"]
#     errors = describe_df.loc["std"]
#     if print_stat == True:
#         print(results)
#     plt.plot(means)   |
#     plt.title("Mean ± STD by paramter")
#     plt.fill_between(range(len(errors)), means - errors, means + errors, alpha = 0.25)
#     plt.axhline(0)

def sum_normalize_data(data):
    """
    scaling values to sum to 1 while preserving their relative proportions
    """
    data_array = np.array(data)
    total = np.sum(data_array)
    normalized_data = data_array / total
    return normalized_data.tolist()

def sum_div_norm(data):
    # Normalize the data so that the sum equals 1
    normalized_data = data / np.sum(data)
    # The sum of normalized values will be 1
    return normalized_data
    

def scale_data_points(data_array):
    # Calculate the area under the curve using the trapezoidal rule with absolute values for each row
    areas = np.trapz(np.abs(data_array), axis=1)

    # Scale the data points row-wise to make the area under the curve equal to 1 for each row
    scaled_data_array = data_array / areas[:, np.newaxis]

    return scaled_data_array

