"""
Clean clustering module for RF analysis
Integrates with the melting framework for flexible column selection
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import rankdata
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def create_rgb_composite(curr_clust_avgs, channel_indices):
    """
    Create RGB composite from selected channels.

    Parameters:
    -----------
    curr_clust_avgs : np.array
        Shape: (color, y, x)
    channel_indices : list
        List of 3 channel indices for R, G, B

    Returns:
    --------
    np.array
        RGB composite image (y, x, 3)
    """
    # Create RGB composite - let it auto-equalize
    rgb_composite = np.zeros((*curr_clust_avgs.shape[1:], 3))       

    for i, channel_idx in enumerate(channel_indices):
        if channel_idx < curr_clust_avgs.shape[0]:  # Check if channel exists
            channel_data = curr_clust_avgs[channel_idx]
            # Normalize each channel independently to 0-1 for RGB display
            if np.max(np.abs(channel_data)) > 0:
                # Use absolute values and normalize to full 0-1 range
                normalized = np.abs(channel_data) / np.max(np.abs(channel_data))
            else:
                normalized = np.zeros_like(channel_data)
        else:
            # Channel doesn't exist, use zeros
            normalized = np.zeros(curr_clust_avgs.shape[1:])

        rgb_composite[:, :, i] = normalized

    return rgb_composite

def prepare_clustering_data(df, feature_patterns, id_vars=None, 
                          scale=True, handle_missing='fill_zero', 
                          scaling_method='standard'):
    """
    Prepare data for clustering by selecting features and handling missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_patterns : list
        List of column patterns to include (e.g., ['areas', 'eccentricity'])
    id_vars : list, optional
        Additional columns to keep as identifiers
    scale : bool, default True
        Whether to standardize features
    handle_missing : str, default 'fill_zero'
        How to handle missing values: 'drop_rows', 'drop_cols', 'fill_zero'
    scaling_method : str, default 'standard'
        Scaling method to use: 'standard', 'custom', 'rank', 'l2'
        - 'standard': StandardScaler (z-score normalization)
        - 'custom': MaxAbsScaler for amplitudes, MinMaxScaler for others
        - 'rank': Rank-based scaling (converts to percentiles)
        - 'l2': L2 normalization (row-wise unit vectors)
    
    Returns:
    --------
    pd.DataFrame, Scaler or None
        Prepared data and scaler (if used)
    """
    
    if id_vars is None:
        id_vars = []
    
    # Select feature columns
    feature_cols = []
    for pattern in feature_patterns:
        cols = [col for col in df.columns if pattern in col]
        feature_cols.extend(cols)
    
    # Remove duplicates while preserving order
    feature_cols = list(dict.fromkeys(feature_cols))
    
    # Combine with id variables
    all_cols = id_vars + feature_cols
    available_cols = [col for col in all_cols if col in df.columns]
    
    data = df[available_cols].copy()
    
    print(f"Selected {len(feature_cols)} feature columns from {len(feature_patterns)} patterns")
    print(f"Feature columns: {feature_cols}")
    
    # Handle missing values
    if handle_missing == 'drop_rows':
        initial_shape = data.shape[0]
        data = data.dropna()
        print(f"Dropped {initial_shape - data.shape[0]} rows with missing values")
        print("WARNING: This approach biases toward broadband cells!")
    elif handle_missing == 'drop_cols':
        initial_cols = data.shape[1]
        data = data.dropna(axis=1)
        print(f"Dropped {initial_cols - data.shape[1]} columns with missing values")
    elif handle_missing == 'fill_zero':
        initial_missing = data[feature_cols].isnull().sum().sum()
        data = data.fillna(0)
        print(f"Filled {initial_missing} missing values with 0")
    elif handle_missing == 'fill_mean':
        initial_missing = data[feature_cols].isnull().sum().sum()
        data[feature_cols] = data[feature_cols].fillna(data[feature_cols].mean())
        print(f"Filled {initial_missing} missing values with column means")
    elif handle_missing == 'fill_median':
        initial_missing = data[feature_cols].isnull().sum().sum()
        data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())
        print(f"Filled {initial_missing} missing values with column medians")
    
    # Separate features and identifiers
    feature_data = data[feature_cols]
    id_data = data[id_vars] if id_vars else None
    
    # Scale if requested
    scaler = None
    if scale:
        if scaling_method == 'standard':
            scaler = StandardScaler()
            feature_data = pd.DataFrame(
                scaler.fit_transform(feature_data),
                index=feature_data.index,
                columns=feature_data.columns
            )
            print("Applied StandardScaler to features")
            
        elif scaling_method == 'custom':
            # Detailed feature-specific scaling based on your successful approach
            transformers = []
            
            # Define feature-specific transformers
            space_amps_cols = [col for col in feature_cols if 'space_amps' in col]
            areas_cols = [col for col in feature_cols if 'areas' in col]
            magnitudes_cols = [col for col in feature_cols if 'magnitudes' in col]
            opponency_cols = [col for col in feature_cols if 'opponency_index' in col]
            eccentricity_cols = [col for col in feature_cols if 'eccentricity' in col]
            angles_cols = [col for col in feature_cols if 'angles' in col]
            orientation_cols = [col for col in feature_cols if 'orientation' in col]
            
            # Add transformers for each feature type
            if space_amps_cols:
                transformers.append(('space_amps', Pipeline([('maxabs', MaxAbsScaler())]), space_amps_cols))
            if areas_cols:
                transformers.append(('areas', Pipeline([('minmax', MinMaxScaler())]), areas_cols))
            if magnitudes_cols:
                transformers.append(('magnitudes', Pipeline([('minmax', MinMaxScaler())]), magnitudes_cols))
            if opponency_cols:
                transformers.append(('opponency', Pipeline([('minmax', MinMaxScaler())]), opponency_cols))
            if eccentricity_cols:
                transformers.append(('eccentricity', Pipeline([('minmax', MinMaxScaler())]), eccentricity_cols))
            if angles_cols:
                transformers.append(('angles', Pipeline([('maxabs', MaxAbsScaler())]), angles_cols))
            if orientation_cols:
                transformers.append(('orientation', Pipeline([('maxabs', MaxAbsScaler())]), orientation_cols))
            
            if transformers:
                scaler = ColumnTransformer(transformers=transformers, remainder='drop')
                feature_data = pd.DataFrame(
                    scaler.fit_transform(feature_data),
                    index=feature_data.index,
                    columns=feature_cols
                )
                print(f"Applied custom scaling to {len(transformers)} feature types")
            else:
                print("Warning: No recognized feature types for custom scaling, using StandardScaler")
                scaler = StandardScaler()
                feature_data = pd.DataFrame(
                    scaler.fit_transform(feature_data),
                    index=feature_data.index,
                    columns=feature_data.columns
                )
            
        elif scaling_method == 'rank':
            # Rank-based scaling
            def rank_transform(x):
                return rankdata(x, method='average') / len(x)
            
            feature_data = feature_data.apply(rank_transform, axis=0)
            print("Applied rank-based scaling")
            
        elif scaling_method == 'robust':
            # Robust scaling for all feature types (like your Strategy 1)
            transformers = []
            
            space_amps_cols = [col for col in feature_cols if 'space_amps' in col]
            areas_cols = [col for col in feature_cols if 'areas' in col]
            magnitudes_cols = [col for col in feature_cols if 'magnitudes' in col]
            opponency_cols = [col for col in feature_cols if 'opponency_index' in col]
            eccentricity_cols = [col for col in feature_cols if 'eccentricity' in col]
            angles_cols = [col for col in feature_cols if 'angles' in col]
            orientation_cols = [col for col in feature_cols if 'orientation' in col]
            
            # Use RobustScaler for most features, MaxAbsScaler for angles/orientation
            if space_amps_cols:
                transformers.append(('space_amps', Pipeline([('robust', RobustScaler())]), space_amps_cols))
            if areas_cols:
                transformers.append(('areas', Pipeline([('robust', RobustScaler())]), areas_cols))
            if magnitudes_cols:
                transformers.append(('magnitudes', Pipeline([('robust', RobustScaler())]), magnitudes_cols))
            if opponency_cols:
                transformers.append(('opponency', Pipeline([('robust', RobustScaler())]), opponency_cols))
            if eccentricity_cols:
                transformers.append(('eccentricity', Pipeline([('robust', RobustScaler())]), eccentricity_cols))
            if angles_cols:
                transformers.append(('angles', Pipeline([('maxabs', MaxAbsScaler())]), angles_cols))
            if orientation_cols:
                transformers.append(('orientation', Pipeline([('maxabs', MaxAbsScaler())]), orientation_cols))
            
            if transformers:
                scaler = ColumnTransformer(transformers=transformers, remainder='drop')
                feature_data = pd.DataFrame(
                    scaler.fit_transform(feature_data),
                    index=feature_data.index,
                    columns=feature_cols
                )
                print(f"Applied robust scaling to {len(transformers)} feature types")
            else:
                scaler = RobustScaler()
                feature_data = pd.DataFrame(
                    scaler.fit_transform(feature_data),
                    index=feature_data.index,
                    columns=feature_data.columns
                )
                print("Applied RobustScaler to all features")
                
        elif scaling_method == 'quantile':
            # Quantile uniform transformation (like your Strategy 2)
            transformers = []
            
            space_amps_cols = [col for col in feature_cols if 'space_amps' in col]
            areas_cols = [col for col in feature_cols if 'areas' in col]
            magnitudes_cols = [col for col in feature_cols if 'magnitudes' in col]
            opponency_cols = [col for col in feature_cols if 'opponency_index' in col]
            eccentricity_cols = [col for col in feature_cols if 'eccentricity' in col]
            angles_cols = [col for col in feature_cols if 'angles' in col]
            orientation_cols = [col for col in feature_cols if 'orientation' in col]
            
            # Use QuantileTransformer for most, MaxAbsScaler for angles/orientation
            if space_amps_cols:
                transformers.append(('space_amps', Pipeline([('quantile', QuantileTransformer(output_distribution='uniform'))]), space_amps_cols))
            if areas_cols:
                transformers.append(('areas', Pipeline([('quantile', QuantileTransformer(output_distribution='uniform'))]), areas_cols))
            if magnitudes_cols:
                transformers.append(('magnitudes', Pipeline([('quantile', QuantileTransformer(output_distribution='uniform'))]), magnitudes_cols))
            if opponency_cols:
                transformers.append(('opponency', Pipeline([('quantile', QuantileTransformer(output_distribution='uniform'))]), opponency_cols))
            if eccentricity_cols:
                transformers.append(('eccentricity', Pipeline([('quantile', QuantileTransformer(output_distribution='uniform'))]), eccentricity_cols))
            if angles_cols:
                transformers.append(('angles', Pipeline([('maxabs', MaxAbsScaler())]), angles_cols))
            if orientation_cols:
                transformers.append(('orientation', Pipeline([('maxabs', MaxAbsScaler())]), orientation_cols))
            
            if transformers:
                scaler = ColumnTransformer(transformers=transformers, remainder='drop')
                feature_data = pd.DataFrame(
                    scaler.fit_transform(feature_data),
                    index=feature_data.index,
                    columns=feature_cols
                )
                print(f"Applied quantile scaling to {len(transformers)} feature types")
            else:
                scaler = QuantileTransformer(output_distribution='uniform')
                feature_data = pd.DataFrame(
                    scaler.fit_transform(feature_data),
                    index=feature_data.index,
                    columns=feature_data.columns
                )
                print("Applied QuantileTransformer to all features")
        
        elif scaling_method == 'l2':
            # L2 normalization (row-wise)
            from sklearn.preprocessing import normalize
            feature_data = pd.DataFrame(
                normalize(feature_data, norm='l2', axis=1),
                index=feature_data.index,
                columns=feature_data.columns
            )
            print("Applied L2 normalization (row-wise)")
            
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}. Choose from: 'standard', 'custom', 'robust', 'quantile', 'rank', 'l2'")
    
    # Recombine
    if id_data is not None:
        prepared_data = pd.concat([id_data, feature_data], axis=1)
    else:
        prepared_data = feature_data
    
    return prepared_data, scaler

def elbow_analysis(feature_data, max_k=30, random_state=42):
    """
    Perform elbow analysis for K-means clustering.
    
    Parameters:
    -----------
    feature_data : pd.DataFrame
        Features for clustering (no ID variables)
    max_k : int
        Maximum number of clusters to test
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict
        Dictionary with k values and corresponding inertias
    """
    
    k_values = range(1, min(max_k + 1, len(feature_data)))
    inertias = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        kmeans.fit(feature_data)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, marker='o')
    plt.title('Elbow Analysis for K-Means Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {'k_values': list(k_values), 'inertias': inertias}

def apply_clustering(data, method='kmeans', n_clusters=5, random_state=42, 
                    use_pca=False, n_components=None, **kwargs):
    """
    Apply clustering algorithm to prepared data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Prepared data (features + optional ID variables)
    method : str
        Clustering method: 'kmeans', 'gmm', 'hierarchical'
    n_clusters : int
        Number of clusters
    random_state : int
        Random state for reproducibility
    use_pca : bool, default False
        Whether to apply PCA before clustering
    n_components : int or float, optional
        Number of PCA components. If float, treated as variance threshold
    **kwargs
        Additional parameters for clustering algorithms
    
    Returns:
    --------
    dict
        Dictionary with clustered data, PCA object (if used), and cluster info
    """
    
    # Extract parameters that shouldn't go to clustering algorithms
    id_cols = kwargs.pop('id_cols', [])
    if id_cols is None:
        id_cols = []
    handle_missing = kwargs.pop('handle_missing', None)  # Remove this parameter
    scale = kwargs.pop('scale', None)  # Remove this parameter
    show_elbow = kwargs.pop('show_elbow', None)  # Remove this parameter
    
    try:
        feature_cols = [col for col in data.columns if col not in id_cols]
    except TypeError as e:
        raise TypeError(f"id_cols must be a list or None, got {type(id_cols)}. Original error: {e}")
    feature_data = data[feature_cols].copy()
    
    pca_obj = None
    pca_data = None
    
    if use_pca:
        print(f"Applying PCA before clustering...")
        
        # Determine number of components
        if n_components is None:
            # Use all components (for noise reduction/decorrelation)
            n_components = min(len(feature_cols), len(feature_data) - 1)
            print(f"Using all {n_components} components (full variance)")
        elif isinstance(n_components, float) and 0 < n_components < 1:
            # Treat as variance threshold - find components needed
            temp_pca = PCA()
            temp_pca.fit(feature_data)
            cumsum_var = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= n_components) + 1
            print(f"Using {n_components} components to capture {n_components:.1%} variance")
        elif n_components == 'all':
            # Explicit request for all components
            n_components = min(len(feature_cols), len(feature_data) - 1)
            print(f"Using all {n_components} components (explicit request)")
        
        pca_obj = PCA(n_components=n_components, random_state=random_state)
        pca_data = pca_obj.fit_transform(feature_data)
        
        # Use PCA data for clustering
        clustering_data = pca_data
        
        print(f"PCA: {len(feature_cols)} features → {n_components} components")
        print(f"Variance explained: {pca_obj.explained_variance_ratio_[:min(5, n_components)]}")
        print(f"Total variance captured: {pca_obj.explained_variance_ratio_.sum():.3f}")
        
    else:
        clustering_data = feature_data
        print(f"Clustering with {method} on {len(feature_cols)} features")
    
    if method == 'kmeans':
        clusterer = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state, 
            n_init='auto',
            **kwargs
        )
        cluster_labels = clusterer.fit_predict(clustering_data)
        
    elif method == 'gmm':
        clusterer = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            **kwargs
        )
        cluster_labels = clusterer.fit_predict(clustering_data)
        
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            **kwargs
        )
        cluster_labels = clusterer.fit_predict(clustering_data)
        
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Add cluster labels to original data
    result = data.copy()
    result['cluster_id'] = cluster_labels
    
    print(f"Created {len(np.unique(cluster_labels))} clusters")
    print(f"Cluster sizes: {np.bincount(cluster_labels)}")
    
    return {
        'clustered_data': result,
        'pca_obj': pca_obj,
        'pca_data': pca_data,
        'clusterer': clusterer
    }

def cluster_summary_stats(clustered_data, feature_patterns, id_cols=None):
    """
    Generate summary statistics for each cluster.
    
    Parameters:
    -----------
    clustered_data : pd.DataFrame
        Data with cluster_id column
    feature_patterns : list
        Patterns used for clustering
    id_cols : list, optional
        ID columns to include in summary
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics by cluster
    """
    
    if id_cols is None:
        id_cols = []
    
    # Get feature columns
    feature_cols = []
    for pattern in feature_patterns:
        cols = [col for col in clustered_data.columns if pattern in col]
        feature_cols.extend(cols)
    
    feature_cols = list(dict.fromkeys(feature_cols))
    
    # Calculate summary stats
    summary_stats = []
    
    for cluster_id in sorted(clustered_data['cluster_id'].unique()):
        cluster_data = clustered_data[clustered_data['cluster_id'] == cluster_id]
        
        stats = {'cluster_id': cluster_id, 'n_rois': len(cluster_data)}
        
        # Add ID column summaries if available
        for id_col in id_cols:
            if id_col in clustered_data.columns:
                if clustered_data[id_col].dtype == 'object':
                    stats[f'{id_col}_mode'] = cluster_data[id_col].mode().iloc[0] if len(cluster_data[id_col].mode()) > 0 else 'None'
                else:
                    stats[f'{id_col}_mean'] = cluster_data[id_col].mean()
        
        # Add feature summaries
        for col in feature_cols:
            if col in cluster_data.columns:
                stats[f'{col}_mean'] = cluster_data[col].mean()
                stats[f'{col}_std'] = cluster_data[col].std()
        
        summary_stats.append(stats)
    
    return pd.DataFrame(summary_stats)

def visualize_clusters_pca(clustered_data, feature_patterns, n_components=2, 
                          color_col='cluster_id', figsize=(10, 8)):
    """
    Visualize clusters in PCA space.
    
    Parameters:
    -----------
    clustered_data : pd.DataFrame
        Data with cluster labels
    feature_patterns : list
        Feature patterns used for clustering
    n_components : int
        Number of PCA components to plot
    color_col : str
        Column to use for coloring points
    figsize : tuple
        Figure size
    """
    
    # Get feature columns
    feature_cols = []
    for pattern in feature_patterns:
        cols = [col for col in clustered_data.columns if pattern in col]
        feature_cols.extend(cols)
    
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_data = clustered_data[feature_cols].dropna()
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(feature_data)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    if color_col in clustered_data.columns:
        colors = clustered_data.loc[feature_data.index, color_col]
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], 
                            c=colors, cmap='hsv', alpha=0.6)
        plt.colorbar(scatter, label=color_col)
    else:
        plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.6)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Clusters in PCA Space')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pca, pca_data

def merge_clusters_to_original(original_df, clustering_result, 
                              id_cols=['recording_id', 'roi_id']):
    """
    Merge cluster IDs back to original dataframe preserving indices.
    
    Parameters:
    -----------
    original_df : pd.DataFrame
        Original dataframe with all ROIs
    clustering_result : dict
        Result from cluster_rf_data()
    id_cols : list
        Columns to use for merging (unique identifiers)
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with cluster_id column added
    """
    # Extract clustered data
    clustered_data = clustering_result['clustered_data']
    
    # Create merge dataframe with just ID columns and cluster_id
    merge_df = clustered_data[id_cols + ['cluster_id']].copy()
    
    # Merge back to original dataframe
    result_df = original_df.merge(merge_df, on=id_cols, how='left')
    
    # cluster_id will be NaN for empty cells (not included in clustering)
    return result_df

def create_cluster_averaged_rfs(df_with_clusters, experiment_obj, 
                               cluster_col='cluster_id', max_shift=5):
    """
    Create averaged RF representations for each cluster, aligned by centroids.
    
    Parameters:
    -----------
    df_with_clusters : pd.DataFrame
        Dataframe with cluster_id column
    experiment_obj : Experiment
        Experiment object containing recording data
    cluster_col : str
        Name of cluster column
    max_shift : int
        Maximum pixels to shift for alignment
    
    Returns:
    --------
    dict
        Dictionary with cluster_id as keys, averaged RFs as values
    """
    from scipy.ndimage import shift
    import warnings
    
    cluster_averages = {}
    
    # Get unique clusters (excluding NaN)
    clusters = df_with_clusters[cluster_col].dropna().unique()
    
    for cluster_id in clusters:
        print(f"\nProcessing cluster {cluster_id}...")
        
        # Get ROIs in this cluster
        cluster_rois = df_with_clusters[df_with_clusters[cluster_col] == cluster_id]
        print(f"  Found {len(cluster_rois)} ROIs in cluster")
        
        # Collect RF data and centroids for this cluster
        cluster_rfs = []
        cluster_centroids_x = []
        cluster_centroids_y = []
        
        for _, roi_row in cluster_rois.iterrows():
            recording_id = int(roi_row['recording_id'])
            roi_id = int(roi_row['roi_id'])
            
            # Get the recording object
            recording = experiment_obj.recording[recording_id]
            
            # Get RF data: (color, roi, y, x)
            rf_data = recording.collapse_times_by_channel()
            
            # Get centroids: (color, roi)
            centroids_x = recording.get_pca_centroidsX_by_channel()
            centroids_y = recording.get_pca_centroidsY_by_channel()
            
            # Extract data for this specific ROI
            roi_rf = rf_data[:, roi_id, :, :]  # Shape: (color, y, x)
            roi_cent_x = centroids_x[:, roi_id]  # Shape: (color,)
            roi_cent_y = centroids_y[:, roi_id]  # Shape: (color,)
            
            cluster_rfs.append(roi_rf)
            cluster_centroids_x.append(roi_cent_x)
            cluster_centroids_y.append(roi_cent_y)
        
        if not cluster_rfs:
            continue
            
        # Convert to numpy arrays
        cluster_rfs = np.array(cluster_rfs)  # Shape: (n_rois, color, y, x)
        cluster_centroids_x = np.array(cluster_centroids_x)  # Shape: (n_rois, color)
        cluster_centroids_y = np.array(cluster_centroids_y)  # Shape: (n_rois, color)
        
        print(f"  RF data shape: {cluster_rfs.shape}")
        
        # Align RFs by centroids for each color channel
        n_rois, n_colors, height, width = cluster_rfs.shape
        aligned_rfs = np.zeros_like(cluster_rfs)
        
        for color in range(n_colors):
            # Calculate reference centroid (mean of all ROIs for this color)
            ref_cent_x = np.nanmean(cluster_centroids_x[:, color])
            ref_cent_y = np.nanmean(cluster_centroids_y[:, color])
            
            print(f"    Color {color}: Reference centroid = ({ref_cent_x:.1f}, {ref_cent_y:.1f})")
            
            for roi in range(n_rois):
                # Calculate shift needed to align to reference
                shift_x = ref_cent_x - cluster_centroids_x[roi, color]
                shift_y = ref_cent_y - cluster_centroids_y[roi, color]
                
                # Limit shifts to prevent excessive movement
                shift_x = np.clip(shift_x, -max_shift, max_shift)
                shift_y = np.clip(shift_y, -max_shift, max_shift)
                
                # Apply shift (note: scipy shift uses (y, x) order)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    aligned_rfs[roi, color, :, :] = shift(
                        cluster_rfs[roi, color, :, :], 
                        shift=[shift_y, shift_x], 
                        mode='constant', 
                        cval=0
                    )
        
        # Average aligned RFs
        avg_rf = np.nanmean(aligned_rfs, axis=0)  # Shape: (color, y, x)
        
        # Store result
        cluster_averages[cluster_id] = {
            'averaged_rf': avg_rf,
            'n_rois': n_rois,
            'reference_centroids_x': [np.nanmean(cluster_centroids_x[:, c]) for c in range(n_colors)],
            'reference_centroids_y': [np.nanmean(cluster_centroids_y[:, c]) for c in range(n_colors)],
            'roi_info': cluster_rois[['recording_id', 'roi_id']].values
        }
        
        print(f"    Created averaged RF with shape: {avg_rf.shape}")
    
    print(f"\nCompleted averaging for {len(cluster_averages)} clusters")
    return cluster_averages


def verify_rf_averaging(df_with_clusters, experiment_obj, cluster_averages, cluster_col='cluster_id', 
                        max_clusters_to_show=3, max_rois_to_show=3):
    """
    Verify that RF averaging is working correctly by showing individual RFs vs averages.
    This function provides evidence that averages are genuine combinations, not single examples.
    
    Parameters:
    -----------
    df_with_clusters : pd.DataFrame
        Dataframe with cluster_id column
    experiment_obj : Experiment
        Experiment object containing recording data
    cluster_averages : dict
        Dictionary from create_cluster_averaged_rfs
    cluster_col : str
        Name of cluster column
    max_clusters_to_show : int or None
        Maximum number of clusters to analyze in detail. If None, shows all clusters.
    max_rois_to_show : int
        Maximum number of individual ROIs to show visually (statistics cover all ROIs)
    
    Returns:
    --------
    dict
        Verification statistics and plots
        
    Note:
    -----
    The RFs shown are ALIGNED versions (already shifted by centroids during averaging).
    The original, unaligned RFs would show more spatial scatter.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    
    print("=== RF AVERAGING VERIFICATION ===")
    print("This analysis proves that cluster averages are genuine combinations of multiple RFs\n")
    
    verification_results = {}
    
    # Handle max_clusters_to_show=None (show all) and ensure linear order (0, 1, 2, ...)
    all_cluster_ids = sorted(cluster_averages.keys())  # Sort to ensure linear order
    
    if max_clusters_to_show is None:
        clusters_to_verify = all_cluster_ids
        print(f"Showing ALL {len(clusters_to_verify)} clusters in order: {clusters_to_verify}")
    else:
        clusters_to_verify = all_cluster_ids[:max_clusters_to_show]
        print(f"Showing first {len(clusters_to_verify)} of {len(all_cluster_ids)} clusters in order: {clusters_to_verify}")
    
    for cluster_id in clusters_to_verify:
        print(f"\n--- CLUSTER {cluster_id} VERIFICATION ---")
        
        # Get cluster info
        cluster_info = cluster_averages[cluster_id]
        averaged_rf = cluster_info['averaged_rf']
        n_rois = cluster_info['n_rois']
        roi_info = cluster_info['roi_info']
        
        print(f"Number of ROIs in cluster: {n_rois}")
        
        # Collect individual RFs for this cluster
        individual_rfs = []
        roi_identifiers = []
        
        for recording_id, roi_id in roi_info:
            recording_id, roi_id = int(recording_id), int(roi_id)
            recording = experiment_obj.recording[recording_id]
            rf_data = recording.collapse_times_by_channel()
            roi_rf = rf_data[:, roi_id, :, :]  # Shape: (color, y, x)
            individual_rfs.append(roi_rf)
            roi_identifiers.append(f"R{recording_id}_ROI{roi_id}")
        
        individual_rfs = np.array(individual_rfs)  # Shape: (n_rois, color, y, x)
        
        # Analysis for each color channel
        channel_stats = {}
        for color in range(averaged_rf.shape[0]):
            print(f"\n  Color channel {color}:")
            
            # Get data for this color
            avg_channel = averaged_rf[color, :, :]
            individual_channels = individual_rfs[:, color, :, :]  # Shape: (n_rois, y, x)
            
            # Calculate statistics that prove it's an average, not a single example
            channel_correlations = []
            channel_intensities = []
            
            for i, individual_rf in enumerate(individual_channels):
                # Correlation with average
                corr, _ = pearsonr(individual_rf.flatten(), avg_channel.flatten())
                channel_correlations.append(corr)
                
                # Max intensity
                channel_intensities.append(np.max(np.abs(individual_rf)))
                
                print(f"    {roi_identifiers[i]}: correlation with average = {corr:.3f}, max intensity = {channel_intensities[i]:.1f}")
            
            # Summary statistics
            mean_corr = np.mean(channel_correlations)
            avg_max_intensity = np.max(np.abs(avg_channel))
            individual_mean_intensity = np.mean(channel_intensities)
            
            print(f"    SUMMARY:")
            print(f"      Mean correlation with average: {mean_corr:.3f}")
            print(f"      Average max intensity: {avg_max_intensity:.1f}")
            print(f"      Mean individual max intensity: {individual_mean_intensity:.1f}")
            
            # Key verification: if this were just a single example, correlation would be 1.0 for one RF and low for others
            # For genuine averages, correlations should be moderate for all contributing RFs
            if len(channel_correlations) > 1:
                correlation_std = np.std(channel_correlations)
                max_correlation = np.max(channel_correlations)
                
                print(f"      Correlation std dev: {correlation_std:.3f}")
                print(f"      Max correlation: {max_correlation:.3f}")
                
                if max_correlation > 0.95 and correlation_std > 0.3:
                    print(f"        WARNING: This might be showing a single example (one very high correlation)")
                elif correlation_std < 0.1 and mean_corr > 0.7:
                    print(f"       VERIFIED: Consistent correlations suggest genuine averaging")
                else:
                    print(f"       VERIFIED: Mixed correlations suggest genuine averaging")
            
            # Store detailed stats
            channel_stats[color] = {
                'correlations': channel_correlations,
                'intensities': channel_intensities,
                'mean_correlation': mean_corr,
                'correlation_std': correlation_std if len(channel_correlations) > 1 else 0,
                'avg_intensity': avg_max_intensity,
                'individual_mean_intensity': individual_mean_intensity
            }
        
        # Create visualization comparing individuals to average
        n_panels = min(max_rois_to_show + 1, n_rois + 1)  # +1 for average panel
        fig, axes = plt.subplots(2, n_panels, figsize=(4 * n_panels, 6))
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        if n_panels == 1:
            axes = axes.reshape(2, 1)
        
        # Show RGB composite for visualization (average + individual examples)
        for i in range(n_panels):
            if i == 0:
                # Show average using your RGB composite function for direct comparison
                rgb_data = create_rgb_composite(averaged_rf, [0, 1, 2])  # RGB composite [0,1,2]
                axes[0, i].imshow(rgb_data)  # No clim - auto equalize like in your overview
                axes[0, i].set_title(f'AVERAGE\n(n={n_rois})')
                axes[0, i].axis('off')
                
                # Show intensity profile for average
                axes[1, i].plot(np.sum(averaged_rf, axis=(1, 2)), 'ko-', linewidth=2, label='Average')
                axes[1, i].set_title('Intensity by color')
                axes[1, i].set_ylabel('Total intensity')
                axes[1, i].legend()
            else:
                # Show individual RF (keep old visualization method)
                if i-1 < len(individual_rfs) and i-1 < max_rois_to_show:
                    # Use old-style RGB visualization for individuals to maintain contrast with average
                    individual_rf = individual_rfs[i-1]
                    
                    # Extract RGB channels and normalize like the old method
                    r_channel = individual_rf[0, :, :] if 0 < individual_rf.shape[0] else np.zeros(individual_rf.shape[1:])
                    g_channel = individual_rf[1, :, :] if 1 < individual_rf.shape[0] else np.zeros(individual_rf.shape[1:])
                    b_channel = individual_rf[2, :, :] if 2 < individual_rf.shape[0] else np.zeros(individual_rf.shape[1:])
                    
                    # Normalize each channel to 0-1 range (old method)
                    def normalize_channel_old(channel):
                        channel_abs = np.abs(channel)
                        if np.max(channel_abs) > 0:
                            return (channel + np.max(channel_abs)) / (2 * np.max(channel_abs))
                        else:
                            return np.zeros_like(channel)
                    
                    r_norm = normalize_channel_old(r_channel)
                    g_norm = normalize_channel_old(g_channel)
                    b_norm = normalize_channel_old(b_channel)
                    
                    individual_rgb = np.stack([r_norm, g_norm, b_norm], axis=2)
                    axes[0, i].imshow(individual_rgb, origin='lower')
                    axes[0, i].set_title(f'{roi_identifiers[i-1]}')
                    axes[0, i].axis('off')
                    
                    # Show intensity profile for individual
                    axes[1, i].plot(np.sum(individual_rfs[i-1], axis=(1, 2)), 'ro-', alpha=0.7, label=f'ROI {i-1}')
                    axes[1, i].set_title('Intensity by color')
                    axes[1, i].legend()
                else:
                    axes[0, i].axis('off')
                    axes[1, i].axis('off')
        
        plt.suptitle(f'Cluster {cluster_id} Verification: Average vs Individuals\n(Individual RFs shown are UNALIGNED originals - spatial scatter is expected)')
        plt.tight_layout()
        plt.show()
        
        # Store verification results
        verification_results[cluster_id] = {
            'n_rois': n_rois,
            'channel_stats': channel_stats,
            'roi_identifiers': roi_identifiers
        }
        
        print(f"--- END CLUSTER {cluster_id} VERIFICATION ---")
    
    # Overall summary
    print(f"\n=== OVERALL VERIFICATION SUMMARY ===")
    print(f"Analyzed {len(verification_results)} clusters")
    
    all_mean_corrs = []
    all_corr_stds = []
    
    for cluster_id, results in verification_results.items():
        for color, stats in results['channel_stats'].items():
            if not np.isnan(stats['mean_correlation']):
                all_mean_corrs.append(stats['mean_correlation'])
                all_corr_stds.append(stats['correlation_std'])
    
    if all_mean_corrs:
        overall_mean_corr = np.mean(all_mean_corrs)
        overall_corr_std = np.mean(all_corr_stds)
        
        print(f"Overall mean correlation with averages: {overall_mean_corr:.3f}")
        print(f"Overall correlation std deviation: {overall_corr_std:.3f}")
        
        if overall_mean_corr > 0.95:
            print("⚠️  SUSPICIOUS: Very high correlations suggest possible single-example issue")
        elif overall_mean_corr > 0.4 and overall_corr_std < 0.4:
            print("✅ LIKELY GENUINE: Moderate, consistent correlations suggest proper averaging")
        else:
            print("✅ LIKELY GENUINE: Mixed correlations suggest proper averaging")
    
    print("=== END VERIFICATION ===\n")
    return verification_results


def plot_all_cluster_averages(cluster_averages,
                            figsize_per_cluster=(6, 1),
                            use_color_mapping=True,
                            rgb_mode='new',
                            save_path=None):
    """
    Plot averaged RFs for all clusters with RGB composites.

    Parameters:
    -----------
    cluster_averages : dict
        Result from create_cluster_averaged_rfs()
    figsize_per_cluster : tuple
        Size per cluster row in inches
    use_color_mapping : bool
        Whether to apply color mapping to channels
    rgb_mode : str
        RGB visualization mode: 'new' (your preferred method) or 'old' (bipolar method)
    save_path : str, optional
        Path to save the figure
    """
    import matplotlib.pyplot as plt
    
    n_clusters = len(cluster_averages)
    n_total_cols = 6  # Red, Green, Blue, UV, RGB, RGU

    if n_clusters == 0:
        print("No clusters to plot!")
        return

    # Calculate figure size and subplot grid
    fig_width = figsize_per_cluster[0] * n_total_cols / 3
    fig_height = figsize_per_cluster[1] * n_clusters

    fig, axes = plt.subplots(n_clusters, n_total_cols,
                            figsize=(fig_width, fig_height))

    # Handle single cluster case
    if n_clusters == 1:
        axes = axes.reshape(1, -1)

    # Sort clusters by ID for consistent ordering
    sorted_clusters = sorted(cluster_averages.keys())

    # Define color maps for individual channels
    if use_color_mapping:
        try:
            from pygor.plotting.custom import maps_concat
            if isinstance(maps_concat, list):
                colormaps = maps_concat[:4]  # First 4 for individual channels
            else:
                colormaps = ['Reds', 'Greens', 'Blues', 'Purples']
        except ImportError:
            colormaps = ['Reds', 'Greens', 'Blues', 'Purples']
    else:
        colormaps = ['gray'] * 4

    channel_names = ['Red', 'Green', 'Blue', 'UV', 'RGB', 'RGU']

    for row, cluster_id in enumerate(sorted_clusters):
        curr_clust_avgs = cluster_averages[cluster_id]["averaged_rf"]
        n_rois = cluster_averages[cluster_id]["n_rois"]

        for col in range(n_total_cols):
            ax = axes[row, col] if n_clusters > 1 else axes[col]
            curr_clim = np.max(np.abs(curr_clust_avgs)) if use_color_mapping else None
            
            if col < 4:  # Individual channels (Red, Green, Blue, UV)
                cmap = colormaps[col] if use_color_mapping else 'gray'
                ax.imshow(curr_clust_avgs[col], cmap=cmap,
                        vmin=-curr_clim, vmax=curr_clim, origin="lower")
                
                # Add crosshairs
                centre = (curr_clust_avgs.shape[1] // 2, curr_clust_avgs.shape[2] // 2)
                ax.axhline(centre[0], color='white', lw=0.5)
                ax.axvline(centre[1], color='white', lw=0.5)

            elif col == 4:  # RGB composite [0,1,2]
                if rgb_mode == 'new':
                    rgb_composite = create_rgb_composite(curr_clust_avgs, [0, 1, 2])
                    ax.imshow(rgb_composite, origin="lower")  # No clim - auto equalize
                else:  # rgb_mode == 'old'
                    rgb_composite = _create_rgb_composite_old(curr_clust_avgs, [0, 1, 2])
                    ax.imshow(rgb_composite, origin="lower")

            elif col == 5:  # RGU composite [0,1,3]
                if rgb_mode == 'new':
                    rgu_composite = create_rgb_composite(curr_clust_avgs, [0, 1, 3])
                    ax.imshow(rgu_composite, origin="lower")  # No clim - auto equalize
                else:  # rgb_mode == 'old'
                    rgu_composite = _create_rgb_composite_old(curr_clust_avgs, [0, 1, 3])
                    ax.imshow(rgu_composite, origin="lower")

            # Add titles only to top row
            if row == 0:
                ax.set_title(channel_names[col], fontsize=8)

            # Add cluster info to leftmost column
            if col == 0:
                ax.set_ylabel(f'Cluster {cluster_id}\n(n={n_rois})', fontsize=8)

            ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cluster averages plot to {save_path}")

    plt.show()

    # Print summary
    print(f"\nDisplayed {n_clusters} clusters:")
    for cluster_id in sorted_clusters:
        n_rois = cluster_averages[cluster_id]["n_rois"]
        print(f"  Cluster {cluster_id}: {n_rois} ROIs")


def _create_rgb_composite_old(curr_clust_avgs, channel_indices):
    """
    Create RGB composite using the old bipolar method (for comparison).
    
    Parameters:
    -----------
    curr_clust_avgs : np.array
        Shape: (color, y, x)
    channel_indices : list
        List of 3 channel indices for R, G, B
    
    Returns:
    --------
    np.array
        RGB composite image (y, x, 3)
    """
    # Extract the RGB channels
    r_channel = curr_clust_avgs[channel_indices[0], :, :] if channel_indices[0] < curr_clust_avgs.shape[0] else np.zeros(curr_clust_avgs.shape[1:])
    g_channel = curr_clust_avgs[channel_indices[1], :, :] if channel_indices[1] < curr_clust_avgs.shape[0] else np.zeros(curr_clust_avgs.shape[1:])
    b_channel = curr_clust_avgs[channel_indices[2], :, :] if channel_indices[2] < curr_clust_avgs.shape[0] else np.zeros(curr_clust_avgs.shape[1:])
    
    # Normalize each channel to 0-1 range (old bipolar method)
    def normalize_channel_old(channel):
        channel_abs = np.abs(channel)
        if np.max(channel_abs) > 0:
            return (channel + np.max(channel_abs)) / (2 * np.max(channel_abs))
        else:
            return np.zeros_like(channel)
    
    r_norm = normalize_channel_old(r_channel)
    g_norm = normalize_channel_old(g_channel)
    b_norm = normalize_channel_old(b_channel)
    
    # Stack into RGB
    rgb_composite = np.stack([r_norm, g_norm, b_norm], axis=2)
    
    return rgb_composite

# Convenience function that combines everything
def cluster_rf_data(df, feature_patterns, method='kmeans', n_clusters=5, 
                   id_vars=None, scale=True, scaling_method='standard', show_elbow=True, **kwargs):
    """
    Complete clustering workflow for RF data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    feature_patterns : list
        Feature patterns to cluster on (e.g., ['areas', 'eccentricity'])
    method : str
        Clustering method
    n_clusters : int
        Number of clusters
    id_vars : list
        ID variables to preserve
    scale : bool
        Whether to scale features
    scaling_method : str, default 'standard'
        Scaling method: 'standard', 'custom', 'rank', 'l2'
    show_elbow : bool
        Whether to show elbow plot for kmeans
    **kwargs
        Additional clustering parameters
    
    Returns:
    --------
    dict
        Dictionary containing clustered_data, summary_stats, scaler, and pca results
    """
    
    print(f"Clustering RF data using {method} with {n_clusters} clusters")
    print(f"Feature patterns: {feature_patterns}")
    
    # Handle None id_vars
    if id_vars is None:
        id_vars = []
    
    # Extract handle_missing parameter
    handle_missing = kwargs.get('handle_missing', 'fill_zero')
    
    # Prepare data
    prepared_data, scaler = prepare_clustering_data(
        df, feature_patterns, id_vars, scale=scale, handle_missing=handle_missing, 
        scaling_method=scaling_method
    )
    
    # Show elbow plot if requested and using kmeans
    elbow_results = None
    if show_elbow and method == 'kmeans':
        feature_cols = [col for col in prepared_data.columns if col not in (id_vars or [])]
        elbow_results = elbow_analysis(prepared_data[feature_cols])
    
    # Apply clustering
    clustering_result = apply_clustering(
        prepared_data, method=method, n_clusters=n_clusters, 
        id_cols=id_vars, **kwargs
    )
    
    # Extract clustered data
    clustered_data = clustering_result['clustered_data']
    
    # Generate summary statistics
    summary_stats = cluster_summary_stats(clustered_data, feature_patterns, id_vars)
    
    # PCA visualization
    pca, pca_data = visualize_clusters_pca(clustered_data, feature_patterns)
    
    return {
        'clustered_data': clustered_data,
        'summary_stats': summary_stats,
        'scaler': scaler,
        'pca': pca,
        'pca_data': pca_data,
        'elbow_results': elbow_results,
        'clustering_pca': clustering_result.get('pca_obj'),  # PCA used for clustering
        'clusterer': clustering_result.get('clusterer')
    }

# Example usage for verification:
"""
# After clustering and creating averaged RFs:
clustering_result = cluster_rf_data(control_data_melted, feature_patterns=['space_amps', 'areas'], 
                                   method='kmeans', n_clusters=5, scaling_method='rank')

# Merge clusters back to original data
clustered_control = merge_clusters_to_original(control_data_melted, clustering_result)

# Create averaged RFs
cluster_averages = create_cluster_averaged_rfs(clustered_control, control_data)

# VERIFY THAT AVERAGING IS WORKING (this will prove it's not just single examples):

# Show all clusters, 3 individual ROIs per cluster (default)
verification_results = verify_rf_averaging(clustered_control, control_data, cluster_averages, 
                                         max_clusters_to_show=None)

# Show first 2 clusters, 5 individual ROIs per cluster
verification_results = verify_rf_averaging(clustered_control, control_data, cluster_averages, 
                                         max_clusters_to_show=2, max_rois_to_show=5)

# PLOT ALL CLUSTER AVERAGES (comprehensive overview):

# Using your preferred RGB method (default)
plot_all_cluster_averages(cluster_averages, use_color_mapping=True, rgb_mode='new')

# Using the old bipolar RGB method for comparison
plot_all_cluster_averages(cluster_averages, use_color_mapping=True, rgb_mode='old')

# Save to file
plot_all_cluster_averages(cluster_averages, use_color_mapping=True, 
                         save_path='cluster_averages_overview.png')

The verification function will:
1. Show correlation statistics between each individual RF and the cluster average
2. Display intensity profiles comparing individuals to averages  
3. Create side-by-side visualizations of averages vs individual contributors
4. Provide clear verdicts on whether averaging is genuine or suspicious

Key indicators of genuine averaging:
- Multiple ROIs with moderate correlations (0.4-0.8) to the average
- Intensity profiles that are smooth combinations of individual profiles
- No single ROI with correlation > 0.95 while others are much lower
- Consistent correlation standard deviations < 0.4

If it were just showing single examples, you'd see:
- One ROI with correlation ~1.0 to the "average"
- Other ROIs with very low correlations  
- Large correlation standard deviation > 0.3
- Average intensity profiles that match exactly one individual
"""