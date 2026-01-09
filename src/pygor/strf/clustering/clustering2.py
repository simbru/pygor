"""
Clean clustering module for RF analysis
Integrates with the melting framework for flexible column selection
"""

from matplotlib import rcParams
import pandas as pd
import numpy as np
import warnings
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

# Configuration for consistency across all plots
sns.set_theme(style="whitegrid", rc={"grid.linewidth": 0.5, "grid.alpha": 0.0})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = False
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_context('paper', font_scale=0.8)

def create_rgb_composite(curr_clust_avgs, channel_indices, mode='proportional'):
    """
    Create RGB composite from selected channels.

    Parameters:
    -----------
    curr_clust_avgs : np.array
        Shape: (color, y, x)
    channel_indices : list
        List of 3 channel indices for R, G, B
    mode : str, default 'proportional'
        'proportional': Standard proportional normalization (current default)
        'grey_baseline': Grey baseline with ON adding color and OFF subtracting color

    Returns:
    --------
    np.array
        RGB composite image (y, x, 3)
    """
    # Extract channels
    channels = []
    for channel_idx in channel_indices:
        if channel_idx < curr_clust_avgs.shape[0]:
            channels.append(curr_clust_avgs[channel_idx])
        else:
            channels.append(np.zeros(curr_clust_avgs.shape[1:]))
    
    # Find global maximum across all channels for scaling
    global_max = max(np.max(np.abs(channel)) for channel in channels) if len(channels) > 0 else 1.0
    
    if mode == 'grey_baseline':
        # Enhanced grey baseline mode with perceptual improvements
        # 1. Start with darker baseline (0.35) for more dynamic range
        rgb_composite = np.full((*curr_clust_avgs.shape[1:], 3), 0.35)
        
        if global_max > 0:
            # 2. Perceptually-uniform scaling based on ITU-R BT.709 luminance sensitivity
            # Green appears brightest (reduce), Blue appears dimmest (boost most), Red intermediate
            saturation_scalars = [1,.75,2]  # [Red, Green, Blue]
            
            for i, channel in enumerate(channels):
                # Use channel-specific saturation scaling with higher range for darker baseline
                scalar = saturation_scalars[i] if i < len(saturation_scalars) else 0.55
                scaled_channel = (channel / global_max) * scalar
                
                # Add scaled values to darker baseline
                # ON responses (positive) make that color channel brighter
                # OFF responses (negative) make that color channel darker
                rgb_composite[:, :, i] += scaled_channel
            
            # Clip to valid range [0, 1]
            rgb_composite = np.clip(rgb_composite, 0.0, 1.0)
            
            # 3. Apply gamma correction for more punchy, saturated colors
            rgb_composite = np.power(rgb_composite, 0.33)  # Gamma < 1 = more contrast/saturation
    
    else:  # 'proportional' mode (default)
        # Standard proportional normalization
        rgb_composite = np.zeros((*curr_clust_avgs.shape[1:], 3))
        
        if global_max > 0:
            for i, channel in enumerate(channels):
                # Proportional normalization - all channels use the same global max
                rgb_composite[:, :, i] = np.abs(channel) / global_max
        # If global_max == 0, rgb_composite remains zeros (black)
    
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
        # First try exact match for singular features
        if pattern in df.columns:
            feature_cols.append(pattern)
        # Then try pattern matching for multi-column features  
        else:
            cols = [col for col in df.columns if pattern in col and col != pattern]
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
            # Robust scaling - apply to all features uniformly
            scaler = RobustScaler()
            feature_data = pd.DataFrame(
                scaler.fit_transform(feature_data),
                index=feature_data.index,
                columns=feature_data.columns
            )
            print("Applied RobustScaler to all features")
                
        elif scaling_method == 'quantile':
            # Quantile uniform transformation - apply to all features uniformly
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
            available_methods = ['standard', 'custom', 'robust', 'quantile', 'rank', 'l2']
            print(f"Available scaling methods:")
            print(f"  - 'standard': StandardScaler (zero mean, unit variance)")
            print(f"  - 'custom': Feature-specific scaling (MaxAbs for some, MinMax for others)")
            print(f"  - 'robust': RobustScaler (median-based, outlier resistant)")
            print(f"  - 'quantile': QuantileTransformer (uniform distribution)")
            print(f"  - 'rank': Rank-based scaling (preserves ordinal relationships)")
            print(f"  - 'l2': L2 normalization (unit norm per sample/ROI)")
            raise ValueError(f"Unknown scaling method: '{scaling_method}'. Choose from: {available_methods}")
    
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
    plt.show()
    
    return {'k_values': list(k_values), 'inertias': inertias}

def _relabel_clusters_by_ipl_depth(clustered_data, verbose=False):
    """
    Relabel clusters so that cluster IDs are ordered by IPL depth (shallow to deep).

    Uses the same logic as the plotting functions to determine IPL-based ordering,
    but applies it to the cluster IDs themselves rather than just for display.

    Parameters:
    -----------
    clustered_data : pd.DataFrame
        Data with cluster_id and depths columns
    verbose : bool, default False
        Whether to print relabeling information

    Returns:
    --------
    pd.DataFrame
        Data with relabeled cluster_id column (0=shallowest, n=deepest)
    """
    if 'depths' not in clustered_data.columns:
        if verbose:
            print("No 'depths' column found - skipping IPL-based relabeling")
        return clustered_data

    # Calculate IPL statistics for each cluster (same logic as plot_cluster_results)
    original_clusters = sorted(clustered_data['cluster_id'].unique())
    cluster_ipl_stats = {}

    for cluster_id in original_clusters:
        cluster_rois = clustered_data[clustered_data['cluster_id'] == cluster_id]
        ipl_depths = cluster_rois['depths'].dropna().values

        if len(ipl_depths) == 0:
            # No depth data for this cluster - assign to middle
            cluster_ipl_stats[cluster_id] = {
                'mean': 45.0,  # Middle depth
                'category': 'unknown',
                'sort_key': (1.5, 45.0)  # Put unknowns after bistratified
            }
            continue

        mean_depth = np.mean(ipl_depths)
        median_depth = np.median(ipl_depths)  # More robust to outliers
        std_depth = np.std(ipl_depths)

        # Detect bistratified clusters (same logic as plot_cluster_results)
        is_bistratified = False
        if len(ipl_depths) >= 10:  # Need sufficient data points
            # Count shallow vs deep ROIs
            shallow_count = np.sum(ipl_depths < 35)
            deep_count = np.sum(ipl_depths > 55)

            # Bistratified if high variability AND presence in both layers
            if (std_depth > 15 and
                min(shallow_count, deep_count) >= 5):
                is_bistratified = True

        # Categorize for sorting - use median depth for robustness
        if is_bistratified:
            category = 'bistratified'
            sort_key = (1, 45 + std_depth/10)  # Middle position
        elif median_depth >= 45:  # Deep/ON-dominant (use median)
            category = 'deep'
            sort_key = (0, -median_depth)  # Deep first, sorted by descending median depth
        else:  # Shallow/OFF-dominant (use median)
            category = 'shallow'
            sort_key = (2, -median_depth)  # Shallow last, sorted by descending median depth

        cluster_ipl_stats[cluster_id] = {
            'mean': mean_depth,
            'median': median_depth,
            'std': std_depth,
            'category': category,
            'sort_key': sort_key
        }

    # Sort clusters by IPL depth (shallow to deep)
    sorted_original_clusters = sorted(original_clusters, key=lambda cid: cluster_ipl_stats[cid]['sort_key'])

    # Create mapping from original cluster IDs to new IPL-ordered IDs
    cluster_id_mapping = {orig_id: new_id for new_id, orig_id in enumerate(sorted_original_clusters)}

    if verbose:
        print("Relabeling clusters by IPL depth (using median for sorting):")
        for new_id, orig_id in enumerate(sorted_original_clusters):
            stats = cluster_ipl_stats[orig_id]
            print(f"  Original cluster {orig_id} → New cluster {new_id} ({stats['category']}, median={stats.get('median', stats['mean']):.1f}, mean={stats['mean']:.1f})")

    # Apply the relabeling
    result = clustered_data.copy()
    result['cluster_id'] = result['cluster_id'].map(cluster_id_mapping)

    return result

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
        available_methods = ['kmeans', 'gmm', 'hierarchical']
        print(f"Available clustering methods:")
        print(f"  - 'kmeans': K-means clustering (spherical clusters, fast)")
        print(f"  - 'gmm': Gaussian Mixture Model (flexible cluster shapes, probabilistic)")
        print(f"  - 'hierarchical': Agglomerative clustering (tree-based, no centroid assumption)")
        raise ValueError(f"Unknown clustering method: '{method}'. Choose from: {available_methods}")
    
    # Add cluster labels to original data
    result = data.copy()
    result['cluster_id'] = cluster_labels

    # Relabel clusters by IPL depth if depths column is available
    if 'depths' in data.columns:
        result = _relabel_clusters_by_ipl_depth(result, verbose=True)

    print(f"Created {len(np.unique(result['cluster_id']))} clusters")
    print(f"Final cluster sizes: {np.bincount(result['cluster_id'])}")

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
        # First try exact match for singular features
        if pattern in clustered_data.columns:
            feature_cols.append(pattern)
        # Then try pattern matching for multi-column features  
        else:
            cols = [col for col in clustered_data.columns if pattern in col and col != pattern]
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
                if cluster_data[id_col].dtype == 'object':
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
                          pca_dims = (0, 1),
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
        # First try exact match for singular features
        if pattern in clustered_data.columns:
            feature_cols.append(pattern)
        # Then try pattern matching for multi-column features  
        else:
            cols = [col for col in clustered_data.columns if pattern in col and col != pattern]
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
        scatter = plt.scatter(pca_data[:, pca_dims[0]], pca_data[:, pca_dims[1]], 
                            c=colors, cmap='nipy_spectral', alpha=0.6,
                            s=rcParams['lines.markersize']/2)
        # plt.colorbar(scatter, label=color_col)
    else:
        plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.6, 
                            s=rcParams['lines.markersize']/2)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[pca_dims[0]]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[pca_dims[1]]:.1%} variance)')
    plt.title('Clusters in PCA Space')
    # plt.show()
    
    return pca, pca_data

def visualize_clusters_umap(clustered_data, feature_patterns, n_components=2,
                           color_col='cluster_id', figsize=(10, 8), **umap_kwargs):
    """
    Visualize clusters in UMAP space.
    
    Parameters:
    -----------
    clustered_data : pd.DataFrame
        Data with cluster labels
    feature_patterns : list
        Feature patterns used for clustering
    n_components : int
        Number of UMAP components to plot
    color_col : str
        Column to use for coloring points
    figsize : tuple
        Figure size
    **umap_kwargs
        Additional UMAP parameters
    """
    
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None, None
    
    # Get feature columns
    feature_cols = []
    for pattern in feature_patterns:
        # First try exact match for singular features
        if pattern in clustered_data.columns:
            feature_cols.append(pattern)
        # Then try pattern matching for multi-column features  
        else:
            cols = [col for col in clustered_data.columns if pattern in col and col != pattern]
            feature_cols.extend(cols)
    
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_data = clustered_data[feature_cols].dropna()
    
    # Apply UMAP
    umap_model = umap.UMAP(n_components=n_components, **umap_kwargs)
    umap_data = umap_model.fit_transform(feature_data)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    if color_col in clustered_data.columns:
        colors = clustered_data.loc[feature_data.index, color_col]
        scatter = plt.scatter(umap_data[:, 0], umap_data[:, 1], 
                            c=colors, cmap='nipy_spectral', alpha=0.6,
                            s=rcParams['lines.markersize']/2)
        # plt.colorbar(scatter, label=color_col)
    else:
        plt.scatter(umap_data[:, 0], umap_data[:, 1], alpha=0.6,
                    s=rcParams['lines.markersize']/2)

    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('Clusters in UMAP Space')
    # plt.show()
    
    return umap_model, umap_data

def visualize_clusters_tsne(clustered_data, feature_patterns, n_components=2,
                           color_col='cluster_id', figsize=(10, 8), **tsne_kwargs):
    """
    Visualize clusters in t-SNE space.
    
    Parameters:
    -----------
    clustered_data : pd.DataFrame
        Data with cluster labels
    feature_patterns : list
        Feature patterns used for clustering
    n_components : int
        Number of t-SNE components to plot
    color_col : str
        Column to use for coloring points
    figsize : tuple
        Figure size
    **tsne_kwargs
        Additional t-SNE parameters
    """
    
    from sklearn.manifold import TSNE
    
    # Get feature columns
    feature_cols = []
    for pattern in feature_patterns:
        # First try exact match for singular features
        if pattern in clustered_data.columns:
            feature_cols.append(pattern)
        # Then try pattern matching for multi-column features  
        else:
            cols = [col for col in clustered_data.columns if pattern in col and col != pattern]
            feature_cols.extend(cols)
    
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_data = clustered_data[feature_cols].dropna()
    
    # Apply t-SNE
    tsne_defaults = {'perplexity': 30, 'random_state': 42}
    tsne_defaults.update(tsne_kwargs)
    tsne_model = TSNE(n_components=n_components, **tsne_defaults)
    tsne_data = tsne_model.fit_transform(feature_data)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    if color_col in clustered_data.columns:
        colors = clustered_data.loc[feature_data.index, color_col]
        scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], 
                            c=colors, cmap='nipy_spectral', alpha=0.6,
                            s=rcParams['lines.markersize']/2)
        # plt.colorbar(scatter, label=color_col)
    else:
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], alpha=0.6,
                    s=rcParams['lines.markersize']/2)
    
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    plt.title('Clusters in t-SNE Space')
    # plt.show()
    
    return tsne_model, tsne_data

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
                               cluster_col='cluster_id', max_shift=5, verbose=True):
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
    verbose : bool, default True
        Whether to print progress messages
    
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
        if verbose:
            print(f"\nProcessing cluster {cluster_id}...")
        
        # Get ROIs in this cluster
        cluster_rois = df_with_clusters[df_with_clusters[cluster_col] == cluster_id]
        if verbose:
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
        
        if verbose:
            print(f"  RF data shape: {cluster_rfs.shape}")
        
        # Align RFs by centroids for each color channel
        n_rois, n_colors, height, width = cluster_rfs.shape
        aligned_rfs = np.zeros_like(cluster_rfs)
        
        for color in range(n_colors):
            # Calculate reference centroid (mean of all ROIs for this color)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
                ref_cent_x = np.nanmean(cluster_centroids_x[:, color])
                ref_cent_y = np.nanmean(cluster_centroids_y[:, color])
            
            if verbose:
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            cluster_averages[cluster_id] = {
                'averaged_rf': avg_rf,
                'n_rois': n_rois,
                'reference_centroids_x': [np.nanmean(cluster_centroids_x[:, c]) for c in range(n_colors)],
                'reference_centroids_y': [np.nanmean(cluster_centroids_y[:, c]) for c in range(n_colors)],
                'roi_info': cluster_rois[['recording_id', 'roi_id']].values
            }
        
        if verbose:
            print(f"    Created averaged RF with shape: {avg_rf.shape}")
    
    if verbose:
        print(f"\nCompleted averaging for {len(cluster_averages)} clusters")
    return cluster_averages


def create_cluster_averaged_strfs(df_with_clusters, experiment_obj, 
                                 cluster_col='cluster_id', max_shift=5):
    """
    Create averaged spacetime kernel (STRF) representations for each cluster, aligned by centroids.
    
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
        Dictionary with cluster_id as keys, averaged STRFs as values
    """
    from scipy.ndimage import shift
    import warnings
    
    cluster_averages = {}
    
    # Get unique clusters (excluding NaN)
    clusters = df_with_clusters[cluster_col].dropna().unique()
    
    for cluster_id in clusters:
        print(f"\nProcessing cluster {cluster_id} STRFs...")
        
        # Get ROIs in this cluster
        cluster_rois = df_with_clusters[df_with_clusters[cluster_col] == cluster_id]
        print(f"  Found {len(cluster_rois)} ROIs in cluster")
        
        # Collect STRF data and centroids for this cluster
        cluster_strfs = []
        cluster_centroids_x = []
        cluster_centroids_y = []
        
        for _, roi_row in cluster_rois.iterrows():
            recording_id = int(roi_row['recording_id'])
            roi_id = int(roi_row['roi_id'])
            
            # Get the recording object
            recording = experiment_obj.recording[recording_id]
            
            # Get STRF data for this ROI: shape depends on multicolor setup
            # Typically: (color*time, y, x) or (time, y, x)
            roi_strf = recording.strfs[roi_id]
            
            # Get centroids for alignment (using collapsed RF centroids)
            rf_data = recording.collapse_times_by_channel()
            centroids_x = recording.get_pca_centroidsX_by_channel()
            centroids_y = recording.get_pca_centroidsY_by_channel()
            
            # Extract centroids for this specific ROI
            roi_cent_x = centroids_x[:, roi_id]  # Shape: (color,)
            roi_cent_y = centroids_y[:, roi_id]  # Shape: (color,)
            
            cluster_strfs.append(roi_strf)
            cluster_centroids_x.append(roi_cent_x)
            cluster_centroids_y.append(roi_cent_y)
        
        if not cluster_strfs:
            continue
            
        # Convert to numpy arrays
        cluster_strfs = np.array(cluster_strfs)  # Shape: (n_rois, time*color, y, x)
        cluster_centroids_x = np.array(cluster_centroids_x)  # Shape: (n_rois, color)
        cluster_centroids_y = np.array(cluster_centroids_y)  # Shape: (n_rois, color)
        
        print(f"  STRF data shape: {cluster_strfs.shape}")
        
        # Calculate reference centroid (mean of all ROIs)
        # Use the first color channel for alignment reference
        ref_cent_x = np.nanmean(cluster_centroids_x[:, 0])
        ref_cent_y = np.nanmean(cluster_centroids_y[:, 0])
        
        print(f"  Reference centroid = ({ref_cent_x:.1f}, {ref_cent_y:.1f})")
        
        # Align STRFs by rolling the spatial dimensions
        n_rois = cluster_strfs.shape[0]
        aligned_strfs = np.zeros_like(cluster_strfs)
        
        for roi in range(n_rois):
            # Calculate shift needed to align to reference (using first color)
            shift_x = ref_cent_x - cluster_centroids_x[roi, 0]
            shift_y = ref_cent_y - cluster_centroids_y[roi, 0]
            
            # Limit shifts to prevent excessive movement
            shift_x = np.clip(shift_x, -max_shift, max_shift)
            shift_y = np.clip(shift_y, -max_shift, max_shift)
            
            # Apply shift to entire spacetime kernel
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aligned_strfs[roi] = shift(
                    cluster_strfs[roi], 
                    shift=[0, shift_y, shift_x],  # Don't shift time dimension
                    mode='constant', 
                    cval=0
                )
        
        # Average aligned STRFs
        avg_strf = np.nanmean(aligned_strfs, axis=0)
        
        # Store result
        cluster_averages[cluster_id] = {
            'averaged_strf': avg_strf,
            'n_rois': n_rois,
            'reference_centroid_x': ref_cent_x,
            'reference_centroid_y': ref_cent_y,
            'roi_info': cluster_rois[['recording_id', 'roi_id']].values
        }
        
        print(f"    Created averaged STRF with shape: {avg_strf.shape}")
    
    print(f"\nCompleted STRF averaging for {len(cluster_averages)} clusters")
    return cluster_averages


def extract_cluster_timecourses(df_with_clusters, experiment_obj, verbose=True, **params):
    """
    Extract RGBU timecourses for each cluster using get_timecourses_dominant_by_channel().
    
    Parameters:
    -----------
    df_with_clusters : pd.DataFrame
        DataFrame with cluster assignments
    experiment_obj : Experiment
        Experiment object containing recording data
    verbose : bool, default True
        Whether to print progress messages
    **params
        Additional parameters (for compatibility, but not used)
    
    Returns:
    --------
    dict
        Dictionary with cluster_id as keys, containing:
        - 'timecourses': RGBU timecourses (4, time) averaged across cluster
        - 'n_rois': number of ROIs in cluster
        - 'individual_timecourses': list of individual ROI timecourses for verification
    """
    cluster_timecourses = {}
    
    # Get unique cluster IDs
    cluster_ids = sorted(df_with_clusters['cluster_id'].unique())
    
    for cluster_id in cluster_ids:
        if verbose:
            print(f"Extracting RGBU timecourses for cluster {cluster_id}...")
        
        # Get ROIs in this cluster
        cluster_rois = df_with_clusters[df_with_clusters['cluster_id'] == cluster_id]
        n_rois = len(cluster_rois)
        
        if n_rois == 0:
            if verbose:
                print(f"  Warning: No ROIs in cluster {cluster_id}")
            continue
        
        # Collect timecourses by color channel for this cluster
        color_timecourses = [[] for _ in range(4)]  # List for each color (R, G, B, UV)
        individual_timecourses = []
        
        for _, roi_row in cluster_rois.iterrows():
            recording_id = int(roi_row['recording_id'])
            roi_id = int(roi_row['roi_id'])
            
            try:
                recording = experiment_obj.recording[recording_id]
                
                # Use get_timecourses_dominant_by_channel()
                # This returns (colour, roi, time) - exactly what we need!
                timecourses_by_channel = recording.get_timecourses_dominant_by_channel()
                
                # Extract timecourses for this specific ROI across all colors
                roi_rgbu = []
                for color_idx in range(4):
                    roi_timecourse = timecourses_by_channel[color_idx, roi_id, :]
                    color_timecourses[color_idx].append(roi_timecourse)
                    roi_rgbu.append(roi_timecourse)
                
                individual_timecourses.append(np.array(roi_rgbu))  # (4, time)
                
            except Exception as e:
                if verbose:
                    print(f"    Error processing ROI {roi_id} from recording {recording_id}: {e}")
                continue
        
        if not any(color_timecourses):
            if verbose:
                print(f"  Warning: No valid timecourses for cluster {cluster_id}")
            continue
        
        # Average timecourses within each color channel
        avg_timecourses = []
        for color_idx in range(4):
            if color_timecourses[color_idx]:
                color_array = np.array(color_timecourses[color_idx])  # (n_rois, time)
                avg_timecourse = np.mean(color_array, axis=0)  # (time,)
                avg_timecourses.append(avg_timecourse)
            else:
                # No data for this color - create zeros
                n_timepoints = len(color_timecourses[0][0]) if color_timecourses[0] else 400
                avg_timecourses.append(np.zeros(n_timepoints))
        
        avg_timecourses = np.array(avg_timecourses)  # (4, time)
        
        cluster_timecourses[cluster_id] = {
            'timecourses': avg_timecourses,
            'n_rois': n_rois,
            'individual_timecourses': individual_timecourses
        }
        
        if verbose:
            print(f"  Extracted RGBU timecourses shape: {avg_timecourses.shape} from {n_rois} ROIs")
    
    return cluster_timecourses


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
                    def normalize_channels_proportional(r_channel, g_channel, b_channel):
                        # Find global max across all channels to preserve relative magnitudes
                        all_channels = [r_channel, g_channel, b_channel]
                        global_max = max(np.max(np.abs(channel)) for channel in all_channels if channel is not None)
                        
                        if global_max > 0:
                            r_norm = (r_channel + global_max) / (2 * global_max) if r_channel is not None else np.zeros_like(g_channel or b_channel)
                            g_norm = (g_channel + global_max) / (2 * global_max) if g_channel is not None else np.zeros_like(r_channel or b_channel)
                            b_norm = (b_channel + global_max) / (2 * global_max) if b_channel is not None else np.zeros_like(r_channel or g_channel)
                        else:
                            r_norm = np.zeros_like(r_channel or g_channel or b_channel)
                            g_norm = np.zeros_like(r_channel or g_channel or b_channel)
                            b_norm = np.zeros_like(r_channel or g_channel or b_channel)
                            
                        return r_norm, g_norm, b_norm
                    
                    r_norm, g_norm, b_norm = normalize_channels_proportional(r_channel, g_channel, b_channel)
                    
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
        # plt.show()
        
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


# def plot_all_cluster_averages(cluster_averages,
#                             figsize_per_cluster=(6, 1),
#                             use_color_mapping=True,
#                             rgb_mode='old',
#                             save_path=None):
#     """
#     Plot averaged RFs for all clusters with RGB composites.

#     Parameters:
#     -----------
#     cluster_averages : dict
#         Result from create_cluster_averaged_rfs()
#     figsize_per_cluster : tuple
#         Size per cluster row in inches
#     use_color_mapping : bool
#         Whether to apply color mapping to channels
#     rgb_mode : str
#         RGB visualization mode: 'new' (my preferred method) or 'old' (bipolar method)
#     save_path : str, optional
#         Path to save the figure
#     """
#     import matplotlib.pyplot as plt
    
#     n_clusters = len(cluster_averages)
#     n_total_cols = 6  # Red, Green, Blue, UV, RGB, RGU

#     if n_clusters == 0:
#         print("No clusters to plot!")
#         return

#     # Calculate figure size and subplot grid
#     fig_width = figsize_per_cluster[0] * n_total_cols / 3
#     fig_height = figsize_per_cluster[1] * n_clusters

#     fig, axes = plt.subplots(n_clusters, n_total_cols,
#                             figsize=(fig_width, fig_height))

#     # Handle single cluster case
#     if n_clusters == 1:
#         axes = axes.reshape(1, -1)

#     # Sort clusters by ID for consistent ordering
#     sorted_clusters = sorted(cluster_averages.keys())

#     # Define color maps for individual channels
#     if use_color_mapping:
#         try:
#             from pygor.plotting.custom import maps_concat
#             if isinstance(maps_concat, list):
#                 colormaps = maps_concat[:4]  # First 4 for individual channels
#             else:
#                 colormaps = ['Reds', 'Greens', 'Blues', 'Purples']
#         except ImportError:
#             colormaps = ['Reds', 'Greens', 'Blues', 'Purples']
#     else:
#         colormaps = ['gray'] * 4

#     channel_names = ['Red', 'Green', 'Blue', 'UV', 'RGB', 'RGU']

#     for row, cluster_id in enumerate(sorted_clusters):
#         curr_clust_avgs = cluster_averages[cluster_id]["averaged_rf"]
#         n_rois = cluster_averages[cluster_id]["n_rois"]

#         for col in range(n_total_cols):
#             ax = axes[row, col] if n_clusters > 1 else axes[col]
#             curr_clim = np.max(np.abs(curr_clust_avgs)) if use_color_mapping else None
            
#             if col < 4:  # Individual channels (Red, Green, Blue, UV)
#                 cmap = colormaps[col] if use_color_mapping else 'gray'
#                 ax.imshow(curr_clust_avgs[col], cmap=cmap,
#                         vmin=-curr_clim, vmax=curr_clim, origin="lower")
                
#                 # Add crosshairs
#                 centre = (curr_clust_avgs.shape[1] // 2, curr_clust_avgs.shape[2] // 2)
#                 ax.axhline(centre[0], color='white', lw=0.5)
#                 ax.axvline(centre[1], color='white', lw=0.5)

#             elif col == 4:  # RGB composite [0,1,2]
#                 if rgb_mode == 'new':
#                     rgb_composite = create_rgb_composite(curr_clust_avgs, [0, 1, 2])
#                     ax.imshow(rgb_composite, origin="lower")  # No clim - auto equalize
#                 else:  # rgb_mode == 'old'
#                     rgb_composite = _create_rgb_composite_old(curr_clust_avgs, [0, 1, 2])
#                     ax.imshow(rgb_composite, origin="lower")

#             elif col == 5:  # RGU composite [0,1,3]
#                 if rgb_mode == 'new':
#                     rgu_composite = create_rgb_composite(curr_clust_avgs, [0, 1, 3])
#                     ax.imshow(rgu_composite, origin="lower")  # No clim - auto equalize
#                 else:  # rgb_mode == 'old'
#                     rgu_composite = _create_rgb_composite_old(curr_clust_avgs, [0, 1, 3])
#                     ax.imshow(rgu_composite, origin="lower")

#             # Add titles only to top row
#             if row == 0:
#                 ax.set_title(channel_names[col], fontsize=8)

#             # Add cluster info to leftmost column
#             if col == 0:
#                 ax.set_ylabel(f'Cluster {cluster_id}\n(n={n_rois})', fontsize=8)

#             ax.axis('off')

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Saved cluster averages plot to {save_path}")
 
#     # plt.show()

#     # Print summary
#     print(f"\nDisplayed {n_clusters} clusters:")
#     for cluster_id in sorted_clusters:
#         n_rois = cluster_averages[cluster_id]["n_rois"]
#         print(f"  Cluster {cluster_id}: {n_rois} ROIs")


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
    def normalize_channels_proportional(r_channel, g_channel, b_channel):
        # Find global max across all channels to preserve relative magnitudes
        all_channels = [r_channel, g_channel, b_channel]
        global_max = max(np.max(np.abs(channel)) for channel in all_channels if channel is not None)
        
        if global_max > 0:
            r_norm = (r_channel + global_max) / (2 * global_max) if r_channel is not None else np.zeros_like(g_channel or b_channel)
            g_norm = (g_channel + global_max) / (2 * global_max) if g_channel is not None else np.zeros_like(r_channel or b_channel)
            b_norm = (b_channel + global_max) / (2 * global_max) if b_channel is not None else np.zeros_like(r_channel or g_channel)
        else:
            r_norm = np.zeros_like(r_channel or g_channel or b_channel)
            g_norm = np.zeros_like(r_channel or g_channel or b_channel)
            b_norm = np.zeros_like(r_channel or g_channel or b_channel)
            
        return r_norm, g_norm, b_norm
    
    r_norm, g_norm, b_norm = normalize_channels_proportional(r_channel, g_channel, b_channel)
    
    # Stack into RGB
    rgb_composite = np.stack([r_norm, g_norm, b_norm], axis=2)
    
    return rgb_composite

from pygor.plotting.scalebar import add_scalebar

def plot_all_cluster_averages_enhanced(df_with_clusters, experiment_obj, cluster_averages,
                                    cluster_timecourses=None, 
                                    metrics_to_plot=None,
                                    metrics_like=None,
                                    figsize_per_cluster=None,
                                    use_color_mapping=True,
                                    rgb_mode='new',
                                    save_path=None,
                                    show_ipl_distribution=True,
                                    sort_by_ipl_depth=True,
                                    hide_noise_clusters=False,
                                    cluster_label_format="{cluster_id}, n = {n_rois}",
                                    silhouette_scores=None,
                                    verbose=False) -> tuple[plt.Figure, np.ndarray]:
    """
    Enhanced plot showing cluster averages with timecourses, IPL distributions, and metrics.

    Parameters:
    -----------
    df_with_clusters : pd.DataFrame
        Dataframe with cluster assignments
    experiment_obj : Experiment
        Experiment object containing recording data
    cluster_averages : dict
        Result from create_cluster_averaged_rfs()
    cluster_timecourses : dict, optional
        Result from extract_cluster_timecourses()
    metrics_to_plot : list, optional
        List of metric column names to plot (e.g., ['space_amps_0', 'areas_0'])
    metrics_like : list, optional
        List of metric prefixes to group and plot (e.g., ['areas', 'space_amps'])
        Each prefix will create a separate column showing all matching metrics
    figsize_per_cluster : tuple
        Size per cluster row in inches
    use_color_mapping : bool
        Whether to apply color mapping to channels
    rgb_mode : str
        RGB visualization mode: 'new' or 'old'
    save_path : str, optional
        Path to save the figure
    show_ipl_distribution : bool
        Whether to show IPL distribution plots
    sort_by_ipl_depth : bool, default True
        Whether to sort clusters by mean IPL depth (shallow to deep). If False, sorts by cluster ID.
    hide_noise_clusters : bool, default False
        Whether to hide clusters with low amplitude timecourses (likely noise clusters). These clusters are identified as having maximum timecourse amplitude < 1.0.
    verbose : bool
        Whether to print debug information
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # First, determine which clusters to show (filter noise clusters if requested)
    all_clusters = list(cluster_averages.keys())
    
    # Sort clusters by IPL stratification pattern (shallow, bistratified, deep)
    def get_cluster_ipl_stats(cluster_id):
        """Calculate IPL depth statistics for a cluster to detect bistratification"""
        cluster_rois = df_with_clusters[df_with_clusters['cluster_id'] == cluster_id]
        ipl_depths = []
        
        for _, roi_row in cluster_rois.iterrows():
            recording_id = int(roi_row['recording_id'])
            roi_id = int(roi_row['roi_id'])
            recording = experiment_obj.recording[recording_id]
            
            if hasattr(recording, 'ipl_depths') and recording.ipl_depths is not None:
                if roi_id < len(recording.ipl_depths):
                    ipl_depth = recording.ipl_depths[roi_id]
                    if not np.isnan(ipl_depth):
                        ipl_depths.append(ipl_depth)
        
        if not ipl_depths:
            return {'mean': float('inf'), 'std': 0, 'is_bistratified': False, 'category': 'unknown'}
        
        ipl_depths = np.array(ipl_depths)
        mean_depth = np.mean(ipl_depths)
        std_depth = np.std(ipl_depths)
        
        # Use mean depth for sorting
        # (modal and median calculations removed for simplicity)
        
        # Detect bistratified clusters: high std AND presence in both shallow and deep layers
        is_bistratified = False
        if len(ipl_depths) >= 3:  # Need enough data points
            shallow_count = np.sum(ipl_depths < 35)  # Expanded shallow layer (OFF)
            deep_count = np.sum(ipl_depths > 45)     # Expanded deep layer (ON)
            total_count = len(ipl_depths)
            
            # More stringent bistratified detection - require substantial presence in both layers
            if (std_depth > 15 and  # Higher std threshold
                shallow_count/total_count > 0.25 and  # Higher presence threshold (25%)
                deep_count/total_count > 0.25 and     # Higher presence threshold (25%)
                min(shallow_count, deep_count) >= 5): # Minimum absolute count in each layer
                is_bistratified = True
        
        # Categorize for sorting - use median depth instead of mean
        # Order: deep → bistratified → shallow, with bistratified forced to middle
        if is_bistratified:
            category = 'bistratified'
            # Force bistratified to middle by using a fixed middle position (45) plus small offset based on std
            sort_key = (1, 45 + std_depth/10)  # Group bistratified in middle, sort by variability
        elif mean_depth >= 45:  # Deep/ON-dominant
            category = 'deep'  
            sort_key = (0, -mean_depth)  # Deep first, then by descending mean depth
        else:  # Shallow/OFF-dominant
            category = 'shallow'
            sort_key = (2, -mean_depth)  # Shallow last, then by descending mean depth
            
        return {
            'mean': mean_depth, 
            'std': std_depth,
            'is_bistratified': is_bistratified,
            'category': category,
            'sort_key': sort_key
        }
    
    # Get all cluster IDs and their IPL statistics
    cluster_ipl_stats = {cid: get_cluster_ipl_stats(cid) for cid in all_clusters}
    
    # Filter out noise clusters if requested
    if hide_noise_clusters and cluster_timecourses is not None:
        # Keep only clusters with high amplitude timecourses (amplitude >= 1.0)
        def is_noise_cluster(cluster_id):
            """Check if cluster has low amplitude timecourses (< 1.0)"""
            if cluster_id not in cluster_timecourses:
                return True  # No timecourse data = noise
            
            timecourses = cluster_timecourses[cluster_id]['timecourses']
            if timecourses is None or len(timecourses) == 0:
                return True  # No timecourse data = noise
            
            # Get max amplitude across all channels/colors
            max_amplitude = 0
            if isinstance(timecourses, dict):
                # Dictionary format: {color: array}
                for color_timecourses in timecourses.values():
                    if color_timecourses is not None and len(color_timecourses) > 0:
                        max_amplitude = max(max_amplitude, np.max(np.abs(color_timecourses)))
            else:
                # Array format: direct numpy array
                if isinstance(timecourses, np.ndarray):
                    max_amplitude = np.max(np.abs(timecourses))
                else:
                    return True  # Unknown format = noise
            
            return max_amplitude < 1  # Low amplitude = noise
        
        filtered_clusters = [cid for cid in all_clusters if not is_noise_cluster(cid)]
        noise_clusters = [cid for cid in all_clusters if is_noise_cluster(cid)]
        
        if noise_clusters:
            print(f"\nHiding {len(noise_clusters)} noise clusters (low amplitude < 1.0):")
            for cid in noise_clusters:
                n_rois = cluster_averages[cid]["n_rois"]
                print(f"  Cluster {cid}: {n_rois} ROIs")
        else:
            print("No noise clusters detected.")
    else:
        filtered_clusters = all_clusters
    
    # Sort the filtered clusters
    if sort_by_ipl_depth:
        # Sort by IPL stratification pattern: shallow -> bistratified -> deep
        sorted_clusters = sorted(filtered_clusters, key=lambda cid: cluster_ipl_stats[cid]['sort_key'])
        if verbose:
            print("IPL sorting categories:")
            for cid in sorted_clusters:
                stats = cluster_ipl_stats[cid] 
                print(f"  Cluster {cid}: {stats['category']} (mean={stats['mean']:.1f}, std={stats['std']:.1f})")
    else:
        # Sort by ID for consistent ordering
        sorted_clusters = sorted(filtered_clusters)

    n_clusters = len(sorted_clusters)
    if n_clusters == 0:
        print("No clusters to plot!")
        return

    # Calculate number of columns needed
    n_spatial_cols = 6  # Red, Green, Blue, UV, RGB, RGU
    n_extra_cols = 0
    
    if cluster_timecourses is not None:
        n_extra_cols += 1  # Timecourses column
    if show_ipl_distribution:
        n_extra_cols += 1  # IPL distribution column
    if metrics_to_plot is not None and len(metrics_to_plot) > 0:
        n_extra_cols += 1  # Single metrics column
    if metrics_like is not None and len(metrics_like) > 0:
        n_extra_cols += len(metrics_like)  # One column per metric group
    
    n_total_cols = n_spatial_cols + n_extra_cols

    # Calculate figure size
    if figsize_per_cluster is None:
        # Automatic scaling to maintain good aspect ratios
        # Target: Make spatial RF subplots roughly square
        
        # The spatial RF images are the limiting factor for aspect ratio
        # Assume RF images are roughly square, so we want square subplots for them
        base_height_per_cluster = 1.0  # Base height per cluster
        
        # Calculate width needed for spatial columns (first 6) to be square
        spatial_width_per_col = base_height_per_cluster  # Square aspect for RF images
        spatial_total_width = n_spatial_cols * spatial_width_per_col
        
        # Extra columns can be wider since they're line plots/histograms
        extra_width_per_col = base_height_per_cluster * 1.5  # Slightly wider for line plots
        extra_total_width = n_extra_cols * extra_width_per_col
        
        fig_width = spatial_total_width + extra_total_width
        fig_height = max(1.0, n_clusters * base_height_per_cluster)
        
        # Add some padding for titles and labels
        fig_width *= 1.1  # 10% padding
        fig_height *= 1.1
    else:
        fig_width = figsize_per_cluster[0]
        fig_height = figsize_per_cluster[1] * n_clusters

    fig, axes = plt.subplots(n_clusters, n_total_cols, figsize=(fig_width, fig_height),
                        sharex="col", layout = "constrained")

    # Keep track of column indices for sharing x-axes
    timecourse_col_idx = None
    ipl_col_idx = None  
    metrics_col_idx = None
    metrics_like_col_indices = []

    # Handle single cluster case
    if n_clusters == 1:
        axes = axes.reshape(1, -1)

    # Define color maps for individual channels
    if use_color_mapping:
        try:
            from pygor.plotting.custom import maps_concat
            if isinstance(maps_concat, list):
                colormaps = maps_concat[:4]
            else:
                colormaps = ['Reds', 'Greens', 'Blues', 'Purples']
        except ImportError:
            colormaps = ['Reds', 'Greens', 'Blues', 'Purples']
    else:
        colormaps = ['gray'] * 4

    # Column headers
    spatial_names = ['Red', 'Green', 'Blue', 'UV', 'RGB', 'RGU']
    extra_names = []
    if cluster_timecourses is not None:
        extra_names.append('Timecourses')
    if show_ipl_distribution:
        extra_names.append('IPL')
    if metrics_to_plot is not None and len(metrics_to_plot) > 0:
        extra_names.append('Metrics')
    if metrics_like is not None and len(metrics_like) > 0:
        for metric_prefix in metrics_like:
            extra_names.append(f'{metric_prefix.title()} Metrics')
    
    all_names = spatial_names + extra_names

    for row, cluster_id in enumerate(sorted_clusters):
        curr_clust_avgs = cluster_averages[cluster_id]["averaged_rf"]
        n_rois = cluster_averages[cluster_id]["n_rois"]
        cluster_rois = df_with_clusters[df_with_clusters['cluster_id'] == cluster_id]
        
        # DEBUG: Check what data we're actually using
        if verbose and row == 0:  # Only print for first cluster
            print(f"\n=== CLUSTER {cluster_id} DATA CHECK ===")
            areas_cols = [col for col in cluster_rois.columns if 'areas_' in col]
            if areas_cols:
                sample_areas = cluster_rois[areas_cols[0]].dropna().head(3).tolist()
                print(f"cluster_rois sample areas: {sample_areas}")
                print(f"cluster_rois areas range: {cluster_rois[areas_cols[0]].min():.3f} to {cluster_rois[areas_cols[0]].max():.3f}")
            time_cols = [col for col in cluster_rois.columns if 'time_amps_' in col]
            if time_cols:
                sample_time = cluster_rois[time_cols[0]].dropna().head(3).tolist()
                print(f"cluster_rois sample time_amps: {sample_time}")
                print(f"cluster_rois time_amps range: {cluster_rois[time_cols[0]].min():.3f} to {cluster_rois[time_cols[0]].max():.3f}")

        col_idx = 0
        
        # Plot spatial RF components (columns 0-5)
        for spatial_col in range(n_spatial_cols):
            ax = axes[row, col_idx] if n_clusters > 1 else axes[col_idx]
            # curr_clim = np.max(np.abs(curr_clust_avgs)) if use_color_mapping else None
            curr_clim = np.percentile(np.abs(curr_clust_avgs), 99) if use_color_mapping else None
            # Add cluster label on the leftmost column
            if spatial_col == 0:
                # Calculate percentage of total cells
                total_cells = sum(cluster_averages[cid]["n_rois"] for cid in cluster_averages.keys())
                percentage = (n_rois / total_cells) * 100 if total_cells > 0 else 0
                
                # Format the cluster label with available variables
                silhouette_score = silhouette_scores.get(cluster_id, "N/A") if silhouette_scores else "N/A"
                label_text = cluster_label_format.format(
                    cluster_id=cluster_id, 
                    n_rois=n_rois,
                    percentage=f"{percentage:.1f}%",
                    silhouette_score=f"{silhouette_score:.3f}" if isinstance(silhouette_score, (int, float)) else silhouette_score
                )
                ax.text(-0.15, 0.5, label_text, transform=ax.transAxes, 
                       verticalalignment='center', horizontalalignment='right',
                       fontsize=plt.rcParams['font.size'] * 0.8)
            
            if spatial_col < 4:  # Individual channels
                cmap = colormaps[spatial_col] if use_color_mapping else 'gray'
                ax.imshow(curr_clust_avgs[spatial_col], cmap=cmap,
                        vmin=-curr_clim, vmax=curr_clim, origin="lower")
                
                # Add crosshairs
                # centre = (curr_clust_avgs.shape[1] // 2, curr_clust_avgs.shape[2] // 2)
                # ax.axhline(centre[0], color='white', lw=0.5)
                # ax.axvline(centre[1], color='white', lw=0.5)

            elif spatial_col == 4:  # RGB composite
                if rgb_mode == 'new':
                    rgb_composite = create_rgb_composite(curr_clust_avgs, [0, 1, 2], mode='proportional')
                    ax.imshow(rgb_composite, origin="lower")
                elif rgb_mode == 'grey_baseline':
                    rgb_composite = create_rgb_composite(curr_clust_avgs, [0, 1, 2], mode='grey_baseline')
                    ax.imshow(rgb_composite, origin="lower")
                else:  # 'old' mode
                    rgb_composite = _create_rgb_composite_old(curr_clust_avgs, [0, 1, 2])
                    ax.imshow(rgb_composite, origin="lower")

            elif spatial_col == 5:  # RGU composite
                if rgb_mode == 'new':
                    rgu_composite = create_rgb_composite(curr_clust_avgs, [0, 1, 3], mode='proportional')
                    ax.imshow(rgu_composite, origin="lower")
                elif rgb_mode == 'grey_baseline':
                    rgu_composite = create_rgb_composite(curr_clust_avgs, [0, 1, 3], mode='grey_baseline')
                    ax.imshow(rgu_composite, origin="lower")
                else:  # 'old' mode
                    rgu_composite = _create_rgb_composite_old(curr_clust_avgs, [0, 1, 3])
                    ax.imshow(rgu_composite, origin="lower")

            # Add titles only to top row
            if row == 0:
                ax.set_title(spatial_names[spatial_col], fontsize=rcParams['font.size']*0.8)

            # Add cluster info to leftmost column
            if spatial_col == 0:
                ax.set_ylabel(f'Cluster {cluster_id}\n(n={n_rois})', fontsize=rcParams['font.size']*0.8)

            ax.axis('off')
            col_idx += 1

        # Plot timecourses if available
        if cluster_timecourses is not None:
            if row == 0:  # Track column index on first row
                timecourse_col_idx = col_idx
            ax = axes[row, col_idx] if n_clusters > 1 else axes[col_idx]
            
            if cluster_id in cluster_timecourses:
                timecourses = cluster_timecourses[cluster_id]['timecourses']
                # plt.plot(timecourses.T) 
                # Plot center/surround timecourses
                time_axis = np.arange(timecourses.shape[-1])
                
                # Typically: index 0=center, 1=surround, 2=noise
                # labels = ['Center', 'Surround', 'Noise', 'Non-center']
                from pygor.plotting.custom import fish_palette
                colors = fish_palette

                for i, color in enumerate(colors):
                    if i < timecourses.shape[0]:
                        if np.ma.is_masked(timecourses) and np.ma.is_masked(timecourses[i]):
                            ax.plot(time_axis, timecourses[i].data, '--', 
                                color=color, alpha=0.5)
                        else:
                            ax.plot(time_axis, timecourses[i], '-', 
                                color=color)

                ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
                # ax.set_xlabel('Time')
                ax.set_ylabel('')
                # ax.set_xticks([])
                
            else:
                ax.text(0.5, 0.5, 'No timecourse\ndata', ha='center', va='center',
                    transform=ax.transAxes)
                ax.axis('off')
            ax.set_yticks([])
            if row == 0:
                ax.set_title('Times', fontsize=8)
            col_idx += 1
            
        # Plot IPL distribution if requested
        if show_ipl_distribution:
            if row == 0:  # Track column index on first row
                ipl_col_idx = col_idx
            ax = axes[row, col_idx] if n_clusters > 1 else axes[col_idx]
            
            # Get IPL depths for ROIs in this cluster
            ipl_depths = []
            for _, roi_row in cluster_rois.iterrows():
                recording_id = int(roi_row['recording_id'])
                roi_id = int(roi_row['roi_id'])
                recording = experiment_obj.recording[recording_id]
                
                if hasattr(recording, 'ipl_depths') and recording.ipl_depths is not None:
                    if roi_id < len(recording.ipl_depths):
                        ipl_depth = recording.ipl_depths[roi_id]
                        if not np.isnan(ipl_depth):
                            ipl_depths.append(ipl_depth)
            
            if ipl_depths and len(ipl_depths) > 1:
                # Calculate appropriate binwidth based on data range
                # ipl_range = max(ipl_depths) - min(ipl_depths)
                # if ipl_range > 0:
                #     # Use adaptive binwidth, but ensure reasonable number of bins
                #     binwidth = max(5, ipl_range / 8)  # Use 8 bins or 5-unit bins, whichever is larger
                #     sns.histplot(y=ipl_depths, binwidth=binwidth, alpha=0.7, 
                #             color='steelblue', edgecolor='black', 
                #             stat="percent",
                #             ax=ax)
                # else:
                    # All values are the same - just show a single bar
                sns.histplot(y=ipl_depths, binwidth=10, alpha=0.7, 
                            binrange=(0,100),
                            color='steelblue', edgecolor='black', 
                            stat="percent",
                            ax=ax)
                ax.set_ylim(0, 100)
                # ax.set_xlabel('IPL Depth')
                ax.set_ylabel('')
                ax.set_xlabel('')
                ax.set_yticks([])
                ax.axhline(45, color='black', linestyle='-', alpha=0.1)
                # ax.margins(x=0.9)
                # ax.grid(True, alpha=0.3)

                # Add statistics
            #     mean_ipl = np.mean(ipl_depths)
            #     std_ipl = np.std(ipl_depths)
            #     ax.axvline(mean_ipl, color='red', linestyle='--', 
            #               label=f'Mean: {mean_ipl:.2f}±{std_ipl:.2f}')
            #     ax.legend(fontsize=6)
            # else:
            #     ax.text(0.5, 0.5, 'No IPL\ndata', ha='center', va='center',
            #            transform=ax.transAxes)
            #     ax.axis('off')
            
            if row == 0:
                ax.set_title('IPL', fontsize=8)
                # share y-axis
            col_idx += 1

        # # Plot metrics if requested
        if metrics_to_plot is not None and len(metrics_to_plot) > 0:
            if row == 0:  # Track column index on first row
                metrics_col_idx = col_idx
            ax = axes[row, col_idx] if n_clusters > 1 else axes[col_idx]
            
            # Calculate mean and std for each metric in this cluster
            metric_means = []
            metric_stds = []
            metric_labels = []
            
            for metric in metrics_to_plot:
                if metric in cluster_rois.columns:
                    values = cluster_rois[metric].dropna()
                    if len(values) > 0:
                        metric_means.append(values.mean())
                        metric_stds.append(values.std())
                        metric_labels.append(metric.replace('_', ' ').title())
            
            if metric_means:
                x_pos = np.arange(len(metric_means))
                ax.errorbar(x_pos, metric_means, yerr=metric_stds, 
                           fmt='o-', capsize=3, capthick=1)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=6)
                ax.set_ylabel('Value')
            else:
                ax.text(0.5, 0.5, 'No metrics\ndata', ha='center', va='center',
                       transform=ax.transAxes)
                ax.axis('off')
            
            if row == 0:
                ax.set_title('Metrics', fontsize=8)
            col_idx += 1

        # Plot metrics_like if requested - one column per metric prefix
        if metrics_like is not None and len(metrics_like) > 0:
            for i, metric_prefix in enumerate(metrics_like):
                if row == 0:  # Track column indices on first row
                    metrics_like_col_indices.append(col_idx)
                ax = axes[row, col_idx] if n_clusters > 1 else axes[col_idx]
                
                # Find all columns that match the pattern: prefix or prefix_*
                # This handles cases like "areas_0", "biphasic_index_0", etc.
                matching_metrics = [col for col in cluster_rois.columns 
                                  if col == metric_prefix or 
                                     (col.startswith(metric_prefix) and 
                                      len(col) > len(metric_prefix) and 
                                      col[len(metric_prefix)] == '_')]
                
                if verbose:
                    print(f"    Metric prefix: '{metric_prefix}'")
                    print(f"    Available columns: {list(cluster_rois.columns)}")
                    print(f"    Matching columns: {matching_metrics}")
                    if matching_metrics:
                        sample_values = cluster_rois[matching_metrics[0]].head(3).tolist()
                        print(f"    Sample values from {matching_metrics[0]}: {sample_values}")
                
                # Calculate mean and std for each matching metric
                metric_means = []
                metric_stds = []
                metric_labels = []
                
                for metric in sorted(matching_metrics):  # Sort to get consistent order
                    values = cluster_rois[metric].dropna()
                    if len(values) > 0:
                        metric_means.append(values.mean())
                        metric_stds.append(values.std())
                        # Extract the suffix (e.g., "0" from "areas_0", "0" from "biphasic_index_0")
                        if metric == metric_prefix:
                            suffix = metric  # Use the full name if exact match
                        else:
                            # Get everything after the prefix and first underscore
                            suffix = metric[len(metric_prefix)+1:]
                        metric_labels.append(suffix)
                
                if metric_means:
                    # sns.boxplot(data=cluster_rois[matching_metrics], 
                    #             ax=ax, palette=fish_palette, fliersize = 0,
                    #             orient='h')
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, message="The palette list has more values")
                        sns.stripplot(data=cluster_rois[matching_metrics], 
                            ax=ax, alpha=0.1, 
                            orient='h',
                            # palette="dark:k",
                            palette = fish_palette,
                            edgecolor = "k",
                        size=rcParams['lines.markersize']/3,
                        linewidth=0.1,
                        # size=3,
                        jitter=False)
                        sns.barplot(data=cluster_rois[matching_metrics],
                            ax=ax, alpha=0.5,
                            orient='h',
                            # palette="dark:k",
                            palette = fish_palette,
                            errorbar='sd',
                            capsize=.4,
                            err_kws={"color": "k", "linewidth": 0.5},
                            linewidth=0,  # No edge on bars
                        )
                        ax.set_yticks([])
                    # Use global axis limits across all clusters for this metric
                    if matching_metrics:
                        # Get global data range across ALL clusters for this metric
                        global_values = df_with_clusters[matching_metrics].values.flatten()
                        global_values = global_values[~np.isnan(global_values)]
                        percentile = 99.5
                        if len(global_values) > 0:
                            vmin = np.percentile(global_values, 100-percentile)
                            vmax = np.percentile(global_values, percentile)
                            ax.set_xlim(vmin, vmax)

                    if verbose:
                        print(f"    Data range: min={df_with_clusters[matching_metrics].min().min():.3f}, max={df_with_clusters[matching_metrics].max().max():.3f}")
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.axis('off')
                
                if row == 0:
                    ax.set_title(f'{metric_prefix.title()} Metrics', fontsize=8)
                col_idx += 1

    # Share x- and y-axes within each metric column type for easier comparison
    # Only relevant when we have multiple rows
    # if n_clusters > 1:
    #     # Share axes for metrics_like columns
    #     for metric_col_idx in metrics_like_col_indices:
    #         try:
    #             column_axes = [axes[row, metric_col_idx] for row in range(n_clusters)]
    #         except Exception:
    #             # Defensive: if axes layout unexpected, skip
    #             continue

    #         # Share x and y with the top row axis
    #         ref_ax = column_axes[0]
    #         for ax_shared in column_axes[1:]:
    #             try:
    #                 ax_shared.sharex(ref_ax)
    #                 ax_shared.sharey(ref_ax)
    #             except Exception:
    #                 pass

            # Align y-limits across the column for consistent scaling
            # try:
            #     ymins = [ax.get_ylim()[0] for ax in column_axes]
            #     ymaxs = [ax.get_ylim()[1] for ax in column_axes]
            #     ymin = np.nanmin(ymins)
            #     ymax = np.nanmax(ymaxs)
            #     for ax_shared in column_axes:
            #         ax_shared.set_ylim(ymin, ymax)
            # except Exception:
            #     pass

        # # Also share axes for the single metrics column if present
        # if metrics_col_idx is not None:
        #     try:
        #         column_axes = [axes[row, metrics_col_idx] for row in range(n_clusters)]
        #         ref_ax = column_axes[0]
        #         for ax_shared in column_axes[1:]:
        #             try:
        #                 ax_shared.sharex(ref_ax)
        #                 ax_shared.sharey(ref_ax)
        #             except Exception:
        #                 pass

        #         # Align y-limits
        #         ymins = [ax.get_ylim()[0] for ax in column_axes]
        #         ymaxs = [ax.get_ylim()[1] for ax in column_axes]
        #         ymin = np.nanmin(ymins)
        #         ymax = np.nanmax(ymaxs)
        #         for ax_shared in column_axes:
        #             ax_shared.set_ylim(ymin, ymax)
        #     except Exception:
        #         pass
    sns.despine(left=False, bottom=False)
    # plt.tight_layout(constrained_layout=True, padding=0.4)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced cluster averages plot to {save_path}")



    # plt.show()

    # Print summary
    print(f"\nDisplayed {n_clusters} clusters:")
    for cluster_id in sorted_clusters:
        n_rois = cluster_averages[cluster_id]["n_rois"]
        print(f"  Cluster {cluster_id}: {n_rois} ROIs")
    return fig, axes


def get_cluster_order_from_result(clustering_result, data, 
                                 sort_by_ipl_depth=True, hide_noise_clusters=False, verbose=False):
    """
    Get the cluster ordering that plot_cluster_results would use.
    
    Parameters:
    -----------
    clustering_result : dict
        Result from clustering function
    data : pd.DataFrame
        Data with 'depths' column for IPL depth information
    sort_by_ipl_depth : bool
        Whether to sort by IPL depth (same as plot_cluster_results)
    hide_noise_clusters : bool
        Whether to hide noise clusters (same as plot_cluster_results)
    
    Returns:
    --------
    list : Cluster IDs in the same order plot_cluster_results would use
    """
    clustered_data = clustering_result['clustered_data']
    all_clusters = sorted(clustered_data['cluster_id'].unique())
    
    # Check for depths column in data
    if 'depths' not in data.columns or not sort_by_ipl_depth:
        # No depths data or not sorting - just return sorted cluster IDs
        return all_clusters
    
    # Calculate IPL statistics (copied from plot_cluster_results logic)
    cluster_ipl_stats = {}
    
    for cluster_id in all_clusters:
        cluster_rois = clustered_data[clustered_data['cluster_id'] == cluster_id].index
        
        # Get depths directly from the data DataFrame using the cluster ROI indices
        cluster_depths = data.loc[cluster_rois, 'depths'].dropna().values
        
        if len(cluster_depths) == 0:
            continue
            
        mean_depth = cluster_depths.mean()
        std_depth = cluster_depths.std()
        
        # Bistratified detection (same logic as plot_cluster_results)
        depth_counts, depth_bins = np.histogram(cluster_depths, bins=20, range=(0, 90))
        peak_indices = []
        for i in range(1, len(depth_counts) - 1):
            if depth_counts[i] > depth_counts[i-1] and depth_counts[i] > depth_counts[i+1]:
                peak_indices.append(i)
        
        significant_peaks = [i for i in peak_indices if depth_counts[i] >= len(cluster_depths) * 0.1]
        is_bistratified = len(significant_peaks) >= 2
        
        # Categorize for sorting (same logic as plot_cluster_results)
        if is_bistratified:
            category = 'bistratified'
            sort_key = (1, 45 + std_depth/10)
        elif mean_depth >= 45:
            category = 'deep'  
            sort_key = (0, -mean_depth)
        else:
            category = 'shallow'
            sort_key = (2, -mean_depth)
            
        cluster_ipl_stats[cluster_id] = {
            'mean': mean_depth,
            'std': std_depth,
            'category': category,
            'sort_key': sort_key
        }
    
    # Filter noise clusters if requested (same logic as plot_cluster_results)
    if hide_noise_clusters:
        # This would require access to timecourse data - skip for now
        filtered_clusters = all_clusters
    else:
        filtered_clusters = all_clusters
    
    # Sort clusters - only include clusters that have IPL stats
    if sort_by_ipl_depth:
        clusters_with_stats = [cid for cid in filtered_clusters if cid in cluster_ipl_stats]
        clusters_without_stats = [cid for cid in filtered_clusters if cid not in cluster_ipl_stats]
        
        sorted_clusters = sorted(clusters_with_stats, key=lambda cid: cluster_ipl_stats[cid]['sort_key'])
        # Add clusters without stats at the end
        sorted_clusters.extend(sorted(clusters_without_stats))
        if verbose:
            print("IPL sorting categories:")
            for cid in sorted_clusters:
                if cid in cluster_ipl_stats:
                    stats = cluster_ipl_stats[cid] 
                    print(f"  Cluster {cid}: {stats['category']} (mean={stats['mean']:.1f}, std={stats['std']:.1f})")
                else:
                    print(f"  Cluster {cid}: no IPL data")
    else:
        sorted_clusters = sorted(filtered_clusters)
    
    return sorted_clusters


def plot_cluster_standardized_heatmap(clustered_data, feature_patterns=None,
                                     cluster_order=None, figsize=(10, 8),
                                     cmap='RdBu_r', center=0, verbose=True):
    """
    Create a heatmap showing standardized cluster profiles.
    
    Each cell shows (cluster_mean - global_mean) / global_std
    Red = above average, Blue = below average, White = average
    
    Parameters:
    -----------
    clustered_data : pd.DataFrame
        Data with cluster assignments and features
    feature_patterns : list, optional
        Feature patterns to include
    cluster_order : list, optional
        Custom order for clusters
    figsize : tuple
        Figure size
    cmap : str
        Colormap (default 'RdBu_r' for red-blue diverging)
    center : float
        Center value for colormap
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # Get features and clusters
    if feature_patterns is None:
        feature_cols = [col for col in clustered_data.select_dtypes(include=[np.number]).columns 
                       if col != 'cluster_id' and not any(id_var in col.lower() 
                       for id_var in ['recording', 'roi', 'condition', 'category'])]
    else:
        feature_cols = []
        for pattern in feature_patterns:
            if pattern in clustered_data.columns:
                feature_cols.append(pattern)
            else:
                cols = [col for col in clustered_data.columns 
                       if pattern in col and col != pattern]
                feature_cols.extend(cols)
        feature_cols = list(dict.fromkeys(feature_cols))
    
    if cluster_order is not None:
        available_clusters = set(clustered_data['cluster_id'].unique())
        cluster_ids = [cid for cid in cluster_order if cid in available_clusters]
        missing_clusters = available_clusters - set(cluster_ids)
        cluster_ids.extend(sorted(missing_clusters))
    else:
        cluster_ids = sorted(clustered_data['cluster_id'].unique())
    
    # Calculate standardized profiles
    standardized_matrix = []
    
    for cluster_id in cluster_ids:
        cluster_row = []
        cluster_data = clustered_data[clustered_data['cluster_id'] == cluster_id]
        
        for feature in feature_cols:
            if feature in clustered_data.columns:
                cluster_mean = cluster_data[feature].mean()
                global_mean = clustered_data[feature].mean()
                global_std = clustered_data[feature].std()
                
                if global_std > 0:
                    z_score = (cluster_mean - global_mean) / global_std
                else:
                    z_score = 0
                
                cluster_row.append(z_score)
            else:
                cluster_row.append(0)
        
        standardized_matrix.append(cluster_row)
    
    # Convert to DataFrame for easier plotting
    heatmap_df = pd.DataFrame(standardized_matrix, 
                             index=[f'Cluster {cid}' for cid in cluster_ids],
                             columns=feature_cols)
    
    if verbose:
        print(f"Standardized profile range: {heatmap_df.values.min():.2f} to {heatmap_df.values.max():.2f}")
        print(f"Features: {len(feature_cols)}, Clusters: {len(cluster_ids)}")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(heatmap_df, 
                cmap=cmap, 
                center=center,
                annot=False,
                fmt='.1f',
                cbar_kws={'label': 'Standard Deviations from Mean'},
                ax=ax)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Standardized Cluster Feature Profiles\n(Red = Above Average, Blue = Below Average)', fontsize=12)
    plt.tight_layout()
    
    return fig, ax


def plot_cluster_two_panel(clustered_data, feature_patterns=None,
                          cluster_order=None, figsize=(15, 8), verbose=True):
    """
    Create a two-panel plot: Feature importance + Cluster means.
    
    Left panel: Which features distinguish clusters best (F-statistics)
    Right panel: Raw cluster means for each feature
    
    Parameters:
    -----------
    clustered_data : pd.DataFrame
        Data with cluster assignments and features
    feature_patterns : list, optional
        Feature patterns to include
    cluster_order : list, optional
        Custom order for clusters
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, (ax1, ax2) : matplotlib objects
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from scipy import stats
    
    # Get features and clusters
    if feature_patterns is None:
        feature_cols = [col for col in clustered_data.select_dtypes(include=[np.number]).columns 
                       if col != 'cluster_id' and not any(id_var in col.lower() 
                       for id_var in ['recording', 'roi', 'condition', 'category'])]
    else:
        feature_cols = []
        for pattern in feature_patterns:
            if pattern in clustered_data.columns:
                feature_cols.append(pattern)
            else:
                cols = [col for col in clustered_data.columns 
                       if pattern in col and col != pattern]
                feature_cols.extend(cols)
        feature_cols = list(dict.fromkeys(feature_cols))
    
    if cluster_order is not None:
        available_clusters = set(clustered_data['cluster_id'].unique())
        cluster_ids = [cid for cid in cluster_order if cid in available_clusters]
        missing_clusters = available_clusters - set(cluster_ids)
        cluster_ids.extend(sorted(missing_clusters))
    else:
        cluster_ids = sorted(clustered_data['cluster_id'].unique())
    
    # Calculate F-statistics for feature importance
    f_stats = {}
    for feature in feature_cols:
        if feature in clustered_data.columns:
            groups = [clustered_data[clustered_data['cluster_id'] == cid][feature].dropna().values 
                     for cid in cluster_ids]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) > 1:
                f_stat, _ = stats.f_oneway(*groups)
                f_stats[feature] = f_stat if not np.isnan(f_stat) else 0
            else:
                f_stats[feature] = 0
    
    # Calculate cluster means matrix
    means_matrix = []
    for cluster_id in cluster_ids:
        cluster_row = []
        cluster_data = clustered_data[clustered_data['cluster_id'] == cluster_id]
        
        for feature in feature_cols:
            if feature in clustered_data.columns:
                cluster_mean = cluster_data[feature].mean()
                cluster_row.append(cluster_mean)
            else:
                cluster_row.append(0)
        
        means_matrix.append(cluster_row)
    
    means_df = pd.DataFrame(means_matrix, 
                           index=[f'Cluster {cid}' for cid in cluster_ids],
                           columns=feature_cols)
    
    if verbose:
        print(f"Feature F-statistics range: {min(f_stats.values()):.1f} to {max(f_stats.values()):.1f}")
        print(f"Cluster means range: {means_df.values.min():.2f} to {means_df.values.max():.2f}")
        print(f"Features: {len(feature_cols)}, Clusters: {len(cluster_ids)}")
    
    # Create two-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left panel: Feature importance (F-statistics)
    f_values = [f_stats[feat] for feat in feature_cols]
    y_pos = np.arange(len(feature_cols))
    
    ax1.barh(y_pos, f_values, alpha=0.7, color='steelblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_cols)
    ax1.set_xlabel('F-statistic (Higher = More Discriminative)')
    ax1.set_title('Feature Importance\n(Which features distinguish clusters?)')
    ax1.invert_yaxis()
    
    # Right panel: Cluster means heatmap
    sns.heatmap(means_df, 
                annot=False,
                fmt='.2f',
                cmap='viridis',
                cbar_kws={'label': 'Mean Value'},
                ax=ax2)
    
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Clusters')
    ax2.set_title('Cluster Mean Values\n(What values does each cluster have?)')
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def generate_cluster_characterization_table(clustered_data, feature_patterns=None,
                                           cluster_order=None, top_n_features=5, verbose=True):
    """
    Generate a table showing the top distinguishing features for each cluster.
    
    For each cluster, shows the features with highest effect sizes that make it unique.
    
    Parameters:
    -----------
    clustered_data : pd.DataFrame
        Data with cluster assignments and features
    feature_patterns : list, optional
        Feature patterns to include
    cluster_order : list, optional
        Custom order for clusters
    top_n_features : int
        Number of top features to show per cluster
    
    Returns:
    --------
    characterization_df : pd.DataFrame
        Table showing top features per cluster with effect sizes and values
    """
    import numpy as np
    import pandas as pd
    
    # Get features and clusters
    if feature_patterns is None:
        feature_cols = [col for col in clustered_data.select_dtypes(include=[np.number]).columns 
                       if col != 'cluster_id' and not any(id_var in col.lower() 
                       for id_var in ['recording', 'roi', 'condition', 'category'])]
    else:
        feature_cols = []
        for pattern in feature_patterns:
            if pattern in clustered_data.columns:
                feature_cols.append(pattern)
            else:
                cols = [col for col in clustered_data.columns 
                       if pattern in col and col != pattern]
                feature_cols.extend(cols)
        feature_cols = list(dict.fromkeys(feature_cols))
    
    if cluster_order is not None:
        available_clusters = set(clustered_data['cluster_id'].unique())
        cluster_ids = [cid for cid in cluster_order if cid in available_clusters]
        missing_clusters = available_clusters - set(cluster_ids)
        cluster_ids.extend(sorted(missing_clusters))
    else:
        cluster_ids = sorted(clustered_data['cluster_id'].unique())
    
    # Calculate effect sizes for each cluster-feature pair
    characterization_rows = []
    
    for cluster_id in cluster_ids:
        cluster_data = clustered_data[clustered_data['cluster_id'] == cluster_id]
        other_data = clustered_data[clustered_data['cluster_id'] != cluster_id]
        
        feature_effects = []
        
        for feature in feature_cols:
            if feature in clustered_data.columns:
                cluster_vals = cluster_data[feature].dropna()
                other_vals = other_data[feature].dropna()
                
                if len(cluster_vals) > 0 and len(other_vals) > 0:
                    # Calculate Cohen's d effect size
                    pooled_std = np.sqrt(((len(cluster_vals)-1)*cluster_vals.var() + 
                                        (len(other_vals)-1)*other_vals.var()) / 
                                       (len(cluster_vals) + len(other_vals) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (cluster_vals.mean() - other_vals.mean()) / pooled_std
                        
                        feature_effects.append({
                            'feature': feature,
                            'effect_size': cohens_d,
                            'abs_effect_size': abs(cohens_d),
                            'cluster_mean': cluster_vals.mean(),
                            'other_mean': other_vals.mean(),
                            'direction': 'Higher' if cohens_d > 0 else 'Lower'
                        })
        
        # Sort by absolute effect size and take top N
        feature_effects.sort(key=lambda x: x['abs_effect_size'], reverse=True)
        top_features = feature_effects[:top_n_features]
        
        for i, feat_info in enumerate(top_features):
            characterization_rows.append({
                'Cluster': f'Cluster {cluster_id}',
                'Rank': i + 1,
                'Feature': feat_info['feature'],
                'Direction': feat_info['direction'],
                'Effect_Size': feat_info['effect_size'],
                'Cluster_Mean': feat_info['cluster_mean'],
                'Other_Mean': feat_info['other_mean'],
                'Description': f"{feat_info['direction']} than other clusters (d={feat_info['effect_size']:.2f})"
            })
    
    characterization_df = pd.DataFrame(characterization_rows)
    
    if verbose:
        print(f"Generated characterization table for {len(cluster_ids)} clusters")
        print(f"Top {top_n_features} features per cluster, {len(feature_cols)} total features analyzed")
        print("\nExample characterizations:")
        for cluster_id in cluster_ids[:3]:  # Show first 3 clusters
            cluster_rows = characterization_df[characterization_df['Cluster'] == f'Cluster {cluster_id}']
            if len(cluster_rows) > 0:
                top_feature = cluster_rows.iloc[0]
                print(f"  {top_feature['Cluster']}: Characterized by {top_feature['Feature']} ({top_feature['Description']})")
    
    return characterization_df


def plot_cluster_feature_dotplot(clustered_data, feature_patterns=None, 
                                threshold=0.1, figsize=(12, 8), 
                                size_scale=50, alpha_range=(0.3, 1.0), 
                                size_metric='variance', show_importance=True, 
                                min_contribution=None, cluster_order=None, 
                                minmax_scaling=None, transpose=False, verbose=True):
    """
    Create a dot plot showing cluster vs feature relationships.
    
    Parameters:
    -----------
    clustered_data : pd.DataFrame
        Data with cluster assignments and features
    feature_patterns : list, optional
        Feature patterns to include. If None, uses all numeric columns except cluster_id
    threshold : float, default 0.1
        Threshold for considering a feature "expressed" (affects circle size)
    figsize : tuple
        Figure size
    size_scale : float
        Scaling factor for circle sizes
    alpha_range : tuple
        Min and max alpha values for color intensity
    size_metric : str, default 'contribution_pct'
        What to use for circle size: 'contribution_pct' or 'transcriptomic_style'
    min_contribution : float, optional
        When using contribution_pct, filter out features with max contribution below this threshold (0-100)
    cluster_order : list, optional
        Custom order for clusters (list of cluster IDs). If None, uses sorted cluster IDs.
    minmax_scaling : str, optional
        Apply min-max scaling to features: '0to1' scales to [0,1], '-1to1' scales to [-1,1], None for no scaling
    transpose : bool, default False
        If True, swap axes so clusters are on x-axis and features on y-axis
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats
    
    def calculate_feature_contributions(data, feature_cols, cluster_ids):
        """Calculate feature contribution metrics across all clusters."""
        contributions = {}
        
        # Calculate ANOVA F-statistics for each feature
        f_stats = {}
        for feature in feature_cols:
            if feature in data.columns:
                groups = [data[data['cluster_id'] == cid][feature].dropna().values 
                         for cid in cluster_ids]
                groups = [g for g in groups if len(g) > 0]  # Remove empty groups
                if len(groups) > 1:
                    f_stat, _ = stats.f_oneway(*groups)
                    f_stats[feature] = f_stat if not np.isnan(f_stat) else 0
                else:
                    f_stats[feature] = 0
        
        # Calculate effect sizes and contributions for each cluster
        for cluster_id in cluster_ids:
            cluster_data = data[data['cluster_id'] == cluster_id]
            other_data = data[data['cluster_id'] != cluster_id]
            
            cluster_contributions = {}
            effect_sizes = {}
            
            for feature in feature_cols:
                if feature in data.columns:
                    cluster_vals = cluster_data[feature].dropna()
                    other_vals = other_data[feature].dropna()
                    
                    if len(cluster_vals) > 0 and len(other_vals) > 0:
                        # Cohen's d effect size
                        pooled_std = np.sqrt(((len(cluster_vals)-1)*cluster_vals.var() + 
                                            (len(other_vals)-1)*other_vals.var()) / 
                                           (len(cluster_vals) + len(other_vals) - 2))
                        
                        if pooled_std > 0:
                            cohens_d = abs(cluster_vals.mean() - other_vals.mean()) / pooled_std
                            effect_sizes[feature] = cohens_d
                        else:
                            effect_sizes[feature] = 0
                    else:
                        effect_sizes[feature] = 0
            
            # Convert to contribution percentages
            total_effect = sum(effect_sizes.values())
            if total_effect > 0:
                cluster_contributions = {f: (effect_sizes[f] / total_effect) * 100 
                                       for f in feature_cols if f in effect_sizes}
            else:
                cluster_contributions = {f: 0 for f in feature_cols}
            
            contributions[cluster_id] = {
                'effect_sizes': effect_sizes,
                'contribution_pct': cluster_contributions,
                'f_stats': f_stats
            }
        
        # Calculate cluster purity/dominance for each feature
        for feature in feature_cols:
            if feature in data.columns:
                feature_values = data[feature].dropna()
                if len(feature_values) > 0:
                    # Define "extreme" as top 20% of values for this feature
                    threshold = np.percentile(feature_values, 80)
                    extreme_indices = data[feature] > threshold
                    
                    for cluster_id in cluster_ids:
                        cluster_mask = data['cluster_id'] == cluster_id
                        
                        # How many extreme values belong to this cluster?
                        extreme_in_cluster = (extreme_indices & cluster_mask).sum()
                        total_extreme = extreme_indices.sum()
                        
                        if total_extreme > 0:
                            purity = extreme_in_cluster / total_extreme
                        else:
                            purity = 0
                        
                        if cluster_id not in contributions:
                            contributions[cluster_id] = {'purity': {}}
                        elif 'purity' not in contributions[cluster_id]:
                            contributions[cluster_id]['purity'] = {}
                        
                        contributions[cluster_id]['purity'][feature] = purity
        
        return contributions
    
    # Get feature columns
    if feature_patterns is None:
        # Use all numeric columns except cluster_id and id columns
        feature_cols = [col for col in clustered_data.select_dtypes(include=[np.number]).columns 
                       if col != 'cluster_id' and not any(id_var in col.lower() for id_var in ['recording', 'roi', 'condition', 'category'])]
    else:
        # Match feature patterns
        feature_cols = []
        for pattern in feature_patterns:
            if pattern in clustered_data.columns:
                feature_cols.append(pattern)
            else:
                cols = [col for col in clustered_data.columns 
                       if pattern in col and col != pattern]
                feature_cols.extend(cols)
        feature_cols = list(dict.fromkeys(feature_cols))
    
    # Get cluster IDs and order them
    if cluster_order is not None:
        # Use custom order, but filter to only include clusters that exist in data
        available_clusters = set(clustered_data['cluster_id'].unique())
        cluster_ids = [cid for cid in cluster_order if cid in available_clusters]
        # Add any missing clusters at the end
        missing_clusters = available_clusters - set(cluster_ids)
        cluster_ids.extend(sorted(missing_clusters))
    else:
        cluster_ids = sorted(clustered_data['cluster_id'].unique())
    
    # Apply min-max scaling if requested
    data_for_calc = clustered_data.copy()
    if minmax_scaling == '0to1':
        if verbose:
            print("Applying 0-1 min-max scaling to features (using absolute values)")
        for feature in feature_cols:
            if feature in data_for_calc.columns:
                # Take absolute values first to scale by amplitude magnitude
                abs_values = np.abs(data_for_calc[feature])
                min_val = abs_values.min()
                max_val = abs_values.max()
                if max_val > min_val:  # Avoid division by zero
                    data_for_calc[feature] = (abs_values - min_val) / (max_val - min_val)
                else:
                    data_for_calc[feature] = 0  # If all values are the same
    elif minmax_scaling == '-1to1':
        if verbose:
            print("Applying smart -1 to 1 min-max scaling to features")
        for feature in feature_cols:
            if feature in data_for_calc.columns:
                min_val = data_for_calc[feature].min()
                max_val = data_for_calc[feature].max()
                if max_val > min_val:  # Avoid division by zero
                    if min_val < 0:
                        # Feature has negative values - use true [-1,1] scaling
                        normalized = (data_for_calc[feature] - min_val) / (max_val - min_val)
                        data_for_calc[feature] = 2 * normalized - 1
                        if verbose:
                            print(f"  {feature}: scaled to [-1,1] (has negative values)")
                    else:
                        # Feature is all positive - use [0,1] scaling
                        data_for_calc[feature] = (data_for_calc[feature] - min_val) / (max_val - min_val)
                        if verbose:
                            print(f"  {feature}: scaled to [0,1] (all positive values)")
                else:
                    data_for_calc[feature] = 0  # If all values are the same
    
    # Calculate contribution metrics if needed
    contributions = None
    if size_metric in ['f_statistic', 'effect_size', 'contribution_pct']:
        contributions = calculate_feature_contributions(data_for_calc, feature_cols, cluster_ids)
    
    # Calculate metrics for each cluster-feature combination
    plot_data = []
    
    for cluster_id in cluster_ids:
        cluster_data = data_for_calc[data_for_calc['cluster_id'] == cluster_id]
        
        for feature in feature_cols:
            if feature in cluster_data.columns:
                values = cluster_data[feature].dropna()
                
                if len(values) > 0:
                    # Multiple metrics for circle size
                    pct_expressing = (np.abs(values) > threshold).mean() * 100
                    variance = values.var()
                    value_range = values.max() - values.min()
                    
                    # Average expression level (color intensity)
                    avg_expression = values.mean()
                    
                    # Calculate additional metrics for new size options
                    global_mean = data_for_calc[feature].mean()
                    global_std = data_for_calc[feature].std()
                    z_score = abs(avg_expression - global_mean) / global_std if global_std > 0 else 0
                    consistency = 1 - (values.std() / abs(avg_expression)) if abs(avg_expression) > 0 else 0
                    consistency = max(0, consistency)  # Ensure non-negative
                    
                    # Transcriptomic-style metrics
                    # Define "active" threshold using absolute values to catch strong responses in either direction
                    global_abs_mean = np.abs(data_for_calc[feature]).mean()
                    global_abs_std = np.abs(data_for_calc[feature]).std()
                    active_threshold = global_abs_mean + 0.5 * global_abs_std
                    
                    # Percentage of cells in this cluster that are "active" for this feature  
                    # Use absolute values to detect strong responses regardless of sign
                    active_cells = np.abs(values) > active_threshold
                    pct_active = active_cells.mean() * 100
                    
                    # Average magnitude among active cells only (preserve original sign)
                    active_values = values[active_cells]
                    avg_active_magnitude = active_values.mean() if len(active_values) > 0 else 0
                    
                    # Add contribution metrics if calculated
                    f_stat = 0
                    effect_size = 0 
                    contribution_pct = 0
                    purity = 0
                    
                    if contributions is not None:
                        f_stat = contributions[cluster_id]['f_stats'].get(feature, 0)
                        effect_size = contributions[cluster_id]['effect_sizes'].get(feature, 0)
                        contribution_pct = contributions[cluster_id]['contribution_pct'].get(feature, 0)
                        purity = contributions[cluster_id]['purity'].get(feature, 0)
                    
                    plot_data.append({
                        'cluster_id': cluster_id,
                        'feature': feature,
                        'pct_expressing': pct_expressing,
                        'variance': variance,
                        'range': value_range,
                        'avg_expression': avg_expression,
                        'abs_avg_expression': np.abs(avg_expression),
                        'f_statistic': f_stat,
                        'effect_size': effect_size,
                        'contribution_pct': contribution_pct,
                        'purity': purity,
                        'z_score': z_score,
                        'consistency': consistency,
                        'pct_active': pct_active,
                        'avg_active_magnitude': avg_active_magnitude
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Filter out low-contribution features if specified
    if size_metric == 'contribution_pct' and min_contribution is not None:
        # Find features where max contribution across clusters is above threshold
        feature_max_contrib = plot_df.groupby('feature')['contribution_pct'].max()
        good_features = feature_max_contrib[feature_max_contrib >= min_contribution].index.tolist()
        
        if len(good_features) == 0:
            print(f"No features with contribution >= {min_contribution}%. Showing all features.")
        else:
            plot_df = plot_df[plot_df['feature'].isin(good_features)]
            feature_cols = good_features
            if verbose:
                print(f"Filtered to {len(good_features)} features with contribution >= {min_contribution}%")
    
    if len(plot_df) == 0:
        print("No data to plot")
        return None, None
    
    if verbose:
        print(f"Percentage range: {plot_df['pct_expressing'].min():.1f}% - {plot_df['pct_expressing'].max():.1f}%")
        print(f"Variance range: {plot_df['variance'].min():.3f} - {plot_df['variance'].max():.3f}")
        print(f"Value range: {plot_df['range'].min():.3f} - {plot_df['range'].max():.3f}")
        print(f"Expression range: {plot_df['avg_expression'].min():.3f} - {plot_df['avg_expression'].max():.3f}")
        if size_metric in ['f_statistic', 'effect_size', 'contribution_pct']:
            print(f"F-statistic range: {plot_df['f_statistic'].min():.3f} - {plot_df['f_statistic'].max():.3f}")
            print(f"Effect size range: {plot_df['effect_size'].min():.3f} - {plot_df['effect_size'].max():.3f}")
            print(f"Contribution % range: {plot_df['contribution_pct'].min():.1f}% - {plot_df['contribution_pct'].max():.1f}%")
            print(f"Purity range: {plot_df['purity'].min():.3f} - {plot_df['purity'].max():.3f}")
        print(f"Features: {len(feature_cols)}, Clusters: {len(cluster_ids)}")
        print(f"Using '{size_metric}' for circle sizes")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize values for visualization
    max_abs_expr = plot_df['abs_avg_expression'].max() if plot_df['abs_avg_expression'].max() > 0 else 1
    
    # Choose size metric and normalize
    # Commented out unused metrics for cleaner interface
    # if size_metric == 'percentage':
    #     size_values = plot_df['pct_expressing'] / 100
    # elif size_metric == 'variance':
    #     max_var = plot_df['variance'].max()
    #     size_values = plot_df['variance'] / max_var if max_var > 0 else plot_df['variance'] * 0
    # elif size_metric == 'range':
    #     max_range = plot_df['range'].max()
    #     size_values = plot_df['range'] / max_range if max_range > 0 else plot_df['range'] * 0
    # elif size_metric == 'f_statistic':
    #     max_f = plot_df['f_statistic'].max()
    #     size_values = plot_df['f_statistic'] / max_f if max_f > 0 else plot_df['f_statistic'] * 0
    # elif size_metric == 'effect_size':
    #     max_effect = plot_df['effect_size'].max()
    #     size_values = plot_df['effect_size'] / max_effect if max_effect > 0 else plot_df['effect_size'] * 0
    
    if size_metric == 'contribution_pct':
        # Use rank for size (where this cluster ranks for each feature)
        size_values = []
        
        for i, row in plot_df.iterrows():
            feature = row['feature']
            cluster_value = row['avg_expression']
            
            # Calculate rank of this cluster for this feature
            feature_data = plot_df[plot_df['feature'] == feature]
            feature_values = feature_data['avg_expression'].values
            
            # Get rank (0=lowest, 1=highest)
            rank = (feature_values < cluster_value).sum() / (len(feature_values) - 1) if len(feature_values) > 1 else 0.5
            size_values.append(rank)
        
        size_values = np.array(size_values)
    # Commented out experimental metrics
    # elif size_metric == 'z_score':
    #     # Use z-score (deviation from expected)
    #     max_z = plot_df['z_score'].max()
    #     size_values = plot_df['z_score'] / max_z if max_z > 0 else plot_df['z_score'] * 0
    # elif size_metric == 'consistency_magnitude':
    #     # Combine consistency and magnitude
    #     consistency_vals = plot_df['consistency']
    #     magnitude_vals = plot_df['abs_avg_expression']
    #     
    #     # Normalize both
    #     max_consistency = consistency_vals.max() if consistency_vals.max() > 0 else 1
    #     max_magnitude = magnitude_vals.max() if magnitude_vals.max() > 0 else 1
    #     
    #     normalized_consistency = consistency_vals / max_consistency
    #     normalized_magnitude = magnitude_vals / max_magnitude
    #     
    #     # Combine (equal weight for now)
    #     size_values = (normalized_consistency + normalized_magnitude) / 2
    elif size_metric == 'transcriptomic_style':
        # Use percentage active for size (like % expressed in transcriptomics)
        size_values = plot_df['pct_active'] / 100  # Convert to 0-1 range
    else:
        # Default to contribution_pct if unknown metric
        size_values = []        
        for i, row in plot_df.iterrows():
            feature = row['feature']
            cluster_value = row['avg_expression']
            
            # Calculate rank of this cluster for this feature
            feature_data = plot_df[plot_df['feature'] == feature]
            feature_values = feature_data['avg_expression'].values
            
            # Get rank (0=lowest, 1=highest)
            rank = (feature_values < cluster_value).sum() / (len(feature_values) - 1) if len(feature_values) > 1 else 0.5
            size_values.append(rank)
        
        size_values = np.array(size_values)
    
    for i, row in plot_df.iterrows():
        if transpose:
            x_pos = cluster_ids.index(row['cluster_id'])
            y_pos = feature_cols.index(row['feature'])
        else:
            x_pos = feature_cols.index(row['feature'])
            y_pos = cluster_ids.index(row['cluster_id'])
        
        # Circle size with enhanced scaling for better contrast
        normalized_size = size_values[i]
        # Linear scaling - no power transformation
        size = normalized_size * size_scale
        
        # Ensure minimum size for visibility
        size = max(size, size_scale * 0.05)
        
        # Color and alpha logic
        if size_metric == 'contribution_pct':
            # Use black dots with purity-based alpha
            color = 'black'
            purity_val = row['purity']
            
            # Scale purity for alpha
            max_purity = plot_df['purity'].max()
            if max_purity > 0:
                scaled_purity = purity_val / max_purity
            else:
                scaled_purity = 0
            
            alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * scaled_purity
        elif size_metric == 'transcriptomic_style':
            avg_active_mag = row['avg_active_magnitude']
            
            if minmax_scaling == '-1to1':
                # Use diverging colormap for -1 to 1 scaled data (preserve sign)
                import matplotlib.cm as cm
                cmap = cm.get_cmap('RdBu_r')  # Red-Blue diverging colormap (reversed)
                
                # Normalize to [0,1] for colormap (0 = -1, 0.5 = 0, 1 = +1)
                normalized_mag = (avg_active_mag + 1) / 2  # Map [-1,1] to [0,1]
                normalized_mag = np.clip(normalized_mag, 0, 1)
                color = cmap(normalized_mag)
            else:
                # Use greyscale based on absolute average active magnitude (0to1 scaling or no scaling)
                # Use absolute values for color intensity to handle negative values
                abs_avg_active_mag = abs(avg_active_mag)
                max_abs_active_mag = plot_df['avg_active_magnitude'].abs().max()
                
                if max_abs_active_mag > 0:
                    # Map to greyscale: 0 = white, max = black
                    grey_intensity = abs_avg_active_mag / max_abs_active_mag
                    # Ensure grey_intensity is in [0,1] range
                    grey_intensity = np.clip(grey_intensity, 0, 1)
                    color = (1 - grey_intensity, 1 - grey_intensity, 1 - grey_intensity)  # RGB greyscale
                else:
                    color = 'white'
            
            alpha = 1.0  # Full opacity
        else:
            # Original color scheme for other metrics
            normalized_expr = row['abs_avg_expression'] / max_abs_expr
            alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * normalized_expr
            color = 'red' if row['avg_expression'] > 0 else 'blue'
        
        # Plot the circle
        ax.scatter(x_pos, y_pos, s=size, c=color, alpha=alpha, 
                  edgecolors='black', linewidth=0.1)
    
    # Customize the plot
    if transpose:
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_xticklabels([str(cid) for cid in cluster_ids])
        ax.set_yticks(range(len(feature_cols)))
        ax.set_yticklabels(feature_cols)
        
        # Flip y-axis to match normal reading order (features top to bottom)
        ax.invert_yaxis()
        
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Features')
    else:
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha='right')
        ax.set_yticks(range(len(cluster_ids)))
        ax.set_yticklabels([str(cid) for cid in cluster_ids])
        
        # Flip y-axis to match cluster average plot order
        ax.invert_yaxis()
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Cluster ID')
    ax.set_title('Cluster-Feature Expression Profile')
    
    # Add grid
    # ax.grid(True, alpha=0.3)
    
    # Create legends based on size_metric
    if size_metric == 'transcriptomic_style':
        # Size legend for percentage active
        sizes = [25, 50, 75, 100]
        size_legend = []
        for s in sizes:
            size_legend.append(ax.scatter([], [], s=(s/100)*size_scale, c='gray', alpha=0.6, 
                                        edgecolors='black', linewidth=0.5))
        
        legend1 = ax.legend(size_legend, [f'{s}%' for s in sizes], 
                           title='% Above\nmean', loc='center left', bbox_to_anchor=(1, 0.7))
        
        # Colorbar for average magnitude
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        if minmax_scaling == '-1to1':
            # Check if we actually have negative values in the scaled data
            min_active_mag = plot_df['avg_active_magnitude'].min()
            has_negative_values = min_active_mag < 0
            
            if has_negative_values:
                # Use diverging colormap for truly bipolar data
                cmap = cm.get_cmap('RdBu_r')  # Red-Blue diverging colormap (reversed)
                # Force colorbar to show full -1 to +1 range
                norm = mcolors.Normalize(vmin=-1, vmax=1)
            else:
                # All features were positive, use greyscale for [0,1] range
                cmap = cm.get_cmap('Greys')
                norm = mcolors.Normalize(vmin=0, vmax=1)
        else:
            # Use greyscale colormap for 0to1 scaling or no scaling
            cmap = cm.get_cmap('Greys')
            
            # Get the range of absolute average active magnitude values for normalization
            max_abs_active_mag = plot_df['avg_active_magnitude'].abs().max()
            min_abs_active_mag = 0  # Always start from 0 for absolute values
            
            # Create a ScalarMappable for the colorbar
            norm = mcolors.Normalize(vmin=min_abs_active_mag, vmax=max_abs_active_mag)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        # Add colorbar - compact and positioned like the old legend
        cbar = plt.colorbar(sm, ax=ax, shrink=0.25, aspect=10, pad=0.05, 
                           anchor=(0, 0.1), panchor=(1.0, 0.1))
        # Create dynamic colorbar title based on scaling and actual data
        if size_metric == 'transcriptomic_style':
            if minmax_scaling == '-1to1':
                # Check if we actually have negative values in the final data
                min_active_mag = plot_df['avg_active_magnitude'].min()
                if min_active_mag < 0:
                    colorbar_title = 'Avg.\nmagnitude'  # Preserves sign, no "abs."
                else:
                    colorbar_title = 'Avg.\nmagnitude'  # All positive after smart scaling
            elif minmax_scaling == '0to1':
                colorbar_title = 'Avg. abs.\nmagnitude'  # Uses absolute values
            else:
                colorbar_title = 'Avg. abs.\nmagnitude'  # Default: absolute values for raw data
        else:
            # For other size_metrics, keep a generic title
            colorbar_title = 'Avg.\nmagnitude'
        
        # Position the label above the colorbar using text annotation
        cbar.ax.text(0.5, 1.05, colorbar_title, transform=cbar.ax.transAxes, 
                     ha='center', va='bottom')
        
        ax.add_artist(legend1)  # Add first legend back
    elif size_metric == 'contribution_pct':
        # Size legend for contribution percentages
        sizes = [25, 50, 75, 100]
        size_legend = []
        for s in sizes:
            size_legend.append(ax.scatter([], [], s=(s/100)*size_scale, c='gray', alpha=0.6, 
                                        edgecolors='black', linewidth=0.5))
        
        legend1 = ax.legend(size_legend, ['Lowest', 'Low', 'High', 'Highest'], 
                           title='Cluster\nRank', loc='center left', bbox_to_anchor=(1, 0.7))
        
        # Alpha legend for cluster dominance
        alphas = [0.3, 0.5, 0.7, 0.9]
        alpha_legend = []
        for a in alphas:
            alpha_legend.append(ax.scatter([], [], c='black', alpha=a, s=100, 
                                         edgecolors='black', linewidth=0.5))
        
        legend2 = ax.legend(alpha_legend, ['Low', 'Medium', 'High', 'Very High'], 
                           title='Cluster\nDominance', loc='center left', bbox_to_anchor=(1, 0.3))
        
        ax.add_artist(legend1)  # Add first legend back
    else:
        # Original legends for other metrics
        sizes = [25, 50, 75, 100]
        size_legend = []
        for s in sizes:
            size_legend.append(ax.scatter([], [], s=(s/100)*size_scale, c='gray', alpha=0.6, 
                                        edgecolors='black', linewidth=0.5))
        
        legend1 = ax.legend(size_legend, [f'{s}%' for s in sizes], 
                           title='% Cells\nExpressing', loc='center left', bbox_to_anchor=(1, 0.7))
        
        # Create legend for colors
        red_patch = ax.scatter([], [], c='red', alpha=0.8, s=100, edgecolors='black', linewidth=0.5)
        blue_patch = ax.scatter([], [], c='blue', alpha=0.8, s=100, edgecolors='black', linewidth=0.5)
        legend2 = ax.legend([red_patch, blue_patch], ['Positive', 'Negative'], 
                           title='Expression\nDirection', loc='center left', bbox_to_anchor=(1, 0.3))
        
        ax.add_artist(legend1)  # Add first legend back
    
    plt.tight_layout()
    sns.despine()
    return fig, ax


def generate_clustering_quality_table(clustered_data, feature_patterns=None, cluster_col='cluster_id'):
    """
    Generate a summary table of clustering quality metrics for each cluster.

    Parameters:
    -----------
    clustered_data : pd.DataFrame
        Data with cluster assignments and features
    feature_patterns : list, optional
        Feature patterns to use for metrics. If None, uses all numeric columns except cluster_id
    cluster_col : str, default 'cluster_id'
        Name of cluster column

    Returns:
    --------
    pd.DataFrame
        Summary table with cluster quality metrics
    """
    from sklearn.metrics import silhouette_samples, calinski_harabasz_score, davies_bouldin_score
    import pandas as pd
    import numpy as np

    # Get feature columns
    if feature_patterns is None:
        feature_cols = [col for col in clustered_data.select_dtypes(include=[np.number]).columns
                       if col != cluster_col and not any(id_var in col.lower() for id_var in ['recording', 'roi', 'condition', 'category'])]
    else:
        # Match feature patterns
        feature_cols = []
        for pattern in feature_patterns:
            if pattern in clustered_data.columns:
                feature_cols.append(pattern)
            else:
                cols = [col for col in clustered_data.columns
                       if pattern in col and col != pattern]
                feature_cols.extend(cols)
        feature_cols = list(dict.fromkeys(feature_cols))

    # Prepare data for sklearn metrics
    X = clustered_data[feature_cols].values
    labels = clustered_data[cluster_col].values

    # Calculate silhouette scores
    silhouette_scores = silhouette_samples(X, labels)

    # Calculate global metrics
    calinski_harabasz = calinski_harabasz_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)

    # Calculate per-cluster metrics
    cluster_metrics = []

    for cluster_id in sorted(clustered_data[cluster_col].unique()):
        cluster_mask = clustered_data[cluster_col] == cluster_id
        cluster_silhouettes = silhouette_scores[cluster_mask]

        # Basic cluster info
        n_samples = cluster_mask.sum()
        pct_total = (n_samples / len(clustered_data)) * 100

        # Silhouette metrics
        mean_silhouette = cluster_silhouettes.mean()
        std_silhouette = cluster_silhouettes.std()
        min_silhouette = cluster_silhouettes.min()

        # Intra-cluster metrics (compactness)
        cluster_data = clustered_data[cluster_mask][feature_cols]
        cluster_centroid = cluster_data.mean()

        # Average distance to centroid
        distances_to_centroid = np.sqrt(((cluster_data - cluster_centroid) ** 2).sum(axis=1))
        avg_distance_to_centroid = distances_to_centroid.mean()

        # Coefficient of variation (relative spread)
        feature_cvs = []
        for col in feature_cols:
            if cluster_data[col].std() > 0:
                cv = cluster_data[col].std() / abs(cluster_data[col].mean()) if cluster_data[col].mean() != 0 else np.inf
                feature_cvs.append(cv)
        avg_cv = np.mean(feature_cvs) if feature_cvs else np.inf

        cluster_metrics.append({
            'cluster_id': cluster_id,
            'n_samples': n_samples,
            'pct_total': pct_total,
            'silhouette_mean': mean_silhouette,
            'silhouette_std': std_silhouette,
            'silhouette_min': min_silhouette,
            'compactness': avg_distance_to_centroid,
            'avg_cv': avg_cv
        })

    # Create DataFrame
    metrics_df = pd.DataFrame(cluster_metrics)

    # Add overall quality assessment
    metrics_df['quality_grade'] = metrics_df['silhouette_mean'].apply(
        lambda x: 'Excellent' if x > 0.7 else
                 'Good' if x > 0.5 else
                 'Fair' if x > 0.25 else 'Poor'
    )

    # Sort by cluster ID for easy reference
    metrics_df = metrics_df.sort_values('cluster_id', ascending=True)

    # Add global metrics as attributes
    metrics_df.attrs['global_calinski_harabasz'] = calinski_harabasz
    metrics_df.attrs['global_davies_bouldin'] = davies_bouldin
    metrics_df.attrs['global_silhouette'] = silhouette_scores.mean()

    return metrics_df


# Convenience function that combines everything
def cluster_rf_data(df, feature_patterns, method='kmeans', n_clusters=5, random_state=42,
                   id_vars=None, scale=True, scaling_method='standard', show_elbow=True, 
                   show_embedding=None, show_silhouette=False, **kwargs):
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
    show_embedding : str or None, default None
        Embedding visualization: None (no plot), 'pca', 'umap', or 'tsne'
    show_silhouette : bool, default False
        Whether to compute and display silhouette analysis
    **kwargs
        Additional clustering parameters
    
    Returns:
    --------
    dict
        Dictionary containing clustered_data, summary_stats, scaler, and embedding results
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
        prepared_data, method=method, n_clusters=n_clusters, random_state=random_state,
        id_cols=id_vars, **kwargs
    )
    
    # Extract clustered data
    clustered_data = clustering_result['clustered_data']
    
    # Generate summary statistics
    summary_stats = cluster_summary_stats(clustered_data, feature_patterns, id_vars)
    
    # Conditional embedding visualization
    embedding_obj = None
    embedding_data = None
    
    if show_embedding is not None:
        if show_embedding.lower() == 'pca':
            embedding_obj, embedding_data = visualize_clusters_pca(clustered_data, feature_patterns)
        elif show_embedding.lower() == 'umap':
            embedding_obj, embedding_data = visualize_clusters_umap(clustered_data, feature_patterns)
        elif show_embedding.lower() == 'tsne':
            embedding_obj, embedding_data = visualize_clusters_tsne(clustered_data, feature_patterns)
        else:
            print(f"Warning: Unknown embedding method '{show_embedding}'. Skipping visualization.")
    
    # Silhouette analysis if requested
    silhouette_results = None
    if show_silhouette:
        from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score
        
        # Get the actual feature columns that were used in clustering
        feature_cols = []
        for pattern in feature_patterns:
            # First try exact match for singular features
            if pattern in clustered_data.columns:
                feature_cols.append(pattern)
            # Then try pattern matching for multi-column features  
            else:
                cols = [col for col in clustered_data.columns 
                       if pattern in col and col != pattern]
                feature_cols.extend(cols)
        
        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))
        print(f"Using {len(feature_cols)} feature columns for silhouette analysis")
        
        # Calculate silhouette scores
        feature_data = clustered_data[feature_cols].dropna()
        if len(feature_data) > 0 and len(clustered_data.loc[feature_data.index, 'cluster_id'].unique()) > 1:
            silhouette_avg = silhouette_score(
                feature_data, 
                clustered_data.loc[feature_data.index, 'cluster_id']
            )
            print(f"Average silhouette score: {silhouette_avg:.3f}")
            
            # Per-cluster silhouette scores
            sample_silhouette_values = silhouette_samples(
                feature_data,
                clustered_data.loc[feature_data.index, 'cluster_id']
            )
            
            cluster_silhouettes = {}
            for cluster_id in clustered_data['cluster_id'].unique():
                cluster_mask = clustered_data.loc[feature_data.index, 'cluster_id'] == cluster_id
                cluster_silhouettes[cluster_id] = np.mean(sample_silhouette_values[cluster_mask])
            
            print("Per-cluster silhouette scores:")
            for cid, score in sorted(cluster_silhouettes.items()):
                print(f"  Cluster {cid}: {score:.3f}")
            
            # Calculate Calinski-Harabasz score (higher is better)
            ch_score = calinski_harabasz_score(
                feature_data,
                clustered_data.loc[feature_data.index, 'cluster_id']
            )
            print(f"Calinski-Harabasz score: {ch_score:.3f} (higher is better)")
            
            silhouette_results = {
                'average_score': silhouette_avg,
                'cluster_scores': cluster_silhouettes,
                'sample_scores': sample_silhouette_values,
                'calinski_harabasz_score': ch_score,
                'feature_columns': feature_cols
            }
        else:
            print("Warning: Cannot compute silhouette scores - insufficient data or clusters")
    
    return {
        'clustered_data': clustered_data,
        'summary_stats': summary_stats,
        'scaler': scaler,
        'embedding': embedding_obj,
        'embedding_data': embedding_data,
        'embedding_method': show_embedding,
        'elbow_results': elbow_results,
        'clustering_pca': clustering_result.get('pca_obj'),  # PCA used for clustering
        'clusterer': clustering_result.get('clusterer'),
        'silhouette_results': silhouette_results
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


# def cluster_and_analyze_comprehensive(df, experiment_obj, feature_patterns, 
#                                      method='kmeans', n_clusters=5, 
#                                      id_vars=['recording_id', 'roi_id'],
#                                      metrics_to_plot=None,
#                                      include_timecourses=True,
#                                      cs_params=None,
#                                      **clustering_kwargs):
#     """
#     Complete workflow: cluster RFs, create averages, extract timecourses, and plot everything.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input dataframe with RF metrics
#     experiment_obj : Experiment
#         Experiment object containing recording data
#     feature_patterns : list
#         Feature patterns for clustering (e.g., ['space_amps', 'areas'])
#     method : str
#         Clustering method ('kmeans', 'gmm', 'hierarchical')
#     n_clusters : int
#         Number of clusters
#     id_vars : list
#         ID variables for merging back to original data
#     metrics_to_plot : list, optional
#         Metric column names to plot (e.g., ['space_amps_0', 'areas_0'])
#     include_timecourses : bool
#         Whether to extract and plot timecourses
#     cs_params : dict, optional
#         Parameters for cs_segment.run()
#     **clustering_kwargs
#         Additional parameters for clustering
    
#     Returns:
#     --------
#     dict
#         Comprehensive results containing:
#         - clustering_result: clustering analysis results
#         - df_with_clusters: dataframe with cluster assignments
#         - cluster_averages: averaged RFs per cluster
#         - cluster_averaged_strfs: averaged STRFs per cluster (if timecourses)
#         - cluster_timecourses: extracted timecourses (if timecourses)
#     """
#     print("=== COMPREHENSIVE CLUSTER ANALYSIS ===")
    
#     # Step 1: Perform clustering
#     print("\n1. Performing clustering...")
#     clustering_result = cluster_rf_data(
#         df, feature_patterns, method=method, n_clusters=n_clusters, 
#         id_vars=id_vars, **clustering_kwargs
#     )
    
#     # Step 2: Merge clusters back to original data
#     print("\n2. Merging cluster assignments...")
#     df_with_clusters = merge_clusters_to_original(df, clustering_result, id_cols=id_vars)
    
#     # Step 3: Create averaged RFs
#     print("\n3. Creating averaged RFs...")
#     cluster_averages = create_cluster_averaged_rfs(df_with_clusters, experiment_obj)
    
#     results = {
#         'clustering_result': clustering_result,
#         'df_with_clusters': df_with_clusters,
#         'cluster_averages': cluster_averages,
#     }
    
#     # Step 4: Extract RGBU timecourses if requested
#     cluster_timecourses = None
#     if include_timecourses:
#         print("\n4. Extracting RGBU timecourses per cluster...")
        
#         # Extract timecourses directly from individual ROI data using get_timecourses_dominant_by_channel
#         cluster_timecourses = extract_cluster_timecourses(df_with_clusters, experiment_obj)
        
#         results['cluster_timecourses'] = cluster_timecourses
    
#     # Step 5: Create comprehensive plot
#     print("\n5. Creating comprehensive visualization...")
#     plot_all_cluster_averages_enhanced(
#         df_with_clusters, experiment_obj, cluster_averages,
#         cluster_timecourses=cluster_timecourses,
#         metrics_to_plot=metrics_to_plot,
#         figsize_per_cluster=(12, 2),
#         show_ipl_distribution=True
#     )
    
#     print("\n=== ANALYSIS COMPLETE ===")
#     print(f"Results available in returned dictionary with keys: {list(results.keys())}")
    
#     return results


def plot_cluster_results(clustering_result, experiment_obj, 
                        original_data=None,
                        metrics_to_plot=None,
                        metrics_like=None,
                        include_timecourses=True,
                        figsize_per_cluster=None,
                        show_ipl_distribution=True,
                        sort_by_ipl_depth=True,
                        hide_noise_clusters=False,
                        cluster_label_format="{cluster_id}, n = {n_rois}",
                        rgb_mode='new',
                        save_path=None,
                        verbose=False):
    """
    Plot comprehensive cluster analysis results from cluster_rf_data().
    
    This function takes the output of cluster_rf_data() and creates enhanced 
    visualizations with RF averages, timecourses, IPL distributions, and metrics.
    
    Parameters:
    -----------
    clustering_result : dict
        Output from cluster_rf_data() containing 'clustered_data' key
    experiment_obj : Experiment
        Experiment object containing recording data
    original_data : pd.DataFrame, optional
        Original unscaled dataframe to use for plotting metrics.
        If None, will use scaled values from clustering_result
    metrics_to_plot : list, optional
        Metric column names to plot (e.g., ['space_amps_0', 'areas_0'])
    metrics_like : list, optional
        Metric prefixes to group and plot (e.g., ['areas', 'space_amps'])
        Each creates a separate column with all matching metrics (e.g., areas_0, areas_1, etc.)
    include_timecourses : bool
        Whether to extract and plot timecourses
    figsize_per_cluster : tuple or None
        Size per cluster row in inches. If None, automatically scales based on number of columns
    show_ipl_distribution : bool
        Whether to show IPL distribution plots
    sort_by_ipl_depth : bool, default True
        Whether to sort clusters by mean IPL depth (shallow to deep). If False, sorts by cluster ID.
    hide_noise_clusters : bool, default False
        Whether to hide clusters with low amplitude timecourses (likely noise clusters). These clusters are identified as having maximum timecourse amplitude < 1.0.
    save_path : str, optional
        Path to save the figure
    verbose : bool, default True
        Whether to print progress messages
    
    Returns:
    --------
    dict
        Dictionary containing:
        - cluster_averages: averaged RFs per cluster
        - cluster_timecourses: extracted timecourses (if requested)
    """
    if verbose:
        print("=== PLOTTING CLUSTER RESULTS ===")
    
    # Extract clustered data
    df_with_clusters = clustering_result['clustered_data']
    
    # Step 3: Create comprehensive plot
    if verbose:
        print("\n3. Creating comprehensive visualization...")
    # Use original data for metrics if provided
    data_for_metrics = df_with_clusters
    if verbose:
        print(f"\n=== DEBUGGING DATA MERGE ===")
        print(f"original_data is None: {original_data is None}")
        if original_data is not None:
            print(f"original_data type: {type(original_data)}")
            print(f"original_data shape: {original_data.shape}")
    
    if original_data is not None:
        if verbose:
            print(f"Starting merge process...")
        # Merge cluster assignments with original data using index
        cluster_assignments = df_with_clusters[['cluster_id']].copy()
        if verbose:
            print(f"cluster_assignments shape: {cluster_assignments.shape}")
            print(f"cluster_assignments columns: {cluster_assignments.columns.tolist()}")
        clusters = df_with_clusters['cluster_id']
        data_for_metrics = original_data
        data_for_metrics['cluster_id'] = clusters
        # data_for_metrics = original_data.join(cluster_assignments, how='left')
    
    # Step 1: Create averaged RFs (using the final data_for_metrics)
    if verbose:
        print("\n1. Creating averaged RFs...")
    cluster_averages = create_cluster_averaged_rfs(data_for_metrics, experiment_obj, verbose=verbose)
    
    results = {
        'cluster_averages': cluster_averages,
    }
    
    # Step 2: Extract RGBU timecourses if requested (using the final data_for_metrics)
    cluster_timecourses = None
    if include_timecourses:
        if verbose:
            print("\n2. Extracting RGBU timecourses per cluster...")
        cluster_timecourses = extract_cluster_timecourses(data_for_metrics, experiment_obj, verbose=verbose)
        results['cluster_timecourses'] = cluster_timecourses
        
        if verbose:
            print(f"  Clustered data shape: {df_with_clusters.shape}")
            print(f"  Original data shape: {original_data.shape}")
            print(f"  Merged data shape: {data_for_metrics.shape}")
            print(f"  Non-null cluster assignments: {data_for_metrics['cluster_id'].notna().sum()}")
            
            # Check if areas columns exist
            areas_cols = [col for col in data_for_metrics.columns if col.startswith('areas_')]
            time_amps_cols = [col for col in data_for_metrics.columns if col.startswith('time_amps_')]
            print(f"  Areas columns in merged data: {areas_cols}")
            print(f"  Time_amps columns in merged data: {time_amps_cols}")
            if areas_cols:
                sample_areas = data_for_metrics[data_for_metrics['cluster_id'].notna()][areas_cols[0]].head(3).tolist()
                print(f"  Sample areas values (from clustered ROIs): {sample_areas}")
            if time_amps_cols:
                sample_time_amps = data_for_metrics[data_for_metrics['cluster_id'].notna()][time_amps_cols[0]].head(3).tolist()
                print(f"  Sample time_amps values (from clustered ROIs): {sample_time_amps}")
    else:
        if verbose:
            print(f"Using clustered data directly (no original_data merge)")
            areas_cols = [col for col in data_for_metrics.columns if col.startswith('areas_')]
            time_amps_cols = [col for col in data_for_metrics.columns if col.startswith('time_amps_')]
            print(f"  Areas columns in clustered data: {areas_cols}")
            print(f"  Time_amps columns in clustered data: {time_amps_cols}")
    
    # Extract silhouette scores if available
    silhouette_scores = None
    if 'silhouette_results' in clustering_result and clustering_result['silhouette_results']:
        silhouette_scores = clustering_result['silhouette_results']['cluster_scores']

    fig, ax = plot_all_cluster_averages_enhanced(
        data_for_metrics, experiment_obj, cluster_averages,
        cluster_timecourses=cluster_timecourses,
        metrics_to_plot=metrics_to_plot,
        metrics_like=metrics_like,
        figsize_per_cluster=figsize_per_cluster,
        show_ipl_distribution=show_ipl_distribution,
        sort_by_ipl_depth=sort_by_ipl_depth,
        hide_noise_clusters=hide_noise_clusters,
        cluster_label_format=cluster_label_format,
        silhouette_scores=silhouette_scores,
        rgb_mode=rgb_mode,
        save_path=save_path,
        verbose=verbose
    )
    
    if verbose:
        print("\n=== PLOTTING COMPLETE ===")
        print(f"Plot results available with keys: {list(results.keys())}")
    
    return fig, ax