import gpmp.num as gnp

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False
#    print(
#        "hnswlib is not installed. estimate_cov_matrix_knn will fall back to classical covariance estimation."
#    )

def estimate_cov_matrix(x):
    """
    Default classical covariance estimation using the sample covariance.
    x: shape = (N, d)
    Returns a (d, d) covariance matrix
    """
    # gpmp.num.cov expects shape (d, N), so we transpose x
    return gnp.cov(x.T)

def estimate_cov_matrix_knn(
    x,
    n_random=50,
    n_neighbors=50,
    ef=100,
    max_ef_construction=200,
    M=16,
):
    """
    Estimate the covariance matrix of x by:
      1) Randomly sampling 'n_random' points from x.
      2) Building an HNSW index (if hnswlib is available).
      3) For each sampled point, finding its n_neighbors nearest neighbors.
      4) Computing each local k-NN covariance.
      5) Averaging the results into a single covariance matrix.

    Fallback: if HNSWLIB_AVAILABLE is False, returns classical covariance.

    Parameters
    ----------
    x : ndarray, shape (N, d)
        The dataset.
    n_random : int, optional
        Number of random points from x to sample for local covariance.
    n_neighbors : int, optional
        Number of neighbors for each point to compute local covariances.
    ef : int, optional
        'ef' parameter for HNSW search (larger means more accurate at some cost).
    max_ef_construction : int, optional
        'ef_construction' parameter for HNSW index building.
    M : int, optional
        HNSW parameter controlling the connectivity of the graph.

    Returns
    -------
    C_avg : ndarray, shape (d, d)
        The averaged covariance matrix (or classical covariance if fallback).
    """
    if not HNSWLIB_AVAILABLE:
        # If import hnswlib failed at module load, fallback to classical
        return estimate_cov_matrix(x)

    N, d = x.shape
    if n_random > N:
        n_random = N

    # 1) Build the HNSW index
    p = hnswlib.Index(space='l2', dim=d)
    p.init_index(max_elements=N, ef_construction=max_ef_construction, M=M)
    p.add_items(x)
    p.set_ef(ef)

    # 2) Randomly select points and compute local covariances
    random_indices = gnp.choice(N, size=n_random, replace=False)
    local_covs = []
    
    for idx in random_indices:
        query_point = x[idx]
        # knn_query returns two arrays: (labels, distances) each of shape (1, k)
        labels, distances = p.knn_query(query_point, k=n_neighbors)
        neighbors_idx = labels[0]        # shape (n_neighbors,)
        neighbors_x = x[neighbors_idx]   # shape (n_neighbors, d)

        # local covariance (gpmp.num.cov => shape (d, n_neighbors))
        C_local = gnp.cov(neighbors_x.T)
        local_covs.append(C_local)

    # 3) Average all local covariance matrices
    C_avg = gnp.mean(local_covs, axis=0)  # shape (d, d)
    return gnp.asarray(C_avg)
