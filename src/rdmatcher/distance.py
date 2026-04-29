import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import time

class GowerKNN(BaseEstimator):
    def __init__(self, weights=None, cat_features=None, logger=None):
        """
        GowerKNN Estimator using Gower Distance for mixed data types.
        Parameters
        ----------
        weights : list or np.ndarray, optional
            Feature weights for distance calculation. If None, equal weights are used.
        cat_features : list, np.ndarray, or None, optional
            Indices or boolean mask of categorical features. If None, auto-detected.
        logger : logging.Logger, optional
            Logger for logging information. If None, a default logger is used.

        Note:
        - Categorical features are identified as object, category, or bool dtypes in DataFrames.
            - If a boolean type is provided, ensure the weights treat the feature as an Asymmetric Binary.
            - If an ordinal categorical feature is present, consider encoding it numerically [0, 1] before using GowerKNN.
        - Numerical features are identified as number dtypes in DataFrames.
        """
        self.weights = weights
        self.cat_features = cat_features
        base = logger if logger is not None else logging.getLogger('rdmatcher')
        if logger is not None:
            self.logger = base.getChild('distance')
            self.logger.setLevel(logging.INFO)
        else:
            # have NO logger output if none provided
            self.logger = logging.getLogger('none')
            self.logger.addHandler(logging.NullHandler())

    def fit(self, X, y=None, seed=42):
        """
        Fit the K-Nearest Neighbors estimator using Gower distance.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : Ignored
            Not used, present for API consistency by convention.
        seed : int, default=42
            Random seed for shuffling the reference pool to ensure reproducibility.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # 1. Data Ingestion. Pandas DataFrame preferred for column handling.
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = np.array(X.columns.tolist())
            n_samples = len(X)
            
            # Identify columns
            if self.cat_features is None:
                cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
                num_cols = X.select_dtypes(include=['number']).columns
            else:
                all_cols = X.columns
                if np.array(self.cat_features).dtype == bool:
                    cat_cols = all_cols[self.cat_features]
                    num_cols = all_cols[~np.array(self.cat_features)]
                else:
                    cat_cols = all_cols[self.cat_features]
                    num_cols = all_cols.difference(cat_cols)

            self.cat_indices_ = [X.columns.get_loc(c) for c in cat_cols]
            self.num_indices_ = [X.columns.get_loc(c) for c in num_cols]
            
            self.logger.info(f"GowerKNN Fitted: {len(num_cols)} numeric, {len(cat_cols)} categorical. ({n_samples} samples)")

            # Extract data
            X_num_raw = X[num_cols].values.astype(np.float32)
            X_cat_raw = X[cat_cols].values 
        else:
            # Fallback for numpy input
            X_vals = np.asarray(X)
            n_samples, n_features = X_vals.shape
            
            if self.cat_features is None:
                self.cat_indices_ = []
                self.num_indices_ = list(range(n_features))
            elif np.array(self.cat_features).dtype == bool:
                self.cat_indices_ = np.where(self.cat_features)[0]
                self.num_indices_ = np.where(~np.array(self.cat_features))[0]
            else:
                self.cat_indices_ = self.cat_features
                all_idx = np.arange(n_features)
                self.num_indices_ = np.setdiff1d(all_idx, self.cat_indices_)

            X_num_raw = X_vals[:, self.num_indices_].astype(np.float32)
            X_cat_raw = X_vals[:, self.cat_indices_]

        # Create a random generator with the seed to ensure reproducibility and be robust to data bias on initial ordering
        rng = np.random.default_rng(seed)
        # Generate a shuffled order for the reference pool
        self.shuffle_idx_ = rng.permutation(len(X))
        # Shuffle all internal data representations by this index
        if X_num_raw is not None:
            X_num_raw = X_num_raw[self.shuffle_idx_]
        if X_cat_raw is not None:
            X_cat_raw = X_cat_raw[self.shuffle_idx_]

        self.n_samples_ = n_samples
        n_features = len(self.num_indices_) + len(self.cat_indices_)

        # Weights
        if self.weights is None:
            self.w_num_ = np.ones(len(self.num_indices_), dtype=np.float32)
            self.w_cat_ = np.ones(len(self.cat_indices_), dtype=np.float32)
        else:
            if len(self.weights) != n_features:
                raise ValueError("Length of weights must match number of features.")
            w = np.array(self.weights, dtype=np.float32)
            self.w_num_ = w[np.array(self.num_indices_, dtype=int)]
            self.w_cat_ = w[np.array(self.cat_indices_, dtype=int)]
        
        # print out the features and their associated weights as a dictionary with {feature_name: weight} for both numerical and categorical features
        # if there are numerical weights:
        if len(self.num_indices_) > 0:
            self.logger.info(f"Feature Weights: Numerical: {dict(zip(self.feature_names_[self.num_indices_], self.w_num_))}") # type: ignore
        if len(self.cat_indices_) > 0:
            self.logger.info(f"Feature Weights: Categorical: {dict(zip(self.feature_names_[self.cat_indices_], self.w_cat_))}") # type: ignore

        self.w_sum_ = self.w_num_.sum() + self.w_cat_.sum()

        # 2. Process Numerical (Range Normalization)
        if len(self.num_indices_) > 0:
            self.num_complete_ = not np.isnan(X_num_raw).any()
            if self.num_complete_:
                self.X_num_mask_ = None 
                X_filled = X_num_raw
                min_vals = np.min(X_filled, axis=0) 
                max_vals = np.max(X_filled, axis=0)
            else:
                self.X_num_mask_ = (~np.isnan(X_num_raw)).astype(np.float32)
                X_filled = np.nan_to_num(X_num_raw, nan=0.0)
                min_vals = np.nanmin(X_num_raw, axis=0) 
                max_vals = np.nanmax(X_num_raw, axis=0)

            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1.0
            self.ranges_ = np.maximum(ranges.astype(np.float32), 1e-8)
            self.X_num_normalized_ = X_filled / self.ranges_
        else:
            self.X_num_normalized_ = None
            self.X_num_mask_ = None
            self.num_complete_ = True

        # 3. Process Categorical (Encoding)
        if len(self.cat_indices_) > 0:
            self.X_cat_, self.cat_encoders_ = self._encode_categorical(X_cat_raw)
            self.cat_complete_ = not (self.X_cat_ == -9).any()
            if self.cat_complete_:
                self.X_cat_mask_ = None
            else:
                self.X_cat_mask_ = (self.X_cat_ >= 0).astype(np.float32)
        else:
            self.X_cat_ = None
            self.X_cat_mask_ = None
            self.cat_complete_ = True
            self.cat_encoders_ = []

        return self

    def _compute_distances_batch(self, queries, n_queries, batch_size=512, Y_ref_num=None, Y_ref_cat=None, Y_ref_num_mask=None, Y_ref_cat_mask=None):
        """
        Memory-Optimized distance computation.
        Loops over features instead of broadcasting to avoid 3D Memory Explosion.
        """
        # Determine Reference Set
        ref_num = Y_ref_num if Y_ref_num is not None else self.X_num_normalized_
        ref_cat = Y_ref_cat if Y_ref_cat is not None else self.X_cat_
        
        # Determine Reference Masks
        ref_num_mask = Y_ref_num_mask if Y_ref_num is not None else self.X_num_mask_
        ref_num_complete = (ref_num_mask is None)
        ref_cat_mask = Y_ref_cat_mask if Y_ref_cat is not None else self.X_cat_mask_
        ref_cat_complete = (ref_cat_mask is None)

        n_samples = ref_num.shape[0] if ref_num is not None else ref_cat.shape[0] if ref_cat is not None else 0
        distances = np.zeros((n_queries, n_samples), dtype=np.float32)

        def _safe_slice(data, indices):
            if isinstance(data, pd.DataFrame):
                return data.iloc[:, indices].values
            return data[:, indices]

        # Pre-process Queries
        has_num = ref_num is not None
        if has_num:
            Q_num_raw = _safe_slice(queries, self.num_indices_).astype(np.float32)
            q_num_has_nan = np.isnan(Q_num_raw).any()
            if q_num_has_nan:
                Q_num_mask = (~np.isnan(Q_num_raw)).astype(np.float32)
                Q_num_filled = np.nan_to_num(Q_num_raw, nan=0.0)
            else:
                Q_num_mask = None
                Q_num_filled = Q_num_raw
            Q_num_norm = Q_num_filled / self.ranges_

        has_cat = ref_cat is not None
        if has_cat:
            Q_cat_raw = _safe_slice(queries, self.cat_indices_)
            Q_cat_encoded = self._encode_query_categorical(Q_cat_raw)
            q_cat_has_missing = (Q_cat_encoded == -9).any()
            if q_cat_has_missing:
                Q_cat_mask = (Q_cat_encoded != -9).astype(np.float32)
            else:
                Q_cat_mask = None

        # Batching Logic 
        self.logger.info(f"Computing Gower distances for {n_queries} queries against {n_samples} controls.")
        self.logger.info(f"Batch size: {batch_size}. Loops per batch: {len(self.num_indices_) + len(self.cat_indices_)}")

        start_time = time.time()

        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            
            # Progress Log
            if (start // batch_size) % 5 == 0:
                elapsed = time.time() - start_time
                self.logger.info(f"Processing batch {start // batch_size + 1}/{(n_queries // batch_size) + 1} ({elapsed:.1f}s elapsed)")

            # Initialize Batch
            batch_numer = np.zeros((end-start, n_samples), dtype=np.float32)
            batch_denom = np.zeros((end-start, n_samples), dtype=np.float32)

            # NUMERICAL FEATURES (Feature-wise Loop)
            if has_num:
                q_chunk = Q_num_norm[start:end] # (B, F_num)
                q_mask_chunk = Q_num_mask[start:end] if q_num_has_nan else None # type: ignore
                
                for k in range(q_chunk.shape[1]):
                    # Extract single feature column (Shape: B, 1) and (1, N)
                    # Broadcasting (B, 1) - (1, N) -> (B, N) is strictly 2D Memory.
                    col_q = q_chunk[:, k:k+1] 
                    col_ref = ref_num[:, k:k+1].T 
                    
                    diff = np.abs(col_q - col_ref)
                    
                    # Handle NaNs
                    weight = self.w_num_[k]
                    
                    if ref_num_complete and not q_num_has_nan:
                        # Fast path: No NaNs
                        batch_numer += diff * weight
                        batch_denom += weight
                    else:
                        # Complex path: Masks
                        m_q = q_mask_chunk[:, k:k+1] if q_mask_chunk is not None else 1.0
                        m_ref = ref_num_mask[:, k:k+1].T if ref_num_mask is not None else 1.0
                        combined_mask = m_q * m_ref
                        
                        batch_numer += (diff * combined_mask) * weight
                        batch_denom += combined_mask * weight

            # CATEGORICAL FEATURES (Feature-wise Loop)
            if has_cat:
                q_chunk = Q_cat_encoded[start:end]
                q_mask_chunk = Q_cat_mask[start:end] if q_cat_has_missing else None # type: ignore
                
                for k in range(q_chunk.shape[1]):
                    col_q = q_chunk[:, k:k+1]
                    col_ref = ref_cat[:, k:k+1].T
                    
                    # Categorical Difference (0 if match, 1 if different)
                    is_diff = (col_q != col_ref).astype(np.float32)
                    
                    weight = self.w_cat_[k]
                    
                    if ref_cat_complete and not q_cat_has_missing:
                         batch_numer += is_diff * weight
                         batch_denom += weight
                    else:
                        m_q = q_mask_chunk[:, k:k+1] if q_mask_chunk is not None else 1.0
                        m_ref = ref_cat_mask[:, k:k+1].T if ref_cat_mask is not None else 1.0
                        combined_mask = m_q * m_ref
                        
                        batch_numer += (is_diff * combined_mask) * weight
                        batch_denom += combined_mask * weight

            # Finalize Batch
            with np.errstate(divide='ignore', invalid='ignore'):
                batch_dists = batch_numer / batch_denom
            
            batch_dists[batch_denom == 0] = 1.0
            distances[start:end] = batch_dists

        return distances

    def _encode_categorical(self, X_cat):
        n_samples, n_feats = X_cat.shape
        X_encoded = np.full((n_samples, n_feats), -9, dtype=np.int32)
        encoders = []
        for col in range(n_feats):
            vals = X_cat[:, col]
            valid_mask = pd.notnull(vals)
            valid_vals = vals[valid_mask].astype(str)
            unique_vals, inverse = np.unique(valid_vals, return_inverse=True)
            encoder = {val: idx for idx, val in enumerate(unique_vals)}
            encoders.append(encoder)
            X_encoded[valid_mask, col] = inverse
        return X_encoded, encoders

    def _encode_query_categorical(self, Q_cat_raw):
        n_queries, n_feats = Q_cat_raw.shape
        Q_encoded = np.full((n_queries, n_feats), -9, dtype=np.int32)
        for col in range(n_feats):
            vals = Q_cat_raw[:, col]
            valid_mask = pd.notnull(vals)
            valid_vals = vals[valid_mask].astype(str)
            encoder = self.cat_encoders_[col]
            codes = [encoder.get(v, -1) for v in valid_vals]
            Q_encoded[valid_mask, col] = codes
        return Q_encoded
    
    def kneighbors(self, query, k=None, return_distance=True, fast_sort=True, batch_size=512, k_pad_mult=3, **kwargs):
        """
        Find the k-nearest neighbors using Gower distance.
        Parameters
        ----------
        query : array-like of shape (n_queries, n_features)
            The input samples to find neighbors for.
        k : int, optional
            The number of neighbors to retrieve. If None, defaults to 1.
        return_distance : bool, default=True
            Whether to return distances along with indices.
        fast_sort : bool, default=True
            Whether to use the optimized sorting method for large k.
        batch_size : int, default=512
            The batch size for distance computations.
        k_pad_mult : int, default=3
            Multiplier for k padding in fast sorting. Higher values use more memory but may be more robust if there are many ties.
        
        Returns
        -------
        distances : ndarray of shape (n_queries, k)
            Array of distances to the nearest neighbors. Returned if return_distance is True.
        indices : ndarray of shape (n_queries, k)
            Indices of the nearest neighbors in the training data.
        """
        if k is None and 'n_neighbors' in kwargs:
            k = kwargs.pop('n_neighbors')
        elif k is None:
            k=1
            self.logger.warning("k not specified in kneighbors(); defaulting to k=1.")
        check_is_fitted(self, ["n_samples_"])
        
        if isinstance(query, pd.DataFrame):
            q_vals = query.values
        else:
            q_vals = np.asarray(query)
        if q_vals.ndim == 1: q_vals = q_vals.reshape(1, -1)
        
        n_queries = q_vals.shape[0]
        distances = self._compute_distances_batch(q_vals, n_queries, batch_size=batch_size)

        self.logger.info("Distance matrix computed. Sorting neighbors...")

        k_pad = min(k * k_pad_mult, self.n_samples_ - 1)
        
        # Get internal (shuffled) indices
        if fast_sort and k_pad < self.n_samples_ - 1:
            unsorted_indices = np.argpartition(distances, k_pad, axis=1)[:, :k_pad]
            row_indices = np.arange(n_queries)[:, None]
            candidate_dists = distances[row_indices, unsorted_indices]
            sort_order = np.lexsort((unsorted_indices, candidate_dists), axis=1)
            final_indices = unsorted_indices[row_indices, sort_order][:, :k]
            final_dists = candidate_dists[row_indices, sort_order][:, :k]
        else:
            full_indices = np.broadcast_to(np.arange(self.n_samples_), (n_queries, self.n_samples_))
            sort_order = np.lexsort((full_indices, distances), axis=1)
            final_indices = sort_order[:, :k]
            row_indices = np.arange(n_queries)[:, None]
            final_dists = distances[row_indices, final_indices]

        # Map back to original indices
        final_indices = self.shuffle_idx_[final_indices]

        if return_distance:
            return final_dists, final_indices
        return final_indices
    
    def cdist(self, XA, XB=None, batch_size=512):
        """
        Computes pairwise Gower distances.
        Parameters
        ----------
        XA : array-like of shape (n_queries, n_features)
            The first set of samples.
        XB : array-like of shape (n_references, n_features), optional
            The second set of samples. If None, uses the fitted data.
        batch_size : int, default=512
            The batch size for distance computations.
        """
        check_is_fitted(self, ["n_samples_"])
        
        # 1. Standardize XA
        if isinstance(XA, pd.DataFrame):
            q_vals = XA.values
        else:
            q_vals = np.asarray(XA)
        if q_vals.ndim == 1: q_vals = q_vals.reshape(1, -1)
        n_queries = q_vals.shape[0]

        # 2. Handle XB (Reference Set)
        Y_ref_num, Y_ref_cat = None, None
        Y_ref_num_mask, Y_ref_cat_mask = None, None
        
        # Track if we are comparing against the internal shuffled data
        using_internal_ref = (XB is None)

        if XB is not None:
            def _safe_slice(data, indices):
                if isinstance(data, pd.DataFrame):
                    return data.iloc[:, indices].values
                return data[:, indices]

            # Process Numerics
            if len(self.num_indices_) > 0:
                XB_num_raw = _safe_slice(XB, self.num_indices_).astype(np.float32)
                if np.isnan(XB_num_raw).any():
                    Y_ref_num_mask = (~np.isnan(XB_num_raw)).astype(np.float32)
                    XB_num_filled = np.nan_to_num(XB_num_raw, nan=0.0)
                else:
                    Y_ref_num_mask = None
                    XB_num_filled = XB_num_raw
                Y_ref_num = XB_num_filled / self.ranges_

            # Process Categoricals
            if len(self.cat_indices_) > 0:
                XB_cat_raw = _safe_slice(XB, self.cat_indices_)
                Y_ref_cat = self._encode_query_categorical(XB_cat_raw)
                
                if (Y_ref_cat == -9).any():
                     Y_ref_cat_mask = (Y_ref_cat != -9).astype(np.float32)
                else:
                     Y_ref_cat_mask = None

        # 3. Compute Distances
        distances = self._compute_distances_batch(
            q_vals, 
            n_queries, 
            batch_size=batch_size,
            Y_ref_num=Y_ref_num,
            Y_ref_cat=Y_ref_cat,
            Y_ref_num_mask=Y_ref_num_mask,
            Y_ref_cat_mask=Y_ref_cat_mask
        )
        
        # 4. Un-shuffle columns if we used the internal reference
        if using_internal_ref:
            # Create a target array of the same shape
            final_distances = np.empty_like(distances)
            # Assign computed columns to their ORIGINAL positions.
            final_distances[:, self.shuffle_idx_] = distances
            
            return final_distances
        
        return distances