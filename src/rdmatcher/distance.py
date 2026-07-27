import logging
import os
import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from scipy.spatial import cKDTree

class GowerKNN(BaseEstimator):
    def __init__(self, weights=None, cat_features=None, n_jobs: Optional[int] = 1, parallel_chunk_size: Optional[int] = None, streaming: str = 'auto', stream_block_size: Optional[int] = None, stream_threshold_gb: float = 1.0, memory_limit_gb: Optional[float] = None, logger=None):
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
        self.n_jobs = n_jobs
        # Optional override for internal parallel chunk size (queries per worker)
        self.parallel_chunk_size = parallel_chunk_size
        # Streaming options
        self.streaming = streaming
        self.stream_block_size = stream_block_size
        # Streaming threshold (GB) used when streaming='auto'
        self.stream_threshold_gb = float(stream_threshold_gb) if stream_threshold_gb is not None else 1.0
        # Memory limit (GB) for guarding vectorized concat operations. If None, default to 4.0
        self.memory_limit_gb = float(memory_limit_gb) if memory_limit_gb is not None else None
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
                    missing = [c for c in self.cat_features if c not in all_cols]
                    if missing:
                        raise ValueError(f"cat_features contains columns not present in X: {missing}")
                    cat_cols = pd.Index(self.cat_features)
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

    def _kneighbors_grouped_l1(self, queries, k, k_pad_mult, n_jobs):
        """Exact top-k accelerator for complete mixed-type Gower data.

        Controls are split by their categorical signature.  Within a group the
        categorical term is constant, so a weighted-L1 tree supplies the only
        candidates that can enter the global top-k.  Boundary candidates are
        expanded and distances are recomputed with the established float32
        Gower accumulation before the final deterministic lexsort.
        """
        if (
            self.X_num_normalized_ is None
            or self.X_cat_ is None
            or not self.num_complete_
            or not self.cat_complete_
            or self.X_num_normalized_.shape[1] == 0
            or self.X_cat_.shape[1] == 0
        ):
            return None

        def _safe_slice(data, indices):
            if isinstance(data, pd.DataFrame):
                return data.iloc[:, indices].values
            return data[:, indices]

        q_num_raw = _safe_slice(queries, self.num_indices_).astype(np.float32)
        if np.isnan(q_num_raw).any():
            return None
        q_num_norm = q_num_raw / self.ranges_
        q_cat = self._encode_query_categorical(_safe_slice(queries, self.cat_indices_))
        if (q_cat == -9).any():
            return None

        if not hasattr(self, "_grouped_l1_positions_"):
            if not hasattr(self, "_grouped_l1_group_ids_"):
                signatures, group_ids, group_sizes = np.unique(
                    self.X_cat_, axis=0, return_inverse=True, return_counts=True
                )
                self._grouped_l1_signatures_ = signatures
                self._grouped_l1_group_ids_ = group_ids
                self._grouped_l1_group_sizes_ = group_sizes
            self._grouped_l1_positions_ = [
                np.flatnonzero(self._grouped_l1_group_ids_ == group).astype(np.int32)
                for group in range(len(self._grouped_l1_signatures_))
            ]

        if not hasattr(self, "_grouped_l1_trees_"):
            scaled_num = (self.X_num_normalized_ * self.w_num_).astype(np.float64, copy=False)
            self._grouped_l1_trees_ = [
                cKDTree(scaled_num[pos]) for pos in self._grouped_l1_positions_
            ]
            self._grouped_l1_max_abs_ = [
                np.max(np.abs(scaled_num[pos]), axis=0)
                for pos in self._grouped_l1_positions_
            ]

        n_queries = q_num_norm.shape[0]
        final_distances = np.empty((n_queries, k), dtype=np.float32)
        final_positions = np.empty((n_queries, k), dtype=np.int32)
        scaled_queries = (q_num_norm * self.w_num_).astype(np.float64, copy=False)
        if n_jobs is None:
            tree_workers = 1
        elif n_jobs == -1:
            tree_workers = os.cpu_count() or 1
        else:
            tree_workers = int(n_jobs)

        constant_denom = np.float32(0.0)
        for weight in self.w_num_:
            constant_denom = np.float32(constant_denom + weight)
        for weight in self.w_cat_:
            constant_denom = np.float32(constant_denom + weight)

        for row in range(n_queries):
            candidate_parts = []
            for positions, tree, max_abs in zip(
                self._grouped_l1_positions_, self._grouped_l1_trees_, self._grouped_l1_max_abs_
            ):
                k_eff = min(int(k), len(positions))
                if k_eff == len(positions):
                    candidate_parts.append(positions)
                    continue
                kth_distance, _ = tree.query(
                    scaled_queries[row], k=k_eff, p=1, workers=tree_workers
                )
                kth_distance = float(np.atleast_1d(kth_distance)[-1])
                # The tree accumulates in float64 while the production Gower
                # kernel accumulates in float32. Retain every plausible kth tie.
                rounding_allowance = (
                    float(os.getenv("RD_MATCHER_GROUPED_L1_ROUNDING_MULT", "16"))
                    * np.finfo(np.float32).eps
                    * float(np.sum(np.abs(scaled_queries[row]) + max_abs))
                )
                local = tree.query_ball_point(
                    scaled_queries[row], kth_distance + rounding_allowance, p=1, workers=tree_workers
                )
                candidate_parts.append(positions[np.asarray(local, dtype=np.intp)])
            candidate_positions = np.concatenate(candidate_parts)

            candidate_distances = np.zeros(candidate_positions.size, dtype=np.float32)
            for feature, weight in enumerate(self.w_num_):
                diff = np.abs(q_num_norm[row, feature] - self.X_num_normalized_[candidate_positions, feature])
                candidate_distances += diff * weight
            for feature, weight in enumerate(self.w_cat_):
                diff = (q_cat[row, feature] != self.X_cat_[candidate_positions, feature]).astype(np.float32)
                candidate_distances += diff * self.w_cat_[feature]
            if constant_denom != 0:
                candidate_distances /= constant_denom
            else:
                candidate_distances.fill(1.0)

            order = np.lexsort((candidate_positions, candidate_distances))[:k]
            final_positions[row] = candidate_positions[order]
            final_distances[row] = candidate_distances[order]

        return final_distances, self.shuffle_idx_[final_positions]

    def _should_use_grouped_l1(self, n_queries, k, n_jobs):
        """Use the exact grouped index only when its candidate set is small."""
        if (
            self.X_num_normalized_ is None
            or self.X_cat_ is None
            or not self.num_complete_
            or not self.cat_complete_
            or self.X_num_normalized_.shape[1] == 0
            or self.X_cat_.shape[1] == 0
            or n_queries < 50
            or self.n_samples_ < 50_000
            or n_jobs not in (None, 1)
        ):
            return False
        if not hasattr(self, "_grouped_l1_group_ids_"):
            signatures, group_ids, group_sizes = np.unique(
                self.X_cat_, axis=0, return_inverse=True, return_counts=True
            )
            self._grouped_l1_signatures_ = signatures
            self._grouped_l1_group_ids_ = group_ids
            self._grouped_l1_group_sizes_ = group_sizes
        group_count = len(self._grouped_l1_group_sizes_)
        if group_count < 2 or group_count > 64:
            return False
        estimated_candidates = int(np.minimum(self._grouped_l1_group_sizes_, int(k)).sum())
        if estimated_candidates > 0.05 * self.n_samples_:
            return False
        if not hasattr(self, "_grouped_l1_positions_"):
            self._grouped_l1_positions_ = [
                np.flatnonzero(self._grouped_l1_group_ids_ == group).astype(np.int32)
                for group in range(group_count)
            ]
        return True

    def _compute_distances_batch(self, queries, n_queries, batch_size=512, Y_ref_num=None, Y_ref_cat=None, Y_ref_num_mask=None, Y_ref_cat_mask=None, n_jobs: Optional[int] = 1):
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
        q_num_has_nan = False
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
        q_cat_has_missing = False
        if has_cat:
            Q_cat_raw = _safe_slice(queries, self.cat_indices_)
            Q_cat_encoded = self._encode_query_categorical(Q_cat_raw)
            q_cat_has_missing = (Q_cat_encoded == -9).any()
            if q_cat_has_missing:
                Q_cat_mask = (Q_cat_encoded != -9).astype(np.float32)
            else:
                Q_cat_mask = None

        # Precompute reference transposes to avoid repeated slicing/transposes in hot loops
        if ref_num is not None:
            ref_num_T = ref_num.T  # shape (F_num, n_samples)
            ref_num_mask_T = ref_num_mask.T if ref_num_mask is not None else None
        else:
            ref_num_T = None
            ref_num_mask_T = None

        if ref_cat is not None:
            ref_cat_T = ref_cat.T  # shape (F_cat, n_samples)
            ref_cat_mask_T = ref_cat_mask.T if ref_cat_mask is not None else None
        else:
            ref_cat_T = None
            ref_cat_mask_T = None

        # Skip zero-weight features to save work
        if has_num and hasattr(self, 'w_num_') and self.w_num_ is not None:
            num_feat_indices = np.where(self.w_num_ != 0)[0]
        else:
            num_feat_indices = np.array([], dtype=int)

        if has_cat and hasattr(self, 'w_cat_') and self.w_cat_ is not None:
            cat_feat_indices = np.where(self.w_cat_ != 0)[0]
        else:
            cat_feat_indices = np.array([], dtype=int)

        # In the common complete-data case every pair has the same Gower
        # denominator.  The former path materialized and updated a second
        # (n_queries x n_samples) float32 matrix once per feature even though
        # all of its entries were identical.  Retain the feature-wise
        # numerator accumulation (and therefore its floating-point order),
        # but use a scalar denominator instead.
        complete_observations = (
            ref_num_complete and not q_num_has_nan
            and ref_cat_complete and not q_cat_has_missing
        )
        # A mixed-completeness input can still have a complete numeric or
        # categorical side.  Those individual feature loops also use the
        # scratch buffer even though the pairwise denominator is required.
        has_complete_feature_block = (
            (has_num and ref_num_complete and not q_num_has_nan)
            or (has_cat and ref_cat_complete and not q_cat_has_missing)
        )
        constant_denom = np.float32(0.0)
        if complete_observations:
            if has_num:
                for kk in num_feat_indices if num_feat_indices.size else range(Q_num_norm.shape[1]):
                    constant_denom = np.float32(constant_denom + self.w_num_[kk])
            if has_cat:
                for kk in cat_feat_indices if cat_feat_indices.size else range(Q_cat_encoded.shape[1]):
                    constant_denom = np.float32(constant_denom + self.w_cat_[kk])

        # Batching Logic 
        self.logger.info(f"Computing Gower distances for {n_queries} queries against {n_samples} controls.")
        self.logger.info(f"Batch size: {batch_size}. Loops per batch: {len(self.num_indices_) + len(self.cat_indices_)}")

        start_time = time.time()

        # Normalize n_jobs semantics: default to single-threaded unless specified
        if n_jobs is None:
            workers = 1
        elif n_jobs == -1:
            workers = os.cpu_count() or 1
        elif isinstance(n_jobs, int) and n_jobs >= 1:
            workers = int(n_jobs)
        else:
            raise ValueError("n_jobs must be None, -1, or an integer >= 1")

        # Determine chunking strategy: by default create chunks based on workers so there
        # are multiple chunks per batch when n_jobs>1. We still respect the outer batch_size
        # as the maximum slice size fed from the caller.
        # Use 'workers' here (requested worker count) — 'max_workers' is computed later
        # as the min(workers, number_of_chunks).
        if workers > 1:
            # If user supplied an explicit parallel_chunk_size on the model, use it
            if hasattr(self, 'parallel_chunk_size') and self.parallel_chunk_size:
                inner_chunk = int(self.parallel_chunk_size)
            else:
                # Evenly partition the n_queries across workers to create chunks
                inner_chunk = max(1, int(math.ceil(n_queries / workers)))
            chunk_size = inner_chunk
        else:
            chunk_size = max(1, int(batch_size))

        chunk_ranges = [(start, min(start + chunk_size, n_queries)) for start in range(0, n_queries, chunk_size)]

        # Cap the number of threads to the number of chunks (no point more threads than chunks)
        max_workers = min(workers, len(chunk_ranges)) if len(chunk_ranges) > 0 else 1

        # If only one worker or one chunk, fall back to serial execution for minimal overhead
        if max_workers <= 1:
            for start, end in chunk_ranges:
                # Progress Log
                if (start // batch_size) % 5 == 0:
                    elapsed = time.time() - start_time
                    self.logger.info(f"Processing batch {start // batch_size + 1}/{(n_queries // batch_size) + 1} ({elapsed:.1f}s elapsed)")

                # Initialize Batch
                batch_numer = np.zeros((end-start, n_samples), dtype=np.float32)
                batch_denom = None if complete_observations else np.zeros((end-start, n_samples), dtype=np.float32)
                # The complete-data kernel needs only one feature-sized scratch
                # matrix.  Reusing it avoids allocating a full (B, N) ``diff``
                # and weighted temporary for every feature.  Keep the missing
                # data kernel unchanged because its masks require additional
                # intermediates to preserve its established semantics.
                work = np.empty_like(batch_numer) if has_complete_feature_block else None
                cat_work = np.empty(batch_numer.shape, dtype=bool) if has_cat and ref_cat_complete and not q_cat_has_missing else None

                # NUMERICAL FEATURES (Feature-wise Loop)
                if has_num:
                    q_chunk = Q_num_norm[start:end]  # (B, F_num)
                    q_mask_chunk = Q_num_mask[start:end] if q_num_has_nan else None  # type: ignore

                    # Iterate only non-zero-weight numeric features when available
                    if num_feat_indices.size:
                        feat_range = num_feat_indices
                    else:
                        feat_range = range(q_chunk.shape[1])

                    for kk in feat_range:
                        # Extract single feature column (Shape: B, 1) and (1, N)
                        col_q = q_chunk[:, kk:kk+1]
                        # Use precomputed transpose for reference
                        col_ref = ref_num_T[kk:kk+1]  # shape (1, n_samples)
                        # Handle NaNs
                        weight = self.w_num_[kk]

                        if ref_num_complete and not q_num_has_nan:
                            np.subtract(col_q, col_ref, out=work)
                            np.abs(work, out=work)
                            np.multiply(work, weight, out=work)
                            batch_numer += work
                            if batch_denom is not None:
                                batch_denom += weight
                        else:
                            diff = np.abs(col_q - col_ref)
                            m_q = q_mask_chunk[:, kk:kk+1] if q_mask_chunk is not None else 1.0
                            m_ref = ref_num_mask_T[kk:kk+1] if ref_num_mask_T is not None else 1.0
                            combined_mask = m_q * m_ref
                            batch_numer += (diff * combined_mask) * weight
                            batch_denom += combined_mask * weight

                # CATEGORICAL FEATURES (Feature-wise Loop)
                if has_cat:
                    q_chunk = Q_cat_encoded[start:end]
                    q_mask_chunk = Q_cat_mask[start:end] if q_cat_has_missing else None  # type: ignore

                    if cat_feat_indices.size:
                        cat_range = cat_feat_indices
                    else:
                        cat_range = range(q_chunk.shape[1])

                    for kk in cat_range:
                        col_q = q_chunk[:, kk:kk+1]
                        col_ref = ref_cat_T[kk:kk+1]

                        weight = self.w_cat_[kk]

                        if ref_cat_complete and not q_cat_has_missing:
                            # Reuse a boolean comparison buffer and the numeric
                            # work buffer rather than allocating ``is_diff``.
                            np.not_equal(col_q, col_ref, out=cat_work)
                            np.multiply(cat_work, weight, out=work)
                            batch_numer += work
                            if batch_denom is not None:
                                batch_denom += weight
                        else:
                            # Categorical Difference (0 if match, 1 if different)
                            is_diff = (col_q != col_ref).astype(np.float32)
                            m_q = q_mask_chunk[:, kk:kk+1] if q_mask_chunk is not None else 1.0
                            m_ref = ref_cat_mask_T[kk:kk+1] if ref_cat_mask_T is not None else 1.0
                            combined_mask = m_q * m_ref
                            batch_numer += (is_diff * combined_mask) * weight
                            batch_denom += combined_mask * weight

                # Finalize Batch
                if batch_denom is None:
                    if constant_denom == 0:
                        batch_dists = np.ones_like(batch_numer)
                    else:
                        batch_dists = batch_numer / constant_denom
                else:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        batch_dists = batch_numer / batch_denom
                    batch_dists[batch_denom == 0] = 1.0
                distances[start:end] = batch_dists

            return distances

        # Parallel execution using ThreadPoolExecutor
        self.logger.info(f"Parallel Gower distance computation using {max_workers} threads over {len(chunk_ranges)} chunks.")

        # Quick memory estimate and warning (not enforced):
        # Complete data uses one float32 scratch plus (when categorical
        # features exist) one boolean comparison buffer.  Missing-data inputs
        # retain their two float32 working matrices.
        if complete_observations:
            est_per_worker_bytes = chunk_size * n_samples * (8 + (1 if has_cat else 0))
        else:
            est_per_worker_bytes = 2 * chunk_size * n_samples * 4
        est_total_bytes = est_per_worker_bytes * max_workers
        est_total_gb = est_total_bytes / (1024 ** 3)
        # Parse memory limit with guard against invalid environment variables
        mem_env = os.getenv('RD_MATCHER_MEMORY_LIMIT_GB', '4')
        # Prefer explicit memory_limit_gb on the model (passed through Matcher); else fallback to env var; else default 4.0
        mem_limit = getattr(self, 'memory_limit_gb', None)
        if mem_limit is not None:
            try:
                mem_limit_gb = float(mem_limit)
            except Exception:
                self.logger.warning(f"memory_limit_gb='{mem_limit}' on model is not a valid float. Falling back to 4 GB.")
                mem_limit_gb = 4.0
        else:
            try:
                mem_limit_gb = float(mem_env)
            except Exception:
                self.logger.warning(f"RD_MATCHER_MEMORY_LIMIT_GB='{mem_env}' is not a valid float. Falling back to 4 GB.")
                mem_limit_gb = 4.0
        if est_total_gb > mem_limit_gb:
            self.logger.warning(
                f"Estimated memory for parallel Gower distance is {est_total_gb:.2f} GB which exceeds the configured warning limit of {mem_limit_gb} GB. "
                f"Consider reducing n_jobs or batch_size to avoid high memory usage. You can set RD_MATCHER_MEMORY_LIMIT_GB to adjust this threshold."
            )

        def _compute_chunk(start, end):
            # Local buffers
            local_numer = np.zeros((end-start, n_samples), dtype=np.float32)
            local_denom = None if complete_observations else np.zeros((end-start, n_samples), dtype=np.float32)
            work = np.empty_like(local_numer) if has_complete_feature_block else None
            cat_work = np.empty(local_numer.shape, dtype=bool) if has_cat and ref_cat_complete and not q_cat_has_missing else None

            # NUMERICAL FEATURES
            if has_num:
                q_chunk = Q_num_norm[start:end]
                q_mask_chunk = Q_num_mask[start:end] if q_num_has_nan else None
                feat_range = num_feat_indices if num_feat_indices.size else range(q_chunk.shape[1])
                for k in feat_range:
                    col_q = q_chunk[:, k:k+1]
                    col_ref = ref_num_T[k:k+1]
                    weight = self.w_num_[k]
                    if ref_num_complete and not q_num_has_nan:
                        np.subtract(col_q, col_ref, out=work)
                        np.abs(work, out=work)
                        np.multiply(work, weight, out=work)
                        local_numer += work
                        if local_denom is not None:
                            local_denom += weight
                    else:
                        diff = np.abs(col_q - col_ref)
                        m_q = q_mask_chunk[:, k:k+1] if q_mask_chunk is not None else 1.0
                        m_ref = ref_num_mask_T[k:k+1] if ref_num_mask_T is not None else 1.0
                        combined_mask = m_q * m_ref
                        local_numer += (diff * combined_mask) * weight
                        local_denom += combined_mask * weight

            # CATEGORICAL FEATURES
            if has_cat:
                q_chunk = Q_cat_encoded[start:end]
                q_mask_chunk = Q_cat_mask[start:end] if q_cat_has_missing else None
                feat_range = cat_feat_indices if cat_feat_indices.size else range(q_chunk.shape[1])
                for k in feat_range:
                    col_q = q_chunk[:, k:k+1]
                    col_ref = ref_cat_T[k:k+1]
                    weight = self.w_cat_[k]
                    if ref_cat_complete and not q_cat_has_missing:
                        np.not_equal(col_q, col_ref, out=cat_work)
                        np.multiply(cat_work, weight, out=work)
                        local_numer += work
                        if local_denom is not None:
                            local_denom += weight
                    else:
                        is_diff = (col_q != col_ref).astype(np.float32)
                        m_q = q_mask_chunk[:, k:k+1] if q_mask_chunk is not None else 1.0
                        m_ref = ref_cat_mask_T[k:k+1] if ref_cat_mask_T is not None else 1.0
                        combined_mask = m_q * m_ref
                        local_numer += (is_diff * combined_mask) * weight
                        local_denom += combined_mask * weight

            if local_denom is None:
                if constant_denom == 0:
                    local_dists = np.ones_like(local_numer)
                else:
                    local_dists = local_numer / constant_denom
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    local_dists = local_numer / local_denom
                local_dists[local_denom == 0] = 1.0
            return start, local_dists

        # Submit all chunks
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for start, end in chunk_ranges:
                # Progress log only for initial scheduling
                if (start // batch_size) % 5 == 0:
                    elapsed = time.time() - start_time
                    # Scheduling logs can be noisy; emit at DEBUG level
                    self.logger.debug(f"Scheduling batch {start // batch_size + 1}/{(n_queries // batch_size) + 1} ({elapsed:.1f}s elapsed)")
                futures.append(exe.submit(_compute_chunk, start, end))

            # Collect results as they complete and write into distances
            for fut in as_completed(futures):
                start_idx, chunk_dists = fut.result()
                end_idx = start_idx + chunk_dists.shape[0]
                distances[start_idx:end_idx] = chunk_dists

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
        
        # Preserve DataFrames so categorical columns keep their original dtype.
        # Converting mixed numeric/categorical frames to .values can coerce
        # integer categories to floats, e.g. 1 -> 1.0, which breaks exact
        # categorical encoding by string key.
        if isinstance(query, pd.DataFrame):
            q_vals = query
        else:
            q_vals = np.asarray(query)
            if q_vals.ndim == 1: q_vals = q_vals.reshape(1, -1)
        
        n_queries = q_vals.shape[0]
        # Allow n_jobs and streaming options to be passed through kwargs; fall back to instance defaults
        n_jobs = kwargs.get('n_jobs', None)
        if n_jobs is None:
            n_jobs = self.n_jobs if hasattr(self, 'n_jobs') else 1

        streaming = kwargs.get('streaming', None)
        if streaming is None:
            streaming = getattr(self, 'streaming', 'auto')

        stream_block_size = kwargs.get('stream_block_size', None)
        if stream_block_size is None:
            stream_block_size = getattr(self, 'stream_block_size', None)

        # This exact path is enabled automatically only for the large,
        # complete-data serial cases for which grouping sharply reduces the
        # candidate graph. Set RD_MATCHER_GROUPED_L1_KNN=0 to disable it or
        # =1 to force it for diagnostics.
        grouped_mode = os.getenv("RD_MATCHER_GROUPED_L1_KNN", "auto").lower()
        use_grouped_l1 = (
            grouped_mode == "1"
            or (grouped_mode == "auto" and self._should_use_grouped_l1(n_queries, k, n_jobs))
        )
        if use_grouped_l1:
            grouped_result = self._kneighbors_grouped_l1(q_vals, k, k_pad_mult, n_jobs)
            if grouped_result is not None:
                final_dists, final_indices = grouped_result
                if return_distance:
                    return final_dists, final_indices
                return final_indices

        # Decide whether to use streaming path
        use_streaming = False
        if streaming == 'on':
            use_streaming = True
        elif streaming == 'off':
            use_streaming = False
        else:  # auto
            # Estimate full matrix size in bytes (float32)
            n_ref_guess = (self.X_num_normalized_.shape[0] if self.X_num_normalized_ is not None else (self.X_cat_.shape[0] if self.X_cat_ is not None else 0))
            est_full_bytes = n_queries * n_ref_guess * 4
            # Stream if estimated full bytes > stream_threshold_gb (convert GB->bytes)
            stream_threshold = float(getattr(self, 'stream_threshold_gb', 1.0))
            use_streaming = est_full_bytes > (stream_threshold * (1024 ** 3))

        if use_streaming:
            final_dists, final_indices = self._kneighbors_streaming(q_vals, n_queries, n_neighbors=k, batch_size=batch_size, k_pad_mult=k_pad_mult, n_jobs=n_jobs, stream_block_size=stream_block_size)
            if return_distance:
                return final_dists, final_indices
            return final_indices
        else:
            distances = self._compute_distances_batch(q_vals, n_queries, batch_size=batch_size, n_jobs=n_jobs)

        self.logger.info("Distance matrix computed. Sorting neighbors...")

        k_pad = min(k * k_pad_mult, self.n_samples_ - 1)

        # The distance kernel may already have used several workers, but the
        # subsequent top-k selection was a single large argpartition over all
        # query rows.  Rows are independent, so partitioning row chunks keeps
        # the exact per-row (distance, internal-index) ordering while allowing
        # the selection work to use the requested workers as well.
        if n_jobs is None:
            selection_workers = 1
        elif n_jobs == -1:
            selection_workers = os.cpu_count() or 1
        elif isinstance(n_jobs, int) and n_jobs >= 1:
            selection_workers = n_jobs
        else:
            raise ValueError("n_jobs must be None, -1, or an integer >= 1")
        
        # Get internal (shuffled) indices
        if fast_sort and k_pad < self.n_samples_ - 1:
            if selection_workers > 1 and n_queries > 1:
                final_indices = np.empty((n_queries, k), dtype=np.int32)
                final_dists = np.empty((n_queries, k), dtype=np.float32)

                def _select_rows(start, end):
                    local_distances = distances[start:end]
                    unsorted_indices = np.argpartition(local_distances, k_pad, axis=1)[:, :k_pad]
                    row_indices = np.arange(end - start)[:, None]
                    candidate_dists = local_distances[row_indices, unsorted_indices]
                    sort_order = np.lexsort((unsorted_indices, candidate_dists), axis=1)
                    return (
                        start,
                        unsorted_indices[row_indices, sort_order][:, :k],
                        candidate_dists[row_indices, sort_order][:, :k],
                    )

                chunk_size = max(1, int(math.ceil(n_queries / min(selection_workers, n_queries))))
                row_chunks = [
                    (start, min(start + chunk_size, n_queries))
                    for start in range(0, n_queries, chunk_size)
                ]
                with ThreadPoolExecutor(max_workers=min(selection_workers, len(row_chunks))) as executor:
                    futures = [executor.submit(_select_rows, start, end) for start, end in row_chunks]
                    for future in as_completed(futures):
                        start, local_indices, local_dists = future.result()
                        end = start + local_indices.shape[0]
                        final_indices[start:end] = local_indices
                        final_dists[start:end] = local_dists
            else:
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

    def _kneighbors_streaming(self, q_vals, n_queries, n_neighbors=1, batch_size=512, k_pad_mult=3, n_jobs=1, stream_block_size=None):
        """
        Streaming top-k implementation: iterate over control/reference blocks
        and maintain per-query top-k candidates incrementally.
        Returns distances and indices arrays of shape (n_queries, n_neighbors).
        """
        # Validate inputs
        if stream_block_size is None:
            stream_block_size = 50000  # default block size
        else:
            stream_block_size = int(stream_block_size)

        n_controls = self.n_samples_
        if n_controls <= 0:
            raise ValueError("No reference controls available for kneighbors().")

        k = int(n_neighbors)
        # Do not request more neighbors than references
        k = min(k, n_controls)
        # k_pad: padded selection size; ensure at least 1
        k_pad = min(k * k_pad_mult, n_controls)
        if k_pad < 1:
            k_pad = 1

        # Pre-allocate running best arrays per query
        best_d = np.full((n_queries, k_pad), np.inf, dtype=np.float32)
        best_i = np.full((n_queries, k_pad), -1, dtype=np.int32)

        # Loop over reference blocks
        for start_ref in range(0, n_controls, stream_block_size):
            end_ref = min(start_ref + stream_block_size, n_controls)

            # Extract reference subset in original (unshuffled) positions then pass to compute_distances
            # Re-use cdist-style pre-processing by calling _compute_distances_batch with Y_ref arrays
            # Prepare Y_ref arrays in the format expected
            # For robustness, build Y_ref_num/Y_ref_cat from internal X arrays
            Y_ref_num = None
            Y_ref_cat = None
            Y_ref_num_mask = None
            Y_ref_cat_mask = None

            if self.X_num_normalized_ is not None:
                Y_ref_num = self.X_num_normalized_[start_ref:end_ref]
                Y_ref_num_mask = self.X_num_mask_[start_ref:end_ref] if self.X_num_mask_ is not None else None

            if self.X_cat_ is not None:
                Y_ref_cat = self.X_cat_[start_ref:end_ref]
                Y_ref_cat_mask = self.X_cat_mask_[start_ref:end_ref] if self.X_cat_mask_ is not None else None

            # Compute distances for all queries against this small ref block
            dblock = self._compute_distances_batch(q_vals, n_queries, batch_size=batch_size, Y_ref_num=Y_ref_num, Y_ref_cat=Y_ref_cat, Y_ref_num_mask=Y_ref_num_mask, Y_ref_cat_mask=Y_ref_cat_mask, n_jobs=n_jobs)

            # dblock shape: (n_queries, block_size)
            # Candidate indices in original control index space
            ref_indices = np.arange(start_ref, end_ref, dtype=np.int32)

            # Reduce this block to its own padded top-k before merging it into
            # the running result.  Concatenating a full ``dblock`` with
            # ``best_d`` used to copy a (n_queries, block_size) distance
            # matrix and an equally large index matrix on every block.  A
            # value outside a block's own top-k cannot enter the global top-k,
            # so only those compact candidates need to be merged.
            block_size = dblock.shape[1]
            row_idx = np.arange(n_queries)[:, None]
            if block_size > k_pad:
                block_part = np.argpartition(dblock, k_pad, axis=1)[:, :k_pad]
                block_d = dblock[row_idx, block_part]
                block_i = (block_part + start_ref).astype(np.int32, copy=False)
            else:
                block_d = dblock
                block_i = np.broadcast_to(ref_indices, (n_queries, block_size))

            concat_d = np.concatenate([best_d, block_d], axis=1)
            concat_i = np.concatenate([best_i, block_i], axis=1)
            if concat_d.shape[1] <= k_pad:
                order = np.argsort(concat_d, axis=1, kind='stable')
                sel = order[:, :k_pad]
            else:
                part = np.argpartition(concat_d, k_pad, axis=1)[:, :k_pad]
                sel_order = np.argsort(concat_d[row_idx, part], axis=1, kind='stable')
                sel = part[row_idx, sel_order]

            best_d = np.take_along_axis(concat_d, sel, axis=1)
            best_i = np.take_along_axis(concat_i, sel, axis=1)

        # After all blocks, final selection to k neighbors and remap shuffled indices
        final_order = np.argsort(best_d, axis=1)
        final_dists = np.take_along_axis(best_d, final_order, axis=1)[:, :k]
        final_idxs = np.take_along_axis(best_i, final_order, axis=1)[:, :k]

        # Map back to original indices (unshuffle)
        final_idxs = self.shuffle_idx_[final_idxs]

        return final_dists, final_idxs

    def kneighbors_subset(self, query, XB, k=None, return_distance=True, batch_size=512, k_pad_mult=3, **kwargs):
        """
        Find top-k neighbors for query rows within an explicit reference subset XB.
        This reuses the incremental top-k logic instead of materializing the full
        query x subset distance matrix when the subset is large.
        """
        if k is None and 'n_neighbors' in kwargs:
            k = kwargs.pop('n_neighbors')
        elif k is None:
            k = 1
            self.logger.warning("k not specified in kneighbors_subset(); defaulting to k=1.")
        check_is_fitted(self, ["n_samples_"])

        if isinstance(query, pd.DataFrame):
            q_vals = query
        else:
            q_vals = np.asarray(query)
            if q_vals.ndim == 1:
                q_vals = q_vals.reshape(1, -1)

        if isinstance(XB, pd.DataFrame):
            n_ref = len(XB)
        else:
            XB = np.asarray(XB)
            if XB.ndim == 1:
                XB = XB.reshape(1, -1)
            n_ref = XB.shape[0]

        if n_ref <= 0:
            raise ValueError("No reference controls available for kneighbors_subset().")

        n_queries = q_vals.shape[0]
        k = min(int(k), n_ref)
        k_pad = min(max(k * k_pad_mult, 1), n_ref)

        n_jobs = kwargs.get('n_jobs', None)
        if n_jobs is None:
            n_jobs = self.n_jobs if hasattr(self, 'n_jobs') else 1

        streaming = kwargs.get('streaming', None)
        if streaming is None:
            streaming = getattr(self, 'streaming', 'auto')

        stream_block_size = kwargs.get('stream_block_size', None)
        if stream_block_size is None:
            stream_block_size = getattr(self, 'stream_block_size', None)
        if stream_block_size is None:
            stream_block_size = 50000
        else:
            stream_block_size = int(stream_block_size)

        use_streaming = False
        if streaming == 'on':
            use_streaming = True
        elif streaming == 'off':
            use_streaming = False
        else:
            est_full_bytes = n_queries * n_ref * 4
            stream_threshold = float(getattr(self, 'stream_threshold_gb', 1.0))
            use_streaming = est_full_bytes > (stream_threshold * (1024 ** 3))

        if not use_streaming:
            distances = self.cdist(q_vals, XB=XB, batch_size=batch_size, n_jobs=n_jobs)
            row_idx = np.arange(n_queries)[:, None]
            if k >= n_ref:
                order = np.argsort(distances, axis=1, kind='stable')
                final_indices = order[:, :k]
            else:
                part = np.argpartition(distances, k - 1, axis=1)[:, :k]
                sel_order = np.argsort(distances[row_idx, part], axis=1, kind='stable')
                final_indices = part[row_idx, sel_order]
            final_dists = distances[row_idx, final_indices]
            if return_distance:
                return final_dists, final_indices
            return final_indices

        best_d = np.full((n_queries, k_pad), np.inf, dtype=np.float32)
        best_i = np.full((n_queries, k_pad), -1, dtype=np.int32)

        for start_ref in range(0, n_ref, stream_block_size):
            end_ref = min(start_ref + stream_block_size, n_ref)
            XB_block = XB.iloc[start_ref:end_ref] if isinstance(XB, pd.DataFrame) else XB[start_ref:end_ref]
            dblock = self.cdist(q_vals, XB=XB_block, batch_size=batch_size, n_jobs=n_jobs)
            block_size = dblock.shape[1]
            ref_indices = np.arange(start_ref, end_ref, dtype=np.int32)

            concat_d = np.concatenate([best_d, dblock.astype(np.float32)], axis=1)
            ids_block_mat = np.broadcast_to(ref_indices[None, :], (n_queries, block_size))
            concat_i = np.concatenate([best_i, ids_block_mat.astype(np.int32)], axis=1)

            if concat_d.shape[1] <= k_pad:
                order = np.argsort(concat_d, axis=1, kind='stable')
                sel = order[:, :k_pad]
            else:
                part = np.argpartition(concat_d, k_pad - 1, axis=1)[:, :k_pad]
                row_idx = np.arange(n_queries)[:, None]
                sel_order = np.argsort(concat_d[row_idx, part], axis=1, kind='stable')
                sel = part[row_idx, sel_order]

            best_d = np.take_along_axis(concat_d, sel, axis=1)
            best_i = np.take_along_axis(concat_i, sel, axis=1)

        final_order = np.argsort(best_d, axis=1, kind='stable')
        final_dists = np.take_along_axis(best_d, final_order, axis=1)[:, :k]
        final_indices = np.take_along_axis(best_i, final_order, axis=1)[:, :k]

        if return_distance:
            return final_dists, final_indices
        return final_indices
    
    def cdist(self, XA, XB=None, batch_size=512, n_jobs: Optional[int] = None):
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
        # Preserve DataFrames so categorical columns keep their original dtype.
        # Converting mixed numeric/categorical frames to .values can coerce
        # integer categories to floats, e.g. 1 -> 1.0, which breaks exact
        # categorical encoding by string key.
        if isinstance(XA, pd.DataFrame):
            q_vals = XA
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
        # Determine n_jobs (explicit argument overrides instance default)
        if n_jobs is None:
            n_jobs = getattr(self, 'n_jobs', 1)

        # Heuristic: only parallelize cdist when user requests >1 workers
        # and the problem size is sufficiently large. Use 1e5 as threshold (user-requested).
        parallel_threshold = int(1e5)
        # Determine reference set size robustly: if XB is None we are comparing against the
        # internally fitted reference pool (self.n_samples_), otherwise use the provided XB rows
        if XB is None:
            n_ref = getattr(self, 'n_samples_', 0)
        else:
            n_ref = Y_ref_num.shape[0] if Y_ref_num is not None else (Y_ref_cat.shape[0] if Y_ref_cat is not None else 0)
        problem_size = n_queries * n_ref

        if (isinstance(n_jobs, int) and n_jobs > 1) and (problem_size >= parallel_threshold):
            distances = self._compute_distances_batch(
                q_vals,
                n_queries,
                batch_size=batch_size,
                Y_ref_num=Y_ref_num,
                Y_ref_cat=Y_ref_cat,
                Y_ref_num_mask=Y_ref_num_mask,
                Y_ref_cat_mask=Y_ref_cat_mask,
                n_jobs=n_jobs
            )
        else:
            # Default serial path
            distances = self._compute_distances_batch(
                q_vals,
                n_queries,
                batch_size=batch_size,
                Y_ref_num=Y_ref_num,
                Y_ref_cat=Y_ref_cat,
                Y_ref_num_mask=Y_ref_num_mask,
                Y_ref_cat_mask=Y_ref_cat_mask,
                n_jobs=1
            )
        
        # 4. Un-shuffle columns if we used the internal reference
        if using_internal_ref:
            # Create a target array of the same shape
            final_distances = np.empty_like(distances)
            # Assign computed columns to their ORIGINAL positions.
            final_distances[:, self.shuffle_idx_] = distances
            
            return final_distances
        
        return distances
