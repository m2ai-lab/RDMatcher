import logging
import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
from typing import Optional


class MahalanobisKNN(BaseEstimator):
    """Lightweight, numeric-only Mahalanobis distance estimator.

    API: fit(X), cdist(queries, Y_ref=None, batch_size=256), kneighbors(queries, n_neighbors)

    Notes:
    - Expects numeric-only inputs. DataFrame inputs with non-numeric dtypes will raise.
    - Uses a whitened-Euclidean internally for speed and stability where possible.
    - Regularizes covariance adaptively if it is singular or near-singular.
    """

    def __init__(
        self,
        regularization: float = 1e-6,
        neighbor_backend: str = "cdist",
        algorithm: str = "auto",
        n_jobs: Optional[int] = 1,
        logger=None,
    ):
        """Initialize MahalanobisKNN.

        Notes:
        - `regularization` is the initial ridge added to the sample covariance.
        - `neighbor_backend='cdist'` preserves the original exact full-distance path.
        - `neighbor_backend='sklearn'` whitens once and uses sklearn's Euclidean
          NearestNeighbors for top-k candidate search.
        """
        self.regularization = float(regularization)
        if neighbor_backend not in {"cdist", "sklearn"}:
            raise ValueError("neighbor_backend must be one of {'cdist', 'sklearn'}")
        self.neighbor_backend = neighbor_backend
        self.algorithm = algorithm
        if n_jobs is None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = -1
        elif isinstance(n_jobs, int) and n_jobs >= 1:
            self.n_jobs = int(n_jobs)
        else:
            raise ValueError("n_jobs must be None, -1, or an integer >= 1")
        # Standard module logger for predictable naming
        base = logging.getLogger('rdmatcher.mahalanobis')
        if logger is not None:
            # allow user-supplied logger to be used as parent
            self.logger = logger.getChild('mahalanobis') if hasattr(logger, 'getChild') else base
        else:
            self.logger = base

    def fit(self, X, y=None, *, cov_source: str = "reference", pooled_X=None, VI=None):
        """Fit the Mahalanobis estimator.

        Parameters
        - X: reference data (controls) as ndarray or DataFrame
        - cov_source: one of {'reference', 'pooled'}; determines whether the covariance
          is computed from the reference X or from pooled_X (treated+controls). Default 'reference'.
        - pooled_X: ndarray/DataFrame used when cov_source=='pooled'
        - VI: optional precomputed precision (inverse covariance) matrix provided by user
        """
        # Accept DataFrame or ndarray
        if isinstance(X, pd.DataFrame):
            # require numeric types
            non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            if non_numeric:
                raise ValueError(f"MahalanobisKNN requires numeric columns only. Non-numeric: {non_numeric}")
            X_arr = X.values.astype(np.float64, copy=False)
            try:
                self.feature_names_in_ = np.array(X.columns.tolist())
            except Exception:
                self.feature_names_in_ = None
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.feature_names_in_ = None

        if X_arr.ndim != 2:
            raise ValueError("X must be 2D array-like")

        n_samples, n_features = X_arr.shape
        if n_samples < 1:
            raise ValueError("X must contain at least one sample")

        self.n_features_in_ = n_features

        # handle missing values by dropping incomplete rows (MatchIt-style) from the reference set
        mask_complete = ~np.isnan(X_arr).any(axis=1)
        if not np.all(mask_complete):
            dropped = np.where(~mask_complete)[0].tolist()
            # log which rows were dropped
            try:
                self.logger.info(f"Dropping {len(dropped)} rows with missing values from fit: {dropped}")
            except Exception:
                pass
        X_complete = X_arr[mask_complete]
        if X_complete.shape[0] < 1:
            raise ValueError("All rows contain missing values; cannot fit MahalanobisKNN on empty data")

        # map internal reference indices back to original input indices
        self._ref_index_map = np.where(mask_complete)[0]

        # decide centering and covariance base
        if cov_source == "pooled":
            if pooled_X is None:
                raise ValueError("cov_source='pooled' requires pooled_X to be provided")
            if isinstance(pooled_X, pd.DataFrame):
                pooled_arr = pooled_X.values.astype(np.float64)
            else:
                pooled_arr = np.asarray(pooled_X, dtype=np.float64)
            if pooled_arr.ndim != 2 or pooled_arr.shape[1] != n_features:
                raise ValueError("pooled_X must be 2D with the same number of features as X")
            self.location_ = np.mean(pooled_arr, axis=0)
            cov_base = pooled_arr
        else:
            # default: compute location and covariance from the reference (controls)
            self.location_ = np.mean(X_complete, axis=0)
            cov_base = X_complete

        # compute covariance / precision / whitener
        if VI is not None:
            VI_arr = np.asarray(VI, dtype=np.float64)
            if VI_arr.shape != (n_features, n_features):
                raise ValueError("VI must be a square matrix with shape (n_features, n_features)")
            cov_inv = VI_arr
            try:
                L = np.linalg.cholesky(cov_inv)
                whitener = L.T
                cov = np.linalg.inv(cov_inv)
            except np.linalg.LinAlgError:
                w, V = np.linalg.eigh(cov_inv)
                w_clipped = np.clip(w, a_min=1e-12, a_max=None)
                whitener = V @ np.diag(np.sqrt(w_clipped)) @ V.T
                cov = np.linalg.inv(cov_inv)
        else:
            reg = float(self.regularization)
            max_reg = 1e-2
            cov = None
            cov_inv = None
            whitener = None
            for attempt in range(0, 10):
                try:
                    cov = np.cov(cov_base.T, bias=False) + np.eye(n_features) * reg
                    cov_inv = np.linalg.inv(cov)
                    try:
                        L = np.linalg.cholesky(cov_inv)
                        whitener = L.T
                    except np.linalg.LinAlgError:
                        w, V = np.linalg.eigh(cov)
                        w_clipped = np.clip(w, a_min=reg, a_max=None)
                        cov_inv = (V / w_clipped) @ V.T
                        whitener = V @ np.diag(1.0 / np.sqrt(w_clipped)) @ V.T
                    break
                except np.linalg.LinAlgError:
                    reg = min(max_reg, reg * 10.0)
                    continue

            if cov_inv is None or whitener is None:
                raise np.linalg.LinAlgError("Failed to compute stable Mahalanobis precision/whitener")

        # Save fitted artifacts
        self.covariance_ = cov
        self.precision_ = cov_inv
        self.whitener_ = whitener

        # Store a whitened copy of the reference data for fast dot-products
        try:
            self._X_ref_whitened = (X_complete - self.location_) @ self.whitener_.T
        except Exception:
            # As a fallback, store unwhitened
            self._X_ref_whitened = (X_complete - self.location_)

        self._nn_model = None
        if self.neighbor_backend == "sklearn":
            self._nn_model = NearestNeighbors(
                metric="euclidean",
                algorithm=self.algorithm,
                n_jobs=self.n_jobs,
            )
            self._nn_model.fit(self._X_ref_whitened)

        # _ref_index_map maps internal positional indices -> original input row indices
        self.n_samples_fit_ = X_complete.shape[0]

        return self

    # Note: Mahalanobis is numeric-only and does not need a safe-slice helper.

    def _whiten_queries(self, queries):
        if isinstance(queries, pd.DataFrame):
            Q_arr = queries.values.astype(np.float64)
        else:
            Q_arr = np.asarray(queries, dtype=np.float64)
        if Q_arr.ndim != 2 or Q_arr.shape[1] != self.n_features_in_:
            raise ValueError("queries must be 2D with same number of features used in fit")

        mask_q_complete = ~np.isnan(Q_arr).any(axis=1)
        if not np.all(mask_q_complete):
            dropped_q = np.where(~mask_q_complete)[0].tolist()
            try:
                self.logger.info(f"Dropping {len(dropped_q)} query rows with missing values for distance computation: {dropped_q}")
            except Exception:
                pass

        if np.any(mask_q_complete):
            Q_complete = Q_arr[mask_q_complete]
            Qw_all = (Q_complete - self.location_) @ self.whitener_.T
        else:
            Qw_all = np.empty((0, self.n_features_in_), dtype=np.float64)

        return Q_arr, mask_q_complete, Qw_all

    def cdist(self, queries, Y_ref=None, batch_size: int = 256, n_jobs: Optional[int] = None):
        """Compute Mahalanobis distances from queries to reference set.

        Parameters
        - queries: 2D array-like (n_queries, n_features)
        - Y_ref: optional reference array; if None uses fitted reference
        - batch_size: number of queries to process per loop to limit memory
        """
        check_is_fitted(self, ["precision_", "whitener_"])
        if n_jobs is None:
            # MahalanobisKNN does not implement internal parallelism; default to 1
            n_jobs = 1

        if Y_ref is None:
            X_ref_whitened = self._X_ref_whitened
            n_ref = X_ref_whitened.shape[0]
        else:
            if isinstance(Y_ref, pd.DataFrame):
                Y_arr = Y_ref.values.astype(np.float64)
            else:
                Y_arr = np.asarray(Y_ref, dtype=np.float64)
            if Y_arr.ndim != 2 or Y_arr.shape[1] != self.n_features_in_:
                raise ValueError("Y_ref must be 2D with same number of features used in fit")
            # Enforce MatchIt-style complete-case policy for Y_ref: callers must
            # pass NaN-free reference data. Returning a modified reference would
            # change output shapes and complicate calling code.
            if np.isnan(Y_arr).any():
                raise ValueError("Y_ref contains missing values (NaN). MahalanobisKNN requires complete-case reference data; please preprocess to remove or impute missing values before calling cdist.")
            X_ref_whitened = (Y_arr - self.location_) @ self.whitener_.T
            n_ref = X_ref_whitened.shape[0]

        Q_arr, mask_q_complete, Qw_all = self._whiten_queries(queries)
        n_queries = Q_arr.shape[0]
        # initialize with NaN so rows with missing queries remain NaN (dropped)
        dists = np.full((n_queries, n_ref), np.nan, dtype=np.float64)

        # Decide block size for references to control memory; pick a conservative block size
        ref_block = max(1, min(n_ref, 8192))

        # iterate only over blocks of the complete-query subset; we map back to original rows
        complete_idx = np.where(mask_q_complete)[0]
        for block_start in range(0, complete_idx.shape[0], batch_size):
            block_end = min(block_start + batch_size, complete_idx.shape[0])
            q_indices = complete_idx[block_start:block_end]
            Qw = Qw_all[block_start:block_end]
            # compute dist to all ref in blocks
            out_block = np.empty((q_indices.shape[0], n_ref), dtype=np.float64)
            for r_start in range(0, n_ref, ref_block):
                r_end = min(r_start + ref_block, n_ref)
                Ref = X_ref_whitened[r_start:r_end]
                # squared distances: ||a||^2 + ||b||^2 - 2 a·b
                a2 = np.sum(Qw * Qw, axis=1)[:, None]
                b2 = np.sum(Ref * Ref, axis=1)[None, :]
                cross = Qw @ Ref.T
                block = a2 + b2 - 2.0 * cross
                # numerical rounding to zero
                block = np.maximum(block, 0.0)
                out_block[:, r_start:r_end] = np.sqrt(block)

            # write block results back into the global array at the original query positions
            dists[q_indices[:, None], np.arange(n_ref)[None, :]] = out_block

        return dists.astype(np.float32)

    def kneighbors(self, queries, n_neighbors=1, batch_size: int = 256, n_jobs: Optional[int] = None):
        check_is_fitted(self, ["precision_", "whitener_"])
        if self.neighbor_backend == "sklearn":
            check_is_fitted(self, ["_nn_model"])
            Q_arr, mask_q_complete, Qw_all = self._whiten_queries(queries)
            k = min(n_neighbors, self._X_ref_whitened.shape[0])
            n_q = Q_arr.shape[0]
            ordered_idx = np.full((n_q, k), -1, dtype=np.int32)
            ordered_d = np.full((n_q, k), np.nan, dtype=np.float32)

            complete_idx = np.where(mask_q_complete)[0]
            if complete_idx.size:
                dists, idxs = self._nn_model.kneighbors(Qw_all, n_neighbors=k, return_distance=True)
                ref_map = self._ref_index_map
                ordered_d[complete_idx] = dists.astype(np.float32)
                ordered_idx[complete_idx] = ref_map[idxs].astype(np.int32)
            return ordered_d, ordered_idx

        dists = self.cdist(queries, batch_size=batch_size, n_jobs=n_jobs)
        # use argpartition for speed then stable sort by (dist, idx)
        k = min(n_neighbors, dists.shape[1])
        # prepare outputs
        n_q = dists.shape[0]
        ordered_idx = np.full((n_q, k), -1, dtype=np.int32)
        ordered_d = np.full((n_q, k), np.nan, dtype=np.float32)

        # Only process rows that have at least one finite distance
        finite_mask = np.isfinite(dists).any(axis=1)
        if np.any(finite_mask):
            dists_proc = dists[finite_mask]
            idx_part = np.argpartition(dists_proc, kth=k-1, axis=1)[:, :k]
            # gather distances
            d_sub = np.take_along_axis(dists_proc, idx_part, axis=1)
            # stable sort each row by (distance, index) to ensure deterministic tie-breaking
            for row_idx_local, row_idx_global in enumerate(np.where(finite_mask)[0]):
                rows = list(zip(d_sub[row_idx_local].tolist(), idx_part[row_idx_local].tolist()))
                rows_sorted = sorted(rows, key=lambda x: (float(x[0]), int(x[1])))
                if rows_sorted:
                    ordered_d[row_idx_global, :len(rows_sorted)] = [r[0] for r in rows_sorted]
                    ordered_idx[row_idx_global, :len(rows_sorted)] = [r[1] for r in rows_sorted]

        # map internal reference indices back to original input indices if available
        try:
            ref_map = self._ref_index_map
            # only map positive indices; -1 stays -1
            mask_valid = ordered_idx >= 0
            ordered_idx_mapped = np.where(mask_valid, ref_map[ordered_idx], -1)
            return ordered_d.astype(np.float32), ordered_idx_mapped.astype(np.int32)
        except Exception:
            return ordered_d.astype(np.float32), ordered_idx.astype(np.int32)
