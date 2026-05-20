import logging
from collections import deque

import numpy as np
import open3d as o3d
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


class ClusterExtractor:
    def __init__(self, verbose: bool = True):
        self.logger = logging.getLogger("ClusterExtractor")
        self.verbose = verbose

    def log_cluster_info(self, labels: np.ndarray, has_noise: bool) -> None:
        if not self.verbose:
            return
        unique_labels, label_count = np.unique(labels, return_counts=True)
        self.logger.info(
            f"Found {len(unique_labels) - (1 if has_noise else 0)} clusters (excluding noise)"
            f" in pointcloud with {len(labels)} points."
        )
        self.logger.info(
            "The number of points in each cluster is: "
            + ", ".join(f"{i}: {count}" for i, count in zip(unique_labels, label_count))
        )

    def extract_clusters_euclidean(
        self,
        pointcloud: o3d.geometry.PointCloud,
        max_distance_to_neighbors: float = 0.15,
        min_points_around_core_point: int = 10,
        min_points_in_cluster: int = 150,
    ) -> np.ndarray:
        """
        Extract clusters using euclidean clustering (DBSCAN).
        This works well when clusters have empty space between them.
        Computation is rather fast.

        Args:
            pointcloud: The input pointcloud for which to extract clusters.
            max_distance_to_neighbors: The maximum distance to consider a point a neighbor.
            min_points_around_core_point: The minimum number of points around a cluster core point.
            min_points_in_cluster: The minimum number of points required in a cluster.

        Returns:
            An array of cluster labels for each point in the input pointcloud.
        """
        labels = np.array(
            pointcloud.cluster_dbscan(
                eps=max_distance_to_neighbors, min_points=min_points_around_core_point, print_progress=False
            )
        )
        unique_labels, label_count = np.unique(labels, return_counts=True)
        # filter out labels below count limit
        tiny_clusters = unique_labels[label_count < min_points_in_cluster]
        labels[np.isin(labels, tiny_clusters)] = -1
        self.log_cluster_info(labels, has_noise=True)
        return labels

    def extract_clusters_k_means(
        self,
        pointcloud: o3d.geometry.PointCloud,
        num_clusters: int = 6,
    ) -> np.ndarray:
        """
        Extract clusters using simple k-means clustering.
        This will determine a fixed number of clusters.
        Clusters tend to be compact, convex regions in space.
        It might split long surfaces into multiple clusters.
        Computation is rather fast.

        Args:
            pointcloud: The input pointcloud for which to extract clusters.
            num_clusters: The number of clusters to extract.

        Returns:
            An array of cluster labels for each point in the input pointcloud.
        """
        self.logger.info("Running k-means clustering...")
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=0,
            n_init=10,
        )
        labels = kmeans.fit_predict(pointcloud.points)
        self.log_cluster_info(labels, has_noise=False)
        return labels

    def extract_clusters_region_growing(
        self,
        pointcloud: o3d.geometry.PointCloud,
        max_distance_to_neighbors: float = 0.2,
        min_points_in_cluster: int = 300,
    ) -> np.ndarray:
        """
        Extract clusters using region growing (normal-based).
        This works well when clusters are mostly flat, or just slightly curved.
        It might merge clusters that have a connection with low curvature between them.
        Computation is rather slow.

        Args:
            pointcloud: The input pointcloud for which to extract clusters.
            max_distance_to_neighbors: The maximum distance to consider a point a neighbor.
            min_points_in_cluster: The minimum number of points required in a cluster.

        Returns:
            An array of cluster labels for each point in the input pointcloud.
        """
        kdtree = o3d.geometry.KDTreeFlann(pointcloud)
        normals = np.asarray(pointcloud.normals)
        labels = -np.ones(len(pointcloud.points), dtype=np.int32)
        current_cluster = 0

        # cosine threshold for angle between normals
        cos_threshold = np.cos(np.deg2rad(10))

        for seed_idx in range(len(pointcloud.points)):
            # already assigned
            if labels[seed_idx] != -1:
                continue

            # add seed to queue and start new cluster
            queue = deque([seed_idx])
            cluster_indices = []
            labels[seed_idx] = current_cluster

            while queue:
                idx = queue.popleft()
                cluster_indices.append(idx)

                # select neighbors within radius
                _, neighbor_indices, _ = kdtree.search_radius_vector_3d(
                    pointcloud.points[idx],
                    max_distance_to_neighbors,
                )

                n_i = normals[idx]
                for nb_idx in neighbor_indices:
                    # already visited
                    if labels[nb_idx] != -1:
                        continue

                    # check normal similarity
                    n_j = normals[nb_idx]
                    similarity = np.dot(n_i, n_j)
                    if similarity < cos_threshold:
                        continue

                    # assign to cluster and add to queue
                    labels[nb_idx] = current_cluster
                    queue.append(nb_idx)

            # ignore if cluster too small
            if len(cluster_indices) < min_points_in_cluster:
                labels[cluster_indices] = -1
            else:
                current_cluster += 1

        self.log_cluster_info(labels, has_noise=True)
        return labels

    def extract_clusters_spectral(
        self,
        pointcloud: o3d.geometry.PointCloud,
        num_clusters: int = 6,
        num_affinity_neighbors: int = 10,
        affinity_sigma: float = 0.4,
    ) -> np.ndarray:
        """
        Extract clusters using spectral clustering.
        This will determine a fixed number of clusters.
        This computes an embedding for the points, which encodes connectivity.
        Then, it runs k-means on that embedding.
        Computation is rather slow.

        Args:
            pointcloud: The input pointcloud for which to extract clusters.
            num_clusters: The number of clusters to extract.
            num_affinity_neighbors: The number of neighbors to consider for the affinity matrix.
            affinity_sigma: The sigma parameter for the Gaussian affinity function.

        Returns:
            An array of cluster labels for each point in the input pointcloud.
        """
        # Construct KDTree for neighbor search
        kdtree = o3d.geometry.KDTreeFlann(pointcloud)

        self.logger.info("Computing affinity matrix...")
        n_points = len(pointcloud.points)
        W = lil_matrix((n_points, n_points))

        for i in range(n_points):
            _, neighbor_indices, neighbor_dist2 = kdtree.search_knn_vector_3d(
                pointcloud.points[i],
                num_affinity_neighbors,
            )

            for j, dist2 in zip(neighbor_indices, neighbor_dist2):
                if i == j:
                    continue

                # Gaussian affinity weight
                weight = np.exp(-dist2 / (2 * affinity_sigma * affinity_sigma))
                W[i, j] = weight
                W[j, i] = weight

        self.logger.info("Computing graph Laplacian...")
        D = np.array(W.sum(axis=1)).flatten()
        # Avoid division by zero
        D_inv_sqrt = 1.0 / np.sqrt(D + 1e-12)

        # Symmetric normalized Laplacian:
        # L = I - D^{-1/2} W D^{-1/2}
        L = W.copy().tocsr()
        for i in range(n_points):
            row_start = L.indptr[i]
            row_end = L.indptr[i + 1]
            cols = L.indices[row_start:row_end]
            L.data[row_start:row_end] *= -D_inv_sqrt[i] * D_inv_sqrt[cols]

        L.setdiag(1.0)

        self.logger.info("Computing eigenvectors for spectral embedding...")

        # Smallest eigenvectors
        _, eigenvectors = eigsh(
            L,
            k=num_clusters + 1,
            which="SM",
        )

        # Skip first eigenvector
        embedding = eigenvectors[:, 1 : num_clusters + 1]

        # Normalize rows
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / (norms + 1e-12)

        self.logger.info("Running k-means in spectral space...")
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=0,
            n_init=10,
        )
        labels = kmeans.fit_predict(embedding)
        self.log_cluster_info(labels, has_noise=False)
        return labels
