from utils.average import (
    apply_pca,
    cluster_features_kmeans,
    cluster_features_birch,
    cluster_features_agglomerative,
    convert_cluster_ids_to_partition,
    pooling
)
from utils.partition_graph import write_partition, find_partition
from utils.cut_features import load_cut_features
from utils.evaluate_partition import evaluate_partition_file
import time
import pandas as pd
from argparse import Namespace

def main(args: Namespace) -> None:
    """
    Main entry point for clustering word-level features into partitions.

    This function handles:
      1. Checking if a partition file already exists.
      2. Loading boundary and feature data.
      3. Applying PCA and pooling to reduce dimensionality.
      4. Running the specified clustering algorithm (kmeans, birch, agglomerative).
      5. Writing the resulting partition file.
      6. Evaluating the partition.

    Args:
        args (Namespace): Command-line arguments containing:
            - partition_type (str): Clustering algorithm ("kmeans", "birch", "agg").
            - language (str): Dataset language ("english" or "mandarin").
            - dataset (str): Dataset split ("train", "dev", etc.).
            - model_name (str): Feature extraction model name.
            - layer (int): Feature layer index.
            - distance_type (Optional[str]): Distance metric ("cos", etc.).
            - threshold (Optional[float]): Distance threshold for graph-based clustering.
            - k (Optional[int]): Number of clusters (for k-means).
            - lmbda (Optional[float]): Regularization parameter (if applicable).

    Returns:
        Optional[Path]: Path to the partition file if written, else None.
    """
    partition_file = find_partition(
        partition_type=args.partition_type,
        language=args.language,
        dataset=args.dataset,
        model_name=args.model_name,
        layer=args.layer,
        distance_type=None,
        threshold=None,
        k=None,
        lmbda=None
    )
    if partition_file is None:
        boundary_df = pd.read_csv(f"Data/alignments/{args.language}/{args.dataset}_boundaries.csv")
        num_clusters = len(set(boundary_df['text'].tolist())) 
        print(f"Number of unique words (clusters) in {args.language} {args.dataset}: {num_clusters}")

        word_features, paths = load_cut_features(args.language, args.dataset, args.model_name, args.layer)
        print(f"Loaded {len(word_features)} word features.")

        start_time = time.time()
        word_features = apply_pca(word_features)
        word_features = pooling(word_features)

        
        if args.partition_type == "kmeans":
            cluster_ids = cluster_features_kmeans(word_features, num_clusters)
        elif args.partition_type == "birch":    
            cluster_ids = cluster_features_birch(word_features, num_clusters)
        elif args.partition_type == "agg":
            cluster_ids = cluster_features_agglomerative(word_features, num_clusters)
        else:
            raise ValueError(f"Unsupported partition type: {args.partition_type}")
        
        partition_membership = convert_cluster_ids_to_partition(cluster_ids)
        total_time = time.time() - start_time
        print(f"Clustering completed in {total_time:.2f} seconds.")
        print(f"Number of clusters formed: {len(partition_membership)}")

        partition_file = write_partition(
            partition_type=args.partition_type,
            partition_membership=partition_membership,
            language=args.language,
            dataset=args.dataset,       
            model_name=args.model_name,
            layer=args.layer,
            distance_type=None, 
            threshold=None,
            word_info = paths,
            total_time = total_time,
            k=None,
            lmbda=None
        )
        
    total_time = int(partition_file.stem.split("_")[-1])
    evaluate_partition_file(
        partition_file, args.language, args.dataset, args.model_name, total_time, 
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run discrete ED graph processing.")
    parser.add_argument("language", type=str, help="Language code for processing.")
    parser.add_argument("dataset", type=str, help="Dataset name for processing.")
    parser.add_argument("model_name", type=str, help="Model name for processing.")
    parser.add_argument("layer", type=int, help="Layer number for feature extraction.")
    parser.add_argument(
        "partition_type",
        type=str,
        choices=["kmeans", "agg", "birch"],
    )

    args = parser.parse_args()
    main(args)