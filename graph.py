from utils.build_graph import build_graph
from utils.partition_graph import CPM_partition, write_partition, find_partition
from utils.evaluate_partition import evaluate_partition_file

from argparse import Namespace
import pandas as pd

def main(args: Namespace) -> None:
    """
    Main entry point for graph-based clustering using CPM (Constant Potts Model).

    This function:
      1. Builds a similarity graph from word-level features.
      2. Applies CPM graph partitioning.
      3. Writes the resulting partition file to disk.
      4. Evaluates the partition.

    Args:
        args (Namespace): Command-line arguments containing:
            - language (str): Dataset language ("english" or "mandarin").
            - dataset (str): Dataset split ("train", "dev", etc.).
            - model_name (str): Feature extraction model name.
            - layer (int): Feature layer index.
            - threshold (float): Edge threshold for similarity graph.
            - distance_type (str): Distance metric ("cos", etc.).
            - k (Optional[int]): Number of clusters (if used).
            - lmbda (Optional[float]): Regularization parameter (if applicable).

    Returns:
        None
    """
    partition_file = find_partition(
        partition_type="graph",
        language=args.language,
        dataset=args.dataset,
        model_name=args.model_name,
        layer=args.layer,
        distance_type=args.distance_type,
        threshold=args.threshold,
        k=args.k,
        lmbda=args.lmbda
    )
    if partition_file is None or args.res != 0.3:
        g, graph_time = build_graph(
            language=args.language,
            dataset=args.dataset,
            model_name=args.model_name,
            layer=args.layer,
            threshold=args.threshold,
            distance_type=args.distance_type,
            k=args.k,
            lmbda=args.lmbda,
        )

        boundary_df = pd.read_csv(f"Data/alignments/{args.language}/{args.dataset}_boundaries.csv")
        num_clusters = len(set(boundary_df['text'].tolist()))

        membership, partition_time = CPM_partition(g, num_clusters, args.res)
        print(f"Graph partitioned in {partition_time:.2f} seconds.")

        partition_file = write_partition(
            partition_type="graph",
            partition_membership=membership,
            language=args.language,
            dataset=args.dataset,       
            model_name=args.model_name,
            layer=args.layer,
            distance_type=args.distance_type,
            threshold=args.threshold,
            word_info = g.vs["name"],
            total_time = graph_time + partition_time,
            k=args.k,
            lmbda=args.lmbda
        )
        total_time = graph_time + partition_time

    total_time = int(partition_file.stem.split("_")[-1])
    evaluate_partition_file(
        partition_file, args.language, args.dataset, args.model_name, total_time,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Builds a similarity graph from cut features.")
    
    parser.add_argument("language", type=str, choices=["english", "mandarin"], help="Language of the dataset.")
    parser.add_argument("dataset", type=str, help="Dataset split (e.g., 'train', 'dev', 'test').")
    parser.add_argument("model_name", type=str, help="Model used for feature extraction (e.g., 'hubert').")
    parser.add_argument("layer", type=int, help="Layer of the model from which features were extracted (e.g., 9).")
    parser.add_argument("distance_type", type=str, choices=["ed", "cos", "dtw"], help="Distance type.")
    parser.add_argument("threshold", type=float, help="Threshold value for building the graph.")

    parser.add_argument("--k", type=int, help="Number of clusters to discretise with.")
    parser.add_argument("--lmbda", type=float, help="Lambda value to use for discretisation.")
    parser.add_argument("--res", type=float, default=0.3, help="Resolution for partitioning graph.")
    
    args = parser.parse_args()

    main(args)