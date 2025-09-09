from utils.build_graph import build_graph

def main(args):
    g, total_time = build_graph(
        language=args.language,
        dataset=args.dataset,
        model_name=args.model_name,
        layer=args.layer,
        threshold=args.threshold,
        distance_type=args.distance_type,
        k=args.k,
        lmbda=args.lmbda,
    )
    print(f"Graph built in {total_time:.2f} seconds.")

    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Builds a similarity graph from cut features.")
    parser.add_argument("language", type=str, choices=["english", "mandarin"], help="Language of the dataset.")
    parser.add_argument("dataset", type=str, help="Dataset split (e.g., 'train', 'dev', 'test').")
    parser.add_argument("model_name", type=str, help="Model used for feature extraction (e.g., 'hubert').")
    parser.add_argument("layer", type=int, help="Layer of the model from which features were extracted (e.g., 9).")
    parser.add_argument(
            "distance_type", type=str, choices=["ed", "cos", "dtw"], help="Distance type."
        )
    parser.add_argument(
        "threshold", type=float, help="Threshold value for building the graph."
    )
    parser.add_argument("--k", type=int, help="Number of clusters to discretise with.")
    parser.add_argument(
        "--lmbda", type=float, help="Lambda value to use for discretisation."
    )

    args = parser.parse_args()

    main(args)