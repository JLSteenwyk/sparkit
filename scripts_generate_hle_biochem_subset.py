from __future__ import annotations

import argparse

from services.eval_service.app.hle_benchmark_generation import write_hle_biochem_subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a benchmark subset from futurehouse/hle-gold-bio-chem."
    )
    parser.add_argument(
        "--output",
        default="benchmarks/hle_gold_bio_chem/questions_bio10_chem10.json",
        help="Destination path for generated subset questions.",
    )
    parser.add_argument("--dataset", default="futurehouse/hle-gold-bio-chem")
    parser.add_argument("--bio-count", type=int, default=10)
    parser.add_argument("--chem-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    count = write_hle_biochem_subset(
        args.output,
        dataset_name=args.dataset,
        bio_count=args.bio_count,
        chem_count=args.chem_count,
        seed=args.seed,
    )
    print(f"Generated {count} questions at {args.output}")


if __name__ == "__main__":
    main()

