from __future__ import annotations

import argparse

from services.eval_service.app.benchmark_generation import write_questions_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the STEM-Exam-200 benchmark question file.")
    parser.add_argument(
        "--output",
        default="benchmarks/stem_exam_200/questions.json",
        help="Destination path for generated benchmark questions.",
    )
    args = parser.parse_args()

    count = write_questions_file(args.output)
    print(f"Generated {count} questions at {args.output}")


if __name__ == "__main__":
    main()
