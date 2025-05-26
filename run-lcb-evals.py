#!/usr/bin/env python3

# This script runs the LiveCodeBench custom evaluator for each of the models
# listed in `MODELS`.
#
# INPUT:
# It assumes that input files (code in the format expected by the LiveCodeBench custom evaluator:
# https://github.com/LiveCodeBench/LiveCodeBench?tab=readme-ov-file#custom-evaluation) is stored
# in the following directory structure, where `data/` is a subdirectory of the working directory.
#
# data/
# ├─ gpt-4.1-2025-04-14/
# │  └─ post-processed/
# │     └─ lcb-formatted.json
# ├─ Llama-3.3-70B-Instruct/
# │  └─ post-processed/
# │     └─ lcb-formatted.json
#
# OUTPUT:
# This script produces a set of LiveCodeBench evaluation files for each
# LiveCodeBench input file in the same directory where it is located.
#
# Example usage:
#   ./run-lcb-evals.py

from dataclasses import dataclass, fields
from typing import Iterator
from pathlib import Path

import glob
import subprocess
import concurrent.futures

MODELS = [
    "claude-3-7-sonnet-20250219",
    "codellama-70b",
    "Codestral-2501",
    "DeepSeek-R1",
    "DeepSeek-V3",
    "gpt-4.1-2025-04-14",
    "Llama-3.3-70B-Instruct",
    "qwen-2.5-coder",
]


@dataclass
class LcbModelEvaluationInfo:
    """Represents an LLM, a LiveCodeBench input file, and an expected LiveCodeBench output file.

    Attributes:
        model (str): An LLM.
        lcb_input_file (str): A path to a `.json` input file for the LiveCodeBench custom
            evaluator, which is detailed here:
            https://github.com/LiveCodeBench/LiveCodeBench?tab=readme-ov-file#custom-evaluation.
        lcb_output_file (str): A path to a `.json` file that should be produced by
            the LiveCodeBench custom evaluator upon a non-exceptional evaluation run.

    """

    model: str
    lcb_input_file_path: str
    lcb_output_file: str

    def __iter__(self) -> Iterator[str]:
        return (getattr(self, field.name) for field in fields(self))


def main() -> None:
    assert Path("data/").is_dir(), (
        "No 'data' folder exists to look for LiveCodeBench input files."
    )
    model_to_lcb_input_files: dict[str, list[str]] = {
        model: _get_lcb_input_files(model) for model in MODELS
    }
    model_evaluation_infos: list[LcbModelEvaluationInfo] = []
    for model, lcb_input_files in model_to_lcb_input_files.items():
        for lcb_input_file in lcb_input_files:
            model_evaluation_infos.append(
                LcbModelEvaluationInfo(
                    model=model,
                    lcb_input_file_path=lcb_input_file,
                    lcb_output_file=_get_lcb_output_file(
                        lcb_input_file
                    ),
                )
            )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_lcb_input = {
            executor.submit(
                _lcb_evaluate, lcb_evaluation_info.lcb_input_file_path
            ): lcb_evaluation_info.lcb_input_file_path
            for lcb_evaluation_info in model_evaluation_infos
        }

        for future in concurrent.futures.as_completed(future_to_lcb_input):
            lcb_input = future_to_lcb_input[future]
            try:
                future.result()
            except Exception as e:
                print(f"LiveCodeBench run for {lcb_input} raised: {str(e)}")

    _check_lcb_evaluation_completeness(model_evaluation_infos)


def _lcb_evaluate(lcb_input_file: str) -> None:
    """Run the LiveCodeBench custom evaluator for the given input file.

    Args:
        input_file (str): The input file for which to run the LiveCodeBench
            custom evaluator.
    """
    lcb_evaluate_cmd = f"python -m lcb_runner.runner.custom_evaluator --custom_output_file {lcb_input_file}"
    subprocess.run(
        lcb_evaluate_cmd,
        shell=True,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    print(f"Finished running: {lcb_evaluate_cmd}")


def _get_lcb_input_files(model: str) -> list[str]:
    """Return the paths to existing LiveCodeBench input files for the given model.

    Args:
        model (str): The model.

    Returns:
        list[str]: The paths to the LiveCodeBench input files for the given model.
    """
    return glob.glob(f"data/{model}/post-processed/*lcb-formatted.json")


def _get_lcb_output_file(lcb_input_file: str) -> str:
    """Return the path to the corresponding LiveCodeBench output file.

    Args:
        lcb_input_file (str): The LiveCodeBench input file.

    Returns:
        The path to the corresponding LiveCodeBench output file.
    """
    lcb_input_file_without_extension, *_ = lcb_input_file.split(".json")
    LCB_OUTPUT_FILE_SUFFIX = "codegeneration_output_eval_all.json"
    return f"{lcb_input_file_without_extension}_{LCB_OUTPUT_FILE_SUFFIX}"


def _check_lcb_evaluation_completeness(
    model_evaluation_infos: list[LcbModelEvaluationInfo],
) -> None:
    """Check the completeness of a LiveCodeBench evaluation run.

    Every LiveCodeBench input file should have an associated output file produced
    by the LiveCodeBench custom evaluator.

    Args:
        model_evaluation_infos (list[LcbModelEvaluationInfo]): A list of model
            evaluation information.
    """
    for _, lcb_input_file_path, lcb_output_file in model_evaluation_infos:
        if not Path(lcb_output_file).is_file():
            print(f"Missing LiveCodeBench output file: {lcb_output_file}")
            print(
                f"Re-run 'python -m lcb_runner.runner.custom_evaluator --custom_output_file {lcb_input_file_path}'"
            )


if __name__ == "__main__":
    main()
