"""Compile an external model: python -m algorithm MODEL.json --output OUTPUT.json."""

import argparse
import json
from pathlib import Path

from .compiler import compile_graph
from .model_input import load_graph


def main():
    parser = argparse.ArgumentParser(description="Compile a caller-supplied physical graph")
    parser.add_argument("model", type=Path, help="External physical graph JSON")
    parser.add_argument("--partition", default="joint_space", help="Partition request name in the input")
    parser.add_argument("--output", type=Path, help="Optional destination for compiled records and plan")
    args = parser.parse_args()
    if args.output is not None and args.output.resolve() == args.model.resolve():
        parser.error("The output must differ from the supplied model path")
    comp = compile_graph(load_graph(args.model), partition=args.partition)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"cmg": comp.cmg, "plan": comp.plan}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(comp.summary(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
