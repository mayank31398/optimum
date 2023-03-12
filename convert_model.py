from argparse import ArgumentParser, Namespace
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model_name", required=True, help="model name")
    parser.add_argument("--save_path", required=True, help="save directory")
    parser.add_argument("--merge_graphs", action="store_true", help="merge the 2 graphs")
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.save_path)

    model = ORTModelForCausalLM.from_pretrained(
        args.model_name, export=True, trust_remote_code=True, use_merged=args.merge_graphs
    )
    model.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()
