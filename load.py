from argparse import ArgumentParser, Namespace
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model_name", required=True, help="model name")
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    model = ORTModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    inputs = tokenizer(["Hello, how are you", "Ayo!"], return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    outputs = tokenizer.batch_decode(outputs)

    print(outputs)


if __name__ == "__main__":
    main()
