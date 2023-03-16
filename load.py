from argparse import ArgumentParser, Namespace
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="onnx-santacoder", help="model name")
    parser.add_argument(
        "--provider",
        default="CUDAExecutionProvider",
        choices=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        help="execution engine",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    model = ORTModelForCausalLM.from_pretrained(args.model_name, provider=args.provider, trust_remote_code=True)

    inputs = tokenizer(["Hello, how are you", "Ayo!"], return_tensors="pt", padding=True)
    if args.provider == "CUDAExecutionProvider":
        for ids in inputs:
            inputs[ids] = inputs[ids].cuda()
    outputs = model.generate(**inputs)
    outputs = tokenizer.batch_decode(outputs)

    print(outputs)


if __name__ == "__main__":
    main()
