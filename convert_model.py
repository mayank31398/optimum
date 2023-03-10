# from pathlib import Path
# import transformers
# from transformers.onnx import FeaturesManager
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # load model and tokenizer
# model_id = "gpt2"
# feature = ""
# model = AutoModelForCausalLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# # load config
# model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
# onnx_config = model_onnx_config(model.config)

# # export
# onnx_inputs, onnx_outputs = transformers.onnx.export(
#     preprocessor=tokenizer,
#     model=model,
#     config=onnx_config,
#     opset=13,
#     output=Path("trfs-model.onnx")
# )

# from optimum.onnxruntime import ORTModelForCausalLM
# from transformers import AutoTokenizer

# # model = ORTModelForCausalLM.from_pretrained("gpt2", export=True, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")
# model = ORTModelForCausalLM.from_pretrained("bigcode/santacoder", export=True, trust_remote_code=True)

# tokenizer.save_pretrained("onnx")
# model.save_pretrained("onnx")
# # # ------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
model_name = "onnx-santacoder"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model = ORTModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
# inputs = tokenizer(["Hello, I am", "ayo, how are you?"], return_tensors="pt", padding=True)
inputs = tokenizer(["def f(x)", "def factorial(x)"], return_tensors="pt", padding=True)
q = model.generate(**inputs)
print(tokenizer.batch_decode(q, skip_special_tokens=True))
# # # # ------------------------------