# ===============================
# inference.py
# ===============================
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, "./checkpoints")

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>
고구마는 줄기인가요 뿌리인가요?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))