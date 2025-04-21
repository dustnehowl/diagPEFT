from datasets import load_dataset

def build_prompt(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text.strip() != "":
        user_prompt = f"{instruction}\n\n{input_text}"
    else:
        user_prompt = instruction

    return {
        "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{output}<|eot_id|>"""
    }

def load_and_format(dataset_name="beomi/KoAlpaca-v1.1a", split="train"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset.map(build_prompt)

if __name__ == "__main__":
    dataset_name = "beomi/KoAlpaca-v1.1a"
    split = "train"
    formatted_dataset = load_and_format(dataset_name, split)
    formatted_dataset.save_to_disk(f"{dataset_name}_{split}")