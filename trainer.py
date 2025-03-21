import google

OPENAI_API_KEY = "<YOUR_OPENAI_API_KEY>"
prompt = "A model that takes in a request for a python code in English, and responds with a thought out Python code that is as efficient as possible."
temperature = .4
number_of_examples = 100

df_path_base = "/content/drive/MyDrive/llama-2-7b-custom-for-tested-python-code/insturctions"  # change to your preferred path


# @title Default title text
# # @title Default title text
# from codeinterpreterapi import CodeInterpreterSession
# import os

# OPENAI_API_KEY = "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"
# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# def main():
#     # create a session
#     session = CodeInterpreterSession()
#     session.start()

#     # generate a response based on user input
#     response = session.generate_response_sync(
#         """1. Generate tests for the following function.
#           2. Run the tests on the function.
#           3. Fix the function according to the tests.
#           4. Save the fixed function to a .py file within the 'functions' dir using python.
#           def rotate_list_right(head, k):
#               # Create a new list to store the rotated elements
#               new_list = []

#               # Loop until the end of the list is reached
#               while head:
#                   # Add the current element to the new list
#                   new_list.append(head.val)

#                   # Move to the next element in the list
#                   head = head.next

#                   # If we have rotated the list k times, break
#                   if k and head is None:
#                       break

#               # Return the new list
#               return new_list
#         """
#     )

#     # output the response (text + image)
#     print("AI: ", response.content)
#     response.show()


# main()


import os
import openai
import random
from tqdm import tqdm


openai.api_key = OPENAI_API_KEY

def generate_example(prompt, prev_examples, temperature=.5):
    messages=[
        {
            "role": "system",
            "content": f"""You are generating data which will be used to train a machine learning model.

            You will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.
            You will do so in this format:
            ```
            prompt
            -----------
            $prompt_goes_here
            -----------

            response
            -----------
            $response_goes_here
            -----------
            ```

            Only one prompt/response pair should be generated per turn.
            For each turn, make the example slightly more complex than the last, while ensuring diversity.
            Make sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.

            Here is the type of model we want to train:
            `{prompt}`"""
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 10:
            prev_examples = random.sample(prev_examples, 10)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = openai.ChatCompletion.create(
        # model="gpt-4",
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=temperature,
        max_tokens=2000,

    )

    return response.choices[0].message['content']

# Generate examples
prev_examples = []
for i in tqdm(range(number_of_examples)):
    # print(f'Generating example {i}')
    example = generate_example(prompt, prev_examples, temperature)
    prev_examples.append(example)

print(prev_examples)

def generate_system_message(prompt):

    response = openai.ChatCompletion.create(
        # model="gpt-4",
        model="gpt-3.5-turbo",
        messages=[
          {
            "role": "system",
            "content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
          },
          {
              "role": "user",
              "content": prompt.strip(),
          }
        ],
        temperature=temperature,
        max_tokens=500,
    )

    return response.choices[0].message['content']

system_message = generate_system_message(prompt)

print(f'The system message is: `{system_message}`. Feel free to re-run this cell if you want a better result.')

import pandas as pd

# Initialize lists to store prompts and responses
prompts = []
responses = []

# Parse out prompts and responses from examples
for example in prev_examples:
  try:
    split_example = example.split('-----------')
    prompts.append(split_example[1].strip())
    responses.append(split_example[3].strip())
  except:
    pass

# Create a DataFrame
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

# Remove duplicates
df = df.drop_duplicates()

print('There are ' + str(len(df)) + ' successfully-generated examples. Here are the first few:')

df.head()


os.makedirs(df_path_base, exist_ok=True)
df_path = 'insturctions_python_code_simple.csv'
df_full_path = os.path.join(df_path_base, df_path)
df.to_csv(df_full_path)

import interpreter


interpreter.auto_run = True
interpreter.api_key = OPENAI_API_KEY
interpreter.model = "gpt-3.5-turbo-16k"
interpreter.temperature = 0.25

try:
  df = pd.read_csv(df_full_path)
except:
  pass

def fix_function(instruction, code, index):
    prompt=f"""You are testing python functions, given a description of the function and its python initial implementation.

            First you will save the python function given in a python file called `function_{index}.py` file inside a directory called 'functions'.
            ***DO NOT PRINT THE FILE NAME AFTER SAVING!!!***
            Second, you will create 2-3 tests and input data to test on, and run them on the original function.
            Then, you will fix the code until all the tests pass.

            Once all the tests pass successfully, you should save the final working function code in `function_{index}.py` and finish.

            Here is the code description:
            `{instruction}`

            Here is the code python implementation:
            `{code}`
            """
    try_again = True
    while try_again:
      try:
        messages = interpreter.chat(prompt, return_messages=True)
        try_again = False
      except:
        try_again = True
        interpreter.reset()
    interpreter.reset()

for index, row in df.iterrows():
    instruction, code = row['prompt'], row['response']
    fix_function(instruction, code, index)


from os import walk, path
mypath = 'functions'
fixed_response = []
filenames = next(walk(mypath), (None, None, []))[2]  # [] if no file
for i, fname in enumerate(filenames):
  with open(path.join(mypath, fname), 'r') as f:
    fixed_response.append('```python\n'+f.read().strip()+'\n```')
df['fixed_response'] = fixed_response
df.drop(columns=['response'], inplace=True)
df.rename(columns={"fixed_response": "response"}, inplace=True)
df.to_csv(df_full_path)

# Split the data into train and test sets, with 90% in the train set
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

# Save the dataframes to .jsonl files
train_df.to_json('train.jsonl', orient='records', lines=True)
test_df.to_json('test.jsonl', orient='records', lines=True)


import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

model_name = "NousResearch/llama-2-7b-chat-hf" # use this if you have access to the official LLaMA 2 model "meta-llama/Llama-2-7b-chat-hf", though keep in mind you'll need to pass a Hugging Face key argument
dataset_name = "/content/train.jsonl"
new_model = "llama-2-7b-custom"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 5
per_device_eval_batch_size = 5
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 5
max_seq_length = None
packing = False
device_map = {"": 0}

# Load datasets
train_dataset = load_dataset('json', data_files='/content/train.jsonl', split="train")
valid_dataset = load_dataset('json', data_files='/content/test.jsonl', split="train")

# Preprocess datasets
train_dataset_mapped = train_dataset.map(lambda examples: {'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)
valid_dataset_mapped = valid_dataset.map(lambda examples: {'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="all",
    evaluation_strategy="steps",
    eval_steps=5  # Evaluate every 20 steps
)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_mapped,
    eval_dataset=valid_dataset_mapped,  # Pass validation dataset here
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()
trainer.model.save_pretrained(new_model)

# Cell 4: Test the model
logging.set_verbosity(logging.CRITICAL)
prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\nWrite a function that reverses a string. [/INST]" # replace the command here with something relevant to your task
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(prompt)
print(result[0]['generated_text'])

from transformers import pipeline

prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\nWrite a function that gets a list of integers and returns a the subset with the largest sum. [/INST]" # replace the command here with something relevant to your task
num_new_tokens = 400  # change to the number of new tokens you want to generate

# Count the number of tokens in the prompt
num_prompt_tokens = len(tokenizer(prompt)['input_ids'])

# Calculate the maximum length for the generation
max_length = num_prompt_tokens + num_new_tokens

gen = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=max_length)
result = gen(prompt)
print(result[0]['generated_text'].replace(prompt, ''))

# Merge and save the fine-tuned model
from google.colab import drive
drive.mount('/content/drive')

model_path = "/content/drive/MyDrive/llama-2-7b-custom-for-tested-python-code"  # change to your preferred path

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Save the merged model
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

from google.colab import drive
from transformers import AutoModelForCausalLM, AutoTokenizer

drive.mount('/content/drive')

model_path = "/content/drive/MyDrive/llama-2-7b-custom-for-tested-python-code"  # change to the path where your model is saved

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

from transformers import pipeline

prompt = """Given the head of a linked list, rotate the list to the right by k places.

You must write an algorithm that runs in O(n) time at most.
"""

prompt = f"""[INST] <<SYS>>
{system_message}
<</SYS>>

{prompt}. [/INST]""" # replace the command here with something relevant to your task
num_new_tokens = 400  # change to the number of new tokens you want to generate

# Count the number of tokens in the prompt
num_prompt_tokens = len(tokenizer(prompt)['input_ids'])

# Calculate the maximum length for the generation
max_length = num_prompt_tokens + num_new_tokens

   # change to your desired prompt
gen = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=max_length)
result = gen(prompt)
print(result[0]['generated_text'].replace(prompt, ''))