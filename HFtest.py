import logging
import sys
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.prompts.prompts import SimpleInputPrompt
import torch
from  llama_index.llms import HuggingFaceLLM
import argparse

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

system_prompt = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# set paramenters
parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='.\\data')
parser.add_argument('--Q',default="What did the author do growing up?")
args = parser.parse_args()
dir = args.dir
Q = args.Q

#load documents
documents = SimpleDirectoryReader(dir).load_data()

# wrap defalut prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
    model_name="StabilityAI/stablelm-tuned-alpha-3b",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# set Logging to debug for more detailed outputs
query_engie = index.as_query_engine()
response = query_engie.query(Q)
print(response)
