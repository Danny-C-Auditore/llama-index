import os

import torch
from transformers import pipeline
from typing import Optional, List, Mapping, Any

from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    LangchainEmbedding,
    ListIndex
)
from llama_index.llms import CustomLLM, CompletionResponse, LLMMetadata, CompletionResponseGen

import argparse



class OurLLM(CustomLLM):

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata"""
        return LLMMetadata(
            context_window=context_window, num_output=num_output
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt_len = len(prompt)
        response = pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]

        #only return new tokens
        text = response[prompt_len:]
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()


if __name__ == '__main__':

    # set context window size
    context_window = 2048
    # set number of output tokens
    num_output = 1024

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None)
    parser.add_argument('--prompt', default='What did author do growing up?')
    parser.add_argument('--model', default="csitfun/llama-7b-logicot")
    parser.add_argument('--device',default="cuda:0")
    parser.add_argument('--window',default=None)
    parser.add_argument('--file',default=None, help="path to the file of document" )
    args = parser.parse_args()
    dir = args.dir
    query = args.prompt
    device = args.device
    window = args.window
    file = args.file

    # store the pipeline or model outside of the LLM class to aovid memory issue
    model_name = args.model
    pipeline = pipeline("text-generation", model=model_name,
                        model_kwargs={"torch_dtype": torch.bfloat16},device_map="auto")
    # define our own LLM
    llm = OurLLM()

    service_context = ServiceContext.from_defaults(
        llm=llm, context_window=int(window), num_output=num_output
    )

    # Load the data
    documents = SimpleDirectoryReader(input_dir=dir).load_data()
    #documents = SimpleDirectoryReader(input_files=[file]).load_data()
    #print(documents)
    index = ListIndex.from_documents(documents, service_context=service_context)

    # Query and print response
    query_engine = index.as_query_engine()
    print("##################### response is below ######################")
    response = query_engine.query(query)
    print(response)
