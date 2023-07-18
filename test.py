
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
    num_output = 256



    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default=None)
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--m', default="openlm-research/open_llama_3b")
    parser.add_argument('--device',default="cuda:0")
    args = parser.parse_args()
    dir = args.directory
    query = args.prompt
    device = args.device

    # store the pipeline or model outside of the LLM class to aovid memory issue
    model_name = args.model
    pipeline = pipeline("text-generation", model=model_name, device=device,
                        model_kwargs={"torch_dtype": torch.bfloat16})
    # define our own LLM
    llm = OurLLM()

    service_context = ServiceContext.from_defaults(
        llm=llm, context_window=context_window, num_output=num_output
    )

    # Load the data
    doucuments = SimpleDirectoryReader(input_dir=dir).load_data()
    index = ListIndex.from_documents(doucuments, service_context=service_context)

    # Query and print response
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)

