import os
import torch
from llama_index.chat_engine import CondenseQuestionChatEngine
from transformers import pipeline
from typing import Optional, List, Mapping, Any
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    LangchainEmbedding,
    ListIndex
)
from llama_index.llms import CustomLLM, CompletionResponse, LLMMetadata, CompletionResponseGen
from llama_index.prompts import Prompt
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

custom_prompt = Prompt("""\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History> 
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
""")

# list of (human_message, ai_message) tuples
# custom_chat_history = [
#     (
#         'Hello, I am asking you questions with options about the data. And you need to give me the right answer from the optinon numbered from A to E ',
#         'Okay, sounds good.'
#     )
# ]
custom_chat_history = [
    
]

if __name__ == '__main__':

    # set context window size
    context_window = 4096
    # set number of output tokens
    num_output = 256
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default="~/llama-index/data")
    parser.add_argument('--prompt', default='What did author do growing up?')
    parser.add_argument('--model', default="daryl149/llama-2-7b-chat-hf")
    parser.add_argument('--device',default="cuda:0")
    parser.add_argument('--window',default=4096)
    parser.add_argument('--file',default="lsat2020.TXT")
    parser.add_argument('--mode', default=0)
    args = parser.parse_args()
    dir = args.dir
    query = args.prompt
    device = args.device
    window = args.window
    file = args.file
    mode = args.mode


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
    if mode == '0':
        documents = SimpleDirectoryReader(input_dir=dir).load_data()
    if mode == '1':
        documents = SimpleDirectoryReader(input_files=[file]).load_data()

    #print(documents)
    index = ListIndex.from_documents(documents, service_context=service_context)

    # Query and print response
    query_engine = index.as_query_engine()
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        condense_question_prompt=custom_prompt,
        chat_history=custom_chat_history,
        verbose=True
    )
    # response = chat_engine.chat(query)
    # print(response)
