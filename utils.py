from openai import AzureOpenAI
from transformers import pipeline
import requests
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def init_az_openai_client():
    azure_oai = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_KEY"),  
        api_version = os.getenv("API_VERSION", "2023-05-15"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    return azure_oai

def init_hf_pipeline(model_name="", tokenizer_name=""):
    return pipeline(
        "question-answering", model=model_name, tokenizer=tokenizer_name
    )

def prepare_custom_api_req(**kwargs):
    req = requests.Request(
        method=kwargs.get("method"), url=kwargs.get("url"), params=kwargs.get("url_params"),
        headers=kwargs.get("request_headers"), json=kwargs.get("json"), auth=kwargs.get("auth")
    )
    return req.prepare()