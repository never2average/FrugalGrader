from utils import init_hf_pipeline, init_az_openai_client, prepare_custom_api_req
import requests



def azureopenai(az_oai_client=None, engine_name="mygpt4", interaction_list=[]):
    if az_oai_client is None:
        az_oai_client = init_az_openai_client()
    
    response = az_oai_client.chat.completions.create(
        model=engine_name,
        messages=interaction_list
    )
    return az_oai_client, response.choices[0].message.content


# Ensure you have done huggingface-cli login else this may fail
def hfinference(hf_pipe=None, model_name="", tokenizer_name="", **kwargs):
    if hf_pipe is None:
        hf_pipe = init_hf_pipeline(model_name=model_name, tokenizer_name=tokenizer_name)

    # Writing code here for parsing shit
    response = ""
    return hf_pipe, response

def custominference(session=None, prep_req=None, **kwargs):
    if session is None:
        session = requests.Session()
    if prep_req is None:
        prep_req = prepare_custom_api_req(**kwargs["request_config"])
    
    response = session.send(prep_req)
    # This is based off of together.ai API schema https://docs.together.ai/docs/inference-rest
    # so change parsing for response as you need
    return None, response.json().output.choices[0].text

executor = {
    "azureopenai": azureopenai,
    "hf-transformers": hfinference,
    "custom-inf": custominference
}