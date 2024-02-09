from mongoengine import Document, StringField, EmbeddedDocument, EmbeddedDocumentListField
from mongoengine import DictField, ListField

from utils import init_az_openai_client, init_hf_pipeline

class Prompt(EmbeddedDocument):
    prompt_type = StringField(choices=["system", "user", "assistant"])
    prompt_template = StringField()
    prompt_default_var_kwargs = DictField()

class PromptLibrary(Document):
    testcase_name = StringField()
    tokenizer_name = StringField()
    model_name = StringField()
    model_provider = StringField()
    interaction_prompts = EmbeddedDocumentListField(Prompt)
    golden_interaction = ListField(StringField())
    interaction_version = StringField()

    def populate_prompts(self, **kwargs):
        # You can extend this fn to cover all types of DFS variations in the interactions
        all_interaction_variations = []
        for interaction in self.interaction_prompts:
            all_interaction_variations.append(
                (interaction.prompt_type, interaction.prompt_template.format(**interaction.prompt_default_var_kwargs))
            )
        return all_interaction_variations
    
    def get_model_connector(self):
        if self.model_provider == "azureopenai":
            return init_az_openai_client()
        elif self.model_provider == "hf-transformers":
            return init_hf_pipeline()
        else:
            return None
