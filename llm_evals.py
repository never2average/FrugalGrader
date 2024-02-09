from collections import defaultdict
from email.policy import default
import mongoengine
from sentence_transformers import CrossEncoder
from prompt_library import PromptLibrary
from provider_funcmaps import executor


mongoengine.connect("ProductionTestCases")

# You can add a weightage here on different questions
# to take a weighted average instead of a simple one
def cross_encoder_scoring(cross_encoder, ans_2x2, model, provider):
    ans_scores = cross_encoder.predict(ans_2x2)
    for i in range(len(ans_scores)):
        print("\n----------------------------\n")
        print(f"Model: {model} by {provider}\n")
        print(f"Real Answer: {ans_2x2[i][0]}\nGenerated Answer: {ans_2x2[i][1]}\nScore (out of 10): {10*ans_scores[i]}")
        print("\n-----------------------------\n")
    return 0 if len(ans_scores) else sum(ans_scores)/len(ans_scores)


all_tests = PromptLibrary.objects(
    testcase_name=input("Please Enter Testcase Name: ").strip(),
    interaction_version=input("Please Enter Testcase Version: ").strip()
).all()

init_model_connectors = defaultdict(None)
print("\n--------------------------------\n")
print(f"Running {len(all_tests)} Test Cases")
print("\n\n\n")

populate_all_test_variations = []
for test in all_tests:
    populate_all_test_variations.append({
        "model_name": test.model_name,
        "tokenizer_name": test.tokenizer_name,
        "provider": test.model_provider,
        "interaction": test.populate_prompts(),
        "golden_answers": test.golden_interaction
    })

quora_model_large = CrossEncoder('cross-encoder/quora-roberta-large',device='cuda')


for idx, testing in enumerate(populate_all_test_variations):
    turns = len([i for i in testing["interaction"] if interaction[0]=="user"])
    print(f"Testcase {idx}:\nNumber of Turns:{turns}")
    final_answers = []
    interaction_stack = []
    ans_2x2 = []
    k = 0
    for interaction in testing["interaction"]:
        interaction_stack.append({interaction[0]:interaction[1]})
        if interaction[0] != "user":
            continue
        init_model_connectors[testing["provider"]], answer_2x2_mat = executor[testing["provider"]](
            az_oai_client=init_model_connectors[testing["provider"]],
            model_name=testing["model_name"],
            tokenizer_name=testing["tokenizer_name"],
            engine_name=testing["model_name"],
            interaction_list=interaction_stack
        )
        ans_2x2.append((testing["golden_answers"][k], answer_2x2_mat))
        interaction_stack.append({"assistant": answer_2x2_mat})
        k += 1

    cross_encoder_scoring(quora_model_large, ans_2x2, testing["model_name"], testing["provider"])