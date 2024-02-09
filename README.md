## 🚀🚀 Frugal Grader

While there are many public benchmarks available for foundational models on [hf.co](hf.co), dev teams need to provide benchmarks on private data+RAG strategies+fine-tuned models/public APIs before deployment in any business usecase. People often end up using GPT4 Autograding on the final LLM responses to generate these reports before deployment. While GPT4 autograding responses may make sense when a developer or a business person is reviewing pipeline performance manually, the approach is stupid and expensive when scaling to a benchmark of even 100 questions on private datasets.

### Why is it expensive?
To make it super clear why this is an expensive process, lets do some quick math. Lets say you have 5 prompt variations each that you want to test out on 5 LLMs (GPT 3.5 + 3 fine-tuned + 1 hugging face public model) on say 100 questions, you end up with a GPT4 autograding bill of **$2500\*(0.01\*3+0.03) = $150**. For context, you could literally hire a human annotator with a grading script for evaluating 2500 question-answer pairs at **$5/hr\*20hr = $100**.

<img width="1271" alt="Screenshot 2024-02-09 at 5 33 15 PM" src="https://github.com/never2average/FrugalGrader/assets/31365087/322d88e6-f791-4c72-a4c3-bb067588d99f">


### Why is it a stupid idea?
- GPT4's autograding output is optimized for human reading and not further consumption into either RL pipelines or analytics dashboard.

- Restricting GPT4 with system prompt to only answer with a single number value will be mostly wrong because the core value in the grading is when it can create a sequence of tokens to guide itself to the correct score.

- While you can do greedy parsing of scores from things like PyDantic or OpenAI API's JSON mode, parsing reliability and format sanity are clear challenges here.
  
- The most obvious flaw in GPT4 autograding is that the generated answer is better than the original golden answer for any answer generated using a GPT4+RAG pipeline.

- The final grading isnt normalized along any axis for scaling. For example: a grade of 4/10 isn't linearly related to 2/10. It does not even follow a log scale or someother mathematical scaling logic. So you will end up having to give the GPT4 feedback to a sentiment analysis BERT.
  

### Our [Hot]Fix
Most generative AI use-cases in enterprise right now including fields like financial services have a limited scope in instruction type with golden answers already in place for benchmarking.

- Using a [cross-encoder model](https://www.sbert.net/examples/applications/cross-encoder/README.html) we can best determine the difference between LLM generated answer and the golden answer. This is well within the core competency of cross-encoder models that are trained on pairwise similarity tasks.

- While the grading isnt a 100% flexible to immediate changes in grading policy, the generated grading scales both in terms of sensibility around the numeric values and .

- While GPT4 autograding will continue to be used in performance evaluations of prompts and LLMs with open-ended questions or completely novel questions, we believe cross-encoders pre-trained on large duplicate question detection datsets (like the quora model we will use here) perform well.

- The model we have currently finetuned for internal usage is the [Quora Model](https://huggingface.co/cross-encoder/quora-roberta-large). While this works perfectly fine out of the box, we recommend running PEFT on ~1000 examples before deployment.

### Installation Instructions
Step 1: [Install MongoDB](https://www.mongodb.com/docs/manual/installation/)

Step 2: Check if Mongo is running properly on localhost
```bash
sudo systemctl status mongod
```

Step 3: Setup the repository
```bash
git clone https://github.com/never2average/FrugalGrader
cd FrugalGrader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Step 4: Populate the prompt library with all your testcases

Step 5: Run evals on full prompt library
```
python3 llm_evals.py
```

#### Repository Roadmap [For Contributors]
- [ ] CLI functions for easy and modular runs.
- [ ] Implement Atlas Dashboard connector for runtime logs.
- [ ] Integrate with PortKey for LLMOps teams.
- [ ] Implement an RLAIF module optimized at grading for PPO.
- [ ] Release PEFT Quora Model for financial services QA autograding. [from onfinance team]

---

**Written by hand cause apparently all AI code documentation tools are so incompetent that they cannot generate a decent README.md for a repository.**
