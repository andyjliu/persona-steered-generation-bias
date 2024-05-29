# persona-steered-generation-bias
Code and data that can be used to replicate results for the ACL 2024 findings paper "Evaluating Large Language Model Biases in Persona-Steered Generation".
## Setup
```
conda create -n persona_steered_generation python==3.11
conda activate persona_steered_generation
pip install -r requirements.txt
git clone https://github.com/tatsu-lab/opinions_qa.git
conda env config vars set OPENAI_API_KEY=<YOUR_KEY_HERE>
```
## Evaluation
`src/generate_stances.py` can be used to, given an OpinionsQA survey, generate a list of stances that can be passed into a prompt for persona-steered generation; all personas are stored in `src/personas_and_topics.json`. Note that this script uses the OpenAI API to generate initial descriptions of each persona: **we recommend verifying that persona descriptions are accurate and manually rewriting any inaccurate descriptions before generating data using them to avoid affecting evaluation**. Once new data is generated and stored in the `persona-tracking-data` directory, it can be automatically evaluated for steerability using the `src/persona_evaluation.py` script. Metric values can then be computed with the `src/analyze_experiment.py` script.

## Replication
The data in the `persona-tracking-results` directory is sourced by running our experimental analysis scripts on the corresponding data directories. The `replicate_figures.ipynb` notebook can then be used to replicate all results and figures from our paper.

## Acknowledgements
Raw data is sourced from [OpinionsQA](https://github.com/tatsu-lab/opinions_qa). The LLM caricature metrics used are modified versions of those found in [CoMPosT](https://github.com/myracheng/lm_caricature). 
