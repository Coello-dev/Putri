# Putri (Prune, Update and Trim)

Official implementaiton of Putri

## Set-up

```
conda create -n SP python=3.12
conda activate SP
pip install -r requirements.txt
```

Install Language Model Evaluation Harness (required for downstream task evaluation):
```bash
cd lm_harness
pip install -e ./

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Usage

```
usage: main.py [-h] --model MODEL [--seed SEED] [--cache_dir CACHE_DIR] [--dense]
               [--pruning_method {putri,2ssp,window_based,shortgpt,blockpruner,evopress}]
               [--sparsity_rate SPARSITY_RATE] [--main_table_results] [--evaluate_inference] [--evaluate_downstream]
               [--evaluate_perplexity] [--evaluate_qualitative] [--local_datasets] [--ablation]
               [--logging {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Pruning of transformer models

options:
  -h, --help            show this help message and exit
  --model MODEL         Specify the model's name or path to be pruned
  --seed SEED           Set a seed for reproducibility (default: 0)
  --cache_dir CACHE_DIR
                        Path to a directory in which a downloaded pretrained model should be cached. This option is not
                        supported when --pruning_method=slicegpt
  --dense               Load the original dense model without pruning
  --pruning_method {putri,2ssp,window_based,shortgpt,blockpruner,evopress,slicegpt}
                        Specify the pruning method to apply
  --sparsity_rate SPARSITY_RATE
                        A floating-point value ranging from 0.0 to 1.0 that determines the target sparsity level for
                        pruning. If set to -1, pruning is performed at all sparsity levels from 0.0 to 1.0 with a step size
                        of 1/N. A value of -2 applies pruning at predefined sparsity levels of 25%, 37.5%, and 50%.)
  --main_table_results  Generate results for the main results table in the paper (Table 1)
  --evaluate_inference  Measure the model's inference time
  --evaluate_downstream
                        Perform downstream task evaluation at 37.5% sparsity
  --evaluate_perplexity
                        Evaluates perplexity on Wikitext2 only
  --evaluate_qualitative
                        Qualitative results
  --local_datasets      Use local datasets stored in the './data/' folder
  --logging {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level (default: INFO)
```

#### Examples
- Dense Model Perplexity Evaluation:
   ```bash
   python main.py --model=meta-llama/Llama-3.2-1B --dense --evaluate_perplexity
   ```

- Pruning at 50% sparsity with `putri` and evaluating perplexity:
   ```bash
   python main.py --model=meta-llama/Llama-3.2-1B --pruning_method=putri --sparsity_rate=0.5 --evaluate_perplexity
   ```
- Pruning at 50% sparsity with `ShortGPT` and evaluating perplexity:
   ```bash
   python main.py --model=meta-llama/Llama-3.2-1B --pruning_method=shortgpt --sparsity_rate=0.5 --evaluate_perplexity
   ```

- Generate main table results for 2SSP at 25%, 37.5% and 50% sparsity on Mistral:
   ```bash
   python main.py --model=meta-llama/Llama-3.2-1B --pruning_method=2ssp --sparsity_rate=-2 --main_table_results
   ```

- Evaluate downstream tasks at 37.5% sparsity:
   ```bash
   python main.py --model=meta-llama/Llama-3.2-1B --pruning_method=2ssp --sparsity_rate=0.375 --evaluate_downstream
   ```

## Supported Pruning Methods

- Putri: Ours
- ShortGPT: [https://arxiv.org/abs/2403.03853](https://arxiv.org/abs/2403.03853)
- Window-Based: [https://arxiv.org/abs/2403.17887](https://arxiv.org/abs/2403.17887)
- BlockPruner: [https://arxiv.org/abs/2406.10594](https://arxiv.org/abs/2406.10594)
- EvoPress: [https://arxiv.org/abs/2410.14649](https://arxiv.org/abs/2410.14649)

---


## Acknowledgments

This repository includes code sourced from the following repositories:
- [2SSP](https://github.com/FabrizioSandri/2SSP)
- [LM Eval](https://github.com/EleutherAI/lm-evaluation-harness)

For more details, refer to the documentation or the associated research paper.

