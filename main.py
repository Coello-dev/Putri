import argparse
import gc
import os
import logging
import math
import time
import torch
import sys

from src.evaluation import *
from src.utilities import *
from src.datasets import *
from src.twossp import *
from src.evopress import *
from src.slicegpt import *
from src.merge import merge, mergeshort
from src.subblockShortGPT import subblockShortGPT
from src.widthmerge import widthmerge
from src.widthmerge_aux import widthmerge_aux

# from src.olica import olica
# from src.replaceme import ReplaceMe

from src.ablations import *

HF_HUB_OFFLINE = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pruning of transformer models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specify the model's name or path to be pruned",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set a seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=False,
        help="Path to a directory in which a downloaded pretrained model should be cached."
        "This option is not supported when --pruning_method=slicegpt",
    )
    parser.add_argument(
        "--dense",
        help="Load the original dense model without pruning",
        action="store_true",
    )
    parser.add_argument(
        "--score_method",
        help="Scoring method for widthmerge",
        type=str,
        choices=[
            "sparsegpt",
            "sparsegpt_norm",
            "l2_norm",
        ],
        default="sparsegpt_norm",
    )
    parser.add_argument(
        "--merge_method",
        help="Merge method for widthmerge",
        type=str,
        choices=[
            "no_merge",
            "min_mse",
            "min_mse_parallel",
        ],
        default="min_mse",
    )
    parser.add_argument(
        "--attention_prune_method",
        help="Pruning method for attention for widthmerge",
        type=str,
        choices=[
            "ppl",
            "ppl_head",
            "ppl_head_rec",
            "kldiv",
            "kldiv_head",
            "cosine",
        ],
        default="ppl_head",
    )
    parser.add_argument(
        "--pruning_method",
        type=str,
        choices=[
            "dense",
            "denseolica",
            "2ssp",
            "window_based",
            "shortgpt",
            "blockpruner",
            "evopress",
            "slicegpt",
            "replaceme",
            "merge",
            "mergeshort",
            "subblockshortgpt",
            "putri",
        ],
        help="Specify the pruning method to apply",
    )
    parser.add_argument(
        "--sparsity_rate",
        type=float,
        help="A floating-point value ranging from 0.0 to 1.0 that determines the target sparsity level for pruning."
        "If set to -1, pruning is performed at all sparsity levels from 0.0 to 1.0 with a step size of 1/N."
        "A value of -2 applies pruning at predefined sparsity levels of 25%%, 37.5%%, and 50%%.)",
    )
    parser.add_argument(
        "--main_table_results",
        help="Generate results for the main results table in the paper (Table 1)",
        action="store_true",
    )
    parser.add_argument(
        "--evaluate_inference",
        help="Measure the model's inference time",
        action="store_true",
    )
    parser.add_argument(
        "--evaluate_downstream",
        help="Perform downstream task evaluation at 37.5%% sparsity",
        action="store_true",
    )
    parser.add_argument(
        "--evaluate_perplexity",
        help="Evaluates perplexity on Wikitext2 only",
        action="store_true",
    )
    parser.add_argument(
        "--evaluate_qualitative",
        help="Qualitative results",
        action="store_true",
    )
    parser.add_argument(
        "--local_datasets",
        help="Use local datasets stored in the './data/' folder",
        action="store_true",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the ablation study experiments",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run simple test experiment",
    )
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    return parser.parse_args()


@torch.no_grad()
def main():

    # Get args
    args = parse_args()

    # Get logger
    logging_level = getattr(logging, args.logging.upper())
    log_folder = f"results/{args.model}/{args.sparsity_rate}/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_path = f"{log_folder}/logging_{args.pruning_method}.log"
    log_all_path = f"{log_folder}/ppl_all_{args.pruning_method}.log"
    log_wiki_path = f"{log_folder}/ppl_wiki_{args.pruning_method}.log"
    log_downstream = f"{log_folder}/zero_shot_{args.pruning_method}.log"

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),  # console
        ],
        force=True,
    )

    # Show info logger, cwd and arguments
    logging.info(f"Logger ready (level: {args.logging.upper()})")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(f"Enviroment: {sys.executable}")
    arguments_string = "Arguments:\n" + "\n".join([
        f" - {k}: {str(v)}" for k, v in args.__dict__.items()
    ])
    logging.debug(arguments_string)

    # Check GPU status
    use_cuda = torch.cuda.is_available()
    logging.info(
        f"Torch version: {torch.__version__} | Cuda available: {use_cuda}"
    )
    logging.info(
        f"GPU (device=0): {torch.cuda.get_device_properties('cuda').name}"
    )
    gc.collect()
    torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info()
    logging.debug(
        f"Memory available: [{free_mem / 1e6:.0f} MB/{total_mem / 1e6:.0f} MB]"
    )

    # Set seed
    set_seed(args.seed)

    # Load the tokenizer
    tokenizer = loadTokenizer(model_name=args.model, cache_dir=args.cache_dir)

    ###################### Datasets
    logging.info("Loading the Datasets")

    # Evaluation datasets
    dataset_wikitext = load_wikitext2(
        args.local_datasets, cache_dir=args.cache_dir
    )
    dataset_c4_train = load_c4(
        train=True, local=args.local_datasets, cache_dir=args.cache_dir
    )
    dataset_c4_val = load_c4(
        train=False, local=args.local_datasets, cache_dir=args.cache_dir
    )
    dataset_fineweb_edu = load_fineweb_edu(
        local=args.local_datasets, cache_dir=args.cache_dir
    )[:500]
    logging.info("Datasets loaded")

    logging.info("Tokenizing the Datasets")
    wikitext_input_ids = tokenizer(
        "\n\n".join(dataset_wikitext["text"]),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    c4_val_input_ids = tokenizer(
        "\n\n".join(dataset_c4_val["text"]),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    fineweb_edu_input_ids = tokenizer(
        "\n\n".join(dataset_fineweb_edu["text"]),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    logging.info("Datasets tokenized")

    # Calibration datasets
    num_calibration_samples_2ssp = 32
    num_calibration_samples = 256  # For SliceGPT, ShortGPT and Window Based

    calibration_dataset = get_calibration(
        dataset_c4_train,
        tokenizer,
        num_samples=num_calibration_samples,
        seq_len=2048,
    )

    calibration_dataset_2ssp = calibration_dataset[
        :num_calibration_samples_2ssp
    ]
    first_calibration_sample = calibration_dataset[0]

    ###################### Dense model
    if args.dense or args.pruning_method == "dense":
        logging.info("Dense model evaluation")
        logging.info("Loading the model")
        model = loadModel(args.model, args.cache_dir)
        logging.debug(model)
        logging.debug(f"Model config type: {model.config.model_type}")
        _ = getModelParams(model, "Dense model")

        if args.evaluate_inference:
            evaluate_inference_time(model, first_calibration_sample)

        if args.evaluate_downstream:
            results = evaluation_downstream(model, args.model)
            with open(log_downstream, "a") as file:
                file.write("Zeroshot results:\n")
                for task in results["results"].keys():
                    task_res = results["results"][task]
                    file.write(
                        f" - {task_res['alias']} : {task_res['acc,none']}\n"
                    )

        if args.main_table_results:
            ppl_wiki, ppl_c4, ppl_edu = evaluation_ppl(
                model,
                wikitext_input_ids,
                c4_val_input_ids,
                fineweb_edu_input_ids,
            )
            with open(log_all_path, "a") as file:
                file.write(
                    f"Perplexity (wikitext2): {ppl_wiki}\nPerplexity (c4): {ppl_c4}\nPerplexity (fineweb-edu): {ppl_edu}\n"
                )

        if args.evaluate_perplexity:
            ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)
            logging.info(f"Perplexity (wikitext2): {ppl}")

        if args.evaluate_qualitative:
            qualitative_results(model, tokenizer, max_length=128)

    ###################### Pruning

    elif args.pruning_method is not None:
        pruning_method = args.pruning_method
        sparsity_rate = args.sparsity_rate

        logging.info("Loading the model")
        model = loadModel(args.model, args.cache_dir)
        dense_model_stats = getModelParams(model, "Dense model")
        num_blocks = len(model.model.layers)
        logging.debug(model)
        logging.debug(f"Model config type: {model.config.model_type}")
        logging.debug(
            f"[Original model] Full number of parameters = {dense_model_stats[0] / 1e6:.1f}M"
        )
        logging.debug(
            f"[Original model] Main model number of parameters = {dense_model_stats[1] / 1e6:.1f}M"
        )

        if pruning_method == "slicegpt":
            del model  # the model will be loaded by the SliceGPT model adapter
            gc.collect()
            torch.cuda.empty_cache()

        if int(round(sparsity_rate)) == -1:  # prune all possible blocks
            pruning_rates = [i / num_blocks for i in range(1, num_blocks - 1)]
        elif int(round(sparsity_rate)) == -2:  # prune at 25%, 37.5%, 50%
            pruning_rates = [0.25, 0.375, 0.5]
        else:  # Prune a single sparsity rate
            pruning_rates = [sparsity_rate]

        for target_sparsity in pruning_rates:
            # Measure pruning time
            start_time = time.time()

            if pruning_method in [
                "window_based",
                "shortgpt",
                "subblockshortgpt",
                "blockpruner",
                "evopress",
                "merge",
                "mergeshort",
            ]:
                target_sparsity_blocks = target_sparsity * num_blocks
                if not target_sparsity_blocks.is_integer():
                    logging.warning(
                        f"Invalid sparsity ({target_sparsity_blocks}) rate for {pruning_method}: must be a multiple of 1/{num_blocks} since model has {num_blocks} blocks."
                    )
                    target_sparsity_blocks = int(
                        math.ceil(target_sparsity_blocks)
                    )
                    logging.warning(
                        f"Rounding to next valid sparsity rate: {target_sparsity_blocks / num_blocks:.6f} ({int(target_sparsity_blocks)} blocks)"
                    )
                else:
                    target_sparsity_blocks = int(target_sparsity_blocks)

                target_sparsity = target_sparsity_blocks / num_blocks

            logging.info(
                f"Pruning rate {target_sparsity * 100} (equivalent of {target_sparsity * num_blocks} blocks)"
            )

            attnMask = mlpMask = None
            if pruning_method == "window_based":
                attnMask = mlpMask = window_based(
                    model, target_sparsity_blocks, calibration_dataset
                )
            elif pruning_method == "shortgpt":
                attnMask = mlpMask = shortGPT(
                    model, target_sparsity_blocks, calibration_dataset
                )
            elif pruning_method == "subblockshortgpt":
                attnMask, mlpMask = subblockShortGPT(
                    model, target_sparsity_blocks, calibration_dataset
                )
            elif pruning_method == "merge":
                attnMask = mlpMask = merge(
                    model, target_sparsity_blocks, calibration_dataset
                )
            elif pruning_method == "mergeshort":
                attnMask, mlpMask = mergeshort(
                    model, target_sparsity_blocks, calibration_dataset
                )
            elif pruning_method == "blockpruner":
                attnMask, mlpMask = blockpruner(
                    model, target_sparsity_blocks, first_calibration_sample
                )
            elif pruning_method == "evopress":
                attnMask, mlpMask = evopress(
                    model,
                    target_sparsity_blocks,
                    tokenizer,
                    dataset_c4_train,
                    drop_entire_block=False,
                )
            elif pruning_method == "putri":
                dataset = calibration_dataset
                model = widthmerge_aux(
                    model,
                    dataset,
                    target_sparsity,
                    merge_method=args.merge_method,
                    score_method=args.score_method,
                    attn_prune_method=args.attention_prune_method,
                )
            elif pruning_method == "slicegpt":
                model_path = os.path.join(args.cache_dir, args.model)
                model = slicegpt(
                    model_path,
                    target_sparsity,
                    calibration_dataset,
                )
            else:
                logging.error("Invalid method provided")
                exit(1)

            end_time = time.time()
            logging.info(f"Pruning Time: {end_time - start_time} s")

            if attnMask is not None:
                logging.debug(
                    f"Pruned blocks: attn ({sum(attnMask)}/{len(attnMask)})={attnMask} mlp ({sum(mlpMask)}/{len(mlpMask)})={mlpMask}"
                )
                maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

            pruned_model_stats = getModelParams(
                model, "Pruned model", attnMask=attnMask, mlpMask=mlpMask
            )
            logging.info(
                f"Pruned ratio (full): [{pruned_model_stats[0] / 1e6:.1f}M/{dense_model_stats[0] / 1e6:.1f}M] ~ {100 - (pruned_model_stats[0] / dense_model_stats[0] * 100):.2f}% sparsity"
            )
            logging.info(
                f"Pruned ratio (only layers): [{pruned_model_stats[1] / 1e6:.1f}M/{dense_model_stats[1] / 1e6:.1f}M] ~ {100 - (pruned_model_stats[1] / dense_model_stats[1] * 100):.2f}% sparsity"
            )

            if args.evaluate_inference:
                evaluate_inference_time(model, first_calibration_sample)

            if args.evaluate_downstream:
                results = evaluation_downstream(model, args.model)
                with open(log_downstream, "a") as file:
                    file.write("Zeroshot results:\n")
                    for task in results["results"].keys():
                        task_res = results["results"][task]
                        file.write(
                            f" - {task_res['alias']} : {task_res['acc,none']}\n"
                        )

            if args.main_table_results:
                ppl_wiki, ppl_c4, ppl_edu = evaluation_ppl(
                    model,
                    wikitext_input_ids,
                    c4_val_input_ids,
                    fineweb_edu_input_ids,
                )
                with open(log_all_path, "a") as file:
                    file.write(
                        f"Perplexity (wikitext2): {ppl_wiki}\nPerplexity (c4): {ppl_c4}\nPerplexity (fineweb-edu): {ppl_edu}\n"
                    )

            if args.evaluate_perplexity:
                ppl_wiki = evaluate_perplexity(
                    model, wikitext_input_ids, seq_len=2048
                )
                logging.info(f"Perplexity (wikitext2): {ppl_wiki}")
                with open(log_wiki_path, "w") as file:
                    file.write(f"Perplexity (wikitext2): {ppl_wiki}")

            if args.evaluate_qualitative:
                qualitative_results(model, tokenizer, max_length=128)

            if attnMask is None:
                reset_mlps_shape(model)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                if pruning_method != "slicegpt":
                    model = loadModel(args.model, args.cache_dir)
            else:
                unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)

    ###################### Ablations
    if args.ablation:
        run_ablations(
            args,
            tokenizer,
            dataset_c4_train,
            wikitext_input_ids,
            calibration_dataset_2ssp,
        )


if __name__ == "__main__":
    main()
