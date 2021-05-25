import os
import torch
import numpy as np
from datasets import load_metric
from torch import cuda
import jsonlines


def main():
    # Setup devices
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    # Minor environment tweak to stop parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ROUGE metrics
    ext_rouge = load_metric("rouge", cache_dir="./metric_caches/ext_metric_cache/", stemming=True)
    pegasus_rouge = load_metric("rouge", cache_dir="./metric_caches/pegasus_metric_cache/", stemming=True)
    pegasus_ind_rouge = load_metric("rouge", cache_dir="./metric_caches/pegasus_ind_metric_cache/", stemming=True)
    t5_rouge = load_metric("rouge", cache_dir="./metric_caches/t5_metric_cache/", stemming=True)
    bart_rouge = load_metric("rouge", cache_dir="./metric_caches/bart_metric_cache/", stemming=True)
    gpt2_rouge = load_metric("rouge", cache_dir="./metric_caches/gpt2_metric_cache/", stemming=True)
    xlnet_rouge = load_metric("rouge", cache_dir="./metric_caches/xlnet_metric_cache/", stemming=True)
    prophetnet_rouge = load_metric("rouge", cache_dir="./metric_caches/prophetnet_metric_cache/", stemming=True)
    led_rouge = load_metric("rouge", cache_dir="./metric_caches/led_metric_cache/", stemming=True)
    mmr_filtered_rouge = load_metric("rouge", cache_dir="./metric_caches/mmr_filtered_metric_cache/", stemming=True)

    sd = jsonlines.open('./outputs/mmr_sum_set.jsonl', 'r')

    for batch in sd:
        print("Ind_number: " + str(batch["ind_number"]))

        ext_rouge.add_batch(predictions=[batch['ext_summary']], references=[batch['gold_summary']])
        pegasus_rouge.add_batch(predictions=[batch['pegasus_summary']], references=[batch['gold_summary']])
        t5_rouge.add_batch(predictions=[batch['t5_summary']], references=[batch['gold_summary']])
        bart_rouge.add_batch(predictions=[batch['bart_summary']], references=[batch['gold_summary']])
        gpt2_rouge.add_batch(predictions=[batch['gpt2_summary']], references=[batch['gold_summary']])
        xlnet_rouge.add_batch(predictions=[batch['xlnet']], references=[batch['gold_summary']])
        prophetnet_rouge.add_batch(predictions=[batch['prophetnet_summary']], references=[batch['gold_summary']])
        pegasus_ind_rouge.add_batch(predictions=[batch['pegasus_ind']], references=[batch['gold_summary']])
        led_rouge.add_batch(predictions=[batch['led_summary']], references=[batch['gold_summary']])
        mmr_filtered_rouge.add_batch(predictions=[batch['mmr_string']], references=[batch['gold_summary']])

    mmr_output = mmr_filtered_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    ext_rouge_output = ext_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    pegasus_rouge_output = pegasus_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    t5_rouge_output = t5_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    bart_rouge_output = bart_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    gpt2_rouge_output = gpt2_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    prophetnet_output = prophetnet_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    xlnet_rouge_output = xlnet_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    pegasus_ind_output = pegasus_ind_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    led_output = led_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])

    mmr_score = {
        "rouge1_fmeasure_mid": round(mmr_output["rouge1"].mid.fmeasure, 4),
        "rouge2_fmeasure_mid": round(mmr_output["rouge2"].mid.fmeasure, 4),
        "rougeL_fmeasure_mid": round(mmr_output["rougeL"].mid.fmeasure, 4),
    }
    ext_score = {
        "rouge1_fmeasure_mid": round(ext_rouge_output["rouge1"].mid.fmeasure, 4),
        "rouge2_fmeasure_mid": round(ext_rouge_output["rouge2"].mid.fmeasure, 4),
        "rougeL_fmeasure_mid": round(ext_rouge_output["rougeL"].mid.fmeasure, 4),
    }
    pegasus_score = {
        "rouge1_fmeasure_mid": round(pegasus_rouge_output["rouge1"].mid.fmeasure, 4),
        "rouge2_fmeasure_mid": round(pegasus_rouge_output["rouge2"].mid.fmeasure, 4),
        "rougeL_fmeasure_mid": round(pegasus_rouge_output["rougeL"].mid.fmeasure, 4),
    }
    t5_score = {
        "rouge1_fmeasure_mid": round(t5_rouge_output["rouge1"].mid.fmeasure, 4),
        "rouge2_fmeasure_mid": round(t5_rouge_output["rouge2"].mid.fmeasure, 4),
        "rougeL_fmeasure_mid": round(t5_rouge_output["rougeL"].mid.fmeasure, 4),
    }
    bart_score = {
        "rouge1_fmeasure_mid": round(bart_rouge_output["rouge1"].mid.fmeasure, 4),
        "rouge2_fmeasure_mid": round(bart_rouge_output["rouge2"].mid.fmeasure, 4),
        "rougeL_fmeasure_mid": round(bart_rouge_output["rougeL"].mid.fmeasure, 4),
    }
    gpt2_score = {
        "rouge1_fmeasure_mid": round(gpt2_rouge_output["rouge1"].mid.fmeasure, 4),
        "rouge2_fmeasure_mid": round(gpt2_rouge_output["rouge2"].mid.fmeasure, 4),
        "rougeL_fmeasure_mid": round(gpt2_rouge_output["rougeL"].mid.fmeasure, 4),
    }

    xlnet_score = {
        "rouge1_fmeasure_mid": round(xlnet_rouge_output["rouge1"].mid.fmeasure, 4),
        "rouge2_fmeasure_mid": round(xlnet_rouge_output["rouge2"].mid.fmeasure, 4),
        "rougeL_fmeasure_mid": round(xlnet_rouge_output["rougeL"].mid.fmeasure, 4),
    }

    prophetnet_score = {
        "rouge1_fmeasure_mid": round(prophetnet_output["rouge1"].mid.fmeasure, 4),
        "rouge2_fmeasure_mid": round(prophetnet_output["rouge2"].mid.fmeasure, 4),
        "rougeL_fmeasure_mid": round(prophetnet_output["rougeL"].mid.fmeasure, 4),
    }

    pegasus_ind_score = {
        "rouge1_fmeasure_mid": round(pegasus_ind_output["rouge1"].mid.fmeasure, 4),
        "rouge2_fmeasure_mid": round(pegasus_ind_output["rouge2"].mid.fmeasure, 4),
        "rougeL_fmeasure_mid": round(pegasus_ind_output["rougeL"].mid.fmeasure, 4),
    }

    led_score = {
        "rouge1_fmeasure_mid": round(led_output["rouge1"].mid.fmeasure, 4),
        "rouge2_fmeasure_mid": round(led_output["rouge2"].mid.fmeasure, 4),
        "rougeL_fmeasure_mid": round(led_output["rougeL"].mid.fmeasure, 4),
    }
    fd = open('./outputs/all_models_rouge_scores.txt', "w+")

    print("MMR: " + str(mmr_score))
    fd.write("MMR: " + str(mmr_score) + '\n')
    print("PEGASUS: " + str(pegasus_score))
    fd.write("PEGASUS: " + str(pegasus_score) + '\n')
    print("EXT: " + str(ext_score))
    fd.write("EXT: " + str(ext_score) + '\n')
    print("T5: " + str(t5_score))
    fd.write("T5: " + str(t5_score) + '\n')
    print("BART: " + str(bart_score))
    fd.write("BART: " + str(bart_score) + '\n')
    print("GPT2: " + str(gpt2_score))
    fd.write("GPT2: " + str(gpt2_score) + '\n')
    print("XLNet: " + str(xlnet_score))
    fd.write("XLNet: " + str(xlnet_score) + '\n')
    print("ProphetNet: " + str(prophetnet_score))
    fd.write("ProphetNet: " + str(prophetnet_score) + '\n')
    print("PEGASUS Ind: " + str(pegasus_ind_score))
    fd.write("PEGASUS Ind: " + str(pegasus_ind_score) + '\n')
    print("LED: " + str(led_score))
    fd.write("LED: " + str(led_score) + '\n')
    fd.close()


if __name__ == "__main__":
    main()
