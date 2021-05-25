import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from torch import cuda
import argparse
import json
import nltk
from MMR_Filter import compute_maximal_marginal_relevance, process_LDA_topics, process_doc2vec_similarity
from summarizer import TransformerSummarizer
from langdetect import detect

nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Summarizer')
    parser.add_argument('--batch_size', type=int, default=1, help='batches to iterate through at a time')
    parser.add_argument('--run_from', type=int, default=1, help='the iteration to start building the dataset from')
    parser.add_argument('--run_until', type=int, default=999999, help='the iteration to stop building the dataset after')
    parser.add_argument('--lambdac', type=float, default=1e-14, help='the lambda constant to use in all MMR calculations')
    parser.add_argument('--rouge', action='store_true')
    parser.add_argument('--cloud', action='store_true')
    args = parser.parse_args()

    # Setup devices
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    # Minor environment tweak to stop parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ROUGE metrics
    if args.rouge:
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

        def rouge_compute():
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

    # PEGASUS
    pegasus_tokenizer = AutoTokenizer.from_pretrained('google/pegasus-multi_news')
    pegasus_model = AutoModelForSeq2SeqLM.from_pretrained('google/pegasus-multi_news')
    pegasus_model.resize_token_embeddings(len(pegasus_tokenizer))
    pegasus_model = pegasus_model.to(device)
    # T5
    t5_summarizer = pipeline("summarization", model="t5-large", tokenizer="t5-large", framework="pt", device=0)
    # BART
    bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn",
                               framework="pt", device=0)
    # GPT2
    gpt2_summarizer = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2")
    # XLNet
    xlnet_summarizer = TransformerSummarizer(transformer_type="XLNet", transformer_model_key="xlnet-base-cased")
    # ProphetNet
    prophetnet_summarizer = pipeline("summarization", model="microsoft/prophetnet-large-uncased-cnndm",
                                     tokenizer="microsoft/prophetnet-large-uncased-cnndm", framework="pt", device=0)
    # LED
    led_summarizer = pipeline("summarization", model="allenai/led-base-16384", tokenizer="allenai/led-base-16384",
                              framework="pt", device=0)

    def collate_fn(batch):
        batch_list = []
        for item in batch:
            batch_list.append({"text": item["text"],
                               "ext_summary": item["ext_summary"],
                               "concat_text": item["concat_text"],
                               "gold_summary": item["gold_summary"]})
        return batch_list

    class SplitTestDataset(Dataset):
        def __init__(self, dataframe, articles_dataframe):
            self.data = dataframe
            self.ext_summary = self.data['ext_summary']

            self.articles_data = articles_dataframe
            self.text = self.articles_data['document']
            self.gold = self.articles_data['summary']

        def __len__(self):
            return len(self.text)

        def __getitem__(self, index):
            text = self.text[index]
            ext_summary = self.ext_summary[index]
            gold = self.gold[index]

            # Here's the only part that really matters:
            #       What get_item returns. The inputs, the attention mask, and the outputs.
            #       These will be different depending on what the model being used takes.
            return {
                'text': text,
                'ext_summary': ext_summary,
                'concat_text': ext_summary + text,
                'gold_summary': gold
            }

    # Load the datasets
    test_set = load_dataset('json', data_files='./matchsum_output/matchsum_output.jsonl', split='train')
    articles_set = load_dataset('multi_news', split='test')
    articles_test_set = SplitTestDataset(test_set, articles_set)
    test_loader = DataLoader(articles_test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_fn)

    # Clear and open the output file
    if args.run_from <= 1:
        open('./outputs/mmr_sum_set.jsonl', 'w+').close()
        sd = open('./outputs/mmr_sum_set.jsonl', 'w+', encoding="utf-8")
    else:
        sd = open('./outputs/mmr_sum_set.jsonl', 'a', encoding="utf-8")

    ind_number = 0

    if args.run_from > 1:
        sd.write('\n')
    for collator in test_loader:
        for batch in collator:
            ind_number = ind_number + 1
            if ind_number >= args.run_from:
                print('\n' + "Iteration: " + str(ind_number))
                # LDA
                lda_topic_words = process_LDA_topics(batch["text"], num_of_topics=5, num_of_words=1,
                                                     mallet=True, cloud=args.cloud)
                # PEGASUS
                pegasus_summary = pegasus_tokenizer.prepare_seq2seq_batch(batch["text"],
                                                                          max_length=pegasus_tokenizer.model_max_length,
                                                                          padding='max_length', truncation=True,
                                                                          return_tensors="pt").to(device)
                pegasus_summary = pegasus_model.generate(**pegasus_summary)
                pegasus_summary = pegasus_tokenizer.batch_decode(pegasus_summary, skip_special_tokens=True)[0]

                t5_summary = ""
                bart_summary = ""
                gpt2_summary = ""
                xlnet_summary = ""
                peg_ind_summary = ""
                prophetnet_summary = ""
                led_summary = ""
                ext_summary = batch["ext_summary"]

                def print_summaries():
                    print("mmr bart_summary : " + bart_summary)
                    print("mmr t5_summary : " + t5_summary)
                    print("mmr prophetnet_summary : " + prophetnet_summary)
                    print("mmr gpt2_summary : " + gpt2_summary)
                    print("mmr led_summary : " + led_summary)
                    print("mmr xlnet_summary : " + xlnet_summary)
                    print("mmr peg_ind_summary : " + peg_ind_summary)
                    print("mmr ext_summary : " + ext_summary)

                for document in (batch['text']).split('|||||'):
                    if document is not None and len(nltk_tokenizer.tokenize(document)) > 3:

                        peg_ids = pegasus_tokenizer.prepare_seq2seq_batch(document + " " + peg_ind_summary,
                                                                          max_length=pegasus_tokenizer.model_max_length,
                                                                          padding='max_length', truncation=True,
                                                                          return_tensors="pt").to(device)
                        peg_ids = pegasus_model.generate(**peg_ids)
                        peg_ind_summary = pegasus_tokenizer.batch_decode(peg_ids, skip_special_tokens=True)[
                                              0] + " " + peg_ind_summary

                        t5_summary = ''.join(t5_summarizer(document[:512], truncation=True, max_length=300, do_sample=False)[0][
                                                 'summary_text']) + " " + t5_summary
                        xlnet_summary = ''.join(xlnet_summarizer(document[:1024], max_length=300)).replace('\n',
                                                                                                           ' ') + " " + xlnet_summary
                        gpt2_summary = ''.join(gpt2_summarizer(document[:1024], max_length=300)).replace('\n',
                                                                                                         ' ') + " " + gpt2_summary
                        if detect(document) == 'en':
                            prophetnet_summary = ''.join(
                                prophetnet_summarizer(document[:1024], truncation=True, max_length=300, do_sample=False)[0][
                                    'summary_text']) + " " + prophetnet_summary
                        else:
                            prophetnet_summary = prophetnet_summary
                        led_summary = ''.join(led_summarizer(document[:1024], truncation=True, max_length=300, do_sample=False)[0][
                                                  'summary_text']) + " " + led_summary
                        bart_summary = ''.join(bart_summarizer(document[:1024], truncation=True, max_length=300, do_sample=False)[0][
                                                   'summary_text']) + " " + bart_summary
                """
                print("bart_summary : " + bart_summary)
                print("t5_summary : " + t5_summary)
                print("prophetnet_summary : " + prophetnet_summary)
                print("gpt2_summary : " + gpt2_summary)
                print("led_summary : " + led_summary)
                print("xlnet_summary : " + xlnet_summary)
                print("peg_ind_summary : " + peg_ind_summary)
                """

                # MMR
                """
                peg_summary_reduced = " ".join(compute_maximal_marginal_relevance(pegasus_summary,
                                                                                      lda_topic_words,
                                                                                      max(1,
                                                                                          len(nltk_tokenizer.tokenize(
                                                                                              pegasus_summary)) - 1),
                                                                                      lambda_constant=args.lambdac,
                                                                                      sim=process_doc2vec_similarity)).replace(
                    '\n', ' ')
                """
                ext_summary_reduced = " ".join(compute_maximal_marginal_relevance(ext_summary,
                                                                                  lda_topic_words,
                                                                                  max(1, len(nltk_tokenizer.tokenize(
                                                                                      batch['ext_summary'])) - 1),
                                                                                  lambda_constant=args.lambdac,
                                                                                  sim=process_doc2vec_similarity)).replace(
                    '\n', ' ')
                peg_ind_summary_reduced = " ".join(compute_maximal_marginal_relevance(peg_ind_summary,
                                                                                      lda_topic_words,
                                                                                      max(1,
                                                                                          len(nltk_tokenizer.tokenize(
                                                                                              peg_ind_summary)) - 1),
                                                                                      lambda_constant=args.lambdac,
                                                                                      sim=process_doc2vec_similarity)).replace(
                    '\n', ' ')
                t5_summary_reduced = " ".join(compute_maximal_marginal_relevance(t5_summary,
                                                                         lda_topic_words,
                                                                         max(1, len(nltk_tokenizer.tokenize(
                                                                             t5_summary)) - 1),
                                                                         lambda_constant=args.lambdac,
                                                                         sim=process_doc2vec_similarity)).replace('\n', ' ')
                xlnet_summary_reduced = " ".join(compute_maximal_marginal_relevance(xlnet_summary,
                                                                            lda_topic_words,
                                                                            max(1, len(nltk_tokenizer.tokenize(
                                                                                xlnet_summary)) - 1),
                                                                            lambda_constant=args.lambdac,
                                                                            sim=process_doc2vec_similarity)).replace('\n', ' ')
                bart_summary_reduced = " ".join(compute_maximal_marginal_relevance(bart_summary,
                                                                           lda_topic_words,
                                                                           max(1, len(nltk_tokenizer.tokenize(
                                                                               bart_summary)) - 1),
                                                                           lambda_constant=args.lambdac,
                                                                           sim=process_doc2vec_similarity)).replace('\n', ' ')

                gpt2_summary_reduced = " ".join(compute_maximal_marginal_relevance(gpt2_summary,
                                                                           lda_topic_words,
                                                                           max(1, len(nltk_tokenizer.tokenize(
                                                                               gpt2_summary)) - 1),
                                                                           lambda_constant=args.lambdac,
                                                                           sim=process_doc2vec_similarity)).replace('\n', ' ')

                prophetnet_summary_reduced = " ".join(compute_maximal_marginal_relevance(prophetnet_summary,
                                                                                 lda_topic_words,
                                                                                 max(1, len(nltk_tokenizer.tokenize(
                                                                                     prophetnet_summary)) - 1),
                                                                                 lambda_constant=args.lambdac,
                                                                                 sim=process_doc2vec_similarity)).replace('\n', ' ')
                led_summary_reduced = " ".join(compute_maximal_marginal_relevance(led_summary,
                                                                          lda_topic_words,
                                                                          max(1, len(nltk_tokenizer.tokenize(
                                                                              led_summary)) - 1),
                                                                          lambda_constant=args.lambdac)).replace('\n', ' ')

                full_sequence = pegasus_summary \
                                + " " + bart_summary_reduced \
                                + " " + t5_summary_reduced \
                                + " " + xlnet_summary_reduced \
                                + " " + peg_ind_summary_reduced \
                                + " " + ext_summary_reduced \
                                + " " + gpt2_summary_reduced \
                                + " " + prophetnet_summary_reduced \
                                + " " + led_summary_reduced

                mmr_list = compute_maximal_marginal_relevance(full_sequence,
                                                              lda_topic_words,
                                                              max(1, len(nltk_tokenizer.tokenize(pegasus_summary)) + 1),
                                                              lambda_constant=1e-19)
                best_string = " ".join(mmr_list).replace('\n', ' ')
                best_string = " ".join(compute_maximal_marginal_relevance(best_string,
                                                                          lda_topic_words,
                                                                          max(1, len(
                                                                              nltk_tokenizer.tokenize(best_string))),
                                                                          lambda_constant=1,
                                                                          sim=process_doc2vec_similarity))

                # OUTPUT writing
                sd.write(json.dumps({"ind_number": ind_number,
                                    "text": batch["text"],
                                     "gold_summary": batch["gold_summary"],
                                     "ext_summary": batch['ext_summary'],
                                     "pegasus_summary": pegasus_summary,
                                     "pegasus_ind": peg_ind_summary,
                                     "xlnet": xlnet_summary,
                                     "t5_summary": t5_summary,
                                     "bart_summary": bart_summary,
                                     "gpt2_summary": gpt2_summary,
                                     "prophetnet_summary": prophetnet_summary,
                                     "led_summary": led_summary,
                                     "mmr_string": best_string}))
                sd.write('\n')
                print("Iteration written to output." + '\n')

                # Add ROUGE batches
                if args.rouge:
                    ext_rouge.add_batch(predictions=[batch['ext_summary']], references=[batch['gold_summary']])
                    pegasus_rouge.add_batch(predictions=[pegasus_summary], references=[batch['gold_summary']])
                    t5_rouge.add_batch(predictions=[t5_summary], references=[batch['gold_summary']])
                    bart_rouge.add_batch(predictions=[bart_summary], references=[batch['gold_summary']])
                    gpt2_rouge.add_batch(predictions=[gpt2_summary], references=[batch['gold_summary']])
                    xlnet_rouge.add_batch(predictions=[xlnet_summary], references=[batch['gold_summary']])
                    prophetnet_rouge.add_batch(predictions=[prophetnet_summary], references=[batch['gold_summary']])
                    pegasus_ind_rouge.add_batch(predictions=[peg_ind_summary], references=[batch['gold_summary']])
                    led_rouge.add_batch(predictions=[led_summary], references=[batch['gold_summary']])
                    mmr_filtered_rouge.add_batch(predictions=[best_string], references=[batch['gold_summary']])

            if ind_number >= args.run_until:
                if args.rouge:
                    rouge_compute()
                exit()

    print()
    if args.rouge:
        rouge_compute()


if __name__ == "__main__":
    main()
