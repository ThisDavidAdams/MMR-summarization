import nltk
from mmr_filter import compute_maximal_marginal_relevance, process_LDA_topics, process_doc2vec_similarity, \
    process_tfidf_similarity
from datasets import load_metric
import jsonlines
import argparse
import skopt
import neptune
import neptunecontrib.monitoring.skopt as sk_utils

nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

SPACE = [
    skopt.space.Real(0, 1, name='lambdac', prior='uniform'),
    skopt.space.Real(0.1, 1, name='percentage', prior='uniform'),
    skopt.space.Integer(1, 10, name='num_topics'),
    skopt.space.Integer(1, 10, name='num_words')
]


def add_mmr(params):
    # Define similarity measure to use for overall mmr combination
    if args.sim == 'doc2vec':
        sim_argument = process_doc2vec_similarity
    else:
        sim_argument = process_tfidf_similarity

    # ROUGE metrics
    mmr_filtered_rouge = load_metric("rouge", cache_dir="./metric_caches/mmr_filtered_metric_cache/", stemming=True)

    ind_number = 0

    # Optimize from pre-generated JSONL file of summaries.
    id = jsonlines.open(args.input_file, 'r')
    for batch in id:
        ind_number = ind_number + 1

        if ind_number >= 1:
            # LDA
            lda_topic_words = process_LDA_topics(batch["text"], num_of_topics=params['num_topics'],
                                                 num_of_words=params['num_words'],
                                                 mallet=True, cloud=args.cloud)
            # PEGASUS
            pegasus_summary = batch["pegasus_summary"]
            t5_summary = batch["t5_summary"]
            bart_summary = batch["bart_summary"]
            gpt2_summary = batch["gpt2_summary"]
            xlnet_summary = batch["xlnet_summary"]
            peg_ind_summary = batch["pegasus_ind"]
            prophetnet_summary = batch["prophetnet_summary"]
            led_summary = batch["led_summary"]
            ext_summary = batch["ext_summary"]

            full_sequence = pegasus_summary \
                            + " " + ext_summary \
                            + " " + peg_ind_summary \
                            + " " + bart_summary \
                            + " " + t5_summary \
                            + " " + xlnet_summary \
                            + " " + ext_summary \
                            + " " + gpt2_summary \
                            + " " + prophetnet_summary \
                            + " " + led_summary

            full_sequence = full_sequence.replace('[X_SEP]', '')

            mmr_length = len(nltk_tokenizer.tokenize(pegasus_summary)) + max(1,
                                                                             int(round(
                                                                                 len(nltk_tokenizer.tokenize(
                                                                                     pegasus_summary)) * params[
                                                                                     'percentage'])))
            best_string = " ".join(compute_maximal_marginal_relevance(full_sequence,
                                                                      lda_topic_words, mmr_length,
                                                                      lambda_constant=params['lambdac'],
                                                                      sim=sim_argument,
                                                                      mmr_percentage=params['percentage'])).replace(
                '\n',
                ' ')
            # Add ROUGE batches
            mmr_filtered_rouge.add_batch(predictions=[best_string], references=[batch['gold_summary']])
        if ind_number >= args.run_until:
            mmr_output = mmr_filtered_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
            return round(mmr_output["rouge1"].mid.fmeasure, 4)

    mmr_output = mmr_filtered_rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    return round(mmr_output["rouge1"].mid.fmeasure, 4)


@skopt.utils.use_named_args(SPACE)
def objective(**params):
    return -1.0 * add_mmr(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarizer')
    parser.add_argument('--project_name', type=str, default='davehead3/example-project-pytorch',
                        help='Neptune project name to initialize.')
    parser.add_argument('--api_token', type=str,
                        default='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Mjg0N2NlYS1iNDJkLTQxZDAtYTQ1NS1jNDU1MWY0YjRiNDkifQ==',
                        help='Neptune api token (should be very long).')
    parser.add_argument('--input_file', type=str, default='outputs/mmr_opt_set.jsonl',
                        help='File to read model summaries from.')
    parser.add_argument('--cloud', action='store_true',
                        help='Whether or not the script is running on Google Colab.')
    parser.add_argument('--sim', type=str, default='doc2vec',
                        help='the similarity to use for MMR. Accepts doc2vec or tfidf')
    parser.add_argument('--run_until', type=int, default=100,
                        help='The iteration to stop the optimization run after.')
    parser.add_argument('--num_calls', type=int, default=20,
                        help='The number of times to run the program for optimization.')
    parser.add_argument('--num_rand_starts', type=int, default=10,
                        help='The number of random starts for the randomization.')
    args = parser.parse_args()

    neptune.init(args.project_name,
                 api_token=args.api_token)
    neptune.create_experiment('hpo-on-any-script', upload_source_files=['*.py'])

    monitor = sk_utils.NeptuneMonitor()
    results = skopt.forest_minimize(objective, SPACE, n_calls=args.num_calls, n_random_starts=args.num_rand_starts)
    best_auc = -1.0 * results.fun
    best_params = results.x

    print('best result: ', best_auc)
    print('best parameters: ', best_params)

    neptune.stop()
