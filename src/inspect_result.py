from argparse import ArgumentParser
from transformers import BertModel
from seaborn import heatmap
import gensim
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import random
import torch

from run_lm import create_args, init_lm
import dataset
from topic_model import load_topic_model, get_top_term_topics
import utils


def cut_sentences_at_length(batch, max_sentence_len):
    assert type(batch) is tuple
    new_batch0 = batch[0][:, :max_sentence_len].clone().detach()  # observations
    new_batch1 = batch[1][:, :max_sentence_len].clone().detach()  # labels
    if batch[2] is not None:
        new_batch2 = batch[2][:, :max_sentence_len].clone().detach()  # attention_mask_all
    else:
        new_batch2 = None
    new_batch3 = [max_sentence_len]  # sentence length
    return new_batch0, new_batch1, new_batch2, new_batch3


def prepare_dev_data(dataset, num_sentences_to_plot, max_sentence_len=None):
    """
    Note: one batch contains `arg.training.batch_size` sentences
    In this case, `arg.training.batch_size` should be set to 1
    Return all the first `num_sentences_to_plot` sentences as a list.
    """
    dev_batches = dataset.get_dev_dataloader()
    batches = []
    i = 0
    for batch in dev_batches:
        if max_sentence_len is not None:
            batch = cut_sentences_at_length(batch, max_sentence_len)
        batches.append(batch)
        i += 1
        if i == num_sentences_to_plot:
            break
    return batches


def translate_ids_to_tokens(token_ids, token2id):
    ids_to_tokens = {token2id[token]: token for token in token2id}
    return [ids_to_tokens[token_id] for token_id in token_ids]


def get_bert_embeddings(bert, input_ids, token_type_ids):
    return bert.embeddings(
        input_ids=input_ids,
        position_ids=None,
        token_type_ids=token_type_ids,
        inputs_embeds=None,
        past_key_values_length=0,
    )


def get_attention_bert(model, input_ids, token_type_ids):
    """
    Get the attention weights for ONE sentence
    """
    assert len(input_ids) == 1, 'the input must contain exactly 1 sentence'
    all_attention_outputs = model.forward(
        input_ids,
        token_type_ids=token_type_ids,
        output_attentions=True,
    ).attentions
    all_attention_outputs = np.array([
        a[0].cpu().detach().numpy()
        for a in all_attention_outputs
    ])  # has shape [num_hidden_layers, num_attention_heads, sentence_len, sentence_len]
    return all_attention_outputs


def plot_attention_bert(lm_model, batches, plot_save_dir):
    for sentence_idx, batch in enumerate(batches):
        batch_size, seq_length = batch[0].size()
        buffered_token_type_ids = lm_model.model.bert.embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        token_type_ids = buffered_token_type_ids_expanded

        all_attention_outputs = get_attention_bert(lm_model.model.bert, batch[0], token_type_ids)

        tokens = translate_ids_to_tokens(batch[0][0].to('cpu').numpy(), dataset.vocab)
        for layer_idx in range(len(all_attention_outputs)):
            for head_idx in range(len(all_attention_outputs[0])):
                plt.figure(figsize=(15, 12))
                ax = plt.axes()
                heatmap(all_attention_outputs[layer_idx][head_idx], xticklabels=tokens, yticklabels=tokens)
                ax.set_title(f"sentence{sentence_idx} layer{layer_idx} head{head_idx}\n{' '.join(tokens)}")
                plt.savefig(
                    os.path.join(plot_save_dir, f"sentence{sentence_idx}_layer{layer_idx}_head{head_idx}.png"))
                plt.show()


def compare_topic_and_non_topic_attention(
        model,
        topic_model,
        num_topics_per_word,
        data,
        max_num_docs,
        stop_tokens=set(),
        ref_sentence_len=None,
        skip_num_docs=0,
        filter_ambiguous_words_threshold=0.0,
):
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    topic_attn_stat = [
        [
            {
                'same_token_attn': 0.0,
                'same_token_cnt': 0,
                'topic_attn': 0.0,
                'topic_cnt': 0,
                'non_topic_attn': 0.0,
                'non_topic_cnt': 0,
                'empty_topic_attn': 0.0,
                'empty_topic_cnt': 0,
                'same_token_attn_fraction_raw': [],
                'topic_attn_fraction_raw': [],
                'non_topic_attn_fraction_raw': [],
                'empty_topic_attn_fraction_raw': [],
                'max_other_topic_attn_fraction_raw': [],
                'same_token_attn_fraction_softmax': [],
                'topic_attn_fraction_softmax': [],
                'non_topic_attn_fraction_softmax': [],
                'empty_topic_attn_fraction_softmax': [],
                'max_other_topic_attn_fraction_softmax': [],
            }
            for head in range(num_heads)
        ]
        for layer in range(num_layers)
    ]
    num_topics_in_docs = []
    same_token_attn_list = []
    topic_attn_list = []
    non_topic_attn_list = []
    empty_topic_attn_list = []

    num_docs = 0
    if type(topic_model) is dict:
        num_topics = len(set(topic_model.values()))
    elif type(topic_model) is gensim.models.ldamodel.LdaModel:
        num_topics = topic_model.get_topics().shape[0]
    else:
        raise NotImplementedError('topic_model should be a gensim.models.ldamodel.LdaModel or dict')

    for doc in data:
        if skip_num_docs > 0:
            skip_num_docs -= 1
            continue
        if type(doc) is tuple:  # a batch (must contain exactly 1 sentence)
            tokens = doc[0][0].to('cpu').numpy()
            words = None
            batch_size, seq_length = doc[0].size()
            buffered_token_type_ids = model.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
            all_attention_outputs = get_attention_bert(model, doc[0], token_type_ids)
            if ref_sentence_len is not None:
                all_attention_outputs = all_attention_outputs * len(tokens) / ref_sentence_len
        else:
            print(f"type(doc): {type(doc)}")
            print(f"doc: {doc}")
            raise NotImplementedError(
                "doc can either be a dict (an entry of wikidata['train']) or a tuple (an entry of DataLoader)"
            )

        n = len(tokens)
        # If a word has one topic, include the topic
        # If a word has multiple topics, only include the topics that are matched with other words in the doc
        matched_topics_of_tokens = [set()] * n
        topics = [
            get_top_term_topics(topic_model, word, num_topics=num_topics_per_word, filter_ambiguous_words_threshold=filter_ambiguous_words_threshold)
            for word in tokens
        ]

        if all_attention_outputs.shape == (num_layers, num_heads, n, n):
            for layer in range(num_layers):
                for head in range(num_heads):
                    for i in range(n):
                        for j in range(n):
                            if (tokens[i] not in stop_tokens) and (tokens[j] not in stop_tokens)\
                            and (words is None or tokens[i] in words and tokens[j] in words):  # heuristics: the word is not broken into pieces by tokenization
                                if i == j or tokens[i] == tokens[j]:
                                    topic_attn_stat[layer][head]['same_token_attn'] += \
                                        all_attention_outputs[layer][head][i][j]
                                    topic_attn_stat[layer][head]['same_token_cnt'] += 1
                                elif len(topics[i]) == 0 or len(topics[j]) == 0:
                                        topic_attn_stat[layer][head]['empty_topic_attn'] += \
                                        all_attention_outputs[layer][head][i][j]
                                        topic_attn_stat[layer][head]['empty_topic_cnt'] += 1
                                elif len(set(topics[i]).intersection(set(topics[j]))) >= 1:
                                    topic_attn_stat[layer][head]['topic_attn'] += all_attention_outputs[layer][head][i][
                                        j]
                                    topic_attn_stat[layer][head]['topic_cnt'] += 1
                                    matched_topics_of_tokens[i] = matched_topics_of_tokens[i].union(
                                                                    set(topics[i]).intersection(set(topics[j]))
                                                                )
                                else:
                                    topic_attn_stat[layer][head]['non_topic_attn'] += \
                                    all_attention_outputs[layer][head][i][j]
                                    topic_attn_stat[layer][head]['non_topic_cnt'] += 1
        else:
            print(f"\t Shape mismatch: all_attention_outputs.shape=={all_attention_outputs.shape}, n=={n}")

        topics_in_doc = set()
        for i in range(n):
            if len(matched_topics_of_tokens[i]) == 0:
                matched_topics_of_tokens[i] = set(topics[i][:1])
            topics_in_doc = topics_in_doc.union(matched_topics_of_tokens[i])
        num_topics_in_docs.append(len(topics_in_doc))

        num_docs += 1
        if num_docs % 100 == 0:
            print(f"Processed {num_docs} documents.")
        if num_docs >= max_num_docs:
            break

    return topic_attn_stat, num_topics_in_docs, same_token_attn_list, topic_attn_list, non_topic_attn_list, empty_topic_attn_list


def save_topic_and_non_topic_attention_results(
        num_topics_per_word,
        inspect_results_dir,
        topic_attn_stat,
        topic_attn_stat_fn,
):
    text_to_save = f"Computing statistics for num_topics_per_word={num_topics_per_word}\n\n"
    print(text_to_save)

    for layer in range(len(topic_attn_stat)):
        for head in range(len(topic_attn_stat[0])):
            if topic_attn_stat[layer][head]['same_token_cnt'] == 0:
                print(f"Warning: unexpected: topic_attn_stat[{layer}][{head}]['same_token_cnt'] == 0")
                topic_attn_stat[layer][head]['same_token_avg_attn'] = 0.0
            else:
                topic_attn_stat[layer][head]['same_token_avg_attn'] = \
                    topic_attn_stat[layer][head]['same_token_attn'] / topic_attn_stat[layer][head]['same_token_cnt']
            if topic_attn_stat[layer][head]['topic_cnt'] == 0:
                topic_attn_stat[layer][head]['topic_avg_attn'] = 0.0
            else:
                topic_attn_stat[layer][head]['topic_avg_attn'] = \
                    topic_attn_stat[layer][head]['topic_attn'] / topic_attn_stat[layer][head]['topic_cnt']
            if topic_attn_stat[layer][head]['non_topic_cnt'] == 0:
                topic_attn_stat[layer][head]['non_topic_avg_attn'] = 0.0
            else:
                topic_attn_stat[layer][head]['non_topic_avg_attn'] = \
                    topic_attn_stat[layer][head]['non_topic_attn'] / topic_attn_stat[layer][head]['non_topic_cnt']
            if topic_attn_stat[layer][head]['empty_topic_cnt'] == 0:
                topic_attn_stat[layer][head]['empty_topic_avg_attn'] = 0.0
            else:
                topic_attn_stat[layer][head]['empty_topic_avg_attn'] = \
                    topic_attn_stat[layer][head]['empty_topic_attn'] / topic_attn_stat[layer][head]['empty_topic_cnt']
            log_str = (
                f"layer {layer} head {head} \
                same_token_avg_attn: {topic_attn_stat[layer][head]['same_token_avg_attn']}, \
                same_token_cnt: {topic_attn_stat[layer][head]['same_token_cnt']}, \
                topic_avg_attn: {topic_attn_stat[layer][head]['topic_avg_attn']}, \
                topic_cnt: {topic_attn_stat[layer][head]['topic_cnt']}, \
                non_topic_avg_attn: {topic_attn_stat[layer][head]['non_topic_avg_attn']}, \
                non_topic_cnt: {topic_attn_stat[layer][head]['non_topic_cnt']}, \
                empty_topic_avg_attn: {topic_attn_stat[layer][head]['empty_topic_avg_attn']}, \
                empty_topic_cnt: {topic_attn_stat[layer][head]['empty_topic_cnt']}, \
                (topic_avg_attn - non_topic_avg_attn): {topic_attn_stat[layer][head]['topic_avg_attn'] - topic_attn_stat[layer][head]['non_topic_avg_attn']}"
            )
            print(log_str)
            text_to_save += log_str + '\n\n'

            log_str = 'Plotting fractions of same vs. different topic attention'
            print(log_str)
            text_to_save += log_str + '\n\n'

    with open(topic_attn_stat_fn, 'wb') as f:
        pickle.dump(topic_attn_stat, f)
    with open(os.path.join(inspect_results_dir, f"topic_attn_stat_top{num_topics_per_word}.txt"), 'wt') as f:
        f.write(text_to_save)


def check_bert_embedding_dot_product_topic(
        model,
        tokenizer,
        topic_model,
        num_topics_per_word,
        inspect_results_dir,
        stop_tokens=set(),
        filter_ambiguous_words_threshold=0.0,
):
    text_to_save = f"Computing embedding dot product for num_topics_per_word={num_topics_per_word}\n\n"
    print(text_to_save)

    if type(model) in [BertModel]:
        emb = model.embeddings.word_embeddings.weight.data.cpu().numpy()
    else:
        raise NotImplementedError('model type not supported')
    tokens = tokenizer.get_vocab().keys()
    tokens = [token for token in tokens if token[0] != '[']  # filter special characters like '[PAD]', '[unused0]'
    # First 1000 tokens are mostly unused placeholders.
    n = len(tokens)
    topics = [
        get_top_term_topics(topic_model, word, num_topics=num_topics_per_word, filter_ambiguous_words_threshold=filter_ambiguous_words_threshold)
        for word in tokens
    ]

    same_token_dot_list = []
    topic_dot_list = []
    non_topic_dot_list = []
    empty_topic_dot_list = []

    cnt = 0
    for i in random.sample(range(n), 10000):
        if tokens[i] not in stop_tokens:
            same_token_dot_list.append(emb[i] @ emb[i])
        for j in random.sample(range(n), 10000):
            if (tokens[i] not in stop_tokens) and (tokens[j] not in stop_tokens):
                if len(topics[i]) == 0 or len(topics[j]) == 0:
                    empty_topic_dot_list.append(emb[i] @ emb[j])
                elif len(set(topics[i]).intersection(set(topics[j]))) >= 1:
                    topic_dot_list.append(emb[i] @ emb[j])
                else:
                    non_topic_dot_list.append(emb[i] @ emb[j])
        cnt += 1
        if cnt % 1000 == 0:
            print(f"Processed {cnt / 100}% of the planned 1e8 embedding dot products.")

    log_str = (
        f"same_token_avg_emb_dot: {np.mean(same_token_dot_list)}, std {np.std(same_token_dot_list)}\n \
        same_topic_diff_token_avg_emb_dot: {np.mean(topic_dot_list)}, std {np.std(topic_dot_list)}\n \
        non_topic_avg_emb_dot: {np.mean(non_topic_dot_list)}, std {np.std(non_topic_dot_list)}\n \
        empty_topic_avg_emb_dot: {np.mean(empty_topic_dot_list)} std {np.std(empty_topic_dot_list)}\n \
        (same_topic_diff_token - diff_topic): {np.mean(topic_dot_list) - np.mean(non_topic_dot_list)}\n"
    )
    print(log_str)
    text_to_save += log_str + '\n\n'

    with open(os.path.join(inspect_results_dir, f"emb_dot_top{num_topics_per_word}.txt"), 'wt') as f:
        f.write(text_to_save)


def check_bert_embedding_weights(lm_model, inspect_results_dir):
    emb = lm_model.model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

    fn = os.path.join(inspect_results_dir, 'embedding.txt')
    with open(fn, 'wt') as f:
        f.write('emb.shape\n')
        f.write(str(emb.shape))
        f.write('\n\n')

        f.write('emb\n')
        f.write(str(emb))
        f.write('\n\n')

        dot_prod = emb @ emb.T
        f.write('dot_prod\n')
        f.write(str(dot_prod))
        f.write('\n\n')

    plt.figure(figsize=(15, 12))
    ax = plt.axes()
    heatmap(dot_prod)
    ax.set_title(f"embedding_dot_prod")
    plt.savefig(os.path.join(inspect_results_dir, f"embedding_dot_prod.png"))
    plt.show()


def check_bert_attention_weights(
        lm_model,
        inspect_results_dir,
        num_topics=None,
):
    bert = lm_model.model.bert
    num_layers = bert.config.num_hidden_layers

    fn = os.path.join(inspect_results_dir, 'attention.txt')
    with open(fn, 'wt') as f:
        for i in range(num_layers):
            attn_weights = bert.encoder.layer[0].attention.self

            # Key
            Wk = attn_weights.key.weight.detach().cpu().numpy()
            f.write(f"layer{i}_key\n")
            f.write(str(Wk))
            f.write('\n\n')
            plt.figure(figsize=(15, 12))
            ax = plt.axes()
            heatmap(Wk)
            ax.set_title(f"layer{i}_key")
            plt.savefig(os.path.join(inspect_results_dir, f"layer{i}_key.png"))
            plt.show()

            # Query
            Wq = attn_weights.query.weight.detach().cpu().numpy()
            f.write(f"layer{i}_query\n")
            f.write(str(Wq))
            f.write('\n\n')
            plt.figure(figsize=(15, 12))
            ax = plt.axes()
            heatmap(Wq)
            ax.set_title(f"layer{i}_query")
            plt.savefig(os.path.join(inspect_results_dir, f"layer{i}_query.png"))
            plt.show()

            # Value
            Wv = attn_weights.value.weight.detach().cpu().numpy()
            f.write(f"layer{i}_value\n")
            f.write(str(Wv))
            f.write('\n\n')
            plt.figure(figsize=(15, 12))
            ax = plt.axes()
            heatmap(Wv)
            ax.set_title(f"layer{i}_value")
            plt.savefig(os.path.join(inspect_results_dir, f"layer{i}_value.png"))
            plt.show()

            # Column-wise dot product
            (m, n) = Wv.shape
            assert m == n
            Wv_column_dot_products = [
                [Wv[:, k].T @ Wv[:, j] for j in range(n)]
                for k in range(n)
            ]
            f.write(f"layer{i}_value column-wise dot product\n")
            f.write(str(Wv_column_dot_products))
            f.write('\n\n')
            plt.figure(figsize=(15, 12))
            ax = plt.axes()
            heatmap(Wv_column_dot_products)
            ax.set_title(f"layer{i}_value column-wise dot product")
            plt.savefig(os.path.join(inspect_results_dir, f"layer{i}_value_column_dot.png"))
            plt.show()

            # Wk.T @ Wq
            WkTWq = Wk.T @ Wq
            f.write(f"layer{i} Wk.T @ Wq\n")
            f.write(str(WkTWq))
            f.write('\n\n')
            plt.figure(figsize=(15, 12))
            ax = plt.axes()
            heatmap(WkTWq)
            ax.set_title(f"layer{i} Wk.T @ Wq")
            plt.savefig(os.path.join(inspect_results_dir, f"layer{i}_WkTWq.png"))
            plt.show()


if __name__ == '__main__':
    # For synthetic dataset

    # Parse args
    argp = ArgumentParser()
    argp.add_argument('config')
    args = argp.parse_args()
    args = create_args(args.config)
    # Only one sentence in a batch, so that displayed sentence length is different for different sentences
    args['training']['batch_size'] = 1
    # Do not mask any token during evaluation
    args['training']['mask_prob'] = 0.0

    dataset = dataset.Dataset(args)

    name_base = args['name']
    for experiment_index in range(args['experiment']['repeat']):
        args['name'] = name_base + str(experiment_index)

        lm_model = init_lm(args)

        if not args['reporting']['random']:
            lm_params_path = utils.get_lm_path_of_args(args)
            if not os.path.exists(lm_params_path):
                print(f"Warning: trained model does not exist: {lm_params_path}")
                continue
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Determine whether CUDA is available
            lm_model.load_state_dict(
                torch.load(lm_params_path, map_location=device),
            )

        if args['lm']['freeze_uniform_attention']:
            lm_model.model.bert.encoder.layer[0].attention.self.key.weight.data = torch.zeros(
                (args['lm']['hidden_dim'], args['lm']['hidden_dim']), device=args['device'])
            lm_model.model.bert.encoder.layer[0].attention.self.key.weight.requires_grad = False
            lm_model.model.bert.encoder.layer[0].attention.self.key.bias.data = torch.zeros(
                (args['lm']['hidden_dim'],), device=args['device'])
            lm_model.model.bert.encoder.layer[0].attention.self.key.bias.requires_grad = False

            lm_model.model.bert.encoder.layer[0].attention.self.query.weight.data = torch.zeros(
                (args['lm']['hidden_dim'], args['lm']['hidden_dim']), device=args['device'])
            lm_model.model.bert.encoder.layer[0].attention.self.query.weight.requires_grad = False
            lm_model.model.bert.encoder.layer[0].attention.self.query.bias.data = torch.zeros(
                (args['lm']['hidden_dim'],), device=args['device'])
            lm_model.model.bert.encoder.layer[0].attention.self.query.bias.requires_grad = False

        lm_model.eval()

        inspect_results_dir = os.path.join(args['reporting']['inspect_results_dir'], args['name'])
        os.makedirs(inspect_results_dir, exist_ok=True)

        # Plot attention
        num_sentences_to_plot = args['reporting']['num_sentences_to_plot']
        batches = prepare_dev_data(dataset, num_sentences_to_plot)

        plot_save_dir = os.path.join(args['reporting']['plot_attention_dir'], args['name'])
        os.makedirs(plot_save_dir, exist_ok=True)

        if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
            plot_attention_bert(lm_model, batches, plot_save_dir)
        else:
            raise NotImplementedError('Model not supported.')

        # Compute topic-wise statistics
        if args['language']['name'] == 'lda':
            max_num_docs = 500
            batches = prepare_dev_data(dataset, max_num_docs, max_sentence_len=100)
            topic_model = load_topic_model("./trained_models/topic10_word100.pkl")

            for num_topics_per_word in [1]:  # should be > 1 if different topics have overlapping words
                topic_attn_stat, num_topics_in_docs, same_token_attn_list, topic_attn_list, non_topic_attn_list, empty_topic_attn_list = \
                                                        compare_topic_and_non_topic_attention(
                                                            lm_model.model.bert,
                                                            topic_model,
                                                            num_topics_per_word,
                                                            batches,
                                                            max_num_docs,
                                                            stop_tokens={'PAD', 'MASK', 'START', 'END'},
                                                            ref_sentence_len=100,
                                                            skip_num_docs=0,
                                                        )

                with open(f"./trained_models/{args['name']}_num_topics_in_docs_top{num_topics_per_word}.pkl", 'wb') as f:
                    pickle.dump(num_topics_in_docs, f)

                save_topic_and_non_topic_attention_results(
                    num_topics_per_word,
                    inspect_results_dir,
                    topic_attn_stat,
                    f"./trained_models/{args['name']}_topic_attn_stat_top{num_topics_per_word}.pkl",
                )

        # Check weights
        if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
            check_bert_embedding_weights(lm_model, inspect_results_dir)
            check_bert_attention_weights(
                lm_model,
                inspect_results_dir,
                args['language']['num_topics'] if 'num_topics' in args['language'] else None,
            )
        else:
            raise NotImplementedError('Model not supported.')
