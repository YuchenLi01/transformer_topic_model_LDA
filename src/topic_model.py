from collections import defaultdict, Counter
import logging
import gensim
import numpy as np
import pickle


def run_lda_wiki():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    id2word = gensim.corpora.Dictionary.load_from_text('wiki_en/_wordids.txt.bz2')
    mm = gensim.corpora.MmCorpus('wiki_en/_tfidf.mm')
    lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, passes=1)
    print('Finished running lda.')
    with open('wiki_en/lda.pkl', 'wb') as f:
        pickle.dump(lda, f)
    print('Saved lda model at wiki_en/lda.pkl.')


def load_topic_model(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        topic_model = pickle.load(f)
    if type(topic_model) is gensim.models.ldamodel.LdaModel:
        topic_model.word2id = {
            topic_model.id2word[word_id]: word_id
            for word_id in topic_model.id2word.keys()
        }
    else:
        assert type(topic_model) is dict, 'topic_model should be a gensim.models.ldamodel.LdaModel or dict'
    return topic_model


def get_top_term_topics(
        topic_model,
        word,  # the string, not the id
        num_topics=None,  # None means keep all topics for the word; can also be int
        filter_ambiguous_words_threshold=0.0,
):
    if type(topic_model) is gensim.models.ldamodel.LdaModel:
        if hasattr(topic_model, 'top_term_topics'):
            return topic_model.top_term_topics[word]
        if word not in topic_model.word2id:
            return []
        topics = sorted(
                topic_model.get_term_topics(topic_model.word2id[word], minimum_probability=0.0),
                key = lambda topic_with_prob: topic_with_prob[1],
                reverse = True,
            )
        if not topics:
            return []
        sum_probs = sum([topic_with_prob[1] for topic_with_prob in topics])
        topics = [
            (topic_with_prob[0], topic_with_prob[1] / sum_probs)
            for topic_with_prob in topics
        ]
        return [
                   topic_with_prob[0] for topic_with_prob in topics
                   if topic_with_prob[1] >= filter_ambiguous_words_threshold
               ][:num_topics]

    if type(topic_model) is dict:
        if filter_ambiguous_words_threshold > 0.0:
            raise NotImplementedError('when topic_model is a dict, currently do not need to filter ambiguous words')
        if word not in topic_model:
            return []
        if type(topic_model[word]) is list:
            return topic_model[word][:num_topics]
        assert type(topic_model[word]) is int, 'topic_model[word] should be int or list<int>'
        return [topic_model[word]]

    raise NotImplementedError('topic_model should be a gensim.models.ldamodel.LdaModel or dict')


def filter_word_distribution_in_topics(
        topic_model,
        top_fraction_to_keep,
        stop_tokens=None,
):
    if type(topic_model) is gensim.models.ldamodel.LdaModel:
        topic_model.top_term_topics = defaultdict(list)
        word_probs_given_topics = topic_model.get_topics()  # shape (num_topics, vocabulary_size)
        for topic in range(topic_model.num_topics):
            word_probs = word_probs_given_topics[topic]
            for i in range(len(word_probs)):
                if stop_tokens is not None and topic_model.id2word[i] in stop_tokens:
                    word_probs[i] = 0.0
            word_probs = dict(zip(range(len(word_probs)), word_probs))
            print(f"topic {topic} has {len(word_probs)} words")
            top_words = sorted(
                word_probs.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:int(len(word_probs) * top_fraction_to_keep) + 1]
            print(f"\t top 10 words are {[topic_model.id2word[word[0]] for word in top_words[:20]]}")
            top_words = dict(top_words)
            for word in top_words:
                topic_model.top_term_topics[topic_model.id2word[word]].append(topic)
        return

    print(f"type(topic_model) == {type(topic_model)}")
    print(f"topic_model == {topic_model}")
    raise NotImplementedError('topic_model should be a gensim.models.ldamodel.LdaModel')
