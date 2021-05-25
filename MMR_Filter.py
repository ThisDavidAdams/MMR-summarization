import os
import string
import gensim
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import gensim.downloader as api
import gensim.corpora as corpora
import nlp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import spacy

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
model = api.load('word2vec-google-news-300')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stop_words = stopwords.words('english')
doc2vec_model = Doc2Vec.load("./models/enwiki_dbow/doc2vec.bin")


def process_LDA_topics(document, num_of_topics=1, num_of_words=1, mallet=False, cloud=False):
    document = document.replace('\n', ' ')
    document = document.replace("'", "")
    document = document.replace('|||||', '')
    sentences = tokenizer.tokenize(document)
    new_words = []
    for sentence in sentences:
        new_words.append(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(new_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[new_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(new_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    topic_string = ""

    if mallet is False:
        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num_of_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=4,
                                                    alpha='auto',
                                                    per_word_topics=True)
        for index, topic in lda_model.show_topics(formatted=False, num_words=num_of_words):
            topic_string = topic_string + " ".join([w[0] for w in topic]) + " "

    else:
        # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip and unzip to C drive ONLY
        # Make sure your system's environment variables include a variable called MALLET_HOME that points to
        #   'C:\mallet-2.8.0', for example
        if cloud is False:
            os.environ['MALLET_HOME'] = 'C:\\mallet-2.0.8'
            mallet_path = 'C:\\mallet-2.0.8\\bin\\mallet'  # update this path
        else:
            os.environ['MALLET_HOME'] = '/content/mallet-2.0.8'
            mallet_path = '/content/mallet-2.0.8/bin/mallet'  # you should NOT need to change this
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_of_topics,
                                                     id2word=id2word, iterations=4)
        for index, topic in ldamallet.show_topics(formatted=False, num_words=num_of_words):
            topic_string = topic_string + " ".join([w[0] for w in topic]) + " "

    return topic_string


def process_LDA_topics_multi_document(document, num_of_topics=1, num_of_words=1, mallet=False, cloud=False):
    document = document.replace('\n', ' ')
    document = document.replace("'", "")
    documents = document.split('|||||')

    topic_string = ""
    for document in documents:
        if document is not None and len(document) >= 3 and document is not '':
            topic_string = topic_string + process_LDA_topics(document, num_of_topics=num_of_topics,
                                                             num_of_words=num_of_words,
                                                             mallet=mallet,
                                                             cloud=cloud) + " "

    return topic_string


def process_tfidf_similarity(document, base_document):
    vectorizer = TfidfVectorizer()
    # To make uniformed vectors, both documents need to be combined first.
    document = [document]
    document.insert(0, base_document)
    embeddings = vectorizer.fit_transform(document)
    cosine_similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
    return cosine_similarities


def preprocess(text):
    # Steps:
    # 1. lowercase
    # 2. Lammetize. (It does not stem. Try to preserve structure not to overwrap with potential acronym).
    # 3. Remove stop words.
    # 4. Remove punctuations.
    # 5. Remove character with the length size of 1.

    lowered = str.lower(text)

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(lowered)

    words = []
    for w in word_tokens:
        if w not in stop_words:
            if w not in string.punctuation:
                if len(w) > 1:
                    lemmatized = lemmatizer.lemmatize(w)
                    words.append(lemmatized)

    return words


def process_doc2vec_similarity(document, base_document):
    # Both pretrained models are publicly available at public repo of jhlau.
    # URL: https://github.com/jhlau/doc2vec

    # Only handle words that appear in the doc2vec pretrained vectors.
    # enwiki_dbow model contains 669549 vocabulary size.
    tokens = preprocess(base_document)
    tokens = list(filter(lambda x: x in doc2vec_model.wv.vocab.keys(), tokens))
    base_vector = doc2vec_model.infer_vector(tokens)

    tokens = preprocess(document)
    tokens = list(filter(lambda x: x in doc2vec_model.wv.vocab.keys(), tokens))
    vector = doc2vec_model.infer_vector(tokens)

    scores = cosine_similarity([base_vector], [vector]).flatten()[0]
    return scores


def process_word_movers_distance(document, query_document):
    document = preprocess(document)
    query_document = preprocess(query_document)
    distance = model.wmdistance(document, query_document)
    return distance


def compute_maximal_marginal_relevance(candidate_string, query, number_of_sentences=1, lambda_constant=0.5,
                                       sim=process_tfidf_similarity):
    """
    Return ranked phrases using MMR. Cosine similarity is used as similarity measure.
    :param sim: similarity function to use for MMR
    :param query: Query sentence
    :param candidate_string: list of candidate sentences
    :param lambda_constant: 0.5 to balance diversity and accuracy. if lambda_constant is high, then higher accuracy. If lambda_constant is low then high diversity.
    :param number_of_sentences: text from which to calculate the number of terms to include in result set
    :return: Ranked phrases with score
    """

    candidate_list = tokenizer.tokenize(candidate_string)
    if not candidate_list or candidate_list is None or len(candidate_list) == 0:
        return ['']

    # Find best sentence to start
    initial_best_sentence = candidate_list[0]
    prev = float("-inf")

    for sent in candidate_list[:]:
        similarity = sim(sent, query)
        if similarity > prev:
            initial_best_sentence = sent
            prev = similarity

    try:
        candidate_list.remove(initial_best_sentence)
    except ValueError:
        pass    # do nothing
    sentences_to_return = [initial_best_sentence]

    # Now find the prescribed number of best sentences
    for i in range(1, number_of_sentences):
        best_line = None
        previous_marginal_relevance = float("-inf")

        for sent in candidate_list[:]:
            # Calculate the Marginal Relevance
            left_side = lambda_constant * sim(sent, query)
            right_values = [float("-inf")]
            for selected_sentence in sentences_to_return:
                right_values.append((1 - lambda_constant) * sim(selected_sentence, query))
            right_side = max(right_values)
            current_marginal_relevance = left_side - right_side

            # Maximize Marginal Relevance
            if current_marginal_relevance > previous_marginal_relevance:
                previous_marginal_relevance = current_marginal_relevance
                best_line = sent
        sentences_to_return += [best_line]
        candidate_list.remove(best_line)

    return sentences_to_return
