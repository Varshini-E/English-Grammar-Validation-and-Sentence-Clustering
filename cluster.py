import sys
import collections
import nltk
import nltk.data
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")

def word_tokenizer(text):
        tokens = word_tokenize(text)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
        return tokens


def cluster_sentences(sentences, nb_of_clusters=5):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                            stop_words=stopwords.words('english'),
                                            max_df=0.9,
                                            min_df=0.1,
                                            lowercase=True)
        #builds a tf-idf matrix for the sentences
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        with open('tfidf.txt', 'w') as f:
            f.write(str(tfidf_matrix) + '\n')

        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(tfidf_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

if __name__ == "__main__":
        tok = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = list(tok.tokenize(sys.argv[1]))
        nclusters= int(sys.argv[2])

        clusters = cluster_sentences(sentences, nclusters)
        stopset = set(stopwords.words('english'))
        for cluster in range(nclusters):
                print("cluster ",cluster,":")
                cset = set()
                token_sentences = []
                for i,sentence in enumerate(clusters[cluster]):
                        sentences[sentence] = sentences[sentence][:-1]
                        if i == 0:
                                cset = set(sentences[sentence].split())
                        else:
                                cset = cset & set(sentences[sentence].split())
                        print("\tsentence ",i,": ",sentences[sentence])
                        token_sentences+=[word_tokenizer(sentences[sentence])]

                print("Keys : ", end='')
                if not cset == set():
                        print(cset.difference(stopset))

                cnt = 0
                j_score = 0.0
                for i in range(len(token_sentences)):
                    for j in range(i+1,len(token_sentences)):
                        j_score+=jaccard_similarity(token_sentences[i],token_sentences[j])
                        cnt+=1

                print("Jaccard Similarity Score: ", end='')
                print("{:.2f}".format((j_score/cnt)))
                print()

        sys.stdout.flush()
