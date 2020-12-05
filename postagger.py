
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import sys
import warnings
warnings.filterwarnings("ignore")
if __name__ == "__main__":
    contents = sys.argv[1]
    tokenized = sent_tokenize(contents)
    test_list=[]
    for i in tokenized:

        # Word tokenizers is used to find the words
        # and punctuation in a string
        wordsList = nltk.word_tokenize(i)

        tagged = nltk.pos_tag(wordsList)
        with open('tokens.txt', 'a') as f:
            f.write(str(wordsList) + '\n' + str(tagged) + '\n')

        test_list+=tagged
    res = [lis[1] for lis in test_list]
    lalrinput = ' '.join([str(elem[0]) for elem in res])
    print(lalrinput)
    