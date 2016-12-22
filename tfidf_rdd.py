"""tfidf_spark.py"""


from __future__ import print_function

import os
import sys
import re
import math

from pyspark import SparkContext

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: tfidf.py <input_path> <output_path>", file=sys.stderr)
        exit(1)

    INPUT_PATH = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]

    REMOVE_PUCT = ur"\b[.,:;'\"!?-]+\B|\B[.,:;'\"!?-]+\b"

    sc = SparkContext("local", "tfidf")

    corpus = sc.wholeTextFiles(INPUT_PATH)

    numFiles = corpus.count()

    doc_word_pair = corpus.map(lambda docs: (os.path.basename(docs[0]), re.sub(REMOVE_PUCT, "", docs[1]).lower().split(" ")))\
                          .flatMapValues(lambda a: a)\
                          .map(lambda x: (x,1))\
                          .reduceByKey(lambda a,b:a+b)\
                          .cache()

    idf = doc_word_pair.keys()\
                       .map(lambda x: (x[1], 1))\
                       .reduceByKey(lambda a,b:a+b)\
                       .map(lambda x:(x[0],math.log10(float(1+numFiles)/(1+x[1]))))

    # idf result: [(u'Fame', 1.2787536009528289)]

    tf = doc_word_pair.map(lambda x: (x[0][1], (x[0][0], x[1])))

    # tf result: [(u'Fame', (u'two_gentlemen_of_verona.txt', 1))]

    ## multiply tf to idf by joining two RDD
    tfidf = tf.fullOuterJoin(idf).map(lambda x: (x, x[1][0][1] * x[1][1]))

    tfidf.saveAsTextFile(OUTPUT_PATH)

    sc.stop()