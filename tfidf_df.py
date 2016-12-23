"""tfidf_df.py"""

from __future__ import print_function

import os
import sys
import re
import math

from pyspark import SparkContext
from pyspark.sql import HiveContext, DataFrameWriter
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: tfidf.py <input_path> <output_path>", file=sys.stderr)
        exit(1)

    INPUT_PATH = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]

    REMOVE_PUCT = ur"\b[.,:;'\"!?-]+\B|\B[.,:;'\"!?-]+\b"

    sc = SparkContext("local", "tfidf")

    sql = HiveContext(sc)

    corpusRDD = sc.wholeTextFiles(INPUT_PATH)

    numFiles = sc.broadcast(corpusRDD.count())

    doc_wordarrayRDD = corpusRDD.map(lambda docs: (os.path.basename(docs[0]), re.sub(REMOVE_PUCT, "", docs[1]).lower().split(" ")))

    schema = StructType([
        StructField("doc_name", StringType(), False),
        StructField("doc_content", ArrayType(StringType()), False)
    ])

    doc_wordarrayDF = sql.createDataFrame(doc_wordarrayRDD, schema)

    tfidf = doc_wordarrayDF.select(doc_wordarrayDF.doc_name, explode(doc_wordarrayDF.doc_content).alias("word") )\
                                     .groupBy("doc_name", "word").agg(count("*").alias("tf"))\
                                     .select("*", count("doc_name").over(Window.partitionBy("word")).alias("df"))\
                                     .select("*", (log10(float(1+numFiles.value) / (1 + col("df")))).alias("idf"))\
                                     .select("*", (col("tf") * col("idf")).alias("tfidf") )
  
    ###################################################################################################################

    tfidf.write.format("parquet").mode("error").save(OUTPUT_PATH);

    sc.stop()

    
