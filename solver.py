from pyspark import SparkContext

sc = SparkContext()

# Otherwise we will get a TON of messages
sc.setLogLevel("ERROR")

next_gen = sc.parallelize([(0, 0)])  # Each successive generation
                                     # Keep track of each generation
                                     # in first slot.
                                     # (gen, pos)
data = sc.parallelize([])  # The entire tree. Initialy unexplored.
                           # [(gen, pos)]
generation = 1             # Keep track of what generation we are on


def next_moves(pos):
    """
    Not quite like the Gamesman API but illustrates the point.
    """
    if pos[1] == 10:
        return [None]
    elif pos[1] == 9:
        return [(generation, pos[1]+1)]
    return [(generation, pos[1]+1), (generation, pos[1]+2)]

while next_gen.count() > 0:
    next_gen = next_gen.flatMap(next_moves)
    next_gen = next_gen.filter(lambda x: x)
    print("next_gen: ", next_gen.collect())
    data = data.union(next_gen)
    generation += 1

print("data (completed): ", data.collect())

# Transform data from:
# [(gen, pos)] -> [(gen, [pos])]
# See: https://databricks.gitbooks.io/databricks-spark-knowledge-base/content/best_practices/prefer_reducebykey_over_groupbykey.html
data = data.groupByKey()
print("data (reformated): ", data.collect())
