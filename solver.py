from pyspark import SparkContext

sc = SparkContext()

# Otherwise we will get a TON of messages
sc.setLogLevel("ERROR")

generation = 0             # Keep track of what generation we are on
generations = [] # RDD's generated going down the tree.
solved_generations = [] # RDD's generated going up the tree.

DWULT = {"L": "L", "W": "W", "T": "T", "D": "D", "U": "U"}

### A simple game, for demonstration. Not actually a tiered game. ###
def primitive(pos):
    return DWULT["L"] if pos == 10 else DWULT["U"]

def gen_moves(pos):
    return [move for move in [1, 2] if pos + move <= 10]

def do_move(pos, move):
    return pos + move

def initial_pos():
    return 0
##################

# TODO: tie/draw logic
# TODO: remoteness
# TODO: writing RDD's to files

def next_moves(state):
    """
    Used going down the tree to generate all states.
    Generate the children for the current generation of this parent state from
    the previous generation.
    state = (pos, (generation, [parent positions], DWULT value))
    """
    pos = state[0]
    # if primitive(pos) != DWULT["U"]:
    #     return [None]
    children = []
    for move in gen_moves(pos):
        new_pos = do_move(pos, move)
        game_val = primitive(new_pos)
        children.append((new_pos, (generation, [pos], game_val)))
    return children

def aggregate_parents(data_a, data_b):
    """
    Used going down the tree to generate all states.
    For a child position of the current generation, combine into a list
    all parent positions of the previous generation that generated it.
    """
    generation, a_parents, game_val = data_a
    b_parents = data_b[1]
    return (generation, a_parents + b_parents, game_val)

def flatmap_parents(state):
    """
    Used going up the tree to solve all states.
    For each parent of the given, solved child state, generate an intermediate state
    for the parent with the child's game value translated to what it means for the
    parent (see child_to_parent_game_value).
    """
    generation, parents, game_val = state[1]
    return [(parent, (generation - 1, [], child_to_parent_game_value(game_val))) for parent in parents]

def child_to_parent_game_value(child_game_val):
    """
    Used going up the tree to solve all states.
    To solve a parent's game value, we need to know what each solved child's
    game value means for it. For example, if a child's game value is a loss, that
    child represents a win for the parent.
    """
    parent_game_val = DWULT["L"]
    if child_game_val == DWULT["L"]:
        parent_game_val = DWULT["W"]
    # TODO: tie/draw logic
    return parent_game_val

def reduce_by_game_value(data_a, data_b):
    """
    Used going up the tree to solve all states.
    Reduce all of a parent's child game values to solve for the parent's game
    value.
    """
    game_val_a = data_a[2]
    game_val_b = data_b[2]
    generation = data_a[0]
    # TODO: tie/draw logic
    if game_val_a == DWULT["W"] or game_val_b == DWULT["W"]:
        # If there is at least one winning move for the parent, its game value
        # is a win
        return (generation, [], DWULT["W"])
    if game_val_a == DWULT["L"] and game_val_b == DWULT["L"]:
        # If all child states represent a loss for the parent, its game value
        # is a loss
        return (generation, [], DWULT["L"])

def merge_data(data_a, data_b):
    """
    Used going up the tree to solve all states.
    Merge intermediate states for a tier/generation that have the solved game
    values for the tier's positions with the originally generated unsolved states
    from going down the tree.
    (pos, (gen, [], solved_value)) + (pos, (gen, [pos's parents], unsolved))
    = (pos, (gen, [pos's parents], solved_value))
    """
    game_val = data_a[2] if data_a[2] != DWULT["U"] else data_b[2]
    return (data_a[0], data_a[1] + data_b[1], game_val)

init_pos = initial_pos()
init_val = primitive(init_pos)
next_gen = sc.parallelize([(initial_pos(), (0, [], init_val))])

### Go down the tree, generating all (potentially unsolved) states ###
while next_gen.count() > 0:
    print("next_gen: ", next_gen.collect()) # for demonstration only
    generations.append(next_gen)
    generation += 1
    next_gen = next_gen.flatMap(next_moves).reduceByKey(aggregate_parents)

### Go up the tree, solving all states ###
intermed_RDD = None
num_gen = len(generations)
while num_gen > 0:
    gen_RDD = generations[num_gen - 1]
    if intermed_RDD is not None:
        gen_RDD = gen_RDD.union(intermed_RDD).reduceByKey(merge_data)
    print("solved_gen: ", gen_RDD.collect()) # for demonstration only
    solved_generations.insert(0, gen_RDD)
    intermed_RDD = gen_RDD.flatMap(flatmap_parents).reduceByKey(reduce_by_game_value)
    num_gen -= 1
