from collections import defaultdict
import numpy as np
import pandas as pd

NUM_MODELS = 10
TRAIN_TARGET_MEAN = 0.37
TEST_TARGET_MEAN = 0.16
REPEAT = 2
DUP_THRESHOLD = 0.5
NOT_DUP_THRESHOLD = 0.1
MAX_UPDATE = 0.2
DUP_UPPER_BOUND = 0.98
NOT_DUP_LOWER_BOUND = 0.01

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

print("Average Ensembling...")
df = pd.read_csv("predictions/preds0.csv")
for i in range(1, NUM_MODELS):
    df["is_duplicate"] = df["is_duplicate"] + pd.read_csv("predictions/preds" + str(i) + ".csv")["is_duplicate"]
df["is_duplicate"] /= NUM_MODELS

print("Adjusting predictions considering the different class inbalance ratio...")
a = TEST_TARGET_MEAN / TRAIN_TARGET_MEAN
b = (1 - TEST_TARGET_MEAN) / (1 - TRAIN_TARGET_MEAN)
df["is_duplicate"] = df["is_duplicate"].apply(lambda x: a*x / (a*x + b*(1 - x)))

test_label = np.array(df["is_duplicate"])

print("Updating the predictions of the pairs with common duplicates..")
for i in range(REPEAT):
    dup_neighbors = defaultdict(set)

    for dup, q1, q2 in zip(df_train["is_duplicate"], df_train["question1"], df_train["question2"]):
        if dup:
            dup_neighbors[q1].add(q2)
            dup_neighbors[q2].add(q1)

    for dup, q1, q2 in zip(test_label, df_test["question1"], df_test["question2"]):
        if dup > DUP_THRESHOLD:
            dup_neighbors[q1].add(q2)
            dup_neighbors[q2].add(q1)

    count = 0
    for index, (q1, q2) in enumerate(zip(df_test["question1"], df_test["question2"])):
        dup_neighbor_count = len(dup_neighbors[q1].intersection(dup_neighbors[q2]))
        if dup_neighbor_count > 0 and test_label[index] < DUP_UPPER_BOUND:
            update = min(MAX_UPDATE, (DUP_UPPER_BOUND - test_label[index]) / 2)
            test_label[index] += update
            count += 1

    print("Updated:", count)

print("Updating the predictions of the pairs with common non-duplicates..")
for i in range(REPEAT):
    not_dup_neighbors = defaultdict(set)

    for dup, q1, q2 in zip(df_train["is_duplicate"], df_train["question1"], df_train["question2"]):
        if not dup:
            not_dup_neighbors[q1].add(q2)
            not_dup_neighbors[q2].add(q1)

    for dup, q1, q2 in zip(test_label, df_test["question1"], df_test["question2"]):
        if dup < NOT_DUP_THRESHOLD:
            not_dup_neighbors[q1].add(q2)
            not_dup_neighbors[q2].add(q1)

    count = 0
    for index, (q1, q2) in enumerate(zip(df_test["question1"], df_test["question2"])):
        dup_neighbor_count = len(not_dup_neighbors[q1].intersection(not_dup_neighbors[q2]))
        if dup_neighbor_count > 0 and test_label[index] > NOT_DUP_LOWER_BOUND:
            update = min(MAX_UPDATE, (test_label[index] - NOT_DUP_LOWER_BOUND) / 2)
            test_label[index] -= update
            count += 1

    print("Updated:", count)

submission = pd.DataFrame({"test_id":df_test["test_id"], "is_duplicate":test_label})
submission.to_csv("predictions/submission.csv", index=False)

