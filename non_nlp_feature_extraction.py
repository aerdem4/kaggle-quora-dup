from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx

NB_CORES = 10
FREQ_UPPER_BOUND = 100
NEIGHBOR_UPPER_BOUND = 5


def create_question_hash(train_df, test_df):
    train_qs = np.dstack([train_df["question1"], train_df["question2"]]).flatten()
    test_qs = np.dstack([test_df["question1"], test_df["question2"]]).flatten()
    all_qs = np.append(train_qs, test_qs)
    all_qs = pd.DataFrame(all_qs)[0].drop_duplicates()
    all_qs.reset_index(inplace=True, drop=True)
    question_dict = pd.Series(all_qs.index.values, index=all_qs.values).to_dict()
    return question_dict


def get_hash(df, hash_dict):
    df["qid1"] = df["question1"].map(hash_dict)
    df["qid2"] = df["question2"].map(hash_dict)
    return df.drop(["question1", "question2"], axis=1)


def get_kcore_dict(df):
    g = nx.Graph()
    g.add_nodes_from(df.qid1)
    edges = list(df[["qid1", "qid2"]].to_records(index=False))
    g.add_edges_from(edges)
    g.remove_edges_from(g.selfloop_edges())

    df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])
    df_output["kcore"] = 0
    for k in range(2, NB_CORES + 1):
        ck = nx.k_core(g, k=k).nodes()
        print("kcore", k)
        df_output.ix[df_output.qid.isin(ck), "kcore"] = k

    return df_output.to_dict()["kcore"]


def get_kcore_features(df, kcore_dict):
    df["kcore1"] = df["qid1"].apply(lambda x: kcore_dict[x])
    df["kcore2"] = df["qid2"].apply(lambda x: kcore_dict[x])
    return df


def convert_to_minmax(df, col):
    sorted_features = np.sort(np.vstack([df[col + "1"], df[col + "2"]]).T)
    df["min_" + col] = sorted_features[:, 0]
    df["max_" + col] = sorted_features[:, 1]
    return df.drop([col + "1", col + "2"], axis=1)


def get_neighbors(train_df, test_df):
    neighbors = defaultdict(set)
    for df in [train_df, test_df]:
        for q1, q2 in zip(df["qid1"], df["qid2"]):
            neighbors[q1].add(q2)
            neighbors[q2].add(q1)
    return neighbors


def get_neighbor_features(df, neighbors):
    common_nc = df.apply(lambda x: len(neighbors[x.qid1].intersection(neighbors[x.qid2])), axis=1)
    min_nc = df.apply(lambda x: min(len(neighbors[x.qid1]), len(neighbors[x.qid2])), axis=1)
    df["common_neighbor_ratio"] = common_nc / min_nc
    df["common_neighbor_count"] = common_nc.apply(lambda x: min(x, NEIGHBOR_UPPER_BOUND))
    return df


def get_freq_features(df, frequency_map):
    df["freq1"] = df["qid1"].map(lambda x: min(frequency_map[x], FREQ_UPPER_BOUND))
    df["freq2"] = df["qid2"].map(lambda x: min(frequency_map[x], FREQ_UPPER_BOUND))
    return df

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("Hashing the questions...")
question_dict = create_question_hash(train_df, test_df)
train_df = get_hash(train_df, question_dict)
test_df = get_hash(test_df, question_dict)
print("Number of unique questions:", len(question_dict))

print("Calculating kcore features...")
all_df = pd.concat([train_df, test_df])
kcore_dict = get_kcore_dict(all_df)
train_df = get_kcore_features(train_df, kcore_dict)
test_df = get_kcore_features(test_df, kcore_dict)
train_df = convert_to_minmax(train_df, "kcore")
test_df = convert_to_minmax(test_df, "kcore")

print("Calculating common neighbor features...")
neighbors = get_neighbors(train_df, test_df)
train_df = get_neighbor_features(train_df, neighbors)
test_df = get_neighbor_features(test_df, neighbors)

print("Calculating frequency features...")
frequency_map = dict(zip(*np.unique(np.vstack((all_df["qid1"], all_df["qid2"])), return_counts=True)))
train_df = get_freq_features(train_df, frequency_map)
test_df = get_freq_features(test_df, frequency_map)
train_df = convert_to_minmax(train_df, "freq")
test_df = convert_to_minmax(test_df, "freq")

cols = ["min_kcore", "max_kcore", "common_neighbor_count", "common_neighbor_ratio", "min_freq", "max_freq"]
train_df.loc[:, cols].to_csv("data/non_nlp_features_train.csv", index=False)
test_df.loc[:, cols].to_csv("data/non_nlp_features_test.csv", index=False)