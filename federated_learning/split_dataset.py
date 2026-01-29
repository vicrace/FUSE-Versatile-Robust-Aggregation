import random
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)       
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    else:
        alpha = 0.5     
        num_topics = 15    
       
        texts = dataset["text"]
        vectorizer = TfidfVectorizer(
            max_features=3000,     
            stop_words="english",
            ngram_range=(1, 2)
        )

        embeddings = vectorizer.fit_transform(texts)  
        kmeans = KMeans(
            n_clusters=num_topics,
            random_state=script_args.seed,
            n_init=10
        )
        topic_labels = kmeans.fit_predict(embeddings)        
        
        topic2indices = defaultdict(list)
        for idx, t in enumerate(topic_labels):
            topic2indices[t].append(idx)

        # Dirichlet non-iid split
        rng = np.random.default_rng(script_args.seed)
        client_indices = [[] for _ in range(fed_args.num_clients)]

        for t, indices in topic2indices.items():
            rng.shuffle(indices)
            proportions = rng.dirichlet(alpha * np.ones(fed_args.num_clients))
            split_points = (np.cumsum(proportions) * len(indices)).astype(int)
            splits = np.split(indices, split_points[:-1])
            for cid, split in enumerate(splits):
                client_indices[cid].extend(split.tolist())

        local_datasets = [dataset.select(client_indices[cid]) for cid in range(fed_args.num_clients)]
        
    return local_datasets

def get_dataset_this_round(dataset, round_idx, fed_args, script_args):
    
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round_idx)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    # num2sample = min(script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps, 10000)
    # num2sample = min(num2sample, len(dataset))
    # random.seed(round_idx)
    # random_idx = random.sample(range(0, len(dataset)), num2sample)
    # dataset_this_round = dataset.select(random_idx)
    
    return dataset_this_round


# def get_edataset_this_round(dataset, round_idx, fed_args, script_args):
    
#     # num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
#     # num2sample = min(num2sample, len(dataset))
#     # random.seed(round_idx)
#     # random_idx = random.sample(range(0, len(dataset)), num2sample)
#     # dataset_this_round = dataset.select(random_idx)

#     num2sample = min(script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps, 1000)
#     num2sample = min(num2sample, len(dataset))
#     random.seed(round_idx)
#     random_idx = random.sample(range(0, len(dataset)), num2sample)
#     dataset_this_round = dataset.select(random_idx)
    
#     return dataset_this_round