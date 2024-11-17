from collections import defaultdict
from data import EvalDataset
from torch.utils.data import DataLoader
import pickle


def process_correlations(df_correlations, config):
    # Initialize required variables
    train_pairs = []
    topic2content = {}
    content2topic = defaultdict(set)
    eval_topics = []
    train_topics = []
    eval_contents = set()
    train_contents = set()

    # Process each row in the DataFrame
    for i in range(len(df_correlations)):
        row = df_correlations.iloc[i]
        t = row["topic_id"]
        cs = row["content_ids"].split(" ")
        fold = row["fold"]
        topic2content[t] = set(cs)

        # Separate training and evaluation topics
        if fold != config.fold:
            train_topics.append(t)
        else:
            eval_topics.append(t)

        # Separate training and evaluation contents
        for c in cs:
            if fold != config.fold:
                train_pairs.append((t, c))
                train_contents.add(c)
            else:
                eval_contents.add(c)

            content2topic[c].add(t)

    # Convert sets to lists and calculate overlap
    content_overlap = (len(eval_contents.intersection(train_contents)) / len(eval_contents)) * 100
    train_contents = sorted(train_contents) # need list for eval dataset
    eval_contents = sorted(eval_contents)

    # Print information
    print(f"training pairs: {len(train_pairs)}")
    print(f"unique topics: {len(topic2content)}")
    print(f"unique contents: {len(content2topic)}")
    print(f"training topics: {len(train_topics)}")
    print(f"evaluation topics: {len(eval_topics)}")
    print(f"training contents: {len(train_contents)}")
    print(f"evaluation contents: {len(eval_contents)}")
    print(f"Content overlap between training and evaluation: {content_overlap:.2f}%")
    
    return train_pairs, topic2content, content2topic, train_topics, eval_topics, train_contents, eval_contents


def prepare_eval_loaders(config, tokenizer, topics, contents, topic2text, content2text):
    dataset_topic = EvalDataset(topics, topic2text, tokenizer, config.max_len)
    loader_topic = DataLoader(
        dataset=dataset_topic, 
        batch_size=config.eval_batch_size, 
        shuffle=False,
        pin_memory=True,
        collate_fn=dataset_topic.collater
    )
    dataset_content = EvalDataset(contents, content2text, tokenizer, config.max_len)
    loader_content = DataLoader(
        dataset=dataset_content, 
        batch_size=config.eval_batch_size, 
        shuffle=False,
        pin_memory=True,
        collate_fn=dataset_content.collater
    )

    print(f"\tTopics (examples, batches): ({len(dataset_topic)}, {len(loader_topic)})")
    print(f"\tContents (examples, batches): ({len(dataset_content)}, {len(loader_content)})")

    return loader_topic, loader_content


def read_pkl(fp):
    with open(fp, "rb") as f:
        data = pickle.load(f)
    return data