import json
import pickle
from collections import Counter, defaultdict
from tqdm import tqdm
from transformers import BertTokenizer

# Paths
TRAIN_Q_PATH = "vqa/questions/v2_OpenEnded_mscoco_train2014_questions.json"
VAL_Q_PATH   = "vqa/questions/v2_OpenEnded_mscoco_val2014_questions.json"
TRAIN_A_PATH = "vqa/annotations/v2_mscoco_train2014_annotations.json"
VAL_A_PATH   = "vqa/annotations/v2_mscoco_val2014_annotations.json"

OUTPUT_TRAIN = "output/vqa_train.pkl"
OUTPUT_VAL   = "output/vqa_val.pkl"
ANS_VOCAB    = "output/answer_vocab.pkl"

def load_questions(q_path):
    with open(q_path, "r") as f:
        q = json.load(f)["questions"]
    return q

def load_annotations(a_path):
    with open(a_path, "r") as f:
        ann = json.load(f)["annotations"]
    return ann

# Builds the Answer Vocab based on the top 1000 answers
def build_answer_vocab(train_annotations, top_k=1000):
    all_answers = []

    for ann in tqdm(train_annotations, desc="Collecting answers"):
        for ans in ann["answers"]:
            all_answers.append(ans["answer"])

    counter = Counter(all_answers)
    top_answers = [a for a, _ in counter.most_common(top_k)]

    ans_to_idx = {ans: i for i, ans in enumerate(top_answers)}
    idx_to_ans = {i: ans for ans, i in ans_to_idx.items()}

    return ans_to_idx, idx_to_ans, set(top_answers)


# Convert Q/A pairs into model-friendly format
def build_dataset(questions, annotations, answer_set, tokenizer):
    # Returns a list of dicts with:
    # 1. image_id: int
    # 2. question: tokenized_ids
    # 3, attention_mask: mask
    # 4. answer: int (answer index)
    ann_map = {a["question_id"]: a for a in annotations}
    dataset = []

    for q in tqdm(questions, desc="Building dataset"):
        qid = q["question_id"]

        if qid not in ann_map:
            continue

        ann = ann_map[qid]

        # find most common answer among the 10 GT answers
        answers = [a["answer"] for a in ann["answers"]]
        answer_counts = Counter(answers)
        final_answer = answer_counts.most_common(1)[0][0]

        # skip if answer not in top 1000
        if final_answer not in answer_set:
            continue

        encoded = tokenizer(
            q["question"],
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )

        dataset.append({
            "image_id": q["image_id"],
            "question": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "answer": final_answer
        })

    return dataset

def main():
    print("Loading questions & annotations...")

    train_q = load_questions(TRAIN_Q_PATH)
    val_q   = load_questions(VAL_Q_PATH)

    train_a = load_annotations(TRAIN_A_PATH)
    val_a   = load_annotations(VAL_A_PATH)

    print("Building answer vocabulary...")
    ans_to_idx, idx_to_ans, answer_set = build_answer_vocab(train_a)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("Processing training set...")
    train_dataset = build_dataset(train_q, train_a, answer_set, tokenizer)

    print("Processing validation set...")
    val_dataset = build_dataset(val_q, val_a, answer_set, tokenizer)

    # Replace answer strings with answer indices
    for item in train_dataset:
        item["answer"] = ans_to_idx[item["answer"]]

    for item in val_dataset:
        if item["answer"] in ans_to_idx:
            item["answer"] = ans_to_idx[item["answer"]]
        else:
            # shouldn't happen but just in case
            item["answer"] = -1

    print("Saving pickles...")
    with open(OUTPUT_TRAIN, "wb") as f:
        pickle.dump(train_dataset, f)

    with open(OUTPUT_VAL, "wb") as f:
        pickle.dump(val_dataset, f)

    with open(ANS_VOCAB, "wb") as f:
        pickle.dump({
            "ans_to_idx": ans_to_idx,
            "idx_to_ans": idx_to_ans
        }, f)

    print("Preprocessed dataset ready.")


if __name__ == "__main__":
    main()
