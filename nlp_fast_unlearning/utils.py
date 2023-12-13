import torch
from torchtext.datasets import DBpedia

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

from torch.utils.data import DataLoader
import os
import random
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dbpedia_class_dict = {
    1: "Company",
    2: "EducationalInstitution",
    3: "Artist",
    4: "Athlete",
    5: "OfficeHolder",
    6: "MeanOfTransportation",
    7: "Building",
    8: "NaturalPlace",
    9: "Village",
    10: "Animal",
    11: "Plant",
    12: "Album",
    13: "Film",
    14: "WrittenWork",
}

CLASSES = dbpedia_class_dict.keys()
n_classes = len(CLASSES)

BATCH_SIZE = 256
retain_percentage = 0.1
retain_to_forget_ratio = 5


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)


class Vocab:
    def __init__(self, train_split):
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator(
            yield_tokens(train_split, self.tokenizer), specials=["<unk>"]
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.vocab_size = len(self.vocab)

    def text_pipeline(self, text):
        return self.vocab(self.tokenizer(text))

    def label_pipeline(self, label):
        return int(label) - 1

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(self.label_pipeline(_label))
            if torch.is_tensor(_text) and _text.dtype == torch.int64:
                processed_text = _text
            else:
                processed_text = torch.tensor(
                    self.text_pipeline(_text), dtype=torch.int64
                )
            text_list.append(processed_text.to(DEVICE))
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(DEVICE), text_list.to(DEVICE), offsets.to(DEVICE)


def get_classwise_dataset(dataset):
    classwise_data = {}
    for class_id in CLASSES:
        classwise_data[class_id] = []

    for label, text in dataset:
        classwise_data[label].append((label, text))

    return classwise_data


def find_error_maximizing_noise(model, classes_to_forget, vocab_class):
    best_noise = {}

    model.train(False)
    for class_id in classes_to_forget:
        best_loss = 0
        print("Searching for error maximizing noise for class ", class_id)
        with torch.no_grad():
            for noise_token in range(vocab_class.vocab_size):
                inputs = noise_token * torch.ones((1, 100)).long()
                labels = torch.zeros(1).long() + class_id
                labels, inputs, offsets = vocab_class.collate_batch(zip(labels, inputs))
                outputs = model(inputs, offsets)
                loss = model.criterion(outputs, labels)
                if loss.item() > best_loss:
                    best_noise[class_id] = inputs
                    best_loss = loss.item()
        print(f"Got loss {best_loss} for {best_noise[class_id]}")

    return best_noise


def get_retain_forget_dl(classwise_dataset, vocab_class, classes_to_forget):
    # retain validation set
    retain_data = []
    for class_id in CLASSES:
        if class_id not in classes_to_forget:
            retain_data.extend(classwise_dataset[class_id])

    # forget validation set
    forget_data = []
    for class_id in CLASSES:
        if class_id in classes_to_forget:
            forget_data.extend(classwise_dataset[class_id])

    retain_dl = DataLoader(
        retain_data, BATCH_SIZE, shuffle=True, collate_fn=vocab_class.collate_batch
    )
    forget_dl = DataLoader(
        forget_data, BATCH_SIZE, shuffle=True, collate_fn=vocab_class.collate_batch
    )

    return retain_dl, forget_dl


def prepare_dbpedia(
    for_baseline_only: bool,
    retain_percentage=retain_percentage,
    classes_to_forget=None,
    model=None,
    retain_to_forget_ratio=retain_to_forget_ratio,
):
    ensure_deterministic()
    train_split, test_split = DBpedia()

    dbpedia_vocab = Vocab(train_split)

    train_dataset = to_map_style_dataset(train_split)
    test_dataset = to_map_style_dataset(test_split)

    num_val = int(len(train_dataset) * 0.05)
    num_train = len(train_dataset) - num_val

    train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])

    if for_baseline_only is True:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=dbpedia_vocab.collate_batch,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=dbpedia_vocab.collate_batch,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=dbpedia_vocab.collate_batch,
        )
        return (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            dbpedia_vocab.vocab_size,
        )
    elif for_baseline_only is False:
        assert classes_to_forget is not None, "Please provide a list of forget classes"
        assert model is not None, "We need the model to infer error-maximizing noise"

        best_noise = find_error_maximizing_noise(
            model, classes_to_forget, dbpedia_vocab
        )

        classwise_train = get_classwise_dataset(train_dataset)
        classwise_val = get_classwise_dataset(val_dataset)
        classwise_test = get_classwise_dataset(test_dataset)

        for class_id in CLASSES:
            print(
                f"Class {class_id} ({dbpedia_class_dict[class_id]}) "
                f"original samples: {len(classwise_train[class_id])}"
            )

        retain_samples = []
        for class_id in CLASSES:
            if class_id not in classes_to_forget:
                retain_len = int(len(classwise_train[class_id]) * retain_percentage)
                retain_samples += classwise_train[class_id][:retain_len]

        noisy_data = []
        for class_id in CLASSES:
            if class_id in classes_to_forget:
                noisy_data.append((class_id, best_noise[class_id]))

        retain_valid_dl, forget_valid_dl = get_retain_forget_dl(
            classwise_val, dbpedia_vocab, classes_to_forget
        )
        retain_test_dl, forget_test_dl = get_retain_forget_dl(
            classwise_test, dbpedia_vocab, classes_to_forget
        )

        return (
            retain_samples,
            noisy_data,
            retain_valid_dl,
            forget_valid_dl,
            retain_test_dl,
            forget_test_dl,
            dbpedia_vocab,
        )
    else:
        raise ValueError(f"{for_baseline_only} is not accepted")


def build_noisy_dl(
    retain_samples,
    noisy_data,
    vocab_class,
    retain_to_forget_ratio=retain_to_forget_ratio,
):
    ensure_deterministic()

    num_noisy_samples = int(len(retain_samples) / retain_to_forget_ratio)
    noisy_samples = noisy_data * num_noisy_samples

    noisy_loader = torch.utils.data.DataLoader(
        noisy_samples + retain_samples,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=vocab_class.collate_batch,
    )

    return noisy_loader


def ensure_deterministic():
    torch.manual_seed(42)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
