from torch import nn
from nlp_fast_unlearning.utils import dbpedia_class_dict

embed_size = 64
n_classes = len(dbpedia_class_dict)


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_size=embed_size, n_classes=n_classes):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_size, sparse=False)
        self.fc = nn.Linear(embed_size, n_classes)
        self.init_weights()
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
