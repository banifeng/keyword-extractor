import json
import logging

from allennlp.data import Tokenizer
from overrides import overrides
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from typing import Dict, List
import re
logger = logging.getLogger(__name__)

@DatasetReader.register("wordpiece_bert_crf_reader")
class WordPieceCrfReader(DatasetReader):
    def __init__(self,
                 tokenizers: Dict[str, Tokenizer]=None,
                 token_indexers: Dict[str, TokenIndexer]=None,
                 lazy: bool = False,
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizers.get("bert_tokenizer")
        self._token_indexers = token_indexers

    @overrides
    def _read(self, file_path: str):
        logger.info("Reading file at %s", file_path)
        with open(file_path) as data_file:
            for line in data_file:
                line = line.rstrip("\n")
                if not line:
                    continue
                line_json = json.loads(line)
                text = line_json['text']
                labels = line_json['labels']
                instance = self.text_to_instance(text, labels)
                yield instance

    @overrides
    def text_to_instance(self,
                         text: str,
                         labels: str = None
                         ) -> Instance:
        passage_tokens = self._tokenizer.tokenize(text)
        # 设置第一个位置为[CLS]
        passage_tokens.insert(0, Token("[CLS]"))
        metadata = {
            "text": text,
            "labels": labels,
        }
        fields = {
            "text_tokens": TextField(passage_tokens, self._token_indexers),
            "metadata": MetadataField(metadata)
        }
        if labels is not None:
            labels = labels.split()
            labels.insert(0, "O")
            assert len(labels) == len(passage_tokens)
            fields["labels"] = SequenceLabelField(labels, TextField(passage_tokens, self._token_indexers))
        return Instance(fields)
