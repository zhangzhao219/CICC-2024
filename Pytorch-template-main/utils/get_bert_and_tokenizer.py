import sys
sys.path.append("..")
import json

from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
)

# from peft import (
#     get_peft_model,
#     TaskType,
#     PrefixTuningConfig,
# )

AutoList = ["ernie-3___0-xbase-zh"]

def getBert(logger, bert_name, custom_config=None):
    logger.info(f"load {bert_name}")
    model_config = AutoConfig.from_pretrained(bert_name)
    model_config.output_hidden_states = True
    bert = AutoModel.from_pretrained(bert_name, config=model_config)
    return bert


def getTokenizer(logger, bert_name):
    logger.info(f"load {bert_name} tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    return tokenizer
