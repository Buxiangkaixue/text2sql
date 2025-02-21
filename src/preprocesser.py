import torch
from datasets import load_dataset
import evaluate

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline
)

import modeltrainer


# load tokenizer
tokenizer = T5Tokenizer.from_pretrained(modeltrainer.MODEL_NAME, model_max_length=modeltrainer.MAX_LENGTH)

# load datasets
train_data = load_dataset('spider', split='train')
validation_data = load_dataset('spider', split='validation')

def format_dataset(example):
    # 使用已分词的 question_toks 和 query_toks
    input_text = 'translate to SQL: ' + ' '.join(example['question_toks'])
    target_text = ' '.join(example['query_toks'])

    return {
        'input': input_text,  # 自然语言问题作为输入
        'target': target_text  # SQL 查询作为目标输出
    }

# 格式化训练集和验证集
train_data = train_data.map(format_dataset, remove_columns=train_data.column_names)
validation_data = validation_data.map(format_dataset, remove_columns=validation_data.column_names)

def convert_to_features(example_batch):
    # 将已分词的 input 和 target 转换为 input_ids 和 labels
    input_encodings = tokenizer.batch_encode_plus(example_batch['input'], padding='max_length', max_length=MAX_LENGTH, truncation=True)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target'], padding='max_length', max_length=MAX_LENGTH, truncation=True)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],  # 标签就是目标的 token IDs
        'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings

# 应用分词处理（这部分依然会根据已经分词的 input 和 target 来生成）
train_data = train_data.map(convert_to_features, batched=True, remove_columns=train_data.column_names)
validation_data = validation_data.map(convert_to_features, batched=True, remove_columns=validation_data.column_names)

columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']

train_data.set_format(type='torch', columns=columns)
validation_data.set_format(type='torch', columns=columns)

