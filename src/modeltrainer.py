from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline
)
import evaluate
import preprocesser


# define constants
MODEL_NAME = 't5-base'
MAX_LENGTH = 64
BATCH_SIZE = 64
NUM_EPOCHS = 5

#load model
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)

# number of trainable parameters
print(model.num_parameters(only_trainable=True)/1e6)

training_args = Seq2SeqTrainingArguments(
    output_dir='./results',  # 模型训练结果保存路径
    logging_dir='./logs',  # 日志保存路径
    per_device_train_batch_size=BATCH_SIZE,  # 每个设备的训练 batch 大小
    num_train_epochs=NUM_EPOCHS,  # 训练的 epoch 数量
    per_device_eval_batch_size=BATCH_SIZE,  # 每个设备的评估 batch 大小
    predict_with_generate=True,  # 在评估阶段是否生成预测结果（即模型的生成任务）
    evaluation_strategy="epoch",  # 评估策略，"epoch"表示每个epoch结束时进行一次评估
    do_train=True,  # 是否进行训练
    do_eval=True,  # 是否进行评估
    logging_steps=5,  # 每多少个训练步骤记录一次日志
    save_strategy="epoch",  # 保存策略，"epoch"表示每个epoch结束时保存一次模型
    overwrite_output_dir=True,  # 如果输出目录存在，是否覆盖它
    save_total_limit=3,  # 最多保存的模型数量，超过后最旧的模型会被删除
    load_best_model_at_end=True,  # 在训练结束时，加载验证集上表现最好的模型
    push_to_hub=False,  # 是否将模型推送到 Hugging Face Hub
    report_to="mlflow",  # 将训练日志报告到mlflow
)

# 使用 evaluate.load 加载 ROUGE 指标
rouge = evaluate.load("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = preprocesser.batch_decode(pred_ids, skip_special_tokens=True)  # 解码模型生成的 token
    labels_ids[labels_ids == -100] = preprocesser.pad_token_id  # 处理 padding 标签
    label_str = preprocesser.batch_decode(labels_ids, skip_special_tokens=True)  # 解码真实标签

    # 计算 ROUGE-2 指标
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])

    # rouge_output 是一个包含 ROUGE-2 结果的字典，直接获取它
    rouge2_fmeasure = round(rouge_output["rouge2"], 4)  # 这里获取的是 ROUGE-2 的 F1 score

    return {
        "rouge2_fmeasure": rouge2_fmeasure,
    }

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=preprocesser.train_data,
    eval_dataset=preprocesser.validation_data,
)