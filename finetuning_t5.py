# 日本語の学習済みT5モデルをファインチューニングします。
# https://huggingface.co/sonoisa/t5-base-japanese

import random
from tqdm import tqdm
import csv
import argparse
import glob
import logging
from itertools import chain
from string import punctuation
import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl

import os
import re

MODEl_NAME = "sonoisa/t5-base-japanese"
model_dir = "./t5_content/model"

# データセットを読み込み、train:val:test=8:1:1に分割する。
file = open('nazokake.csv','r')
reader = csv.reader(file)
datasets = []
for row in reader:
    datasets.append(row)
random.shuffle(datasets) # ジャンルごとにわかれてたからシャッフルしておく。

with open("./t5_content/data/train.csv","w",encoding="utf-8") as file_train, \
    open("./t5_content/data/val.csv","w",encoding="utf-8") as file_val, \
    open("./t5_content/data/test.csv","w",encoding="utf-8") as file_test:
    writer_train = csv.writer(file_train)
    writer_val = csv.writer(file_val)
    writer_test = csv.writer(file_test)
    for i,data in tqdm(enumerate(datasets)):
        text_a = data[0] + "とかけまして、" 
        text_b = data[1] + "とときます。"
        answer = "そのこころは、どちらも" + data[2]
        if answer[-1] != "。" and answer[-1] != "！" and answer[-1] != "？":
            answer = answer + "。"
        
        if i < len(datasets) * 0.8:
            writer_train.writerow([text_a,text_b,answer])
        elif i < len(datasets) * (0.8 + 0.1):
            writer_val.writerow([text_a,text_b,answer])
        else:
            writer_test.writerow([text_a,text_b,answer])

# GPU使う?
use_gpu = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ハイパーパラメータ
args_dict = dict(
    data_dir="./t5_content/data",  # データセットのディレクトリ
    model_name_or_path=MODEl_NAME,
    tokenizer_name_or_path=MODEl_NAME,

    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    # max_input_length=64,
    # max_target_length=512,
    # train_batch_size=8,
    # eval_batch_size=8,
    # num_train_epochs=10,

    n_gpu=1 if use_gpu else 0,
    early_stop_callback=False,
    fp_16=False,
    opt_level='01',
    max_grad_norm=1.0
)

# CSVをデータセットとして読み込む。
class CsvDataset():
    def __init__(self, tokenizer, data_dir, type_path, input_max_len=126, target_max_len=126):
        self.file_path = os.path.join(data_dir, type_path)
        
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()
  
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _make_record(self, text_a, text_b, answer):
        # なぞかけタスク用の入出力形式に変換する。
        texts = text_a + text_b
        input = f"{texts}"
        target = f"{answer}"
        return input, target
  
    def _build(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split(",")
                assert len(line) == 3
                assert len(line[0]) > 0
                assert len(line[1]) > 0
                assert len(line[2]) > 0

                text_a = line[0]
                text_b = line[1]
                answer = line[2]

                input, target = self._make_record(text_a, text_b, answer)

                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input], max_length=self.input_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target], max_length=self.target_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)

# 学習の処理
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

        # 事前学習済みモデルの読み込み
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)

        # トークナイザーの読み込み
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path, is_fast=True)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]

        # All labels set to -100 are ignored (masked), 
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.hparams.learning_rate, 
                          eps=self.hparams.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_dataset(self, tokenizer, type_path, args):
        """データセットを作成する"""
        return CsvDataset(
            tokenizer=tokenizer, 
            data_dir=args.data_dir, 
            type_path=type_path, 
            input_max_len=args.max_input_length,
            target_max_len=args.max_target_length)
    
    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                             type_path="train.csv", args=self.hparams)
            self.train_dataset = train_dataset

            val_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                           type_path="val.csv", args=self.hparams)
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
            )

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset, 
                          batch_size=self.hparams.train_batch_size, 
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset, 
                          batch_size=self.hparams.eval_batch_size, 
                          num_workers=4)

# 学習に用いるハイパーパラメータを設定する
args_dict.update({
    "max_input_length":  32,  # 入力文の最大トークン数
    "max_target_length": 64,  # 出力文の最大トークン数
    "train_batch_size":  8,
    "eval_batch_size":   8,
    "num_train_epochs":  10,
    })
args = argparse.Namespace(**args_dict)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    amp_backend=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
)

# ファインチューニング
model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

# 最終エポックのモデルを保存
model.tokenizer.save_pretrained(model_dir)
model.model.save_pretrained(model_dir)

del model

# 学習したモデルを読み込んで、テストデータで試す。
trained_model = T5ForConditionalGeneration.from_pretrained(model_dir)
trained_model.cuda()

# トークナイザ―
tokenizer = T5Tokenizer.from_pretrained(model_dir, is_fast=True)

# テストデータの読み込み
test_dataset = CsvDataset(tokenizer, args_dict["data_dir"], "test.csv", 
                          input_max_len=args.max_input_length, 
                          target_max_len=args.max_target_length)

test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

trained_model.eval()

inputs = []
outputs = []
targets = []

for batch in tqdm(test_loader):
    input_ids = batch['source_ids']
    input_mask = batch['source_mask']
    if use_gpu:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    output = trained_model.generate(input_ids=input_ids, 
        attention_mask=input_mask, 
        max_length=args.max_target_length,
        repetition_penalty=10.0,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
        )

    output_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False) 
                for ids in output]
    target_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                               clean_up_tokenization_spaces=False) 
                for ids in batch["target_ids"]]
    input_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                               clean_up_tokenization_spaces=False) 
                for ids in input_ids]

    inputs.extend(input_text)
    outputs.extend(output_text)
    targets.extend(target_text)

results = []
for output, target, input in zip(outputs, targets, inputs):
    print("input:     " + input)
    print("generated: " + output)
    print("good:    " + target)
    print()
    result = [input,output,target]
    results.append(result)

file_output = open("results.txt","w")
for r in results:
    line = "input:            " + r[0] + "\ngenerated:        " + r[1] + "\ngood:             " + r[2] + "\n\n"
    file_output.write(line)

file_output.close()
