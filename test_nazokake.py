import os
import csv
import argparse
import torch
import textwrap
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_dir = './t5_content/model'
# トークナイザー（SentencePiece）
tokenizer = T5Tokenizer.from_pretrained(model_dir, is_fast=True)

# 学習済みモデル
trained_model = T5ForConditionalGeneration.from_pretrained(model_dir)

# GPUの利用有無
use_gpu = False
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ハイパーパラメータ
args_dict = dict(
    data_dir="./t5_content/data",  # データセットのディレクトリ
    model_name_or_path=model_dir,
    tokenizer_name_or_path=model_dir,

    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    max_input_length=32,
    max_target_length=64,

    n_gpu=1 if use_gpu else 0,
    early_stop_callback=False,
    fp_16=False,
    opt_level='01',
    max_grad_norm=1.0
)
args = argparse.Namespace(**args_dict)

MAX_SOURCE_LENGTH = args.max_input_length   # 入力される記事本文の最大トークン数
MAX_TARGET_LENGTH = args.max_target_length  # 生成されるタイトルの最大トークン数

# 推論モード設定
trained_model.eval()

# 前処理とトークナイズを行う
inputs = []
labels = []
file = open("odai.csv","r")
reader = csv.reader(file)
for row in reader:
    inputs.append(row[0])
    labels.append(row[1])
batch = tokenizer.batch_encode_plus(
    inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, 
    padding="longest", return_tensors="pt")

input_ids = batch['input_ids']
input_mask = batch['attention_mask']
if use_gpu:
    input_ids = input_ids.cuda()
    input_mask = input_mask.cuda()

# 生成処理を行う
outputs = trained_model.generate(
    input_ids=input_ids, attention_mask=input_mask, 
    max_length=MAX_TARGET_LENGTH,
    # temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
    # num_beams=10,  # ビームサーチの探索幅
    # diversity_penalty=1.0,  # 生成結果の多様性を生み出すためのペナルティパラメータ
    # num_beam_groups=10,  # ビームサーチのグループ
    # num_return_sequences=10,  # 生成する文の数
    repetition_penalty=8.0,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
)

# 生成されたトークン列を文字列に変換する
generated = [tokenizer.decode(ids, skip_special_tokens=True, 
                                     clean_up_tokenization_spaces=False) 
                    for ids in outputs]

# 生成された文章を表示する
for i, answer in enumerate(generated):
    print("input_txt: " + inputs[i])
    print("generated: " + answer)
    print("example_A: " + labels[i])
    print()