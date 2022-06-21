# コーパスを作成します。

import requests
from bs4 import BeautifulSoup
import re
import csv

URL = "https://help-nandemo.com/nazokake-300sen/"

# Responseオブジェクト生成
response = requests.get(URL)
# 文字化け防止
response.encoding = response.apparent_encoding
# BeautifulSoupオブジェクト生成
soup = BeautifulSoup(response.text, "html.parser")

# webページ情報を取得し、text[]にpタグで分割して格納する。
elems = soup.select("p")
texts = []
for elem in elems:
    texts.append(elem)

# 必要な情報だけにする処理
pattern = "\d+\.【(.*?)】とかけて【(.*?)】とときます。"
pattern2 = "どちらも『(.*?)』"
nazokakes = [] # [1単語目,2単語目,答え]
text_a = ""
text_b = ""
answer = ""
for text in texts:
    new_text = str(text)
    new_text = re.sub(r'<.*?>',"",new_text)
    if len(new_text)>7:
        if re.match(pattern,new_text):
            text_a = re.match(pattern,new_text).group(1)
            text_b = re.match(pattern,new_text).group(2)
        if text_a and text_b and re.match(pattern2,new_text):
            answer = re.match(pattern2,new_text).group(1)
            answer = re.sub("（.*）","",answer)

        if text_a and text_b and answer:
            nazokakes.append([text_a,text_b,answer])
            nazokakes.append([text_b,text_a,answer])
            # 変数の初期化
            text_a = ""
            text_b = ""
            answer = ""

print("データ数: " + str(len(nazokakes)))

file = open("nazokake.csv","w")
writer = csv.writer(file)
for n in nazokakes:
    writer.writerow(n)
file.close()
