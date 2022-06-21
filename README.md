# なぞかけの自動生成
なぞかけの自動生成を試す。  
「〇〇とかけまして、△△とときます。」を入力とし、「そのこころは、～～。」を出力にしたい。  

データセットとして、[なぞかけ 厳選300選](https://help-nandemo.com/nazokake-300sen/)を参考にした。  

日本語 事前学習済みモデル : T5
https://huggingface.co/sonoisa/t5-base-japanese

**プログラム**  
make_corpus.py : なぞかけのコーパス(nazokake.csv)を作成する。カラムは[単語1,単語2,答え]。 
   
finetuning_t5.py : 事前学習済みの日本語T5モデルをファインチューニングする。ファインチューニングしたモデルは、実行後につくられるt5_content/modelに保存される。テストデータによる推論結果は、result.txtに出力。 
   
test_nazokake.py : finetuning_t5.pyを実行しモデルが保存された後、ファインチューニングしたモデルを使って、なぞかけの推論ができる。そのとき使うなぞかけは、odai.csvにある。(nazokake.csvにはないなぞかけ)
