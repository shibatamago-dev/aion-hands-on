# Microsoft Aion 1.0 Instruct ハンズオン結果

- 対象: `microsoft/Aion-1.0-Instruct-2.3B`（MIT、2.3B、32Kコンテキスト）
- 実施日: 2026-11-19
- 実施環境: Google Colab 無料枠 T4 GPU、アカウントは pochi のみ使用
- ノートブック: https://colab.research.google.com/drive/1otJq4qM_6EWNOrVc8W9aodAKUYgXTwMU?authuser=1
- 未実施: 公開・投稿・GitHub/Zennへの保存は一切していない

## 環境

- GPU: Tesla T4 15360 MiB（Driver 580.82.07 / CUDA 13.0）
- Python 3.13.16
- 実使用バージョン: transformers 5.23.0、accelerate 1.16.0、torch 2.14.1+cu130
- Colab 標準時点で transformers 5.22.1 が入っており、`pip install -U "transformers>=5.0" accelerate` で 5.23.0 に更新（エラーなし）

## モデルロード

- 公式クイックスタートどおり `trust_remote_code=True`、`dtype="auto"`、`device_map="auto"` でロード成功
- device: cuda:0、dtype: torch.bfloat16、パラメータ数 約2.289B
- GPUメモリ確保 約4.58GB
- 生成は再現性のため全て `do_sample=False`、チャットテンプレート使用

## 基本タスク結果（Aion）

### EN1 英語入門説明（入力49トークン）
正常な3文程度の LLM 入門説明を生成。再実行でも同一出力（決定的生成を確認）。

### JP1 日本語要約（入力254トークン）
図書館の郷土資料デジタル化の文章を要約。要点は正しいが、「3行で」の指示に対して出力は4文だった。
また事業費4,800万円の金額自体は落ち、補助金で「半額を賄う」のみ残った。

出力:
> 町立図書館は4月から、明治～昭和初期の写真1万2千点と地元新聞縮刷版320冊をデジタル化。資料の劣化により10年以内に閲覧できなくなるものも多い。来年3月から館内端末で無料公開予定。国の文化財保存補助金で半額を賄う。

### JP2 ビジネスメール書き換え（入力95トークン）
文面自体は丁寧で適切。ただし頼んでいない「ポイント解説」まで付け、max_new_tokens=256 に達して文末が途切れた。実務利用では出力の冗長さが弱点。

### JP3 指示追従（入力73トークン）
「猫」「雨」「傘」を全て使い、ちょうど3文で物語を生成。条件は満たした（内容の自然さはやや不自然：雨は傘を通り抜け、猫は安心する）。

### JP4 文書QA（入力188トークン）
2問とも正解（最大部門=クラウド事業 5億1,000万円、営業利益率=11.4%）。

### JP5 計算（入力94トークン）
計算過程つきで正解（120×3 + 80×5 = 760円）。ただし単純な計算に対して非常に冗長な段階分け出力だった。

## ツールコール

- `get_weather(city)` を tools として渡し「東京の今日の天気を教えてください。」と入力（入力180トークン）
- 生出力: `<|tool_call|>{"name": "get_weather", "arguments": {"city": "Tokyo"}}<|/tool_call|><|end|>`
- 有効なJSONで正しくツールコール形式を出力。ただし city は「東京」ではなく「Tokyo」だった。

## 長文（needle）テスト

フィラー段落を繰り返し、末尾近くに「合言葉は『青いイルカ42番』です。」を1回挿入して質問。

| 入力トークン | 結果 |
|---|---|
| 3,993 | 正解「青いイルカ42番」 |
| 11,993 | CUDA OOM で失敗（12.86 GiB の追加確保を試行） |
| 26,953 | CUDA OOM で失敗（64.95 GiB の追加確保を試行） |

- これは「32Kコンテキスト対応」と「T4 16GB 上で公式クイックスタート相当の素朴な Transformers 実行が 32K まで通る」は別だ、という結果。
- OOM 後もセッションは正常で、短い生成は継続できた。
- 原因の断定はしない。attention 実装（sdpa 等）指定での再試行を第2フェーズで行う。

正確なエラーテキスト（12K時）:
```
OutOfMemoryError CUDA out of memory. Tried to allocate 12.86 GiB. GPU 0 has a total capacity of 14.56 GiB of which 7.44 GiB is free. Including non-PyTorch memory, this process has 7.12 GiB memory in use. Of the allocated memory 5.71 GiB is allocated by PyTorch, and 1.28 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

## 第2フェーズ: Qwen2.5-3B-Instruct との比較

同一プロンプト・同一 max_new_tokens=256・do_sample=False、各モデルのチャットテンプレート使用。
Qwen はロード成功（bf16、GPU確保 約6.21GB、attn_implementation は sdpa）。

| 課題 | Aion 2.3B | Qwen2.5-3B |
|---|---|---|
| EN1 入門説明 | 簡潔に完結（再実行でも同一出力） | 詳しいが冗長で 256 トークンで途切れた |
| JP1 要約 | 要点は正しいが金額（4,800万円）とネット公開検討が脱落、「3行で」に対し4文 | 金額・ネット公開とも保持し、より完全。ただしこちらも1段落4文 |
| JP2 リライト | 文面は適切だが未依頼の解説が付き、途中で切れた | 文面＋短い締めで完結。署名欄の自己紹介まで補った |
| JP3 指示追従 | 「猫・雨・傘」＋ちょうど3文を達成 | 3語は使ったが2文で終了し、文数条件を外した |
| JP4 文書QA | 2問とも正解 | 2問とも正解 |
| JP5 計算 | 正解 760円（冗長な段階分け） | 正解 760円は過程内にあるが結論文が途切れた |
| ツールコール | 成功。city は "Tokyo" | 成功。city は "東京"（日本語のまま） |

総評: 基本品質は互角に近い。Aion は短く完結させる力（EN1、JP3 の文数遵守）で勝る場面があり、Qwen は情報の保持量（JP1）で勝る場面があった。どちらも冗長化して 256 トークンで切れる癖は共通。

## 第2フェーズ: 長文の sdpa 再試行

- Aion を `attn_implementation="sdpa"` で再ロード: 拒否されず成功（attn_implementation=sdpa と記録）。
- しかし 11,993 トークンの needle テストは再び OOM。追加確保要求は 12.86 GiB で、デフォルト設定時と完全に同一だった。
- つまりこの OOM は attention 実装の切り替えでは解消しない。要求量が sdpa でも変わらないことから、ボトルネックは attention のスコア行列以外（語彙数 200,029 の大きさなど）にある可能性はあるが、本検証では原因の特定まではしていない。
- expandable_segments バリアントは、sdpa が受理されたため事前に決めた分岐条件に該当せず未実施。

## 残り

- 再現スクリプト: `aion_hands_on.py`（本フォルダ）
- Zenn下書き: `zenn-draft.md`（本フォルダ、ローカルのみ・未push）
- GitHub 再現リポジトリ作成と zenn-content への push は Takai 判断待ち
