# Aion 1.0 Instruct ハンズオン

Microsoft の Windows オンデバイス向け SLM「Aion 1.0 Instruct」（2.3B、32K コンテキスト、MIT）の重み公開を受けて、Google Colab の無料 T4 で実際に動かした検証の再現リポジトリです。

- モデル: [microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B)
- 対応する Zenn 記事は下書きとして準備中です（公開後にリンクを追加します）

## 検証内容

- 公式クイックスタートでのロード（transformers 5.x、`trust_remote_code=True`）
- 日本語タスク 5 本（要約、ビジネスメール書き換え、指示追従、文書 QA、計算）と英語 1 本
- Qwen2.5-3B-Instruct との同一プロンプト比較
- ツールコール（`get_weather`）の形式確認
- 長文 needle テスト（約 4K / 12K / 27K トークン）

## 主な結果

- 基本タスクとツールコールは実用圏。ツールコールの引数 city は「Tokyo」表記になった
- 長文は約 4K トークンでは正解したが、約 12K トークンで CUDA OOM（T4 16GB）。`attn_implementation="sdpa"` でも追加確保要求は同じ 12.86 GiB で解消しなかった
- 詳細と正確なエラーテキストは [RESULTS.md](RESULTS.md) にあります

## ファイル

- `aion_hands_on.ipynb` — Colab で実行したノートブック（出力つき）
- `aion_hands_on.py` — 同内容の再現スクリプト
- `RESULTS.md` — 検証結果のまとめ

## 実行方法

```bash
pip install -U "transformers>=5.0" accelerate
python aion_hands_on.py
```

GPU メモリ 16GB 級を想定しています。生成は再現性のため `do_sample=False` で統一しています。
