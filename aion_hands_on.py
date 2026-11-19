"""Microsoft Aion 1.0 Instruct (2.3B) hands-on reproduction script.

Run on Google Colab (free T4) or any GPU with ~16GB+ VRAM.
Requires: transformers >= 5.0, accelerate, torch

Usage:
    pip install -U "transformers>=5.0" accelerate
    python aion_hands_on.py

Notes:
- Generation uses do_sample=False for reproducibility.
- Long-context needle test: ~4k tokens succeeds on T4; ~12k tokens
  OOMs on T4 (16GB) with the naive Transformers path, even with
  attn_implementation="sdpa" (both tried to allocate an extra
  12.86 GiB). See RESULTS.md for details.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/Aion-1.0-Instruct-2.3B"

PROMPTS = {
    "EN1": "Give me a short introduction to large language models.",
    "JP1": (
        "次の文章を3行で要約してください。\n\n"
        "町立図書館は今年4月から、所蔵する郷土資料のデジタル化を始めた。対象は明治から昭和初期にかけての写真約1万2千点と、"
        "地元新聞の縮刷版320冊である。資料は劣化が進んでおり、このままでは10年以内に閲覧できなくなるものも多いという。"
        "デジタル化した資料は来年3月から館内の端末で無料公開される予定で、著作権の確認が取れたものについては"
        "インターネット公開も検討している。事業費は総額4,800万円で、その半分を国の文化財保存補助金がまかなう。"
        "館長は『地域の記憶を次の世代に残すための最後の機会だ』と話している。"
    ),
    "JP2": (
        "次の文を、取引先に送る丁寧なビジネスメールの文面に書き換えてください。\n\n"
        "この資料、明日までにざっと見といて。気になるとこあったらメモしといて。"
    ),
    "JP3": "「猫」「雨」「傘」の3つの言葉をすべて使って、ちょうど3文で短い物語を書いてください。",
    "JP4": (
        "次の文章を読んで質問に答えてください。\n\n"
        "第3四半期の売上は前年同期比12%増の8億4,000万円だった。部門別では、クラウド事業が5億1,000万円で全体の約61%を占め、"
        "次いでサポート事業が2億2,000万円、ハードウェア事業が1億1,000万円だった。営業利益は9,600万円で、利益率は11.4%である。\n\n"
        "質問1: 売上が最も大きい部門とその金額は？ 質問2: 営業利益率は何%？"
    ),
    "JP5": (
        "ある店では、りんご1個120円、みかん1個80円で売っています。"
        "りんごを3個とみかんを5個買うと合計はいくらですか。計算過程も書いてください。"
    ),
}


def generate(tokenizer, model, content, max_new_tokens=256, tools=None):
    messages = [{"role": "user", "content": content}]
    kwargs = dict(
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if tools is not None:
        kwargs["tools"] = tools
    inputs = tokenizer.apply_chat_template(messages, **kwargs).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"loaded: device={model.device}, dtype={model.dtype}")

    for name, prompt in PROMPTS.items():
        print(f"\n===== {name} =====")
        print(generate(tokenizer, model, prompt))

    # Tool calling
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    print("\n===== TOOL =====")
    print(generate(tokenizer, model, "東京の今日の天気を教えてください。",
                   max_new_tokens=128, tools=tools))

    # Long-context needle test (~4k tokens)
    filler = (
        "春の公園では桜が咲き、子どもたちが走り回っている。図書館の窓からは穏やかな光が差し込み、"
        "静かな時間が流れている。駅前の商店街では季節の野菜や果物が並び、買い物客でにぎわっている。"
    )
    needle = "合言葉は『青いイルカ42番』です。"
    context = filler * 48 + needle + filler * 2
    print("\n===== NEEDLE ~4k =====")
    print(generate(
        tokenizer, model,
        context + "\n\n合言葉は何ですか。文中の言葉で答えてください。",
        max_new_tokens=64,
    ))


if __name__ == "__main__":
    main()
