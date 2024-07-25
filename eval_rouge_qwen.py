import torch
from tqdm import tqdm
import os
from torch.nn import CrossEntropyLoss 
from kv_cache_qwen import ElasticCache, LocalCache, H2OCache
import json
device = "cuda"

import argparse
import torch

from cache_generate_qwen import generate, sample, greedy_search
import types

from qwen_generation_utils import make_context
from rouge import Rouge
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def main(args):


    with open(args.data_path, "r") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()

    parent_class = model.__class__.__bases__[0]
    model.generate = types.MethodType(generate, model)
    model.sample = types.MethodType(sample, model)
    model.greedy_search = types.MethodType(greedy_search, model)

    k_seq_dim = v_seq_dim = 1

    os.makedirs('logs_detail/', exist_ok=True)

    data = data[:args.eval_samples]
    score_all = []
    for item in tqdm(data):
        if args.method == "elastic":
            kv_cache = ElasticCache(
                start_size=args.start_size,
                recent_size=args.recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                ratio=args.ratio,
                layer_num=32
            )
        elif args.method == "local":
            kv_cache = LocalCache(
                start_size=args.start_size,
                recent_size=args.recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                ratio=args.ratio
            )
        elif args.method == "h2o":
            kv_cache = H2OCache(
                start_size=args.start_size,
                recent_size=args.recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                ratio=args.ratio
            )

        image_path = os.path.join(args.image_path, item["image"])
        question = item['question']
        answer = item['answer']

        question = question.replace('<image>', '')

        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': question}
        ])

        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=None,
            system="You are a helpful assistant.",
            max_window_size=None,
            chat_format='chatml',
        )

        input_ids = torch.tensor([context_tokens]).cuda()
        answer_ids = tokenizer.encode(answer, return_tensors='pt').cuda()[:, 1:]
        past_key_values = None

        num_of_token = 0
        output_ids = model.generate(
            input_ids,
            do_sample=True if (args.temperature > 0 and args.ratio == 0) else False,
            temperature=args.temperature if args.ratio == 0 else 0,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            use_cache=True,
            kv_cache_criteria=kv_cache,
            attention_mask=None)
    
        outputs_generate = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        rouge = Rouge()
        scores = rouge.get_scores(outputs_generate, answer)
        score_all.append(scores[0]['rouge-l']['f'])
    
    rouge = sum(score_all) / len(score_all)
    with open(f"logs_rouge_qwen/{args.exp_name}.txt", "a") as f:
        f.write(f"{rouge}\n")
        f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/qwen_vl_chat/")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./playground/data/mm-vet/rouge-qwen-detail.json")
    parser.add_argument("--image-path", type=str, default="./playground/data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--start-size", type=int, default=1)
    parser.add_argument("--recent-size", type=int, default=2047)
    parser.add_argument("--eval-samples", type=int, default=218)
    parser.add_argument("--exp-name", type=str, default='')
    parser.add_argument("--method", type=str, default="elastic")
    parser.add_argument("--ratio", type=float, default=0.0)
    args = parser.parse_args()
    main(args)
