import torch
from tqdm import tqdm
import os
from torch.nn import CrossEntropyLoss 
import json
device = "cuda"

import argparse
import torch
from cache_generate import generate, sample, greedy_search
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

    outputs_data_json = []
    model_name = args.model_path.split('/')[-1]
    dataset_name = args.data_path.split('/')[-1].split('.')[0]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()

    os.makedirs('logs_temp/', exist_ok=True)

    data = data[:args.eval_samples]

    for item in tqdm(data):
        image_path = os.path.join(args.image_path, item["image"])
        question = item['question']
        answer = item['answer']

        output_data_json = {}
        output_data_json['image'] = item['image']
        output_data_json['question'] = question

        if "detail_1k" in args.data_path:
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
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            use_cache=True)
    
        outputs_generate = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        output_data_json['answer'] = outputs_generate
        outputs_data_json.append(output_data_json)
    
    with open('./playground/data/' + dataset_name + '/rouge-' + model_name + '-' + dataset_name + '.json', 'w') as f:
        json.dump(outputs_data_json, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/qwen-vl-chat")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./playground/data/detail_1k/detail_1k.json")
    parser.add_argument("--image-path", type=str, default="./playground/data/detail_1k/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
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
    parser.add_argument("--method", type=str, default='elastic')
    args = parser.parse_args()
    main(args)
