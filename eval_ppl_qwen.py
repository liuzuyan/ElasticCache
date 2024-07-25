import torch
from tqdm import tqdm
import os
from torch.nn import CrossEntropyLoss
from kv_cache_qwen import ElasticCache, LocalCache, H2OCache
import json
device = "cuda"
import argparse
import torch
from qwen_generation_utils import make_context
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

    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None

    k_seq_dim = v_seq_dim = 1

    os.makedirs('logs_ppl_qwen/', exist_ok=True)

    data = data[:args.eval_samples]
            
    nlls = []

    for item in tqdm(data):
        if args.method == "elastic":
            kv_cache = ElasticCache(
                start_size=args.start_size,
                recent_size=args.recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                ratio=args.ratio,
                layer_num=32,
            )
        elif args.method == "local":
            kv_cache = LocalCache(
                start_size=args.start_size,
                recent_size=args.recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                ratio=args.ratio,
            )
        elif args.method == "h2o":
            kv_cache = H2OCache(
                start_size=args.start_size,
                recent_size=args.recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                ratio=args.ratio,
            )
        image_path = os.path.join(args.image_path, item["image"])
        question = item['question']
        answer = item['answer']
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

        for idx in range(0, answer_ids.shape[-1] - 1):

            with torch.no_grad():
                if past_key_values is None:
                    outputs = model(
                        input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True,
                    )
                    logits = outputs.logits.view(-1, model.config.vocab_size)
                    num_of_token += logits.shape[0]
                    past_key_values = outputs.past_key_values
                    attentions = outputs.attentions

                    logits = logits[-1].view(-1, model.config.vocab_size)
                    label = answer_ids[:, idx : idx + 1].to(logits.device).view(-1)
                    neg_log_likelihood = loss_fn(logits, label)
                    if kv_cache is not None:
                        past_key_values = kv_cache(past_key_values, num_of_token, attentions)
                else:
                    cur_input_ids = answer_ids[:, idx - 1: idx]
                    outputs = model(
                        cur_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True,
                        attention_mask=(cur_input_ids != 0),
                    )
                    logits = outputs.logits.view(-1, model.config.vocab_size)
                    num_of_token += logits.shape[0]
                    past_key_values = outputs.past_key_values
                    attentions = outputs.attentions

                    label = answer_ids[:, idx : idx + 1].to(logits.device).view(-1)
                    neg_log_likelihood = loss_fn(logits, label)
                    if kv_cache is not None:
                        past_key_values = kv_cache(past_key_values, num_of_token, attentions)
                        
                nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())
    with open(f"logs_ppl_qwen/{args.exp_name}.txt", "a") as f:
        f.write(f"{ppl.item()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/qwen-vl-chat")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./playground/data/mm-vet/mm-vet.json")
    parser.add_argument("--image-path", type=str, default="./playground/data/mm-vet/images/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--start-size", type=int, default=1)
    parser.add_argument("--recent-size", type=int, default=2047)
    parser.add_argument("--eval-samples", type=int, default=218)
    parser.add_argument("--exp-name", type=str, default='')
    parser.add_argument("--method", type=str, default="elastic")
    parser.add_argument("--ratio", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
