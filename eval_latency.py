import torch
from tqdm import tqdm
import os
from torch.nn import CrossEntropyLoss
from kv_cache import ElasticCache, LocalCache, H2OCache
import json
device = "cuda"
import time
import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def main(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    past_key_values = None

    k_seq_dim = v_seq_dim = 2
    os.makedirs('logs/', exist_ok=True)

    if args.method == "elastic":
        kv_cache = ElasticCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=args.ratio,
            layer_num=(32 if "7b" in model_name else 40) * args.batch_size
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

    input_ids = torch.ones([1, 900], dtype=int).cuda()
    answer_ids = torch.ones([1, 512], dtype=int).cuda()
    past_key_values = None
    try:
        kv_cache.score_sum = torch.zeros_like(kv_cache.score_sum).cuda()
        kv_cache.flag = True
    except:
        print('cannot reset kv_cache')
        pass
    num_of_token = 0
    start_time = time.time()
    print("start here")
    for idx in range(0, answer_ids.shape[-1] - 1):
        with torch.no_grad():
            if past_key_values is None:
                time2 = time.time()
                outputs = model(
                    input_ids.repeat(args.batch_size, 1),
                    images=None,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )
                logits = outputs.logits.view(args.batch_size, -1, model.config.vocab_size)
                num_of_token += logits.shape[1]
                past_key_values = outputs.past_key_values
                attentions = outputs.attentions
            
                if kv_cache is not None:
                    past_key_values = kv_cache(past_key_values, num_of_token, attentions)
                time3 = time.time()
                print('time: ', time3 - time2)
            else:

                cur_input_ids = answer_ids[:, idx - 1: idx]
                outputs = model(
                    cur_input_ids.repeat(args.batch_size, 1),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )
                logits = outputs.logits.view(args.batch_size, -1, model.config.vocab_size)
                num_of_token += logits.shape[1]
                past_key_values = outputs.past_key_values
                attentions = outputs.attentions

                if kv_cache is not None:
                    past_key_values = kv_cache(past_key_values, num_of_token, attentions)
    end_time = time.time()
    print('time: ', end_time - start_time)
                            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/stu6/models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
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
    parser.add_argument("--exp-name", type=str, default='')
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--method", type=str, default="elastic")
    parser.add_argument("--ratio", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
