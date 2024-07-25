# Evaluation

In Elastic Cache, we evaluate our method on three models(llava 7b/13b and qwen-vl-chat), two datasets(llava_detail_1k and MM-Vet). We use two metrics: PPL and ROUGE. Result will be saved at ./logs\_\<METRIC\>\_\<MODEl\>/

## Llava-7b

### MM-Vet

To eval PPL, run the following scripts. Method should be elastic/h2o/local

```bash
python3 eval_ppl.py\
	--model-path ./models/llava-v1.5-7b \
	--data-path ./playground/data/mm-vet/mm-vet.json \
	--image-path ./playground/data/mm-vet/images \
	--eval-samples 218 \
	--method "elastic" \
	--ratio 0.2 \
	--exp-name "llava-7b-ppl-mmvet"
```

To eval ROUGE, run the following scripts. Method should be elastic/h2o/local

```bash
# you should firstly generate refernce texts with full cache, and then run the evaluation
python3 convert_rouge_llava.py \
	--model-path ./models/llava-v1.5-7b \
	--data-path ./playground/data/mm-vet/mm-vet.json \
	--image-path ./playground/data/mm-vet/images 
	
python3 eval_rouge.py \
	--model-path ./models/llava-v1.5-7b \
	--data-path ./playground/data/mm-vet/rouge-llava-v1.5-7b-mm-vet.json \
	--image-path ./playground/data/mm-vet/images \
	--eval-samples 218 \
    --ratio 0.2 \
    --exp-name "llava-7b-rouge-mmvet"
```

## Llava-13b

### MM-Vet

To eval PPL, run the following scripts. Method should be elastic/h2o/local

```bash
python3 eval_ppl.py\
	--model-path ./models/llava-v1.5-13b \
	--data-path ./playground/data/mm-vet/mm-vet.json \
	--image-path ./playground/data/mm-vet/images \
	--eval-samples 218 \
	--method "elastic" \
	--ratio 0.2 \
	--exp-name "llava-13b-ppl-mmvet"
```

To eval ROUGE, run the following scripts. Method should be elastic/h2o/local

```bash
# you should firstly generate refernce texts with full cache, and then run the evaluation
python3 convert_rouge_llava.py \
	--model-path ./models/llava-v1.5-13b \
	--data-path ./playground/data/mm-vet/mm-vet.json \
	--image-path ./playground/data/mm-vet/images 
	
python3 eval_rouge.py \
	--model-path ./models/llava-v1.5-13b \
	--data-path ./playground/data/mm-vet/rouge-llava-v1.5-13b-mm-vet.json \
	--image-path ./playground/data/mm-vet/images \
	--eval-samples 218 \
    --ratio 0.2 \
    --exp-name "llava-13b-rouge-mmvet"
```


## Qwen-VL-Chat

### MM-Vet

To eval PPL, run the following scripts. Method should be elastic/h2o/local

```bash
python3 eval_ppl_qwen.py\
	--model-path ./models/qwen-vl-chat \
	--data-path ./playground/data/mm-vet/mm-vet.json \
	--image-path ./playground/data/mm-vet/images \
	--eval-samples 218 \
	--method "elastic" \
	--ratio 0.2 \
	--exp-name "qwen-ppl-mmvet"
```

To eval ROUGE, run the following scripts. Method should be elastic/h2o/local

```bash
# you should firstly generate refernce texts with full cache, and then run the evaluation
python3 convert_rouge_qwen.py \
	--model-path ./models/qwen-vl-chat \
	--data-path ./playground/data/mm-vet/mm-vet.json \
	--image-path ./playground/data/mm-vet/images
	
python3 eval_rouge_qwen.py \
	--model-path ./models/qwen-vl-chat \
	--data-path ./playground/data/mm-vet/rouge-qwen-vl-chat-mm-vet.json \
	--image-path ./playground/data/mm-vet/images \
	--eval-samples 218 \
    --ratio 0.2 \
    --exp-name "qwen-rouge-mmvet"
```

# Generation

 To generate, you can run

  ```bash
  python3 eval_generate.py \
  	--model-path ./models/llava-v1.5-13b \
	--data-path ./playground/data/mm-vet/mm-vet.json \
	--image-path ./playground/data/mm-vet/images
  	--method "elastic" \
    --ratio 0.2 
  ```

# Latency

To eval latency, you can run

  ```bash
  python3 eval_latency.py \
  	--model-path ./models/llava-v1.5-13b \
	--batch-size 8 \
  	--method "elastic" \
    --ratio 0.2 
  ```
