python3 eval_ppl.py\
	--model-path ./models/llava-v1.5-7b \
	--data-path ./playground/data/mm-vet/mm-vet.json \
	--image-path ./playground/data/mm-vet/images \
	--eval-samples 218 \
	--method "elastic" \
	--ratio 0.2 \
	--exp-name "llava-7b-ppl-mmvet"