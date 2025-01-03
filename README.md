> This project utilizes [Axolotl](https://github.com/axolotl-ai-cloud/axolotl?tab=readme-ov-file#dataset) for finetuning tasks.

## Environment Setup : Axolotl
```shell
# python >= 3.10
conda create -n axolotl python=3.10

# pytorch >= 2.3.1
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install nvcc
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit cuda-nvcc
echo $CONDA_PREFIX
## add to bashrc
vim ~/.bashrc
export CUDA_HOME= # paste the conda prefix
export PATH=$CUDA_HOME/bin:$PATH

# install axolotl
pip3 install packaging
pip3 install --no-build-isolation -e '.[flash-attn,deepspeed]'

# log in to huggingface
huggingface-cli login
```

## Part 1 : Sentiment Analysis with IMDB dataset and LoRA
In this part, Llama3-8B model is finetuned with the IMDB dataset to perform sentiment analysis. 
- Dataset : [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb)

#### Dataset preperation
```shell
python load_imdb.py
```

#### Finetune model
```shell
axolotl train sentiment_analysis_LoRA/model.yml
```

#### Inference
```shell
python sentiment_analysis_LoRA/inference.py
```
#### Results
```shell
# Non-fine-tuned llama3 result
Finally, store the cake in an airtight container to maintain its freshness and quality.

# DPO fine-tuned llama3 result
Summarize this paragraph : “The process of baking a cake begins by gathering the necessary ingredients and preheating the oven. The baking pan is prepared by coating it with cooking spray or lining it with parchment paper. Next, the dry ingredients are mixed together in one bowl, and the wet ingredients are combined in another. The dry ingredients are added to the wet ingredients in alternate batches, stirring each time until thoroughly incorporated. After adding any additional flavorings or colorings, the batter is poured into the prepared pan and smoothed out evenly. The cake is then baked at the appropriate temperature for the designated amount of time, typically around 20 minutes. Once done, the cake is removed from the oven and allowed to cool completely before serving or decorating.”

```

## Part 2 : Human preference learning with DPO
In this part, Llama3-8B model is finetuned with DPO for summarization tasks.
- Dataset : [fozziethebeat/alpaca_messages_2k_dpo_test](https://huggingface.co/datasets/fozziethebeat/alpaca_messages_2k_dpo_test)

#### Finetune model
```shell
axolotl train summarization_DPO/model.yml
```

#### Inference
```shell
axolotl inference summarization_DPO/model.yml --lora_model_dir="./model"
```
#### Results
```shell
Review: I love this movie! This movie is sad, but it is good for learning about life.
Is this movie review positive or negative? Answer with a single word - positive or negative:positive
```