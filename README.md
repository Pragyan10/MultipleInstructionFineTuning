# Multiple Instruction FineTuning on Large Language Models
The task we accomplish in this exercise is to fine tune a large langauge model on different instructions and see how they differ from each other. 

## Model Detail 
For the task we use Llama 2 instruct model from hugging face -> https://huggingface.co/NousResearch/Llama-2-7b-chat-hf. We call this as our base model for the experiment.  

## Datasets details 
For the purpose of fine tuning and testing the models we use 2 different datasets listed below and also create 10 out of sample instances which can be found in [3]
Datasets:
- Twitter Sentiment Analysis Dataset -> https://huggingface.co/carblacac/twitter-sentiment-analysis
- Alpaca Dataset -> https://huggingface.co/datasets/tatsu-lab/alpaca

## Directory detail
There are 4 different directories included in the repository. 
- EvaluationNotebooks: Consists of all the notebooks used for evaluating the base, single instruction fine tune, and two instructions fine tuned models.  
    [1]. FineTunedModelEvaluation-OnAlpacaData.ipnyb -> Evaluation of the models done on Alpaca Dataset
    [2]. FineTunedModelEvaluation-OnCustomData.ipnyb -> Evaluation of the models done on Custom Instruction Twitter Sentiment Dataset
    [3]. FineTunedModelEvaluation-OnOutOfSampleData.ipnyb -> Evaluation of the models done on 10 different out of sample instructions.
- FineTuningNotebook: Consistis of notebooks used to fine tune the Llama2 model on multiple instructions. 
    [4]. FineTunningLlama2-AlpacaAndTwitterDataset-AlpacaAndCustomInstruction.ipynb -> Fine tuning Llama2 on a combiniation of dataset from Custom instruction Twitter Sentiment Dataset and Alpaca Dataset. 
    [5]. FineTunningLlama2-TwitterDataset-CustomInstruction.ipynb -> Fine tuning Llama2 on a Custom instruction Twitter Sentiment Dataset only.
- llama2-finetunedSentimentClassificationOneInstruction: Consists the fine tuned model's files -> fine tuned on custom instruction twitter dataset only.
- llama2-finetunedSentimentClassificationTwoInstruction: Consists the fine tuned model's files -> fine tuned on custom instruction twitter dataset and alpaca dataset.

## Models used for analysis:
[6] Base Model 
[7] Fine Tuned Model on Custom Instruction Twitter Sentiment Dataset only (FT-CustomOnly)
[8] Fine Tuned Model on Custom Instruction Twitter Sentiment Dataset and Alpaca Dataset (FT-BothInstruction)

## Training Details:
- FT-CustomOnly -> Training Time 48 mins with 8k data 
- FT-BothInstruction -> Training Time 38 mins with 8k data
Both training completed using 3 RTX 6000 GPU - used with accelerator. 

# Results and Analysis
We study the models behavior on three settings. First, when all the models are tested on Custom Instruction Twitter Sentiment Dataset, and Second, on 10 custom built instructions generated. Additionally, we also see when all the models are tested on the Alpaca Dataset. 

## [A] Comparing the models - on Custom Instruction Twitter Sentiment Dataset -> [2]
The results suggest that the base model is not able to handle the sentiment classification instruction well as the metric scores are low. FT-CustomOnly and FT-BothInstruction perform the same since they both were trained on the same instruction dataset. Both the models perform very good. We also notice that BLEU score is not a good metric for doing sentiment analysis since it only has one number as out. 

| Metric    | Base Model    |   FT-CustomOnly  | FT-BothInstruction    |
|--------------|--------------|--------------|--------------|
| BLEU | 0.00 | 0.00 | 0.00 |
| ROUGE | 0.65 | 0.95 | 0.95 |
| BERTScore | 0.992 | 0.999 | 0.999 |

## [B] Comparing the models - on 10 custom built Instructions -> [3]
The results for the out of sample is different comparing to [A]. We notice that the base model outperforms the fine tuned model. Even in fine tuned model, we see that the FT-CustomOnly performs better in comparsion to the FT-BothInstruction. This maybe due to the multi instruction fine tuning which makes the model prediction a little random. 

| Metric    | Base Model    |   FT-CustomOnly  | FT-BothInstruction    |
|--------------|--------------|--------------|--------------|
| BLEU | 0.25 | 0.09 | 0.00 |
| ROUGE | 0.28 | 0.08 | 0.03 |
| BERTScore | 0.88 | 0.81 | 0.78 |


## Additional Analysis 
## [C] Comparing the models - on the Alpaca Dataset -> [1]
Similart to [B], the results for the out of sample is different comparing to [A]. We notice that the base model outperforms the fine tuned model. Even in fine tuned model, we see that the FT-CustomOnly performs better in comparsion to the FT-BothInstruction. This maybe due to the multi instruction fine tuning which makes the model prediction a little random. 

| Metric    | Base Model    |   FT-CustomOnly  | FT-BothInstruction    |
|--------------|--------------|--------------|--------------|
| BLEU | 0.41 | 0.29 | 0.00 |
| ROUGE | 0.35 | 0.21 | 0.05 |
| BERTScore | 0.88 | 0.85 | 0.79 |




