## Latvian Twitter Sentiment
This is the repository for the paper **Pretraining and Fine-Tuning Strategies for Sentiment Analysis of Latvian Tweets**. Documentation updation in-progress.

### 

### Data
- Since the twiiter data cannot be shared directly due to twitter terms, kindly refer https://github.com/pmarcis/latvian-tweet-corpus for the data.

### Requirements
- Pretrained Latvian Language model (link will updated here)
-  
### How to run
- Either download the pretrained models or train new bert models using 
  - mBERT https://github.com/huggingface/transformers/blob/master/examples/language-modeling/
  - For pretraining ALBERT and ELECTRA https://github.com/shoarora/lmtuners/tree/master/experiments/disc_lm_small
- Update the config file with the pretrained models link 
- Update train.py for the locations of the dataset
- train.py expects pre-processed and pre-split data files( train, dev, test)

`
cd bert-sentiment/src;
python train.py
`
- for prediction

`
python predict.py
`

### Performance Metrices

### Publications

### Acknowledgement
The project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement no. 812997.
