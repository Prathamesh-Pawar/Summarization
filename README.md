# Summarization

This project showcases an attempt to summarize and simplify scientific documents by fine tuning language models on custom preprocessed data

### Installation

#### Clone the repository: 

`git clone https://github.com/Prathamesh-Pawar/Summarization.git`

#### Change to the project directory: 

`cd Summarization`

#### Install dependencies: 

`pip install -r requirements.txt`

#### Get Huggingface login

Generate hugginface token to use the pretrained models and datasets.

#### To generate a small sample of pre-processed data, follow these steps:

This will generate a JSON with 'topic', 'summary' and 4 preprocessed contents

`python data.py`

#### To use Summarization, follow these steps:

`python test.py -t "This is the input text for summarization." -m "Choose from list of models"`

list of models:
1. PrathameshPawar/pegasus_raw
2. PrathameshPawar/pegasus_traditional
3. PrathameshPawar/pegasus_custom
4. PrathameshPawar/pegasus_combined
5. PrathameshPawar/bart_raw
6. PrathameshPawar/bart_traditional
7. PrathameshPawar/bart_custom
8. PrathameshPawar/bart_combined

#### To train Summarization, follow these steps:

This may not run, if you do not posses the sufficient cuda core and memory if you desire to run it on CPU change the `fp16=False` under Seq2SeqTrainingArgument.

`python train.py --dataset 3500_train_trad.json --model google/pegasus`

#### To evaluate the Summarization, follow these steps:

`python evaluation.py`

### Contributing follow these steps:

If you would like to contribute to Summarization, you can follow these steps:


Fork the repository.
Create a new branch for your contribution: 
`git checkout -b feature/your-feature-branch`

Make your changes and commit them: 
`git commit -m "Add your commit message here"`

Push your changes to your forked repository: 
`git push origin feature/your-feature-branch`

Create a pull request to the main repository, explaining your changes and their significance.

### Contact
For any questions, suggestions, or concerns, please feel free to contact the project maintainer at pawar.prath@northeastern.edu and sharan.a@northeastern.edu

We appreciate your interest and contributions to Summarization! Thank you for your support.


### Links to models and datasets

## Models

### Pegasus 

1. pegasus_raw :        https://huggingface.co/PrathameshPawar/pegasus_raw/tree/main
2. pegasus_traditional: https://huggingface.co/PrathameshPawar/pegaus_traditional/tree/main
3. pegasus_custom:      https://huggingface.co/PrathameshPawar/pegasus_custom/tree/main
4. pegasus_combined:    https://huggingface.co/PrathameshPawar/pegasus_combined/tree/main

### Bart

1. bart_raw :        https://huggingface.co/PrathameshPawar/bart_raw/tree/main
2. bart_traditional: https://huggingface.co/PrathameshPawar/bart_traditional/tree/main
3. bart_custom:      https://huggingface.co/PrathameshPawar/bart_custom/tree/main
4. bart_combined:    https://huggingface.co/PrathameshPawar/bart_combined/tree/main


## Datasets

1. Train: https://huggingface.co/datasets/PrathameshPawar/10ktesttrain/tree/main
2. Test:  https://huggingface.co/datasets/PrathameshPawar/summary_2k/tree/main


