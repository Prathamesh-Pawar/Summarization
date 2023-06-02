import argparse
import re
from rouge import Rouge
rouge = Rouge()
from transformers import pipeline
from datasets import load_dataset
from huggingface_hub import notebook_login




def split_into_chunks(text, max_length):
    """
    Splits a string into chunks of text with complete sentences, where each chunk
    has a maximum length of `max_length` characters.
    """
    sentences = re.findall(r'[^\n.!?]+[.!?]', text)  # Split into sentences
    chunks = []
    current_chunk = ''

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            # If adding the sentence doesn't exceed max_length, add to current chunk
            current_chunk += sentence
        else:
            # If adding the sentence exceeds max_length, start a new chunk
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def get_chunks(input_text, summarizer, max_length=1025):
    chunks = split_into_chunks(input_text, max_length)
    summary_temps=[]
    for i in chunks:
        summary_temps.append(summarizer(i,max_length=32))
    summary_temps_ = [i[0]['summary_text'] for i in summary_temps]
    return '. '.join(summary_temps_)




def rouge_score_generation(generated_summary, reference_summary):
    scores = rouge.get_scores(generated_summary, reference_summary)
    rouge_1 = scores[0]['rouge-1']['f']
    rouge_2 = scores[0]['rouge-2']['f']
    rouge_l = scores[0]['rouge-l']['f']
    print("ROUGE-1: {:.2f}".format(rouge_1 * 100))
    print("ROUGE-2: {:.2f}".format(rouge_2 * 100))
    print("ROUGE-L: {:.2f}".format(rouge_l * 100))
    return True



class SummaryEvaluator:
    def __init__(self, data_files, dataset):
        self.rouge = Rouge()
        notebook_login()
        self.data_files = data_files
        self.dataset = dataset

    
    def run(self):
        # Initialize Bart-base model as a control summary
        print("Scores for bart-base")
        summarizer = pipeline("summarization", model="facebook/bart-base")
        bart_base_summary = get_chunks(self.dataset['test']['content'][0],summarizer)
        rouge_base = rouge_score_generation(self.dataset['test']['summary'][0], bart_base_summary)

        # Initialize Bart-custom model
        print("Scores for bart_custom")
        summarizer = pipeline("summarization", model="PrathameshPawar/bart_custom")
        bart_custom_summary = get_chunks(self.dataset['test']['custom_approach'][0],summarizer)
        rouge_custom = rouge_score_generation(self.dataset['test']['summary'][0], bart_custom_summary)

        # Initialize Pegasus model as a control summary
        print("Scores for pegasus-base")
        summarizer = pipeline("summarization", model="google/pegasus-arxiv")
        pegasus_base_summary = get_chunks(self.dataset['test']['content'][0],summarizer)
        rouge_base = rouge_score_generation(self.dataset['test']['summary'][0], pegasus_base_summary)

        # Initialize Pegasus-custom model
        print("Scores for pegasus_custom")
        summarizer = pipeline("summarization", model="PrathameshPawar/pegasus_custom")
        pegasus_custom_summary = get_chunks(self.dataset['test']['custom_approach'][0],summarizer)
        rouge_custom = rouge_score_generation(self.dataset['test']['summary'][0], pegasus_custom_summary)

if __name__ == '__main__':

    data_files = { "test": "1000_test.json" }
    dataset = load_dataset('PrathameshPawar/summary_2k', data_files=data_files)

    summary_evaluator = SummaryEvaluator(data_files, dataset)
    summary_evaluator.run()
