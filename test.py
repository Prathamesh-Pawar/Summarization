import argparse
import re
from rouge import Rouge
from transformers import pipeline

from huggingface_hub import notebook_login
notebook_login()

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

def get_chunks(input_text, model_name, max_length=1025):
    summarizer = pipeline("summarization", model=model_name)
    chunks = split_into_chunks(input_text, max_length)
    summary_temps=[]
    
    for i in chunks:
        summary_temps.append(summarizer(i,max_length=32))
        
    summary_temps_ = [i[0]['summary_text'] for i in summary_temps]
        
    return '. '.join(summary_temps_)

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Summarize text using BART model")

    # Add arguments
    parser.add_argument("-t", "--text", type=str, required=True, help="Input text for summarization")
    parser.add_argument("-m", "--model", type=str, required=True, help="Name of the BART model to use")

    # Parse command line arguments
    args = parser.parse_args()

    # Retrieve values of arguments
    text = args.text
    model_name = args.model

    bart_base_summary = get_chunks(text, model_name)
    print(bart_base_summary)
