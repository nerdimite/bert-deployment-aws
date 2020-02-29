from transformers import BertTokenizer
import torch
import torch.nn

def preprocess(sentence, maxlen=64):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)

    # Inserting the CLS and SEP token at the beginning and end of the sentence
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # Padding/truncating the sentences to the maximum length
    if len(tokens) < maxlen:
        tokens = tokens + ['[PAD]' for _ in range(maxlen - len(tokens))]
    else:
        tokens = tokens[:maxlen-1] + ['[SEP]']

    # Convert the sequence to ids with BERT Vocabulary
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Converting the list to a pytorch tensor
    tokens_ids_tensor = torch.tensor(tokens_ids).unsqueeze(0)

    # Obtaining the attention mask
    attn_mask = (tokens_ids_tensor != 0).long()

    return tokens_ids_tensor, attn_mask