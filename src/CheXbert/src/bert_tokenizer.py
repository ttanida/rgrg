import os

import pandas as pd
from transformers import BertTokenizer
import json
from tqdm import tqdm
import argparse

def get_impressions_from_csv(path):	
    df = pd.read_csv(path)
    imp = df['Report Impression']
    imp = imp.str.strip()
    imp = imp.replace('\n',' ', regex=True)
    imp = imp.replace('\s+', ' ', regex=True)
    imp = imp.str.strip()
    return imp

def tokenize(impressions, tokenizer):
    new_impressions = []
    print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
    for i in tqdm(range(impressions.shape[0])):
        try:
            tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
        except Exception:
            txt_file_name = os.path.join("/u/home/tanida/region-guided-chest-x-ray-report-generation/src/", "failed_tokenizations.txt")
            with open(txt_file_name, "a") as f:
                f.write(f"Failed tokenization for {impressions.iloc[i]} at index {i}\n")
            tokenized_imp = None
        if tokenized_imp:  # not an empty report
            res = tokenizer.encode_plus(tokenized_imp)['input_ids']
            if len(res) > 512:  # length exceeds maximum size
                # print("report length bigger than 512")
                res = res[:511] + [tokenizer.sep_token_id]
            new_impressions.append(res)
        else:  # an empty report
            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return new_impressions

def load_list(path):
        with open(path, 'r') as filehandle:
                impressions = json.load(filehandle)
                return impressions

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Tokenize radiology report impressions and save as a list.')
        parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                            help='path to csv containing reports. The reports should be \
                            under the \"Report Impression\" column')
        parser.add_argument('-o', '--output_path', type=str, nargs='?', required=True,
                            help='path to intended output file')
        args = parser.parse_args()
        csv_path = args.data
        out_path = args.output_path
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        impressions = get_impressions_from_csv(csv_path)
        new_impressions = tokenize(impressions, tokenizer)
        with open(out_path, 'w') as filehandle:
                json.dump(new_impressions, filehandle)
