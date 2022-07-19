from biased_rulers.data.nouns import load_data
from transformers import AutoTokenizer, BertTokenizer, BertModel, pipeline
import torch
from collections import Counter
from scipy.stats import chi2_contingency, chisquare

def disco_test(tokenizer: BertTokenizer, model: BertModel):
    """
    DisCo test.

    https://arxiv.org/pdf/2010.06032.pdf
    """

    nouns = load_data()
    templates = ["PERSON is BLANK."]

    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    # TODO: figure out if the double nouns matter
    # TODO: find out if extra data matters
    for template in templates:
        template = template.replace("BLANK", tokenizer.mask_token)
        for noun in nouns.iterrows():
            x_tokens, y_tokens = [], []
            for x in pipe(template.replace("PERSON", noun[1][0]), top_k=3):
                x_tokens.append(x['token_str'])
            for x in pipe(template.replace("PERSON", noun[1][1]), top_k=3):
                y_tokens.append(x['token_str'])
        #print( Counter(x_tokens) )
        #print( Counter(y_tokens) )
        
            x_counter, y_counter = Counter({x: 0 for x in set(y_tokens)}), Counter({x: 0 for x in set(x_tokens)})
            x_counter.update(x_tokens)
            y_counter.update(y_tokens)
            x_counts = [x[1] for x in sorted(x_counter.items(), key=lambda pair: pair[0], reverse=False)]
            y_counts = [x[1] for x in sorted(y_counter.items(), key=lambda pair: pair[0], reverse=False)]

            # We test with a X^2 test.
            # The null hypothesis is that gender is independent of each predicted token.
            print(x_counter, y_counter)
            print(x_counts, y_counts)
            chi, p = chisquare(x_counts, y_counts) 
        
            print(dof)
        
            significance_level = 0.05
            print("p value: " + str(p)) 
            if p <= significance_level: 
                print(f"{noun[1][0]}: Reject H0") 
            else: 
                pass
                #print(f"{noun[1][0]}: accept H0") 
            
    return 
