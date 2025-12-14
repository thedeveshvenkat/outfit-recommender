import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv("clothes_processed.csv")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP on {device}...")
model, preprocess = clip.load("ViT-B/32", device=device)

# define prompts for each category. [0] is male, compared against the others (female) in each list
prompt_map = {
    "label_Shorts": ["men's basketball shorts", "women's short jean shorts", "women's daisy dukes"],
    "label_Pants": ["men's pants", "women's leggings", "women's yoga pants"],
    "label_T-Shirt": ["men's t-shirt", "women's blouse", "women's crop top"],
    "label_Longsleeve": ["men's sweater", "women's blouse", "women's cardigan"],
    "label_Hoodie": ["men's hoodie", "women's cropped hoodie", "pink women's hoodie"],
    "label_Blazer": ["men's suit jacket", "women's blazer", "fitted women's blazer"],
    "label_Polo": ["men's polo shirt", "women's polo", "fitted women's top"],
    "label_Shoes": ["men's shoe", "women's high heel", "women's flat shoe"],
    "label_Shirt": ["men's button up shirt", "women's blouse", "women's dress shirt"],
    # just in case, skirts and dresses have already been taken care of in the preprocessing
    "label_Skirt": ["kilt", "women's skirt", "miniskirt"], 
    "label_Dress": ["men's robe", "women's dress", "gown"],
}


df["is_female"] = 0 

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = row["filepath"]
    
    active_label = None
    for col in df.columns:
        if col.startswith("label_") and row[col] == 1:
            active_label = col
            break
            
    if active_label and active_label in prompt_map:
        prompts = prompt_map[active_label]
        
        try:
            # prepare image
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            
            # prepare text
            text_inputs = torch.cat([clip.tokenize(f"a photo of {p}") for p in prompts]).to(device)
            
            with torch.no_grad():
                # find similarity
                logits_per_image, _ = model(image, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                

                # since [0] is the male prompt, if sum of other probabilities > male probability, the piece is female
                male_score = probs[0]
                female_score = np.sum(probs[1:])
                
                if female_score > male_score:
                    df.at[idx, "is_female"] = 1
                    
        except Exception as e:
            print(f"Error on {image_path}: {e}")
            
    elif active_label in ["label_Skirt", "label_Dress"]:
        df.at[idx, "is_female"] = 1

df.to_csv("clothes_universal_tagged.csv", index=False)