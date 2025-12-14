import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# gender selection
if "gender_pref" not in st.session_state:
    st.set_page_config(layout="centered")
    st.title("Select Your Style Preference")
    col1, col2 = st.columns(2)
    if col1.button("Values / Masculine Style", use_container_width=True):
        st.session_state["gender_pref"] = "Male"
        st.rerun()
    if col2.button("Values / Feminine Style", use_container_width=True):
        st.session_state["gender_pref"] = "Female"
        st.rerun()
    st.stop()

@st.cache_data
def load_data(gender_pref):
    clothes = pd.read_csv("clothes_universal_tagged.csv")
    embeddings_array = np.load("embeddings.npy")
    
    # keeps everything
    mask = pd.Series([True] * len(clothes))
    
    # filtering
    if gender_pref == "Male":
        # drop the female one hot encoded columns
        if "label_Skirt" in clothes.columns: mask &= (clothes["label_Skirt"] == 0)
        if "label_Dress" in clothes.columns: mask &= (clothes["label_Dress"] == 0)
        
        # filters based off labeler.py
        if "is_female" in clothes.columns:
            mask &= (clothes["is_female"] == 0)

    clothes = clothes[mask].reset_index(drop=True)
    embeddings_array = embeddings_array[mask]

    # features vector creating/combining
    one_hot_cols = [c for c in clothes.columns if c.startswith("label_")]
    one_hot_array = clothes[one_hot_cols].values
    full_features = np.concatenate([one_hot_array, embeddings_array], axis=1)
    
    tops = clothes[
        (clothes["label_T-Shirt"] == 1) | (clothes["label_Longsleeve"] == 1) | 
        (clothes["label_Shirt"] == 1) | (clothes["label_Hoodie"] == 1) | 
        (clothes["label_Polo"] == 1) | (clothes["label_Blazer"] == 1)
    ]
    bottoms = clothes[
        (clothes["label_Pants"] == 1) | (clothes["label_Shorts"] == 1)
    ]
    shoes = clothes[clothes["label_Shoes"] == 1]
    
    return clothes, full_features, tops, bottoms, shoes

clothes, full_features, tops, bottoms, shoes = load_data(st.session_state["gender_pref"])

def save_feedback(top_f, bot_f, shoe_f, like):
    with open("user_feedback.csv", "a") as f:
        f.write(f"{top_f},{bot_f},{shoe_f},{like}\n")

def get_feature_by_filename(filename):
    matches = clothes.index[clothes['filepath'] == filename].tolist()
    if not matches: return None
    return full_features[matches[0]]

def train_model():
    try:
        feedback = pd.read_csv("user_feedback.csv", names=["top", "bot", "shoe", "like"])
    except FileNotFoundError: return None
    
    if len(feedback) < 5 or len(feedback["like"].unique()) < 2: return None

    X, y = [], []
    for _, row in feedback.iterrows():
        ft = get_feature_by_filename(row['top'])
        fb = get_feature_by_filename(row['bot'])
        fs = get_feature_by_filename(row['shoe'])
        if ft is not None and fb is not None and fs is not None:
            X.append(np.concatenate([ft, fb, fs]))
            y.append(row['like'])
            
    if len(X) < 5: return None
    clf = RandomForestClassifier(n_estimators=100, random_state=4)
    clf.fit(np.vstack(X), y)
    return clf

if "df_outfits" not in st.session_state:
    num_outfits = 200
    outfits_list = []
    
    if len(shoes) > 0 and len(tops) > 0 and len(bottoms) > 0:
        for _ in range(num_outfits):
            t = tops.sample(1)
            b = bottoms.sample(1)
            s = shoes.sample(1)
            
            outfits_list.append({
                "top_file": t.iloc[0]["filepath"],
                "bottom_file": b.iloc[0]["filepath"],
                "shoe_file": s.iloc[0]["filepath"],
                "features": np.concatenate([
                    full_features[t.index[0]], 
                    full_features[b.index[0]], 
                    full_features[s.index[0]]
                ])
            })
        st.session_state["df_outfits"] = pd.DataFrame(outfits_list)
    else:
        st.error(f"Not enough data. Found {len(shoes)} shoes, {len(tops)} tops, {len(bottoms)} bottoms.")
        st.stop()


# display

clf = train_model()
if clf and not st.session_state["df_outfits"].empty:
    feats = np.vstack(st.session_state["df_outfits"]["features"].values)
    st.session_state["df_outfits"]["prob"] = clf.predict_proba(feats)[:, 1]
    st.session_state["df_outfits"] = st.session_state["df_outfits"].sort_values("prob", ascending=False).reset_index(drop=True)

if "idx" not in st.session_state: st.session_state.idx = 0
idx = st.session_state.idx

if idx >= len(st.session_state["df_outfits"]):
    st.write("Out of outfits")
    if st.button("Generate More"):
        del st.session_state["df_outfits"]
        st.session_state.idx = 0
        st.rerun()
    st.stop()

row = st.session_state["df_outfits"].iloc[idx]

def on_like():
    save_feedback(row["top_file"], row["bottom_file"], row["shoe_file"], 1)
    st.session_state.idx += 1

def on_dislike():
    save_feedback(row["top_file"], row["bottom_file"], row["shoe_file"], 0)
    st.session_state.idx += 1
    
def on_reset():
    with open("user_feedback.csv", "w") as f: f.write("")
    del st.session_state["df_outfits"]
    st.session_state.idx = 0

c1, c2, c3 = st.columns(3)
c1.image(Image.open(row["top_file"]), caption="Top")
c2.image(Image.open(row["bottom_file"]), caption="Bottom")
c3.image(Image.open(row["shoe_file"]), caption="Shoes")

b1, b2, b3 = st.columns(3)
b1.button("Like", on_click=on_like, use_container_width=True)
b2.button("Dislike", on_click=on_dislike, use_container_width=True)
b3.button("Reset", on_click=on_reset, use_container_width=True)