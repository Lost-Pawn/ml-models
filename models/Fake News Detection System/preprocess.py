import re
import pandas as pd
import config


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # strip urls if any slip in
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()  # collapse extra spaces
    return text


def load_and_clean(path=None):
    path = path or config.RAW_DATA_PATH
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    return df


if __name__ == "__main__":
    df = load_and_clean()
    # checked a sample row after cleaning and punctuation, casing were
    # all gone as expected, 8000 rows survived the dropna and length filter
    print(df[["text", "clean_text"]].head(2))
    print(f"rows after cleaning, {len(df)}")
