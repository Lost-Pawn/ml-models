import json
import pickle
import random

from preprocess import clean_text

# Everything here is loaded once, at startup, so each reply during the chat
# loop is just a quick lookup instead of retraining or reloading anything.
with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("responses.pkl", "rb") as f:
    responses = pickle.load(f)

# model_info.json is written by train_model.py from the actual training run,
# so the numbers shown at startup are never hard-coded and can't go stale.
try:
    with open("model_info.json", "r") as f:
        model_info = json.load(f)
except FileNotFoundError:
    model_info = None

# Below this confidence the bot admits it does not understand instead of
# guessing. This was chosen by testing thresholds 0.20-0.40 against held-out
# test predictions (see train_model.py / project report): at 0.30, only 3
# out of 113 test examples were wrong-but-confident-enough to slip through,
# while accuracy on the answers the bot actually gives climbs to ~95%.
# Random/gibberish input scored around 0.10-0.12 in testing, well below
# this line, so genuine nonsense is reliably caught.
CONFIDENCE_THRESHOLD = 0.30

EXIT_COMMANDS = {"quit", "exit", "bye", "goodbye"}

# A short list of example topics shown in the fallback message so an
# unrecognized message still points the user somewhere useful instead of
# just saying "I don't understand".
SAMPLE_TOPICS = [
    "store hours", "pricing", "your order status", "cancelling an order",
    "contacting support", "general questions",
]


def get_response(user_input):
    """
    Clean the input, classify its intent, and return an appropriate reply.
    Never raises on empty, whitespace-only, or punctuation-only input.
    """
    cleaned = clean_text(user_input)

    if not cleaned:
        return ("I didn't catch any actual words there. "
                "Could you type your question in a few words?")

    vector = vectorizer.transform([cleaned])

    probabilities = model.predict_proba(vector)[0]
    best_index = probabilities.argmax()
    confidence = probabilities[best_index]
    predicted_tag = model.classes_[best_index]

    if confidence < CONFIDENCE_THRESHOLD:
        topics = ", ".join(random.sample(SAMPLE_TOPICS, k=3))
        return (
            "I'm not quite sure what you mean, could you rephrase that? "
            f"For example, you can ask me about {topics}."
        )

    return random.choice(responses[predicted_tag])


def print_banner():
    print("=" * 40)
    print("        ML CHATBOT")
    print("=" * 40)
    if model_info:
        print(f"Model: {model_info['model']}")
        print(f"Intents: {model_info['num_intents']}")
        print(f"Training examples: {model_info['num_training_examples']}")
        print(f"Cross-validation accuracy: "
              f"{model_info['cv_mean_accuracy']*100:.1f}% "
              f"(+/- {model_info['cv_std_accuracy']*100:.1f}%)")
    print()
    print("Bot: Hello! How can I help you? (type 'quit' to exit)")
    print()


def main():
    print_banner()
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in EXIT_COMMANDS:
            print("Bot: Goodbye! Have a great day.")
            break
        reply = get_response(user_input)
        print("Bot:", reply)


if __name__ == "__main__":
    main()
