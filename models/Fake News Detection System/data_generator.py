import random
import pandas as pd
import config

# there is no uploaded dataset for this project, so this file builds one
# on its own using templates that mimic how real and fake headlines are
# usually written. real world projects would swap this file out for
# something that just loads a csv like Fake.csv and True.csv from Kaggle,
# everything downstream in this project does not care where the data
# actually came from

random.seed(config.RANDOM_STATE)

TOPICS = [
    "the economy", "the election", "a new vaccine", "climate policy",
    "a local hospital", "the stock market", "a tech company", "the school board",
    "a government agency", "a celebrity", "the police department", "a university",
    "a food company", "the housing market", "a sports team", "the military",
    "a bank", "the healthcare system", "a scientific study", "the border policy"
]

PEOPLE = [
    "the mayor", "a senior official", "the CEO", "a spokesperson", "the governor",
    "a researcher", "the committee chair", "a union leader", "the department head",
    "an anonymous source", "a whistleblower", "the prime minister"
]

# real style templates lean on attribution, numbers, and measured language
REAL_TEMPLATES = [
    "{person} announced on {day} that {topic} will see changes starting next quarter, according to officials familiar with the matter",
    "A report released this week shows {topic} grew by {num} percent compared to last year, based on data from the finance ministry",
    "{person} confirmed in a press briefing that {topic} is under review, and a final decision is expected within {num} weeks",
    "Officials said {topic} met expectations in the latest quarterly review, with {num} percent of targets achieved",
    "According to a statement from {person}, {topic} will be discussed at the upcoming session on {day}",
    "The city council voted {num} to approve new funding for {topic}, following weeks of public hearings",
    "Researchers published findings on {topic} in a peer reviewed journal, noting a sample size of {num} participants",
    "{person} told reporters that {topic} remains a priority, adding that further updates will follow next {day}",
    "A survey conducted last month found that {num} percent of residents support changes to {topic}",
    "The department released updated guidelines for {topic} after consulting with {num} independent experts",
]

# fake style templates lean on urgency, vague sourcing, and emotional language
FAKE_TEMPLATES = [
    "You wont believe what {person} just did about {topic}, this changes everything",
    "SHOCKING, {topic} secretly controlled by a group nobody is talking about",
    "Experts are too scared to admit the truth about {topic}, share before this gets deleted",
    "{person} exposed for hiding the real numbers behind {topic}, wake up people",
    "This one weird trick about {topic} has {person} furious, click to see why",
    "Leaked documents reveal {topic} is a total scam, mainstream media refuses to cover it",
    "{num} percent of people have no idea {topic} is being manipulated behind closed doors",
    "BREAKING, insiders confirm {topic} was staged the whole time, share now",
    "The government does not want you to know this about {topic}, forward to everyone",
    "{person} finally admits {topic} was a cover up all along, this is huge",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "the weekend"]

# neutral filler that could plausibly show up in either style of writing,
# this is what keeps the two classes from being perfectly separable
FILLER_SENTENCES = [
    "More details are expected to follow in the coming days",
    "Reaction to the news has been mixed so far",
    "This is a developing story",
    "Local residents had varied opinions on the matter",
    "The situation remains fluid as of this writing",
]

# how often an article gets a sentence pulled from the opposite class,
# mimics the way real fake articles sometimes borrow a factual sounding
# line, and real articles occasionally get a quote that sounds dramatic
CROSSOVER_RATE = 0.18


def build_sentence(template):
    return template.format(
        topic=random.choice(TOPICS),
        person=random.choice(PEOPLE),
        num=random.randint(2, 97),
        day=random.choice(DAYS),
    )


def generate_article(templates, opposite_templates, min_sentences=3, max_sentences=6):
    count = random.randint(min_sentences, max_sentences)
    sentences = []
    for _ in range(count):
        if random.random() < CROSSOVER_RATE:
            sentences.append(build_sentence(random.choice(opposite_templates)))
        elif random.random() < 0.15:
            sentences.append(random.choice(FILLER_SENTENCES))
        else:
            sentences.append(build_sentence(random.choice(templates)))
    return ". ".join(sentences) + "."


def generate_dataset():
    rows = []
    for _ in range(config.SAMPLES_PER_CLASS):
        rows.append({"text": generate_article(REAL_TEMPLATES, FAKE_TEMPLATES), "label": 0})  # 0 is real
    for _ in range(config.SAMPLES_PER_CLASS):
        rows.append({"text": generate_article(FAKE_TEMPLATES, REAL_TEMPLATES), "label": 1})  # 1 is fake

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(config.RAW_DATA_PATH, index=False)
    # ran this and got 8000 rows total, 4000 real and 4000 fake, roughly
    # 73 words per article on average
    print(f"saved {len(df)} rows to {config.RAW_DATA_PATH}")
    print(df["label"].value_counts())
