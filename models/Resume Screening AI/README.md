# Resume Screening AI

A small machine learning project that reads a resume and predicts which
job category it best fits, for example Data Science, Software Development,
Human Resources, Sales, Web Designing or Mechanical Engineering. It works
using TF-IDF text vectorization together with a Logistic Regression
classifier from scikit learn.

## Files in this project

`generate_dataset.py` builds a synthetic dataset of resumes and saves it
as `data/resumes.csv`. Since a real company resume database usually isn't
available for practice projects, this creates realistic sounding resumes
by mixing skills, tools and summary lines for six categories.

`preprocess.py` holds the text cleaning function used everywhere else in
the project, so training and prediction always clean text the same way.

`train_model.py` loads the dataset, cleans it, converts it into TF-IDF
features, trains the classifier, prints accuracy and a classification
report, then saves the trained model, vectorizer and label encoder into
the `model` folder.

`predict.py` loads those saved files and predicts the category for new
resume text, along with a confidence score.

## How to run it

Install the dependencies first.

```
pip install -r requirements.txt
```

Then run the three steps in order.

```
python generate_dataset.py
python train_model.py
python predict.py
```

## A note on the results

Since the dataset here is synthetic and built from templates, the model
scores a perfect accuracy on the test split, which is printed as a comment
inside `train_model.py`. Real resumes are much messier and less templated,
so anyone plugging in an actual resume dataset should expect the accuracy
to be lower, and that is normal.

## Ideas to extend this

Swap `data/resumes.csv` with a real labeled resume dataset if you have one.
Try a different model such as Random Forest or a Support Vector Machine
and compare the results. Add a simple web interface using Streamlit or
Flask so resumes can be uploaded and screened through a browser instead
of the command line.
