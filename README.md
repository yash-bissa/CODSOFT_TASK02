# Movie Genre Classification

CodSoft Machine Learning Internship · Task 1

TF-IDF converts English plot summaries into unigram and bigram features. Class-balanced logistic regression learns one of the dataset's genres. The app shows the highest-scoring genre and the next two alternatives.

## Run on Windows

Extract the project under `D:\CodSoft`, open PowerShell in `CODSOFT_TASK1`, and paste:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_data.py
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:8001](http://127.0.0.1:8001). Keep PowerShell open while using the app. The first page load trains the model. Restarting the app retrains it.

If the automatic download is unavailable, download and extract the [assignment dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb). Place the following files directly in the project folder's `data` subfolder: `data/train_data.txt`, `data/test_data_solution.txt`. Create the subfolder if needed.

On Linux or macOS, use `python3 -m venv .venv` and replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Demo

Load the detective plot or paste an original plot of your own. Click **Predict genre** and compare the three genre scores. Click the theme button at the top to switch between dark and light modes. The browser remembers the selection.

## Evaluation

The vectorizer and classifier learn only from `train_data.txt`. Evaluation uses `test_data_solution.txt`, the dataset's separate labeled test split. Accuracy measures overall correctness, and macro F1 gives each genre equal weight.

Some plots span several genres. Model scores reflect the training labels and may be uncertain on unfamiliar or very short summaries.

## Project files

- `app.py` serves the app and prediction API.
- `model.py` trains, evaluates, and runs the model.
- `index.html` contains the responsive interface and theme controls.
- `setup_data.py` downloads the assignment dataset.
- `requirements.txt` lists Python dependencies.

The repository excludes the downloaded dataset and local environment. Follow the setup instructions to reproduce training and measurements.
