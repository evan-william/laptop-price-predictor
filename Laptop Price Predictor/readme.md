# Laptop Price AI Predictor

A sleek, dark-themed web app that uses Machine Learning to estimate the market value of a laptop based on its specs. Built with a professional **Scikit-Learn Pipeline** and **Streamlit**.

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt

```


2. **Train the AI:**
```bash
python src/train.py

```


3. **Launch the App:**
```bash
streamlit run src/app.py

```



---

## How it Works

* **The Brain:** A Linear Regression model trained on laptop specs (RAM, SSD, Weight, Screen Size).
* **The Pipeline:** Uses a `Pipeline` to scale data and predict price in one step.
* **The UI:** A dashboard built with Streamlit and custom CSS.
* **The Quality:** Includes a `pytest` suite to ensure the AI logic is actually accurate before it goes live.

---

## Folder Structure

* `src/app.py`: The web dashboard.
* `src/train.py`: The script that creates the AI.
* `tests/`: Professional unit tests using Pytest.
* `models/`: Where the saved AI "brain" (`.joblib`) lives.

---

## Author

**Evan William** Version 1.0 (2026)