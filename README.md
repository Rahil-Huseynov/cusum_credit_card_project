# CUSUM Credit Card Anomaly Detection

This project uses the CUSUM algorithm to detect unusual credit card transactions.

## 1. Requirements

Make sure Python 3 is installed.

Check your Python version:

```bash
python3 --version
```

## 2. Open the Project

Extract the ZIP file and open the project folder in the terminal:

```bash
cd cusum_credit_card_project
```

## 3. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

## 4. Install Required Packages

```bash
pip install -r requirements.txt
```

## 5. Run the Program

```bash
python src/cusum_detector.py
```

## 6. Run the Tests

```bash
python -m unittest discover -s tests -v
```

## 7. Results

The program saves its output in the `results` folder.

This folder contains detected anomalies, transaction results, and other output files.

## Note

Run all commands from the main project folder. If you have a problem, check that Python is installed and all required packages are installed correctly.
