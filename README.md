# 🎬 Movie Success Predictor

A machine learning–powered web application that predicts whether a movie will be a **Success or Flop**, along with projected revenue and probability metrics, based on key film attributes.

This project was developed as part of **Project & Portfolio IV** and serves as a **Proof of Concept** for a larger predictive analytics platform focused on film performance forecasting.

---

## 📌 Project Overview

The Movie Success Predictor allows users to input structured movie data such as budget, genre, runtime, and cast score. Using a trained machine learning model, the application generates predictive insights including:

- Success vs. Flop classification
- Success probability percentage
- Estimated box office revenue

The goal of this system is to simulate how studios and investors evaluate film performance prior to release.

---

## 🧠 Technologies Used

- **Python**
- **Flask** (Web Framework)
- **Scikit-learn** (Machine Learning)
- **Pandas / NumPy** (Data Processing)
- **HTML / CSS** (Frontend UI)
- **Jinja2** (Templating)
- **Pickle** (Model Serialization)

---

## ⚙️ Features (Proof of Concept)

### Input Interface
Users can enter:

- Budget
- Genre
- Runtime
- Cast Score
- Release Year

### Prediction Outputs

- 🎯 Success vs. Flop classification
- 📊 Success probability %
- 💰 Revenue prediction estimate

### Analytics (Prototype)

- Revenue forecast visualization
- Feature importance chart
- Budget impact graph
- Genre performance analysis

---

## 🏗️ System Architecture

1. User inputs film attributes via web form
2. Flask processes the request
3. Pretrained ML model loads from `.pkl` file
4. Data is transformed to model format
5. Prediction + probabilities are generated
6. Results displayed in dashboard UI

---

## 🚀 How to Run the App Locally

```bash
git clone https://github.com/aslawant/project_and_portfolio_4.git
cd project_and_portfolio_4
