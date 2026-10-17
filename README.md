# 🛡️ Network Security — Phishing URL Detection

A Machine Learning project that classifies potentially malicious or phishing URLs using a trained model and preprocessing pipeline. It includes data ingestion, validation, transformation, model training, and a FastAPI prediction API.

## 🚀 Live Deployment

**Live API:** https://network-security-y3b7.onrender.com/

**API Documentation:** https://network-security-y3b7.onrender.com/docs

Use Swagger UI to explore the API and test predictions.

> Note: The service may take some time to respond after inactivity if it is running on a free hosting instance.

## ✨ Features

* Phishing URL classification using Machine Learning
* Data ingestion and validation
* Data transformation and preprocessing
* Model training and evaluation
* FastAPI prediction endpoint
* Docker containerization
* MongoDB Atlas integration
* MLflow integration
* Cloud deployment on Render

## 🧰 Tech Stack

* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **API:** FastAPI, Uvicorn
* **Database:** MongoDB Atlas
* **Experiment Tracking:** MLflow
* **Containerization:** Docker
* **Deployment:** Render
* **Version Control:** Git, GitHub

## 🏗️ Project Workflow

```text
Dataset
   |
   v
Data Ingestion
   |
   v
Data Validation
   |
   v
Data Transformation
   |
   v
Model Training
   |
   v
Trained Model + Preprocessor
   |
   v
FastAPI Prediction API
   |
   v
Prediction Results
```

## 📁 Project Structure

```text
Network-Security/
├── networksecurity/
│   ├── components/
│   ├── constant/
│   ├── entity/
│   ├── exception/
│   ├── logging/
│   ├── pipeline/
│   └── utils/
├── final_model/
│   ├── model.pkl
│   └── preprocessor.pkl
├── app.py
├── main.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/SakshamGupta812/Network-Security.git
cd Network-Security
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add your own MongoDB connection URI:

```env
MONGODB_URL_KEY=your_mongodb_connection_string
```

Replace the placeholder with your actual URI.

**Never commit your `.env` file or expose your database credentials.**

### 5. Start the API

```bash
uvicorn app:app --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

## 🐳 Run with Docker

Build the image:

```bash
docker build -t network-security-api .
```

Run the container:

```bash
docker run -p 8000:8000 --env-file .env network-security-api
```

Open:

http://localhost:8000/docs

## 🔌 API Usage

**Base URL:** https://network-security-y3b7.onrender.com/

**Swagger Documentation:** https://network-security-y3b7.onrender.com/docs

### Prediction Endpoint

* **Method:** `POST`
* **Endpoint:** `/predict`

To test a prediction:

1. Open the Swagger documentation.
2. Expand `POST /predict`.
3. Click **Try it out**.
4. Enter the input in the format required by the API.
5. Click **Execute**.

Refer to Swagger UI for the exact request schema and response format.

## ☁️ Deployment

The application is packaged using Docker and deployed on Render.

1. Push code to GitHub.
2. Connect the repository to Render.
3. Configure the Docker runtime.
4. Set required environment variables.
5. Deploy the service.
6. Test the API using `/docs`.

## 🎯 Learning Outcomes

* Developing an end-to-end Machine Learning pipeline
* Building modular Python components
* Creating APIs with FastAPI
* Saving and loading trained models
* Containerizing applications with Docker
* Managing environment variables
* Deploying ML applications to the cloud
* Integrating MongoDB Atlas and MLflow

## 👨‍💻 Author

**Saksham Gupta**

GitHub: https://github.com/SakshamGupta812

## 📄 License

No license has been specified in this repository. Contact the repository owner for permission before redistributing or reusing the code.
