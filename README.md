# Bank Notes Detection API with Flask, Google Cloud and TensorFlow

This repository contains a simple Flask web application that serves as an API for Bank Notes Detection (IDR). The application uses my pre-trained CNN TensorFlow models to predict the Bank Notes in uploaded images.

## Getting Started

### Prerequisites
- Flask
- gunicorn
- tensorflow
- numpy
- pillow
- flask_cors

### Installation
1. Clone the repository: `https://github.com/ibnuarz/BankNotes_API_TensorFlow.git`
2. Install dependencies: `pip install -r requirements.txt`

### Usage
1. Run the Flask application: `python prebanknote.py`
2. Upload an image file using the provided HTML form or make POST requests to the `/` endpoint.
3. Receive JSON responses with Bank Notes detection.
4. If you run on local you can use local server example `http://127.0.0.1:5000/` but in this case i use endpoint from google cloud

### Important
1. You can use my model or your model to predict (CNN) , here my model `https://drive.google.com/file/d/1y35f6FeXTObu21lehEMqzgCfRE2aUatU/view?usp=sharing`

## License
This project is licensed under the MIT License
