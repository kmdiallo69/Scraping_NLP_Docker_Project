# 🛡️ ToxiGuard French Tweets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.78.0-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

A complete machine learning pipeline for detecting toxic content in French tweets. This project combines web scraping, natural language processing, and API deployment to create a robust toxicity classification system.

## 🎯 Features

- **🔍 Smart Web Scraping**: Automated Twitter data collection using Selenium
- **🤖 Auto-Labeling**: Leverages Detoxify for intelligent data annotation
- **📊 ML Pipeline**: SVM classifier with TF-IDF vectorization
- **🚀 API Deployment**: RESTful API built with FastAPI
- **🐳 Docker Ready**: Containerized for easy deployment
- **🌐 Multi-language Support**: Optimized for French content

## 🏗️ Project Structure

```
ToxiGuard-French-Tweets/
├── 📁 data/                    # Dataset storage
├── 📁 fastapi/                # API application
│   ├── 📁 model/              # Trained models
│   ├── 🐳 Dockerfile          # Container configuration
│   ├── 🚀 main.py             # FastAPI application
│   └── 📋 requirements.txt    # API dependencies
├── 🔧 helpers.py              # Data processing utilities
├── 🤖 model.py                # ML model training
├── 🕷️ scraping.py             # Twitter scraping logic
├── 📋 requirements.txt        # Main dependencies
└── 📖 README.md               # This file
```

## ⚡ Quick Start

### 📋 Prerequisites

- Python 3.8+
- Chrome browser (for Selenium)
- Docker (optional, for containerized deployment)

### 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ToxiGuard-French-Tweets.git
   cd ToxiGuard-French-Tweets
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Chrome WebDriver**
   The script automatically downloads ChromeDriver using `webdriver-manager`.

### 🏃‍♂️ Running the Pipeline

#### 1. Data Collection & Model Training

```bash
python scraping.py
```

This script will:
- Prompt you to customize keywords (or use defaults)
- Collect 500 tweets per keyword
- Clean and process the data
- Auto-label using Detoxify
- Train an SVM classifier
- Save the model for API use

#### 2. API Deployment

**Option A: Local Development**
```bash
cd fastapi
python main.py
```

**Option B: Docker Deployment**
```bash
cd fastapi
docker build -t toxiguard-api .
docker run -d -p 8000:80 --name toxiguard-container toxiguard-api
```

### 🌐 API Usage

Once running, access:
- **API Documentation**: http://localhost:8000/docs
- **Interactive Testing**: http://localhost:8000/docs

**Example API Call:**
```bash
curl -X GET "http://localhost:8000/predict/Votre%20texte%20ici"
```

**Response:**
```json
{
  "text": "Votre texte ici",
  "toxicity": "Non-Toxique"
}
```

## 🔬 Technical Details

### 🕷️ Web Scraping
- **Framework**: Selenium WebDriver
- **Target**: Twitter French content
- **Features**: Auto-scroll, duplicate detection, rate limiting

### 🤖 Machine Learning
- **Labeling**: Detoxify multilingual model
- **Vectorization**: TF-IDF (1-4 n-grams)
- **Classifier**: Support Vector Machine (SVM)
- **Optimization**: GridSearchCV for hyperparameters

### 🌐 API Framework
- **Backend**: FastAPI
- **Serialization**: Pydantic models
- **Documentation**: Automatic OpenAPI/Swagger
- **Performance**: Async support

## 📊 Model Performance

The SVM classifier achieves competitive performance on French toxicity detection:
- **Vectorization**: TF-IDF with 1-4 n-grams
- **Optimization**: Grid search with cross-validation
- **Classes**: Toxic / Non-Toxic binary classification

## 🛠️ Configuration

### Customizing Keywords
Edit the `words` list in `scraping.py` or use the interactive prompt:

```python
words = ['your', 'custom', 'keywords', 'here']
```

### Adjusting Tweet Count
Modify `number_tweets` in `scraping.py`:

```python
number_tweets = 1000  # Collect 1000 tweets per keyword
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Important Notes

- **Rate Limiting**: Twitter scraping respects platform limits
- **Data Privacy**: Collected data is processed locally
- **Language Support**: Optimized for French content
- **Model Updates**: Retrain periodically for best performance

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Detoxify](https://github.com/unitaryai/detoxify) for multilingual toxicity detection
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Selenium](https://selenium-python.readthedocs.io/) for web automation

## 📞 Support

If you encounter any issues or have questions:
- Create an [Issue](https://github.com/yourusername/ToxiGuard-French-Tweets/issues)
- Check existing documentation
- Review the API docs at `/docs`

---

**⭐ If you find this project useful, please give it a star!**
