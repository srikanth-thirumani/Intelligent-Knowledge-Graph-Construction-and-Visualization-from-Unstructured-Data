<h1 align="center">Knowledge Graph Analysis Project</h1>
<h3 align="center">An intelligent text analysis and visualization tool powered by NLP and Gemini AI</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Flask-2.0+-green.svg" alt="Flask Version">
  <img src="https://img.shields.io/badge/spaCy-3.0+-orange.svg" alt="spaCy Version">
  <img src="https://img.shields.io/badge/Google-Gemini_AI-red.svg" alt="Gemini AI">
</p>

## 🌟 Features

### 📥 Multiple Input Methods
- 📝 Direct text input
- 📄 PDF document upload
- 📋 Word document upload
- 🔗 URL content extraction
- 📺 YouTube video transcript analysis

### 🔍 Text Analysis
- 🏷️ Named Entity Recognition (NER)
- 🔀 Relationship extraction using Google's Gemini AI
- 📊 Topic modeling
- 😊 Sentiment analysis
- 📝 Text summarization

### 📊 Visualizations
- 🕸️ Interactive knowledge graph
- ☁️ Word cloud
- 📈 Word frequency charts
- 📉 Sentiment analysis charts
- 🔍 Topic modeling results
- 🌲 Decision tree visualization
- ⏱️ Timeline extraction (for dates in content)

### 📑 Report Generation
- 📊 Comprehensive PDF reports with visualizations
- 💡 Explanations and insights derived from the knowledge graph
- 📋 Entity and relationship tables

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Pip package manager
- Google Gemini API key

### Installation

1. Clone the repository
```bash
git clone https://github.com/srikanth-thirumani/knowledge-graph-analysis.git
cd knowledge-graph-analysis
```

2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Run the application
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## 💻 Usage

1. Input your Google Gemini API key
2. Select your input method (text, PDF, document, URL, or YouTube)
3. Submit your content
4. Explore the generated knowledge graph, visualizations, and insights
5. Download the comprehensive PDF report

## 🛠️ Technologies Used

- **Backend**: Flask, Python
- **NLP**: spaCy, TextBlob, Gensim
- **AI**: Google Generative AI (Gemini)
- **Visualization**: NetworkX, Matplotlib, Plotly, Seaborn
- **Document Processing**: PyPDF2, python-docx, BeautifulSoup
- **Report Generation**: ReportLab

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/a746293a-19fa-4a63-9a3e-a88cb98686a8)


## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/knowledge-graph-analysis/issues).

## 📝 License

This project is [MIT](LICENSE) licensed.

## 🙏 Acknowledgments

- Google for providing the Gemini AI API
- The spaCy team for their excellent NLP library
- All the open-source libraries that made this project possible
