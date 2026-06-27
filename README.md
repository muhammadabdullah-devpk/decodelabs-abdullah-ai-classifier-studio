# Abdullah AI — Data Classification Studio 🚀
> A Premium, Real-Time Interactive Machine Learning Playground for Supervised Classification and Decision Boundary Visualization.

---

## 📌 Project Overview
**Abdullah AI Classifier Studio** is a professional, client-side standalone (with optional Express/SQLite backend) machine learning dashboard. It is designed to bridge the gap between machine learning theory and practical intuition. 

Through a visually stunning glassmorphism dashboard, users can upload custom CSV datasets, adjust split configurations, tune hyperparameters, train four classic classification algorithms, and instantly view the plotted coordinates along with their mathematical **decision boundaries** and **ROC curves**.

---

## 🎨 Design System & Visual Aesthetics
*   **Modern Dark Theme:** Styled with custom space-cadet blues, deep indigo backgrounds, and neon cyberpunk accents.
*   **Glassmorphism Layout:** Custom blurred frosted card backdrops with fine border glows.
*   **Micro-Animations:** Fluid transitions on input focus, sliders, and button hover states.
*   **Console Logging Drawer:** Retro terminal emulator showing real-time computation logs (e.g. `Trained KNN in 1.45ms`).

---

## 🌟 Key Features

### 1. Interactive Data Studio
*   **Classic Datasets:** Embedded Iris Flower dataset, Synthetic Linear separation, interleaving Moons, and nested circular rings.
*   **Custom CSV Parser:** Drop a CSV containing feature columns and binary labels (`0` or `1`). Auto-detects delimiters (comma or semicolon) and strips trailing white space/quotes.
*   **Statistical Table Summary:** Live row counters, feature list, and target category ratio.

### 2. Algorithmic Control Center
Tune parameters in real time and see how they shape the decision logic:
*   **k-Nearest Neighbors (k-NN):** Configure neighbor count (`k`) and distance metrics (Euclidean, Manhattan).
*   **Decision Tree:** Configure tree splitting metrics, maximum split depth, and minimum split samples.
*   **Logistic Regression:** Adjust learning rate and training epochs to watch Gradient Descent converge.
*   **Gaussian Naive Bayes:** Dynamic classification using Gaussian probability distribution parameters.

### 3. High-Performance Visualization Engine
*   **Decision Boundary Grid:** Samples coordinates on a $60 \times 60$ resolution grid, runs inferences, and shades the background to visualize the model's split decisions.
*   **Performance Metrics:** Real-time Accuracy, F1-Score, Precision, and Recall.
*   **Interactive Confusion Matrix:** Visual grid showing True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
*   **ROC/AUC Graph:** Computes classification threshold sweeps to plot the ROC curve and measure Area Under the Curve (AUC).

### 4. Predictor Sandbox
*   Enter custom feature coordinate values and get instant classification results with confidence percentages.

### 5. Persistent Local Catalog
*   Saves trained model configurations (accuracy, parameters, algorithm, date) directly inside the browser's database (`localStorage`) to compare runs.

---

## 🛠 Technology Stack

### Frontend Client
*   **Core:** HTML5 Semantic Markup, Vanilla JavaScript (ES6+).
*   **Styling:** Vanilla CSS3 (Custom Variables, Flexbox/Grid layouts, Glassmorphism gradients).
*   **Graphics:** HTML5 Canvas API (High-performance rendering).

### Backend Server (Optional Persistence)
*   **Runtime:** Node.js, Express.js.
*   **Database:** SQLite (`better-sqlite3` database engine with WAL logging).
*   **Security:** JWT Token authorization, Password hashing (`bcryptjs`), CORS.

### DevOps & Containerization
*   **Docker:** Multi-stage builder containing compilation dependencies for SQLite native modules.
*   **Docker Compose:** Volume mounts and port mapping setup.

---

## ⚡ Quick Start & Run Guide

### Option 1: Standalone Client Mode (Python Server) — *Recommended*
Since Python is installed on your computer, you can run the application instantly with **zero dependencies**:
1.  Open PowerShell in the project directory.
2.  Start the Python local server:
    ```bash
    python -m http.server 3000 --directory "public"
    ```
3.  Open your browser and navigate to: **[http://localhost:3000](http://localhost:3000)**.
4.  **Note:** Make sure to refresh with `Ctrl + F5` to bypass browser caching.

---

### Option 2: Full-Stack Mode (Node.js & SQLite Backend)
To run with active database services and user authentication:
1.  Ensure Node.js is installed.
2.  Install dependencies:
    ```bash
    cd server
    npm install
    ```
3.  Start server:
    ```bash
    npm start
    ```
4.  Navigate to: **[http://localhost:3000](http://localhost:3000)**.

---

### Option 3: Container Deployment (Docker)
To run inside isolated containers:
1.  Ensure Docker and Docker Compose are installed.
2.  Build and run containers:
    ```bash
    docker-compose up --build -d
    ```
3.  Navigate to: **[http://localhost:3000](http://localhost:3000)**.

---

## 📐 Mathematical Overview of Implemented Algorithms

### 1. k-Nearest Neighbors (k-NN)
Computes distance metrics from testing coordinate $\mathbf{x}$ to training vectors $\mathbf{x}_i$:
$$\text{Euclidean Distance: } d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^d (x_j - x_{i,j})^2}$$
$$\text{Manhattan Distance: } d(\mathbf{x}, \mathbf{x}_i) = \sum_{j=1}^d |x_j - x_{i,j}|$$
Selects class label based on the majority vote of the top $k$ nearest neighbors.

### 2. Decision Tree Classifier
Splits nodes by calculating Information Gain via Entropy reduction. Entropy of a dataset $D$:
$$H(D) = -\sum_{c \in C} p(c) \log_2 p(c)$$
Information Gain of a split on feature $A$ at threshold $T$:
$$IG(D, A) = H(D) - \sum_{v \in \{\text{left}, \text{right}\}} \frac{|D_v|}{|D|} H(D_v)$$
Recursively builds split nodes until reaching `maxDepth` or `minSamplesSplit`.

### 3. Logistic Regression
Updates parameter weights $\mathbf{w}$ and bias $b$ using gradient descent. Sigmoid probability:
$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
Optimized using Stochastic Gradient Descent (SGD) under Cross-Entropy Loss:
$$\text{Loss} = -y \log(\hat{y}) - (1-y)\log(1-\hat{y})$$

### 4. Gaussian Naive Bayes
Calculates class priors $P(c)$ and applies Bayes theorem using the Gaussian Probability Density Function:
$$P(x_i | c) = \frac{1}{\sqrt{2\pi\sigma_c^2}} \exp\left( -\frac{(x_i - \mu_c)^2}{2\sigma_c^2} \right)$$
$$P(c | \mathbf{x}) \propto P(c) \prod_{i=1}^d P(x_i | c)$$

---

## 👥 Developer Profile — Muhammad Abdullah
*   **Name:** Muhammad Abdullah
*   **Role:** AI Engineer | Machine Learning Engineer | LLM Engineer | Full Stack AI Developer | CS Student
*   **Location:** Lahore, Pakistan
*   **Email:** [meharabdullah4337@gmail.com](mailto:meharabdullah4337@gmail.com)
*   **GitHub:** [github.com/muhammadabdullah-devpk](https://github.com/muhammadabdullah-devpk)
*   **LinkedIn:** [linkedin.com/in/muhammad-abdullah-devpk](https://linkedin.com/in/muhammad-abdullah-devpk)

### 🚀 About Me
I am a passionate Computer Science student focused on Artificial Intelligence, Machine Learning, Large Language Models (LLMs), NLP, and Full Stack AI Development. I enjoy building real-world AI applications rather than just completing academic assignments. My goal is to become a world-class AI Engineer capable of designing scalable, production-ready intelligent systems that solve real business problems.

### 💻 Key Tech Stack
*   **Languages:** Python (Advanced), C++, SQL, JavaScript, HTML5/CSS3.
*   **AI & ML:** Machine Learning, NLP, Prompt Engineering, RAG Pipelines, AI Agents & Chatbots, Scikit-learn, Pandas, NumPy, Model Evaluation.
*   **Web & DB:** Flask, Django, REST APIs, JWT, SQLite, MySQL.
*   **DevOps & Tools:** Git, GitHub, Docker, Linux, VS Code, Colab, Postman.

### 🎯 Career Goals
Actively working towards specializing as a:
*   AI / Machine Learning Engineer
*   LLM / Generative AI Engineer
*   Full Stack AI Developer

---

## 📄 License
This project is licensed under the MIT License.

