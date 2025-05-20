# AI-Powered Hand Digit Classifier 🤖✍️🔢

An interactive web application that uses a Convolutional Neural Network (CNN) to recognize handwritten digits drawn by a user in real-time. This project was developed as part of the Naan Mudhalvan initiative.

**Deployed Application:** [https://hand-digit-recognition-by-genesis.streamlit.app/](https://hand-digit-recognition-by-genesis.streamlit.app/)

<!-- Placeholder for screenshot/GIF - to be added later -->
<!-- ![Screenshot of the Digit Recognizer App](app_screenshot.png) -->

## 📌 Project Overview

Experience the magic of AI firsthand! This "AI-Powered Hand Digit Classifier" translates the complex world of deep learning into an interactive experience. Using a robust Convolutional Neural Network (CNN), the application instantly recognizes handwritten digits (0-9) drawn on a canvas, displaying predictions and probability insights via a user-friendly Streamlit interface.

It’s designed to be more than just a classifier — it’s an engaging and educational tool that makes understanding machine vision intuitive and fun.

## 🚀 Key Features

*   ✅ **Interactive Drawing Canvas:** Freely draw digits (0-9) with adjustable stroke width on a digital canvas.
*   🧠 **Real-Time CNN Prediction:** Instantly see the classification result from our robust, pre-trained Convolutional Neural Network.
*   📊 **Probability Insights:** Understand the model's confidence with a clear visual breakdown of prediction probabilities for each digit, powered by Altair charts.
*   📱 **Mobile-Responsive Design:** Enjoy a smooth experience on mobile, with a friendly prompt for optimal viewing in landscape mode.
*   🌐 **Live & Accessible:** Deployed on Streamlit Community Cloud for easy access and real-time interaction anywhere.

## 🧠 Model Details & Preprocessing

This project utilizes a Convolutional Neural Network (CNN) for digit classification, with specific preprocessing for real-time user input.

### CNN Architecture

The CNN, built with TensorFlow/Keras, has the following architecture:

1.  **Input Layer:** Accepts 28x28 grayscale images (shape: `(28, 28, 1)`).
2.  **Conv2D Layer 1:** 32 filters, kernel size `(3, 3)`, ReLU activation.
3.  **Conv2D Layer 2:** 64 filters, kernel size `(3, 3)`, ReLU activation.
4.  **MaxPooling2D Layer:** Pool size `(2, 2)`.
5.  **Dropout Layer:** Rate of `0.25` (for regularization).
6.  **Flatten Layer:** Converts 2D feature maps to a 1D vector.
7.  **Dense Layer (Hidden):** 128 neurons, ReLU activation.
8.  **Dropout Layer:** Rate of `0.5`.
9.  **Output Layer (Dense):** 10 neurons (for digits 0-9), Softmax activation (for probability distribution).

### Training & Performance

*   **Dataset:** Trained on the standard MNIST dataset.
*   **Callbacks:** Utilized `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` for robust training and to prevent overfitting.
*   🧪 **Achieved Test Accuracy:** Approximately **99.36%** on the MNIST test set.

*For detailed training logs and comprehensive evaluation metrics (confusion matrix, classification report), please refer to the `nexus_trainer.ipynb` notebook.*

### Preprocessing for User-Drawn Digits

A critical component for the interactive application (`nexus_guesser.py`) is the preprocessing pipeline applied to digits drawn on the canvas. This pipeline transforms raw user input to closely match the characteristics of the MNIST training data:

1.  **Bounding Box Cropping:** Isolates the actual digit from empty canvas space.
2.  **Aspect Ratio Preserving Resize:** Scales the cropped digit to fit within a smaller internal dimension (e.g., 16x16).
3.  **Centering:** Pastes the resized digit onto the center of a new 28x28 black canvas.
4.  **Color Inversion:** Converts drawings (typically black on white) to white digit on a black background, matching MNIST.
5.  **Normalization:** Scales pixel values to the [0, 1] range.

This multi-step preprocessing is vital for the model to accurately classify varied user inputs.
## 🛠️ Tech Stack

This project leverages a combination of powerful Python libraries and frameworks:

*   **Core Language:**
    *   🐍 **Python 3.11**

*   **Deep Learning & Numerical Computation:**
    *   🧠 **TensorFlow:** The primary deep learning framework used for building and training the CNN.
    *   🧱 **Keras:** High-level API within TensorFlow for defining and managing neural network architectures.
    *   🔢 **NumPy:** Essential for numerical operations, especially array manipulation for image data.

*   **Web Application & User Interface:**
    *   🎈 **Streamlit:** Used to create and deploy the interactive web application.
    *   ✍️ **`streamlit-drawable-canvas`:** A Streamlit component enabling the drawing canvas for user input.

*   **Data Handling & Image Processing:**
    *   🐼 **Pandas:** Utilized for structuring data for the Altair probability chart.
    *   🖼️ **Pillow (PIL):** Crucial for image manipulation tasks in the preprocessing pipeline for user-drawn digits (resizing, color conversion, etc.).

*   **Data Visualization:**
    *   📊 **Altair:** For creating the interactive bar chart displaying prediction probabilities.
    *   📉 **Matplotlib & Seaborn:** Used in the Jupyter Notebook (`nexus_trainer.ipynb`) for plotting training history, confusion matrices, and sample images.

*   **Development Environment:**
    *   📓 **Jupyter Notebook:** For model development, training, and experimentation.
    *   💻 **VS Code (or your preferred IDE):** For developing the Streamlit application script.

## 🚀 Getting Started

Follow these instructions to set up and run the project locally on your machine.

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python:** Version 3.9 or newer (the project was developed with Python 3.11).
*   **pip:** Python package installer (usually comes with Python).
*   **Git:** For cloning the repository from GitHub.

### Installation & Setup

1.  **Clone the Repository:**
    Open your terminal or command prompt and clone this project:
    ```bash
    git clone https://github.com/Abinanthan-CG/Hand-Digit-Recognition.git
    cd Hand-Digit-Recognition
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.
    ```bash
    # Create a virtual environment (you can name it 'venv' or anything you like)
    python -m venv venv
    ```
    Activate the virtual environment:
    *   On Windows (Command Prompt/PowerShell):
        ```cmd
        venv\Scripts\activate
        ```
    *   On macOS/Linux (bash/zsh):
        ```bash
        source venv/bin/activate
        ```
    Your terminal prompt should change to indicate the virtual environment is active (e.g., `(venv)`).

3.  **Install Dependencies:**
    With your virtual environment activated, install all the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    This command will download and install all the libraries listed in `requirements.txt`, such as TensorFlow, Streamlit, NumPy, etc., at their specified versions.

---

**Why these additions?**

*   **Prerequisites:** It's helpful to explicitly state what users need *before* they even start cloning (Python, pip, Git).
*   **Cloning Step:** Essential for anyone new to the project.
*   **Virtual Environment:** Strongly encouraging a virtual environment is a best practice in Python development. It prevents dependency hell.
*   **Clear Instructions:** Numbered steps make it easy to follow.
*   **Context for `pip install`:** Placing the `pip install -r requirements.txt` command within the context of an activated virtual environment is key.

This section now provides comprehensive guidance for someone to get your project up and running locally.

### 🖥️ Running the Application

Once you have installed the dependencies and ensured the trained model file is present:

1.  **Navigate to the Project Directory:**
    Ensure your terminal is still in the `Hand-Digit-Recognition` directory where `nexus_guesser.py` is located.

2.  **Activate Virtual Environment (if not already active):**
    *   Windows: `venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`

3.  **Launch the Streamlit Web Application:**
    ```bash
    streamlit run nexus_guesser.py
    ```
    This command will start a local web server, and the application should automatically open in your default web browser. You can then interact with the drawing canvas and see the digit predictions.

### 🧠 Retraining the Model (Optional)

The repository includes a pre-trained model (`models/mnist_cnn_model.h5`) that the Streamlit application uses by default. However, if you wish to experiment with the model architecture, training parameters, or retrain it from scratch:

1.  **Ensure all dependencies from `requirements.txt` are installed**, including those specific to training like `matplotlib` and `seaborn`.
2.  **Open and run the Jupyter Notebook:** `nexus_trainer.ipynb`.
    *   This notebook contains all the code for data loading, preprocessing, model definition, training with callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau), and evaluation.
    *   Running all cells in this notebook will train the model and save the best version as `models/mnist_cnn_best_model.h5`, which is then copied/renamed to `models/mnist_cnn_model.h5` for the Streamlit app to use.

## 🚀 Future Scope

While this application successfully demonstrates real-time handwritten digit recognition, there are several exciting avenues for future development and enhancement:

1.  **Enhanced Model Robustness & Capabilities:**
    *   **Diverse Datasets:** Train the CNN on more varied and challenging datasets beyond MNIST (e.g., EMNIST for letters and digits, or custom datasets from real-world scanned documents) to improve its generalization to a wider array of handwriting styles and noise.
    *   **Advanced Data Augmentation:** Implement more sophisticated data augmentation techniques during training (like elastic distortions or perspective transforms) to further bolster model resilience.

2.  **Expanded Recognition Features:**
    *   **Multi-Digit Recognition:** Develop capabilities to recognize sequences of multiple digits drawn on the canvas or present within an uploaded image. This would involve adding a character segmentation step before individual digit classification.
    *   **Alphanumeric Support:** Extend the model's recognition abilities to include handwritten alphabetic characters (A-Z, a-z), transforming it into a more comprehensive alphanumeric recognizer.
    *   **Image Upload Functionality:** Allow users to upload existing image files containing handwritten digits for classification, in addition to the live drawing canvas.

3.  **Advanced Model Exploration:**
    *   **State-of-the-Art Architectures:** Experiment with more complex and modern CNN architectures (e.g., variants of ResNet, EfficientNet) or incorporate attention mechanisms to potentially achieve higher accuracy and better feature extraction.
    *   **Model Optimization for Edge Devices:** Explore techniques like model quantization or pruning (e.g., using TensorFlow Lite) to create smaller, faster models suitable for deployment on resource-constrained environments like mobile devices.

4.  **Improved User Interaction & Learning:**
    *   **Feedback Mechanism:** Implement a feature allowing users to provide feedback on incorrect predictions. This data could potentially be collected for active learning and future model retraining cycles.
    *   **Enhanced Drawing Tools:** Offer users more control over the drawing experience, such as different stroke thicknesses, colors, or more sophisticated erasing options.

These potential improvements could significantly expand the application's utility, accuracy, and user engagement, pushing the boundaries of this digit recognition project.

## 🧑‍💻 Author

This project was created and developed by **Abinanthan D**. Connect with me:

<a href="https://github.com/Abinanthan-CG" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Profile" />
</a>
<a href="https://www.linkedin.com/in/abinanthan-d-6a157529a/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Profile" />
</a>

## 🙏 Acknowledgements

This project was made possible by and builds upon the work of many incredible open-source communities and resources. Special thanks to:

*   **The Naan Mudhalvan Initiative (Government of Tamil Nadu):** For providing the platform and encouragement to undertake this project.
*   **TensorFlow & Keras Team:** For developing and maintaining powerful and accessible deep learning libraries.
*   **Streamlit Team:** For creating an intuitive framework that makes building interactive data applications straightforward.
*   **The Creators of the MNIST Dataset:** For providing a foundational dataset that has propelled research in machine learning and computer vision for decades.
*   **The developers of `streamlit-drawable-canvas`, `Altair`, `Pillow`, `NumPy`, `Pandas`, and other open-source Python libraries** used in this project.
*   *(Optional: Any specific tutorials, mentors, or individuals who provided significant help).*

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Abinanthan-CG/Hand-Digit-Recognition/blob/main/LICENSE) file for details.
