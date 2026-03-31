👇



🚧 Helmet Detection using Deep Learning

📌 Overview



This project builds an end-to-end computer vision system to automatically detect whether a person is wearing a safety helmet. It leverages deep learning to improve workplace safety by enabling scalable and real-time monitoring.



Manual safety checks are often inconsistent and inefficient. This solution automates the process, reducing risk and ensuring compliance in industrial environments such as construction sites and factories.



🎯 Problem Statement



Ensuring that workers consistently wear safety helmets is critical but challenging due to:



Manual monitoring limitations

Human error

Lack of scalability

💡 Solution



A deep learning-based image classification model that classifies images into:



✅ Helmet

❌ No Helmet



The system is designed to:



Work in real-world conditions

Handle variations in lighting and angles

Be easily extendable to real-time deployment

📊 Dataset

Total Images: \~4,125

Classes:

Helmet: 3,161

No Helmet: 964

Data Processing

Image resizing (224x224)

Normalization

Data augmentation:

Rotation

Zoom

Horizontal flip

🧠 Model Architecture

Transfer Learning using MobileNetV2

Frozen base layers for feature extraction

Custom classification head:

Dense layer (ReLU)

Dropout (0.5)

Sigmoid output

🏗️ Architecture Diagram



⚙️ Tech Stack

Python

TensorFlow / Keras

OpenCV

NumPy / Pandas

Matplotlib / Seaborn

📁 Project Structure

helmet-detection/

│

├── data/

│   ├── train/

│   └── val/

│

├── notebooks/

│   └── Version1\_HelmNet\_Full\_Code.ipynb

│

├── src/

│   ├── train.py

│   ├── model.py

│   ├── preprocess.py

│   └── inference.py

│

├── models/

│   └── helmet\_model.h5

│

├── experiments/

│

├── requirements.txt

└── README.md

🚀 Getting Started

1️⃣ Clone Repository

git clone https://github.com/yourusername/helmet-detection.git

cd helmet-detection

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Train Model

python src/train.py

4️⃣ Run Inference

python src/inference.py

🧪 Experiments

Experiment	Model	Augmentation	Accuracy

Baseline	CNN	No	\~80%

Improved	CNN	Yes	\~85%

Final	MobileNetV2	Yes	\~90%+

📈 Results

Achieved \~90% accuracy on validation data

Good generalization across different environments

Low false negatives (important for safety use-case)

📸 Sample Predictions



(Add sample images here for better impact)



🔮 Future Improvements

Real-time video detection (CCTV integration)

Edge deployment (Raspberry Pi / Jetson Nano)

Multi-class PPE detection (helmet, vest, gloves)

Object detection using YOLO

🤝 Contributing



Contributions are welcome! Feel free to fork the repo and submit a pull request.

