# DEEP LEARNING LAB (0021)

## Problem Statement

**Face Detection System for Identifying Hostellers and Non-Hostellers with Name and Registration Number Classification.**

### TEAM:

1. Trisha Sharma 229309215
2. Anahita Bhandari 229309186

---

## Project Document: Face Detection and Recognition System for Classroom Students

### Objective

The primary objective of this project is to develop a face detection and recognition system that identifies students from a class of 20 individuals. The system will classify students based on their name and registration number, providing an efficient and automated method for managing attendance and monitoring student presence.

---

### Scope

1. **Face Detection:** Utilize advanced algorithms to detect student faces in various settings with high accuracy.
2. **Identification and Classification:** Implement facial recognition technology to identify and classify students based on their name and registration number.
3. **Attendance Management:** Automate the process of logging student attendance and monitoring student presence in real-time.
4. **Database Integration:** Maintain a database to store facial data, names, and registration numbers for real-time and future use.

---

### Methodology

1. **Data Collection:**
    - Collect facial images of all 20 students in diverse conditions (lighting, angles, etc.) to build a robust training dataset.
    - Ensure the dataset is balanced and representative of the various scenarios the system might encounter.
  
2. **Model Development:**
    - **Convolutional Neural Networks (CNNs):** Utilize CNNs for face detection and recognition tasks.
    - **Pre-trained Models:** Leverage pre-trained deep learning models (e.g., **FaceNet**, **VGGFace**, or **ResNet**) to enhance the system’s accuracy and reduce training time.
    - **OpenCV and Dlib:** Use OpenCV for real-time image processing and Dlib for face detection and feature extraction.
  
3. **System Integration:**
    - Deploy the system to automatically capture images or video at class entry points or during attendance sessions.
    - Connect the face recognition system to the existing student information database for automated attendance logging.
  
4. **Testing and Validation:**
    - Perform extensive testing using the dataset to ensure accuracy under various environmental conditions.
    - Validate the system with different students, expressions, lighting, and angles to achieve real-time performance and reliability.

---

### Tech Stack

#### Frontend:

- **React.js** (for user interface development and real-time status visualization).
- **HTML5/CSS3** (for basic layout and design).

#### Backend:

- **Node.js** (for server-side logic, handling API requests, and connecting to the database).
- **Express.js** (for setting up RESTful APIs to manage facial recognition requests and responses).
- **Python 3.x** (for image processing, face recognition, and machine learning model implementation).

#### Machine Learning & Computer Vision:

- **TensorFlow** (for building and training the neural network models).
- **Keras** (for high-level neural network API, making model development faster).
- **OpenCV** (for real-time face detection and image processing).
- **Dlib** (for face detection and feature extraction).
- **FaceNet or VGGFace2** (for facial recognition using pre-trained deep learning models).

#### Database:

- **MongoDB** (for storing student facial data, names, and registration numbers in a NoSQL database).
- **Firebase Firestore** (for real-time database management, enabling quick updates and querying).

#### Cloud & Deployment:

- **Google Cloud Platform (GCP)** (for hosting the machine learning model and database).
- **Docker** (for containerizing the application, ensuring it runs consistently across environments).
- **Heroku** or **AWS EC2** (for deploying the application to the cloud).

#### Version Control & Collaboration:

- **Git** (for version control).
- **GitHub** (for collaboration, issue tracking, and project management).

---

### Key Features

1. **Real-time Face Detection:**
    - Detect student faces in real-time with high accuracy, using optimized deep learning models.
  
2. **Automated Attendance:**
    - Automatically log student attendance upon identification, reducing manual errors.
  
3. **High Accuracy Identification:**
    - Accurately classify students by matching facial data to their name and registration number stored in the database.
  
4. **Scalability:**
    - The system is designed to be scalable, allowing for future expansion to include additional classes, departments, or even the entire campus.

---

### Novelty Points

- **Automated Attendance:** The system provides an automated and accurate alternative to traditional manual roll calls, improving efficiency.
- **Real-Time Monitoring:** It offers real-time monitoring of student presence in the classroom.
- **Expandable:** The system can easily be expanded beyond the current classroom to other parts of the campus or to a larger number of students.

---

### Conclusion

This face detection and recognition system offers a secure, efficient, and automated solution for identifying students and managing classroom attendance. With its real-time processing, it significantly reduces administrative work while providing an accurate and reliable method for monitoring student presence. The system’s scalability also allows for future expansion across different classes or departments, making it a valuable addition to any educational institution.

---

### Future Work

- Integrate with **IoT devices** for additional capabilities such as door access control or security cameras.
- Explore **multimodal biometric systems** that combine face recognition with fingerprint or voice recognition for enhanced security.
- Extend the system to manage not just attendance but also student performance tracking and reporting.

---

### Research Papers Reviewed

- [Face Recognition: A Literature Survey](https://arxiv.org/pdf/0812.02575)
- [A Review of Face Recognition Technology](https://www.researchgate.net/publication/343118558_A_Review_of_Face_Recognition_Technology)
- [A survey on deep learning techniques for image and video semantic segmentation](https://www.sciencedirect.com/science/article/pii/S2590005619300141)
- [Deep Learning for Face Recognition: A Critical Analysis](https://ieeexplore.ieee.org/document/8356995)
- [Face Recognition System: A Comprehensive Study](https://www.ijert.org/research/face-recognition-system-IJERTV8IS050150.pdf)
- [Face Recognition Based Attendance Management System: A Comprehensive Review](https://www.sciencedirect.com/science/article/pii/S1877050923000546)
- [Face Recognition Techniques: A Comprehensive Review](https://www.sciencegate.app/keyword/502229)
- [A Survey on Deep Face Recognition in the Wild: Recent Advances and New Frontiers](https://www.mdpi.com/1424-8220/20/2/342)
- [Face Recognition Methods and Their Applications: A Review](https://link.springer.com/article/10.1007/s10559-024-00655-w)
- [A comprehensive survey on deep learning-based face recognition: From traditional to advanced approaches](https://www.sciencedirect.com/science/article/pii/S2542660524000313)

### Dataset
- [Dataset](https://www.notion.so/Dataset-10cb23a0120e8048949af7f6b39b5c12?pvs=21)

### Workflow Diagram
- [WorkFlow Diagram](https://www.notion.so/WorkFlow-Diagram-10cb23a0120e8088954fccd6831b9dbb?pvs=21)
