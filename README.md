# Coursera AI for Medicine Specialization (offered by deeplearning.ai)

Programming assignments, labs and quizzes from all courses in the Coursera [AI for Medicine Specialization](https://www.coursera.org/specializations/ai-for-medicine) offered by `deeplearning.ai`.

**Instructors**: `Pranav Rajpurkar`, `Bora Uyumazturk`, `Amirhossein Kiani` and `Eddy Shyu`.

## Specialization Info

AI is transforming the practice of medicine. It’s helping doctors diagnose patients more accurately, make predictions about patients’ future health, and recommend better treatments. In this Specialization, you’ll gain practical experience applying machine learning to concrete problems in medicine. You’ll learn how to:

- Diagnose diseases from X-Rays and 3D MRI brain images
- Predict patient survival rates more accurately using tree-based models
- Estimate treatment effects on patients using data from randomized trials
- Automate the task of labeling medical datasets using natural language processing

## Credits

This repo contains my work for this specialization. All the code base, quiz questions, screenshot, and images, are taken from, unless specified, [AI for Medicine Specialization](https://www.coursera.org/specializations/ai-for-medicine).

## Programming Assignments

### Course 1: [AI for Medical Diagnosis](https://www.coursera.org/learn/ai-for-medical-diagnosis)
  
  - Week 1: [Chest X-Ray Medical Diagnosis with Deep Learning](AI%20for%20Medical%20Diagnosis/Week%201/C1W1_A1_Chest%20X-Ray%20Medical%20Diagnosis%20with%20Deep%20Learning.ipynb)
  - Week 2: [Evaluation of Diagnostic Models](AI%20for%20Medical%20Diagnosis/Week%202/C1W2_A1_Evaluation%20of%20Diagnostic%20Models.ipynb)
  - Week 3: [Brain Tumor Auto-Segmentation for Magnetic Resonance Imaging (MRI)](AI%20for%20Medical%20Diagnosis/Week%203/C1M3_A1_Brain%20Tumor%20Auto-Segmentation%20for%20Magnetic%20Resonance%20Imaging.ipynb)

### Course 2: [AI for Medical Prognosis](https://www.coursera.org/learn/ai-for-medical-prognosis)
  
  - Week 1: [Diagnosing Diseases using Linear Risk Models](AI%20for%20Medical%20Prognosis/Week%201/C2W1_A1_Build%20and%20Evaluate%20a%20Linear%20Risk%20model.ipynb)
  - Week 2: [Risk Models Using Machine Learning](AI%20for%20Medical%20Prognosis/Week%202/C2W2_A1_Risk%20Models%20Using%20Tree-based%20Models.ipynb)
  - Week 3: [Non-Parametric Estimators for Survival Analysis](AI%20for%20Medical%20Prognosis/Week%203/C2W3_A1_Survival%20Estimates%20that%20Varies%20with%20Time.ipynb)
  - Week 4: [Cox Proportional Hazards and Random Survival Forests](AI%20for%20Medical%20Prognosis/Week%204/C2W4_A1_Cox%20Proportional%20Hazards%20and%20Random%20Survival%20Forests.ipynb)

### Course 3: [AI For Medical Treatment](https://www.coursera.org/learn/ai-for-medical-treatment)
  - Week 1: [Estimating Treatment Effect Using Machine Learning](AI%20For%20Medical%20Treatment/Week%201/C3W1_A1_Estimating%20Treatment%20Effect%20Using%20Machine%20Learning.ipynb)
  - Week 2: [Natural Language Entity Extraction](AI%20For%20Medical%20Treatment/Week%202/C3W2_A1_Natural%20Language%20Entity%20Extraction.ipynb)
  - Week 3: [ML Interpretation](AI%20For%20Medical%20Treatment/Week%203/C3W3_A1_ML%20Interpretation.ipynb)

## Syllabus

### Course 1: AI For Medical Diagnosis

How can AI be applied to medical imaging to diagnose diseases? In this first course, you’ll learn about the nuances of working with both 2D and 3D medical image data, for multi-class classification and image segmentation. You’ll then apply what you’ve learned to classify diseases in x-ray images and segment tumors in 3D MRI brain images. Finally, you’ll learn how to properly evaluate the performance of your models.

#### Week 1:
- Introduction: A conversation with Andrew Ng
- Diagnosis examples
- Model training on chest x-rays
- Training, prediction, and loss
- Class imbalance
- Binary cross entropy loss function
- Resampling methods
- Multi-task loss
- Transfer learning and data augmentation
- Model testing

#### Week 2:
- Introduction: A conversation with Andrew Ng
- Evaluation metrics
- Accuracy in terms of conditional probability
- Sensitivity, specificity, and prevalence
- Confusion matrix
- ROC curve
- Threshold (operating point)
- Confidence intervals
- Width of confidence intervals and sample size
- Using a sample to estimate the population 

#### Week 3:
- Introduction: A conversation with Andrew Ng
- Representing MRI data
- Image registration
- 2D and 3D segmentation
- 3D U-Net
- Data augmentation for segmentation
- Loss function for image segmentation
- Soft dice loss
- External validation
- Retrospective vs. prospective data
- Working with cleaned vs. raw data
- Measuring patient outcomes
- Algorithmic bias
- Model influence on medical decision-making

---

### Course 2: AI For Medical Prognosis

Machine learning is a powerful tool for prognosis, a branch of medicine that specializes in predicting the future health of patients. First, you’ll walk through multiple examples of prognostic tasks. You’ll then use decision trees to model non-linear relationships, which are commonly observed in medical data, and apply them to predicting mortality rates more accurately. Finally, you’ll learn how to handle missing data, a key real-world challenge.

#### Week 1:
- Introduction: A conversation with Andrew Ng
- Examples of prognostic tasks
- Patient profile to risk score
- Risk score for atrial fibrillation
- Liver disease mortality
- Calculate 10-year risk of heart disease
- Risk score computation
- Evaluating prognostic models
- Concordant pairs
- Risk ties
- Permissible pairs
- C-index interpretation

#### Week 2:
- Decision trees for prognosis
- Predicting mortality risk
- Dividing the input space
- Non-linear associations
- Class boundaries of a decision tree
- Random forest
- Ensemble methods
- Survival data
- Problems with dropping incomplete rows
- Dropping incomplete case changes the distribution
- Imputation
- Mean imputation
- Regression imputation

#### Week 3:
- Survival function
- Censoring
- Collecting time data
- Heart attack data
- Estimating the survival function
- Using censored data
- Chain rule of conditional probability
- Derivation
- Calculating probabilities from the data
- Comparing estimates
- Kaplan Meier Estimate
 
#### Week 4:
- Hazard functions
- Survival to hazard
- Cumulative hazard
- Individualized predictions
- Individual vs. baseline hazard
- Smoker vs. non-smoker
- Effect of age on hazard
- Factor risk increase or decrease
- Survival trees
- Nelson Aelen estimator
- Mortality score
- Evaluating survival models
- Permissible pair examples
- Harrell’s concordance index

---

### Course 3: AI For Medical Treatment

Medical treatment may impact patients differently based on their existing health conditions. In this final course, you’ll estimate treatment effects using data from randomized control trials and applying tree-based models. In the second week, you’ll apply machine learning interpretation methods to explain the decision-making of complex machine learning models. In the final week of this course, you’ll use natural language entity extraction and question-answering methods to automate the task of labeling medical datasets.

#### Week 1:
- Treatment effect estimation
- Randomized control trials
- Average risk reductio
- Individualized treatment effect
- T-Learner and S-Learner
- C-for-benefit

#### Week 2:
- Information extraction from medical reports
- Rules-based label extraction
- Text matching
- Negation detection
- Dependency parsing
- Question-Answering with BERT
 
#### Week 3:
- Machine Learning Interpretation
- Interpret CNN models with GradCAM
- Aggregate and Individual feature importance
- Permutation Importance
- Shapley Values
- Interpret random forest models