# Garment Employee Productivity Prediction using Machine Learning through Gaussian-Process-Classification

## Abstract

The garment industry plays a crucial role in the global economy, and the performance of garment workers significantly impacts production efficiency. Employee productivity is a critical factor that determines the overall success of manufacturing facilities. This research focuses on predicting the productivity of garment workers using various machine learning classification algorithms: **Gaussian Process Classification (GPC)**, **Random Forest (RF)**, **Decision Tree**, **K-Nearest Neighbor (KNN)**, **Gaussian Naïve Bayes (GNB)**, and **Support Vector Machine (SVM)**. Additionally, feature reduction techniques such as **Principal Component Analysis (PCA)** and **Probabilistic PCA** are utilized to improve model performance. After evaluating different machine learning models, the best-performing algorithms are **Random Forest** and **LightGBM**, both achieving an accuracy of **0.81**.

---

## Introduction

The garment industry faces various challenges affecting employee productivity, including training, empowerment, teamwork, and the internal operational system within manufacturing facilities. Machine learning algorithms provide an innovative approach to predicting employee productivity, offering insights to help optimize factory operations. This project explores how machine learning models can forecast whether garment workers will meet their production targets and provides an understanding of the factors influencing productivity.

---

## Literature Review

Enhancing employee productivity is a priority for manufacturing industries aiming for high-performance standards. Previous research combined multiple classification algorithms, including **Random Forest (RF)**, **Support Vector Machine (SVM)**, and **Naïve Bayes (NB)**, with ensemble methods like **AdaBoost** and **Bagging**. These studies found that **Random Forest** consistently outperforms other algorithms with high accuracy (up to 98.3%) and low **Root Mean Square Error (RMSE)**, making it an ideal model for predicting employee productivity. Our study expands on this by incorporating a broader set of machine learning classifiers and ensemble learning techniques.

---

## Dataset

The dataset used for this project is the **Garment Employee Productivity** dataset, which contains **1197 instances** and **15 features**. The dataset is publicly available as a zip file at the following URL:
[Garment Employee Productivity Dataset](https://archive.ics.uci.edu/static/public/597/productivity+prediction+of+garment+employees.zip)

### Data Description:
1. **date** - Date in Month-Day-Year format
2. **day** - Day of the week the workers were employed
3. **quarter** - A division of the month into weeks (1-5)
4. **department** - The department the worker belongs to
5. **team_no** - The team number assigned to the worker
6. **no_of_workers** - Number of workers in a team
7. **no_of_style_change** - Number of style changes during production
8. **targeted_productivity** - The production target set for the team
9. **smv** - Standard Minute Value (time allocated for each product)
10. **wip** - Work in progress (unfinished products)
11. **over_time** - Overtime hours worked by the team
12. **incentive** - Monetary incentive given for meeting production goals (in Bangladeshi Taka)
13. **idle_time** - Time spent due to production interruptions
14. **idle_men** - Number of idle workers due to disruptions
15. **actual_productivity** - The actual percentage of product tasks achieved (0-1 scale)

The target variable, **actual_productivity**, is categorized as **0** (did not meet the target) or **1** (met the target).

---

## Classification Models and Performance

The following machine learning models were tested to predict employee productivity:

### Without Boosting:
- **Gaussian Process Classification (GPC)**: 0.71 accuracy
- **Decision Tree**: 0.63 accuracy
- **Logistic Regression**: 0.70 accuracy
- **SVM**: 0.71 accuracy
- **Naïve Bayes**: 0.66 accuracy
- **Random Forest (RF)**: 0.81 accuracy
- **KNN**: 0.68 accuracy

### With Boosting:
- **AdaBoost**: 0.75 accuracy
- **GBDT**: 0.79 accuracy
- **LightGBM**: 0.81 accuracy
- **XGBoost**: 0.80 accuracy

---

## Discussion

This research explores the effectiveness of different machine learning algorithms and ensemble techniques to predict the productivity of garment employees. The findings indicate that **Random Forest** and **LightGBM** consistently achieved the highest accuracy of **0.81**, outperforming other models. While **Random Forest** is a robust model for classification tasks, incorporating boosting techniques like **AdaBoost** and **LightGBM** further enhances predictive performance.

Comparing this study with prior research, our approach stands out by utilizing a broader array of algorithms (including **Gaussian Process Classification**) and ensemble methods. Notably, this study represents a novel combination of **GPC**, **Random Forest**, **SVM**, and other classification models, along with feature reduction techniques, which has not been previously explored in the domain of garment employee productivity prediction.

---

## Installation

To run the code for this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/garment-productivity-prediction.git
   ```

2. Install the required dependencies:
   ```bash
   cd garment-productivity-prediction
   pip install -r requirements.txt
   ```

3. Download the dataset from the UCI repository and extract it in the project directory:
   [Garment Employee Productivity Dataset](https://archive.ics.uci.edu/static/public/597/productivity+prediction+of+garment+employees.zip)

---

## Usage

1. **Preprocess the data**:
   - The `data_preprocessing.py` script handles data cleaning, encoding, and feature selection.

2. **Train the models**:
   - The `train_models.py` script allows training of different classification models (Random Forest, SVM, Decision Tree, etc.) and evaluates their performance.

3. **Evaluate the models**:
   - The `evaluate_models.py` script compares model performance using accuracy and other metrics such as **precision**, **recall**, and **F1-score**.

---

## Future Work

Future enhancements to this project could involve:
- **Hyperparameter tuning**: Further optimizing the machine learning models to achieve even higher performance.
- **Deep Learning**: Exploring deep learning techniques such as neural networks and their potential for predicting productivity.
- **Real-time prediction**: Integrating the model with real-time factory data for on-the-fly productivity predictions.

---

## Contributing

If you'd like to contribute to this project, please fork the repository, make your changes, and submit a pull request. We welcome all suggestions, improvements, and enhancements.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**LinkedIn** [Guassian Application](https://www.linkedin.com/posts/ernest-braimoh-29284b141_github-akindreamgaussian-process-classification-and-other-algorithms-activity-7306304049190981633-4zv4?utm_source=share&utm_medium=member_desktop&rcm=ACoAACJ5f84BSF16YQBlNnzy86sMhIc99PdU8l0)
**Medium** [Guassian Application](https://medium.com/@akindream/sure-heres-a-medium-post-draft-for-your-project-on-predicting-garment-employee-productivity-using-4e8567f15d48)
