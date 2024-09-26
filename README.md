# Exploring-Sleep-Quality-Influential-Factors-and-Implications
This project investigates sleep quality by applying advanced machine learning techniques like Principal Component Analysis (PCA), Support Vector Machines (SVM), and Association Rule Mining. It aims to identify key variables affecting sleep quality and disorders, with a focus on actionable insights for healthcare interventions and lifestyle improvements.
### Python, Machine Learning, and Data Visualization

This project investigates sleep quality by applying advanced machine learning techniques like Principal Component Analysis (PCA), Support Vector Machines (SVM), and Association Rule Mining. It aims to identify key variables affecting sleep quality and disorders, with a focus on actionable insights for healthcare interventions and lifestyle improvements.

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Research Methodology](#research-methodology)
- [Key Findings](#key-findings)
- [Results](#results)
- [Dataset](#dataset)
- [Publication](#publication)
- [Contributors](#contributors)
- [License](#license)

## Introduction
Sleep quality plays a crucial role in mental, physical, and emotional well-being. This research uses machine learning algorithms to analyze sleep patterns and identify key influential factors like occupation, stress level, BMI, and heart rate. The findings provide insights into potential lifestyle and medical interventions to improve sleep health.

The project was presented at HINT 24, Manipal Institute of Technology, and has been accepted for publication by Springer (pending final review).

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn` (PCA, SVM)
  - `mlxtend` (Association Rule Mining)
  
- **Machine Learning Techniques**:
  - Principal Component Analysis (PCA)
  - Support Vector Machines (SVM)
  - Histogram-Based Feature Selection
  - Recursive Feature Elimination (RFE)
  - Random Forest Feature Importance
  - Association Rule Mining

## Research Methodology
1. **Data Collection**: The dataset used for this study was sourced from Kaggle and included variables such as sleep duration, BMI, stress level, and physical activity.
2. **Preprocessing**: Data cleaning and feature encoding were applied to make the dataset compatible with machine learning algorithms.
3. **Machine Learning**: Applied PCA for dimensionality reduction and SVM for classification of sleep disorders with an 85% accuracy rate. Association Rule Mining and histogram-based feature selection were used to reveal patterns and critical features.
4. **Statistical Analysis**: Chi-squared and Cramér's V tests were conducted to assess the strength and significance of associations between variables.

## Key Findings
- **Precision Improvement**: The application of PCA and feature selection improved data analysis precision by 30%.
- **Influential Factors**: Key factors like sleep duration, occupation, stress levels, and BMI were found to significantly impact sleep quality and disorders.
- **Actionable Insights**: These findings provide valuable insights for healthcare professionals to develop targeted interventions for improving sleep health.


## Results
The analysis identified the following key findings:
- Positive correlation between sleep duration and quality.
- Stress level and BMI were found to be strong indicators of sleep disorders.
- Occupation had a notable impact on health metrics such as blood pressure and heart rate.

Visualizations of these findings can be found in the `visualizations/` folder.

## Dataset
The dataset used in this research is publicly available on Kaggle:
- [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

## Publication
This research was presented at HINT 24, Manipal Institute of Technology, and has been accepted for publication by Springer, pending final review.

## Contributors
- **Sinchana HR** (Lead Researcher)
- **Shreya C S**
- **Shreya Mannapur**
- **Dr. Abhilash C B** (Advisor)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
