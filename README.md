<html>
<body>
  <h1>Heart Failure Prediction</h1>
  <h2>Machine Learning Task</h2>
  <p>
    The machine learning task that I intend to solve for this project is predicting heart failure 
    mortality based on a patient’s medical information such as the presence of certain cardiovascular 
    diseases (CVDs) or some quantifiable test about how well their body is functioning. According 
    to the World Health Organization (WHO), CVDs are the leading cause of death globally 
    accounting for approximately 32% (17.9 million lives) of all deaths worldwide each year. Out of 
    the 17.9 million lives taken by CVD, approximately 85% are a direct result of heart attacks and 
    strokes. Therefore, it would be beneficial to implement an algorithm that can predict if a heart 
    failure event were to occur would that individual survive given certain medical information. 
    These predictions can assist medical practitioners in determining if something can be done about 
    their patient’s health to possibly avoid mortality caused by heart failure.  
    This type of problem would require supervised training because I intend to focus on 
    specific features that were previously determined to be effective in predicting mortality in the 
    event of heart failure. Since there are specific features that I intend to focus on, an unsupervised 
    approach to this problem would not make sense. Aside from supervised training, this problem is 
    a binary classification problem as opposed to a regression problem because I am not trying to 
    predict a continuous value. Instead, I want to be able to classify the samples within the dataset 
    into two classes (death caused by heart failure/heart failure survivor) based on specific medical 
    information. This problem will also require a change of basis because the data is inseparable in 
    lower dimensions.
  </p>
  <h2>Data Utilized</h2>
  <p>
    The data available for my problem was collected by the Faisalabad Institute of 
    Cardiology at the Allied Hospital in Faisalabad located in Punjab, Pakistan. These medical 
    records were collected over the course of a few months from April of 2015 to December of that 
    year. Consisting of 105 female patients and 194 male patients (making the sample total 299), 
    with patient ages ranging from 40 to 95 years old. This information was compiled by Davide 
    Chicco and Giuseppe Jurman for their research on how machine learning can be used to predict a 
    patient’s survival from heart failure just by utilizing two features: serum creatinine and ejection 
    fraction. Although they were arguing that it was possible to predict survival from those two 
    factors the dataset does include other features as well. In total there are 13 features in the dataset 
    and some of them are anemia, high blood pressure, diabetes, sex, smoking, platelet count, etc. 
    This data was provided by a Kaggle user named Larxel with a Creative Commons by 4.0 license. 
    The data was made available in a .csv document format. 
  </p>
  <p>Data URL: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data</p>
  <h2>Approach</h2>
  <p>
    For this project, the model that was used is Regularized Logistic Regression with Cross-Validation. The Logistic Regression model was optimized using the Iterative Re-weighted Least Squares (IRLS) algorithm. More generally the Logistic Regression model was chosen because the goal of the dataset sourced for this project was to predict a patient’s mortality given information such as clinical, health, and lifestyle information. Since the target of the dataset, a feature called death event, was given as binary values the machine learning problem we will be tackling is a binary classification problem. Therefore, Logistic Regression, a binary classification algorithm, was chosen as the primary algorithm that would be implemented. Regularization and cross-validation were added to prevent the Logistic Regression model from overfitting during training. Overfitting will cause the model to have a high training performance, but then perform poorly on data it has not seen previously during training.
  </p>
  <p>*** See Further Discussion in Final Report ***</p>
</body>
</html>
