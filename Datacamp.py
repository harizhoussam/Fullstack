##The unprocessed dataset has been loaded into a DataFrame df. 
#Explore it in the IPython Shell with the .head() method. 
#You will see that there are certain data points labeled with a '?'. 
#These denote missing values. As you saw in the video, different datasets encode missing values in different ways. 
#Sometimes it may be a '9999', other times a 0 - real-world data can be very messy! 
#If you're lucky, the missing values will already be encoded as NaN. 
#We use NaN because it is an efficient and simplified way of internally representing missing data, 
#and it lets us take advantage of pandas methods such as .dropna() and .fillna(), 
#as well as scikit-learn's Imputation transformer Imputer().

#In this exercise, your job is to convert the '?'s to NaNs, and then drop the rows that contain them from the DataFrame. 

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))





# You'll now practice setting up a pipeline with two steps: the imputation step, 
# followed by the instantiation of a classifier. You've seen three classifiers in
 # this course so far: k-NN, logistic regression, and the decision tree. You will
 # now be introduced to a fourth one - the Support Vector Machine, or SVM. For now,
 # do not worry about how it works under the hood. It works exactly as you would 
 # expect of the scikit-learn estimators that you have worked with previously, in 
 # that it has the same .fit() and .predict() methods as before.
 
 # Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', SVC())]     
        

 # Having setup the steps of the pipeline in the previous exercise, you will now
 # use it on the voting dataset to classify a Congressman's party affiliation.
 # What makes pipelines so incredibly useful is the simple interface that they provide.
 # You can use the .fit() and .predict() methods on pipelines just as you did with your
 # classifiers and regressors!


# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))
