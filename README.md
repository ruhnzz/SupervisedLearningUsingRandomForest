# SupervisedLearningUsingRandomForest
statement: Scenario - You are working for an architecture firm, and your task is to build a model that predicts the energy efficiency rating of buildings based on features like wall area, roof area, overall height, etc.

RandomForestRegressor is a machine learning model in the scikit-learn library used for regression tasks.
It is based on the Random Forest algorithm so RandomForestRegressor is the regression version of the Random Forest algorithm. It predicts continuous numeric values, like house prices, temperatures, or energy efficiency ratings. supervised learnign has classification for classes like to predict yes or no regression to predict continous values

pandas – Used for handling data in tables (DataFrames).
numpy – Helps in working with numbers, arrays, and generating random data.
matplotlib.pyplot – For plotting graphs.
seaborn – A statistical plotting library (built on matplotlib), makes plots prettier.
warnings – Lets you suppress warning messages in the output.
train_test_split – Helps split data into training and testing sets.
RandomForestRegressor – Machine Learning model to predict continuous values.
mean_squared_error – A metric to evaluate how far off predictions are from actual values.

warnings.filterwarnings('ignore') => This line hides warning messages that can clutter your output. It’s useful in notebooks or scripts to keep things clean.

np.random.seed(0) => Makes sure the random numbers are reproducible (same each time you run the code).
it’s important to get the same random numbers every time you run the script.
So, by setting a seed, you "lock in" the random behavior

np.random.randint(low, high, size) => gives numbers between them if size = 5 we get 5 randoms numbers between high and low
np.random.randint(low, high, size) => to genrate random integer values
np.random.uniform(low, high, size) => to generate random flaot values

pd.DataFrame(data) – We are converting the dictionary data into a table, and storing that table in a variable called df.

X = df.drop('EnergyEfficiency', axis=1) =>All the input features (independent variables),selecting all columns except EnergyEfficiency
df.drop(labels, axis)
labels: the name(s) of the row(s) or column(s) you want to drop.
axis: this tells pandas whether to drop rows (0) or columns (1)
eg:  df.drop(0, axis=0) removes row with index 0 , df.drop('column_name', axis=1) removes that column

y = df['EnergyEfficiency'] =>y: The target variable (EnergyEfficiency) we want to predict select only that col
X → a 2D array-like structure (also still a pandas DataFrame)
y → a 1D array-like structure (a pandas Series)

sns.pairplot(...) – Creates scatter plots showing how each feature relates to EnergyEfficiency
df	The whole dataset (pandas DataFrame)
x_vars	List of feature columns for the x-axis
y_vars	Target column (EnergyEfficiency) for y-axis
kind	Type of plot (scatter)
height	Size of each subplot vertically (in inches)
aspect	Width/Height ratio (1 = square plots)
sns.pairplot(df) => this will give all teh possible combination but we only require evry input featurre combination with target so we specify x nad y variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_test_split(...): Randomly splits the data into:
X_train, y_train → for training the model.
X_test, y_test → for testing (evaluating) the model.
test_size=0.2: 20% of data goes to testing, 80% to training.
random_state=42: random_state is like a seed value Every time you run the code, you'll get the same train-test split.42 is just a commonly used number (but you can use any number)

model = RandomForestRegressor() => Initializes the model
model.fit(X_train, y_train) =>  Trains the model using the training data (X_train, y_train).

predictions = model.predict(X_test) => Uses the trained model to predict values for the test set
mse = mean_squared_error(y_test, predictions) =>Calculates how far off the predicted values are from the actual values
print(f"Mean Squared Error: {mse}")

evaluation metrics:
classisfication: accuracy precision 
regression: MSE Rsquare MAE

plt.figure(...): Set figure size.
plt.scatter(...): Plot true values vs. predicted values.
xlabel/ylabel/title: Add labels and title.
plt.plot(...): Add a dashed diagonal line for reference (perfect predictions lie on this line).
plt.show(): Display the plot.













