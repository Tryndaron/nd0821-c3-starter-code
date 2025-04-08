# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Aaron Roggenland developed this model in APril 2025. This Modell is a logistic Regression model. Parameters: max_iter=1000 and Random_State = 42
## Intended Use
This Modell is intended to be used to predict, if a person has a salary above or below 50k. 

## Training Data
The data the Modell got trained on is the census Dataset from the machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income). Whitespaces got removed from the Dataset.
The data got split into 80/20 train and test split. Categorical features wehre encoded with a OneHotEncode. The label is encoded with a LabelBinarizer.


## Evaluation Data
For evaluation the test dataset, which consist of 20% of the entire dataset, got used.


## Metrics
_Please include the metrics used and your model's performance on those metrics._
THe Metrics used are Precision, Recall and Fbeta. The performance can be seen in src/model/slice_output.txt 

## Ethical Considerations
There are many features like sex, native-country and race, that should not be used in cases, where equal oppurtunity is a big issue.

## Caveats and Recommendations
unbalanced dataset