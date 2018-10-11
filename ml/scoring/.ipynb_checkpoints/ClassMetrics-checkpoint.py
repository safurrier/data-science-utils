import sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy.stats import chisquare
from functools import reduce
import itertools
import matplotlib.style as style
import matplotlib.pyplot as plt
import seaborn as sns

sklearn.warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
style.use('fivethirtyeight')


def difference_predicted_actual(df, target_col, predict_col):
    """ 
    Return the difference in number of predicted response values and actual values
    + difference indicates model is predicting response value more than the actual --> overpredicting
    - difference indicates model is predicting response value less than the actual --> underpredicting
    """
    response_values_df = find_predicted_and_actual_instances(df, target_col, predict_col)
    response_values_df['difference'] = response_values_df.predicted_count - response_values_df.actual_count 
    return response_values_df.sort_values(by='class')

def chisquare_preds(df, target_col, pred_col):
    """ Run a Chi Square test on the predicted classes vs. actual classes"""
    observed_expected = difference_predicted_actual(df, target_col, pred_col)
    return chisquare(f_obs= observed_expected.predicted_count, f_exp= observed_expected.actual_count)


def multiclass_metrics(df, target_col, predicted_col):
    """ Compute predicted vs. actual counts, accuracy, and other metrics 
    on a classification predictions and return in a single dataframe"""
    # Arguments to pass to functions
    args = [df, target_col, predicted_col]
    # Compute the metrics using these arguments 
    dfs_to_concat = [find_predicted_and_actual_instances(*args), 
                     response_classification_accuracy(*args), 
                     class_model_metrics(*args)]
    # Merge the metrics into one dataframe
    all_metrics_df = reduce(lambda x, y: pd.merge(x, y, on='class'), dfs_to_concat)
    return all_metrics_df
        
def find_predicted_and_actual_instances(df, target_col, predict_col):
    """
    Look through a target column and predicted columns to find the counts of predicted vs actual
    """
    # Unique response values
    response_vals = df[target_col].unique()
    predicted_vals = df[predict_col].value_counts()
    # Missing values in predictions
    missing_vals = [value for value in response_vals if value not in list(predicted_vals.index)]
    # Series to insert with missing values and 0 counts
    missing_val_series = pd.Series(0*len(missing_vals), index=missing_vals)
    # Predicted and actual response value counts
    predicted_instances = predicted_vals.append(missing_val_series)
    predicted_instances.name = 'predicted_count'
    actual_instances = df[target_col].value_counts()
    actual_instances.name = 'actual_count'
    # Dataframe with info
    val_counts_df = (pd.concat([actual_instances, predicted_instances], axis=1)
                     .reset_index()
                     .rename(columns={'index':'class'})
                     .sort_values(by='class')
                    )
    return val_counts_df   

def response_classification_accuracy(df, target_col, predicted_col):
    # Unique values of response
    response_vals = list(df[target_col].unique())
    # 
    response_vals_accuracy = [classification_response_accuracy(df, target_col, predicted_col, val) for val in response_vals] 

    response_vals_accuracy_df = (pd.DataFrame.from_dict(dict(zip(response_vals, response_vals_accuracy)), orient='index')
                                          .reset_index()
                                          .rename(columns={'index':'class', 0:'classification_accuracy'})
                                          .sort_values(by='class')
                                         )
    return response_vals_accuracy_df

def classification_response_accuracy(df, target_col, predict_col, response_val):
    """ Find the classification error for a given response value"""
    # Reduce to df where response value is actual target response
    response_val_df = df[df[target_col] == response_val]
    # Reduce to correct predictions
    correct_guesses = response_val_df[target_col].where(response_val_df[target_col] == response_val_df[predict_col])
    # Check if any correct guesses. 0 if no correct guesses, otherwise acc is number of correct/total instances of company
    try:
        # Accuracy is count of correct guesses divided by total guesses
        accuracy = correct_guesses.value_counts()[0]/len(correct_guesses)
    except KeyError:
        accuracy = 0.0
    return accuracy

def class_model_metrics(df, target_col, predicted_col):
    """ Compute Precision, Recall and F1 Scores for target classes and return metrics in a dataframe"""
    labels = sorted(df[target_col].unique())
    f1 = f1_score(df[target_col], df[predicted_col], average=None)
    precision = precision_score(df[target_col], df[predicted_col], average=None)
    recall = recall_score(df[target_col], df[predicted_col], average=None)
    metrics_dict = {
        'class':labels,
        'f1':f1,
        'precision':precision,
        'recall':recall
    }
    return pd.DataFrame(metrics_dict)

class ClassMetrics(object):
    """
    Take a dataframe with predictions and compute various metrics
    Plot common metrics and analysis
    """
    def __init__(self, df, target_col, predicted_col):
        self.df = df
        self.y_true = df[target_col].values
        self.y_pred = df[predicted_col].values
        self.target_col = target_col
        self.predicted_col = predicted_col
        # Class names sorted alphabetically
        self.class_names = sorted(list(df[target_col].unique()))
        # Multiclass Metrics Df
        #self.metrics_df = multiclass_metrics(self.df, target_col, predicted_col)
        # Max of the max response value counts of predictions and actual
        #self.majority_class_count = self.metrics_df[['actual_count','predicted_count']].max().max()
    
    np.set_printoptions(precision=2)
    
    @property
    def confusion_matrix(self):
        # Confusion matrix from sklearn
        return sklearn.metrics.confusion_matrix(self.y_true, self.y_pred)
    
    @property
    def metrics_df(self):
        """ A dataframe of various classification metrics by class"""
        return multiclass_metrics(self.df, self.target_col, self.predicted_col)
    
    @property
    def _majority_class_count(self):
        """ Count of the majority class count"""
        return self.metrics_df[['actual_count','predicted_count']].max().max()
    
    @property
    def count_comparison(self):
        """ Dataframe of actual class counts compared to predicted class counts"""
        return difference_predicted_actual(self.df, self.target_col, self.predicted_col)
    
    @property
    def chi_square_test(self):
        """ Return the results of runnning a chi square test on the predicted vs. actual class counts"""
        return chisquare_preds(self.df, self.target_col, self.predicted_col)
        
    def plot_confusion_matrix(self,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = self.confusion_matrix
        classes=self.class_names
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        #print(cm)
        plt.figure(figsize=(12,8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.figsize=(12,8)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.show()
        
    def plot_predicted_distribution(self):
        g = sns.countplot(x=self.y_pred, order=self.class_names)
        g.set_xticklabels(g.get_xticklabels(), rotation=30)
        g.set_ylim(0,1.1*self._majority_class_count)
        g.set_title('Predicted Distriubtion of Classes')
        plt.show()
    
    def plot_target_distribution(self):
        g = sns.countplot(x=self.y_true, order=self.class_names)
        g.set_xticklabels(g.get_xticklabels(), rotation=30)
        g.set_ylim(0,1.1*self._majority_class_count)
        g.set_title('Actual Distriubtion of Classes')
        plt.show()
    
    def plot_distributions(self):
        self.plot_predicted_distribution()
        self.plot_target_distribution()
        
    def plot_class_metrics(self, classes='all', metrics='all'):
        """
        Given a data frame with class names and metrics, return a bar chart plotting these

        Metrics: 'classification_accuracy','f1','precision','recall'

        Can specify which classes to chart as well as specific classes
        """
        if classes=='all':
            classes=list(self.metrics_df['class'].unique())
        df = self.metrics_df[self.metrics_df['class'].isin(classes)]
        df.set_index('class', inplace=True)
        df = df.loc[:,  ['classification_accuracy','f1','precision','recall']]
        if metrics != 'all':
            try:
                df = df.loc[:,  metrics]
            except KeyError:
                return print("Metrics must be a list combination of 'classification_accuracy','f1','precision','recall'")
            df.plot(kind='bar', title='Metrics for '+ " ".join(classes), figsize=(12,8), ylim=(0, 1))
        else:
            df.plot(kind='bar', title='Metrics for '+ ", ".join(classes), figsize=(12,8), ylim=(0, 1)) 