data-science-utils
==============================

Various code to aid in data science projects for tasks involving data cleaning, 
ETL, EDA, NLP, viz, feature engineering, feature selection, etc.


Project Organization
------------

    ├── custom_transformers.py             <- Various custom transformers in the form of sklearn TransformerMixins
    │                                                                   For building reproducible ETL pipelines
    │
    ├── data_checks.py                              <- Code to validate various data checks
    │
    ├── pandas_utils.py                             <- Code for a variety of tasks dealing with Pandas DataFrames
    │                                                                   Includes null value profiling, duplicate column checks, associative 
    │                                                                   relationship functions (mutual information, correlations, grouped means)
	│                                                                   quick NaN replacement, quantile binning of columns, categoric or continuous
	│                                                                   column search, etc. 
	│                                                                   
	├── text_utils.py                                    <- Code for dealing with text. Includes distributed loading of text corpus, 
    |                                                                    entity statement extraction, sentiment analysis, etc.	
    │
    ├── feature_engineering                      <- Code to facilitate feature engineering, most often by measuring relationship between 
	|                                                                    data and a target or by manipulating existing data. Includes code for feature value
	|                                                                    target mean, grouped searching by target mean, feature matching for specific values,
	|                                                                    one hot encoding groups of feature values above a certain minimum target mean etc.
	|                                                                    Build interaction terms based on a dictionary of the form {base_feature:[interacting_terms]}
	|                                                                    in conjunction with transformer DFInteractionTerms
    │
    ├── feature_selection                           <- Code to log feature lists, as well extract logged feature lists, interaction terms from a given featurelist
	|                                                                    and rank features given multiple feature lists with features in order of importance. 
    │
    ├── code_gists                                       <- Code gists with commonly used code (change to root directory, connect to database, profile data, etc)
