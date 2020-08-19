# -*- coding: utf-8 -*-
"""
Exercise2: Review the script below and apply the provided style guide to improve it:
https://google.github.io/styleguide/pyguide.html

Purpose of script: exploratory data analysis of titanic test dataset, imputation, feature engineering
Comparison of feature distribution against test data
source: https://www.kaggle.com/c/titanic/data
"""

import os
#change working directory
os.chdir('/Users/richardleyshon/Documents/DSCA_reproducibility_dev/exercises/activity_2')

import pandas as pd
#adjust console presentation of output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#import test data
titanic_train = pd.read_csv('titanic/train.csv')
#cache unedited df for use in training later
pd.to_pickle(titanic_train, 'cache/titanic_train.pkl')


#dimensions of dataframe
titanic_train.shape
#891 x 12

#summary statistics of numeric columns
titanic_train.describe()
# =============================================================================
# PassengerId, is it unique?
# Survived, can see from quartiles that majority did not survive. Would hope that the test DF would be based on the same distribution.
# Pclass, relatively fewer 1st class passengers. Majority are 3rd class. 
# Age, floating point. Estimated ages include a .5 value. There is a degree of missingness, only 714 observations. What proportion is missing? What proportion is estimated?
# Sibsp, number of siblings or spouses aboard. Mistresses and fiances were ignored. Majority recorded having no siblings. Mean is higher than median, large outliers influenced? Max is 8 siblings.
# Parch, number of parents / children on board. Children travelling with nanny only Parch == 0. Large outlier influence on mean as in SibSp. 
# Fare, minimum is zero. How many? Are these the crew? Do they have a unique pclass and age profile? Mean is double the median, high large outlier influence. Maximum is £512, whopping for 1912.
# =============================================================================

#summary stats for object columns
titanic_train.describe(include = 'object')
# =============================================================================
# Name column has no duplicates, I happen to know the full dataset has 2 non unique values. Either in test df or split. Likely of no consequence for this analysis.
# Sex appears to be mostly male. Look at proportion male:female. 
# ticket ref is object, column is a mix of character / numeric values. Interestingly, 230 values are non-unique - family tickets?
# Cabin number has a high degree of missingness. Only 204 values. This is a shame as it would have been very interesting to analyse survival based on distance from the deck level. Perhaps isolate this group and investigate?
# Embarked - port of embarkation has 3 values. Most common one is S = Southampton.  2 missing values here.
# =============================================================================

'''some questions identified to pursue further. My main concern would be on the method of splitting 
the test dataset. No detail on how this has been done and hopefully not just based on distribution
of survived column alone. 
'''
# explore the pairwise correlations within the data using pearson's correlation coefficient
titanic_train.corr(method ='pearson') 
# some interesting trends observed, Only Parch & Fare indicate a positive correlation with Survival
# neither of which would be indicated as strong (corr of 0.7 or higher)
# all other numeric feautures indicate weak negative correlations against survived
# suggests a lot of noise in observed trends in features against survival. Weak though they are,
# the largest coefficients observed (+ or -) against survived are Pclass & Fare.
# Within the matrix as a whole, Pclass & Fare has the strongest observed coefficient of -0.549500
# Not a strong correlation though (+/-0.7 threshold). A moderate negative correlation. 
# Next up would be SibSp & Parch with a weak positive correlation of 0.414838.

########PassengerId###########
# is it unique?
titanic_train.PassengerId.value_counts(dropna = False)
# appears so.

########Survived###########
# distribution?
titanic_train.Survived.value_counts(normalize = True)
# 62% didn't survive. Will chek this against the test set. 

########Pclass###########
# distribution?
from matplotlib import pyplot as plt
plt.hist(titanic_train.Pclass, bins = 3)
# 3rd class by far the most common pclass. Interesting that there appears to be more 1st than 2nd class.
titanic_train.Pclass.value_counts(normalize  = True) 
# =============================================================================
# 3    0.551066
# 1    0.242424
# 2    0.206510
# =============================================================================

########Age###########
titanic_train.Age.value_counts(dropna = False, normalize = True)
# NaN      0.198653. Nearly 20% missing data. Parch / Sibsp won't offer the required directionality to help inform imputation. 
# what prop are estimated and equal to 1 or older
len(titanic_train.query('Age >= 1.0 & Age % 1 == 0.5')) / len(titanic_train)
# 2 percent of data are estimated age. Combined with #NAs, this makes for a column with significant quality issues. 
# Create a feature that records estimated ages - age_estimated
import numpy as np
titanic_train.insert((titanic_train.columns.get_loc("Age") + 1), "age_estimated", np.where((titanic_train.Age >= 1.0) & (titanic_train.Age % 1 == 0.5), True, False))
# introduce age bins in order to explore diffreential survival rates
# school age from https://en.wikipedia.org/wiki/Raising_of_school_leaving_age_in_England_and_Wales#19th_century
# define labels for the age categories
cut_labels = ['pre_school', 'school', 'adolescent', 'age_of_majority', 'pensioners']
# define boundary limits for the boundaries
cut_bins = [0, 4.5, 13.5, 20.5, 69.5, 80]
# insert an age quantile column  to investigate differential survival
titanic_train.insert((titanic_train.columns.get_loc("Age") + 1), "age_discrete", value = pd.cut(titanic_train.Age, bins=cut_bins, labels=cut_labels))

# tidy up
del cut_bins, cut_labels

#need to investigate differential survival among the age categories, interesting to look at pre school survival rates
preschool = titanic_train.query('age_discrete == "pre_school"')
# eda on this dataset
preschool.corr(method ='pearson')
# interesting that a strong negative correlation is returned for SibSp x Survived -0.714502 . Could this indicate that passengers of smaller families (with preschoolers) were more likely to survive than individuals with larger families?
preschool.Survived.value_counts(normalize = True)
# =============================================================================
# 1    0.675
# 0    0.325
# =============================================================================
len(preschool)
# small number of observations, high correlation observed could be a relic of the data. 
# Not observed correlation within total df. Perhaps school age?
schoolchildren = titanic_train.query('age_discrete == "school"')
schoolchildren.corr(method = 'pearson')
len(schoolchildren)
# again, a small subset. Moderate -ve correlation between SibSp & survived, -0.668927 
# more pronounced influence of pclass x survived here -0.526789 than in preschool -0.325941. Both small subsets however.
schoolchildren.Survived.value_counts(normalize = True)
# =============================================================================
# 0    0.516129
# 1    0.483871
# =============================================================================
# survival in this age group appears to have declined in comparison to preschool.  

# school to adolescence?
adolescents = titanic_train.query('age_discrete == "adolescent"')
adolescents.corr(method = 'pearson')
# negligible correlation with SibSp x survived in this subset. -0.054019 
# moderate correlation between PClass x survived -0.439298, appears to be obeying law of central limit theorem/
len(adolescents)
# more observations. Could be that a moderate relationship detected between PClass & survival could hold more weight
# adolescent survival by gender
adolescents.groupby('Sex').Survived.value_counts().plot.bar()
adolescents.groupby('Sex').Survived.value_counts(normalize = True)
# =============================================================================
# Sex     Survived
# female  1           0.744186
#         0           0.255814
# male    0           0.878788
#         1           0.121212
# =============================================================================
# clear distinction in survival rate by sex
# adolescent survival by class?
adolescents.groupby('Pclass').Survived.value_counts().plot.bar()
# visually, the survival differential is in favour of the 1st class adolescents
adolescents.groupby('Pclass').Survived.value_counts(normalize = True)
# =============================================================================
# Pclass  Survived
# 1       1           0.823529
#         0           0.176471
# 2       0           0.529412
#         1           0.470588
# 3       0           0.760000
#         1           0.240000
# =============================================================================
# could be an interesting candidate for chi2 test. Difference between the classes appears worhwhile investigating.

# create a crosstab to analyse
tab = pd.crosstab(adolescents.Pclass, adolescents.Survived)
# unable to run chi2 as there is an observed frequency in the contingency table of below 5. Assumption would be violated.
# tidy up
del adolescents, preschool, schoolchildren, tab



########Sibsp###########
# reset the plot canvass
plt.clf()
# distribution?
plt.hist(titanic_train.SibSp)
#small number of observations at 8 sibsp skewing the mean over the median. Frequency for these?
titanic_train.SibSp.value_counts(dropna = False)
# Interesting that sibsp == 8 returns a count of 7. The missing 1 must be in the test df?
# reset the plot canvass
plt.clf()
# outlier analysis
plt.boxplot(titanic_train.SibSp)
# boxplot is flagging sibsp values > 2 as outliers. This coincides with 1.5 * inter quartile range (> 2.5). Would see no valid reason to omit these outliers, but to be aware that this is a column with significant skew. 

########Parch###########
# reset the plot canvass
plt.clf()
# distribution?
plt.hist(titanic_train.Parch)
# some large values with low frequencies here.
titanic_train.Parch.value_counts(dropna = False)
# =============================================================================
# 0    678
# 1    118
# 2     80
# 5      5
# 3      5
# 4      4
# 6      1
# this result indicates an imbalance in train test split. 1 observation for Parch for a value of 6. The remaining 5 must be in the test group. 
# these are all likely to qualify as outliers. Implications for logistic regression model training vs test. 
# Possibly a candidate for outlier removal, but see no legitimate reason for doing so apart from improving model accuracy for parch rows < 6
# =============================================================================
# reset the plot canvass
plt.clf()
# outlier analysis
plt.boxplot(titanic_train.Parch)
# boxplot is flagging sibsp values > 2 as outliers. This coincides with 1.5 * inter quartile range (> 2.5). Would see no valid reason to omit these outliers, but to be aware that this is a column with significant skew. 
#f lags everything above a 0 as outlier. Coincides with 1.5 * IQR. Families were in the minority and most of the families on board were small ones.

########Fare###########
# reset the plot canvass
plt.clf()
#distribution
plt.hist(titanic_train.Fare, bins = 20)
# previously established minimum fare is 0. Idea being that these were crew. From this histogram it is clear a significant prop paid little or nothing.
titanic_train.Fare.value_counts(dropna = False)
# =============================================================================
# 8.0500      43
# 13.0000     42
# 7.8958      38
# 7.7500      34
# 26.0000     31
# 10.5000     24
# 7.9250      18
# 7.7750      16
# 26.5500     15
# 0.0000      15
#only 15 paid 0. So this wouldn't represent a total crew. 
# =============================================================================
# still, lets investigate the zero fares further
zero_fares = titanic_train.query('Fare == 0')
# visual inspection of the df, all male, all emarked at Southampton, 1 survived, a mixture of classes. 4 have 'LINE' as ticket reference.  All have 0 family on board.
# all but 3 have no cabin number recorded. I wonder how often 'LINE' occurs in the total training set?
titanic_train.query("Ticket == 'LINE'")  
# no, this is not more widely recorded than within this group.
# There's not a great deal more to be said about this group without additional info. 
#clean up
del zero_fares

########Sex###########
# distribution of gender
titanic_train.Sex.value_counts(normalize = True)
#overall 65% male. Did women and children go first make a difference here? 
titanic_train.groupby(titanic_train.Sex).Survived.value_counts(normalize = True)
#although the gender imbalance is evident, ~3/4 females survived, whereas ~1/5 males survived. 
titanic_train.groupby(titanic_train.Sex).Survived.value_counts().plot.bar()
# so gender would be an important feature for log regression. 

########Ticket###########
# add a duplicated tickets column
titanic_train.insert((titanic_train.columns.get_loc("Ticket") + 1), "duplicated_tickets", titanic_train.groupby('Ticket')['Ticket'].transform('count'))
# isolate duplicate tickets
duplicated_tickets = titanic_train.query("duplicated_tickets > 1")
duplicated_tickets.Ticket.value_counts()
duplicated_tickets.describe()
# Interesting here that there are 0s for SibSp & also for Parch separately. Are there any for both columns?
len(duplicated_tickets.query('SibSp == 0 & Parch == 0') )
# yes, 75 of the duplicated tickets were 0 for both family indicators. Metadata indicates that these could be mistresses, fiancees and nannies.   
# 20% of passengers with duplicated tickets are 0 for both family indicators
duplicated_tickets.Sex.value_counts(normalize = True)
# =============================================================================
# female    0.523256
# male      0.476744
# =============================================================================
# Interesting that in this subset the gender balance is slightly in favour of females. This contrasts the training df as a whole.
duplicated_tickets.groupby('Sex').Survived.value_counts(normalize = True)
# Closer to 20% males survived here and female survival rate looks the same as larger group. No apparent radical departure.
# No further question springs to mind on this with the data to hand. 
# Clean up 
del duplicated_tickets

########Cabin number###########
# value counts with NA
# =============================================================================
titanic_train.Cabin.value_counts(dropna = False)
# NaN                687
# G6                   4
# B96 B98              4
# C23 C25 C27          4
# Interesting that there are a number of observations with multiple rooms recorded. Possible reasons - families, group bookings and so on. 
# Isolate and observe the SibSp & Parch values for that group.
# =============================================================================
# insert a cabin string length column
titanic_train.insert((titanic_train.columns.get_loc("Cabin") + 1), "Cabin_str_len", titanic_train.Cabin.str.len(), True) 
# filter on that column
multi_cabins = titanic_train.query("Cabin_str_len > 5")
# observe some summary stats
multi_cabins.describe()
# all first class. SibSp and Parch both have minimum of 0.
multi_cabins.SibSp.value_counts(dropna = False)
# =============================================================================
# 1    9
# 0    6
# 3    3
# 2    2
# 6 rows with 0 siblings / spouses on board
# =============================================================================
multi_cabins.Parch.value_counts(dropna = False)
# =============================================================================
# 2    12
# 1     4
# 0     3
# 4     1
# 3 rows with 0 parents / children on board
# =============================================================================
# Are there any passengers with 0 for both Parch & SibSp?
multi_cabins.query("SibSp == 0 and Parch == 0")
# =============================================================================
#      PassengerId  Survived  Pclass                      Name   Sex   Age  SibSp  Parch    Ticket  Fare        Cabin  Cabin_str_len Embarked
# 789          790         0       1  Guggenheim, Mr. Benjamin  male  46.0      0      0  PC 17593  79.2      B82 B84            7.0        C
# 872          873         0       1  Carlsson, Mr. Frans Olof  male  33.0      0      0       695   5.0  B51 B53 B55           11.0        S
# 2 rows of 20.
# Guggenheim boarded the Titanic at Cherbourg with his valet Victor Giglio and his "mistress" Mrs 
# Aubart. Guggenheim and Giglio's ticket was 17593 and cost £79 4s1. Mr Guggenheim's chauffeur René 
# Pernot travelled in second class.
# Despite Etches best efforts Guggenheim soon returned to his room (B-82) and changed into his finest 
# evening wear, his valet, Mr Giglio did likewise. He was later heard to remark 'We've dressed up in 
# our best and are prepared to go down like gentlemen.
# https://www.encyclopedia-titanica.org/titanic-victim/benjamin-guggenheim.html
# There doesn't appear to be any supporting info as to why Frans Carlsson would have had multiple cabins associated, unlike Guggenheim who had an entourage. 
# =============================================================================
# 10% of passengers with multiple rooms are 0 for both family indicators.
# I wonder how widespread the 'down with the ship' ethos was among the male passengers. Could look at
# survival rates among males by age categories? Classes?
#tidy up
del multi_cabins

########Embarked###########
# distribution?
titanic_train.Embarked.value_counts(dropna = False, normalize = True)
# Interesting that the vast majority of passengers were boarded at Southampton. Let's look at the NaNs.
titanic_train[titanic_train.Embarked.isna()]
# =============================================================================
#      PassengerId  Survived  Pclass                                       Name     Sex   Age  SibSp  Parch  Ticket  Fare Cabin Embarked
# 61            62         1       1                        Icard, Miss. Amelie  female  38.0      0      0  113572  80.0   B28      NaN
# 829          830         1       1  Stone, Mrs. George Nelson (Martha Evelyn)  female  62.0      0      0  113572  80.0   B28      NaN
# =============================================================================
# interesting they both had no family on board but shared a ticket and cabin. 
# =============================================================================
# Mrs Stone boarded the Titanic in Southampton on 10 April 1912 and was travelling in first class 
# with her maid Amelie Icard. She occupied cabin B-28.
# https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html
# =============================================================================
# This could be evidence to amend the embarked column for Martha. Several sites online indicate this also.
# Safe to assume Amelie boarded at same port?
# same website indicates Amelie boarded at S too
# https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html

# replace the NaNs in this column with 'S'
titanic_train.Embarked = titanic_train.Embarked.fillna('S')

########Name###########
# By sorting on names it is possible to see patterns observed in the tickets that may indicate families.
# Could be possible to group family units together under group size
# insert a column for families with children
titanic_train.insert((titanic_train.columns.get_loc('duplicated_tickets') + 1), 'families_w_children', (titanic_train.duplicated_tickets > 1) & (titanic_train.Parch > 0))
# insert a column for families with no children on board ()
titanic_train.insert((titanic_train.columns.get_loc('families_w_children') + 1), 'families_no_children', (titanic_train.duplicated_tickets > 1) & (titanic_train.SibSp > 0) & (titanic_train.Parch == 0))
# observe pairwise trends 
pair_corr = titanic_train.corr(method = 'pearson')
# engineered columns show trends with family indicators as expected, no sig influence on survival
#tidy up
del pair_corr


pd.to_pickle(titanic_train, 'cache/titanic_engineered.pkl')

# comparison with test data distributions

titanic_test = pd.read_csv('titanic/test.csv')
# cache for future use
pd.to_pickle(titanic_test, 'cache/titanic_test.pkl')
len(titanic_test) / len(titanic_train)
# 0.4691358024691358, nearly 50% training data. Higher than expected. 

# compare Pclass
titanic_train.Pclass.value_counts(normalize = True)
# =============================================================================
# 3    0.551066
# 1    0.242424
# 2    0.206510
# =============================================================================
titanic_test.Pclass.value_counts(normalize = True)
# =============================================================================
# 3    0.521531
# 1    0.255981
# 2    0.222488
# =============================================================================
# seems to have slightly lower proportion 3rd class in test df

# compare Sex ratios
titanic_train.Sex.value_counts(normalize = True)
# =============================================================================
# male      0.647587
# female    0.352413
# =============================================================================

titanic_test.Sex.value_counts(normalize = True)
# =============================================================================
# male      0.636364
# female    0.363636
# =============================================================================
# 1.1 % towards female in test df

# reset the plot canvass
plt.clf()
# Compare distributions of Age
plt.hist(titanic_train.Age, alpha = 0.5, label = 'train')
plt.hist(titanic_test.Age, alpha = 0.5, label = 'test')
plt.legend(loc='upper right')
# distribution appears to be similar. Perhaps reverse proportions in ~20yrs vs ~30yrs. But overall likely to be of same distribution.
titanic_train.Age.describe() 
# =============================================================================
# count    714.000000
# mean      29.699118
# std       14.526497
# min        0.420000
# 25%       20.125000
# 50%       28.000000
# 75%       38.000000
# max       80.000000
# =============================================================================
titanic_test.Age.describe()
# =============================================================================
# count    332.000000
# mean      30.272590
# std       14.181209
# min        0.170000
# 25%       21.000000
# 50%       27.000000
# 75%       39.000000
# max       76.000000
# =============================================================================
# central tendency and distribution summary stats all very similar


# compare sibsp dists
# reset the plot canvass
plt.clf()
# Compare distributions of Sibsp
plt.hist(titanic_train.SibSp, alpha = 0.5, label = 'train')
plt.hist(titanic_test.SibSp, alpha = 0.5, label = 'test')
plt.legend(loc='upper right')
# visually analagous
titanic_train.SibSp.describe()
# =============================================================================
# count    891.000000
# mean       0.523008
# std        1.102743
# min        0.000000
# 25%        0.000000
# 50%        0.000000
# 75%        1.000000
# max        8.000000
# =============================================================================
titanic_test.SibSp.describe()
# =============================================================================
# count    418.000000
# mean       0.447368
# std        0.896760
# min        0.000000
# 25%        0.000000
# 50%        0.000000
# 75%        1.000000
# max        8.000000
# =============================================================================
# quartiles are identical, similar means but std smaller in test

# now Parch
# reset the plot canvass
plt.clf()
# Compare distributions of Parch
plt.hist(titanic_train.Parch, alpha = 0.5, label = 'train')
plt.hist(titanic_test.Parch, alpha = 0.5, label = 'test')
plt.legend(loc='upper right')
# visually analagous
# Perhaps the test set here shows consistent skew to the right of training set, though analagous in proportion
titanic_train.Parch.describe()
# =============================================================================
# count    891.000000
# mean       0.381594
# std        0.806057
# min        0.000000
# 25%        0.000000
# 50%        0.000000
# 75%        0.000000
# max        6.000000
# =============================================================================
titanic_test.Parch.describe()
# =============================================================================
# count    418.000000
# mean       0.392344
# std        0.981429
# min        0.000000
# 25%        0.000000
# 50%        0.000000
# 75%        0.000000
# max        9.000000
# =============================================================================
# quartiles not influenced by the larger range observed within the test set

# fare
# reset the plot canvass
plt.clf()
# Compare distributions of Fare
plt.hist(titanic_train.Fare, alpha = 0.5, label = 'train')
plt.hist(titanic_test.Fare, alpha = 0.5, label = 'test')
plt.legend(loc='upper right')
# looks to be a few outlier high values in the test set, have they affected the central tendency?
titanic_train.Fare.describe()
# =============================================================================
# count    891.000000
# mean      32.204208
# std       49.693429
# min        0.000000
# 25%        7.910400
# 50%       14.454200
# 75%       31.000000
# max      512.329200
# =============================================================================
titanic_test.Fare.describe()
# =============================================================================
# count    417.000000
# mean      35.627188
# std       55.907576
# min        0.000000
# 25%        7.895800
# 50%       14.454200
# 75%       31.500000
# max      512.329200
# =============================================================================
# the quartiles are almost identical but the mean appears to have been influenced upwards by the high outliers within the test set

# Embarked
titanic_train.Embarked.value_counts(normalize = True)
# =============================================================================
# S    0.725028
# C    0.188552
# Q    0.086420
# =============================================================================

titanic_test.Embarked.value_counts(normalize = True)
# =============================================================================
# S    0.645933
# C    0.244019
# Q    0.110048
# =============================================================================
# rel. proportions of Sothhampton are 8% lower in test set. Could this be significant? Could be candidate for Chi2 test.

# tidy up
del titanic_test, titanic_train




