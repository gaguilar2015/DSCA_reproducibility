# -*- coding: utf-8 -*-
"""
Exercise2: Review the script below and apply the provided style guide to
improve it:
https://google.github.io/styleguide/pyguide.html
Purpose of script: exploratory data analysis of titanic test dataset,
imputation, feature engineering
Comparison of feature distribution against test data
source: https://www.kaggle.com/c/titanic/data
"""
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
#change working directory
os.chdir('/Users/richardleyshon/Documents/DSCA_reproducibility_dev/exercises/activity_2')


#adjust console presentation of output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#import test data
TITANIC_TRAIN = pd.read_csv('titanic/train.csv')
#cache unedited df for use in training later
pd.to_pickle(TITANIC_TRAIN, 'cache/TITANIC_TRAIN.pkl')


#dimensions of dataframe
TITANIC_TRAIN.shape
#891 x 12

#summary statistics of numeric columns
TITANIC_TRAIN.describe()
# =============================================================================
# PassengerId, is it unique?
# Survived, can see from quartiles that majority did not survive. Would hope
# that the test DF would be based on the same distribution.
# Pclass, relatively fewer 1st class passengers. Majority are 3rd class.
# Age, floating point. Estimated ages include a .5 value. There is a degree of
# missingness, only 714 observations. What proportion is missing? What
# proportion is estimated?
# Sibsp, number of siblings or spouses aboard. Mistresses and fiances were
# ignored. Majority recorded having no siblings. Mean is higher than median,
# large outliers influenced? Max is 8 siblings.
# Parch, number of parents / children on board. Children travelling with nanny
# only Parch == 0. Large outlier influence on mean as in SibSp.
# Fare, minimum is zero. How many? Are these the crew? Do they have a unique
# pclass and age profile? Mean is double the median, high large outlier
# influence. Maximum is £512, whopping for 1912.
# =============================================================================

#summary stats for object columns
TITANIC_TRAIN.describe(include='object')
# =============================================================================
# Name column has no duplicates, I happen to know the full dataset has 2 non
# unique values. Either in test df or split. Likely of no consequence for this
# analysis.
# Sex appears to be mostly male. Look at proportion male:female.
# ticket ref is object, column is a mix of character / numeric values.
# Interestingly, 230 values are non-unique - family tickets?
# Cabin number has a high degree of missingness. Only 204 values. This is a
# shame as it would have been very interesting to analyse survival based on
# distance from the deck level. Perhaps isolate this group and investigate?
# Embarked - port of embarkation has 3 values. Most common one is S =
# Southampton.  2 missing values here.
# =============================================================================

'''some questions identified to pursue further. My main concern would be on
the method of splitting the test dataset. No detail on how this has been done
and hopefully not just based on distribution of survived column alone.'''
# explore the pairwise correlations within the data using pearson's
# correlation coefficient
TITANIC_TRAIN.corr(method='pearson')
'''some interesting trends observed, Only Parch & Fare indicate a positive
correlation with Survival neither of which would be indicated as strong
(corr of 0.7 or higher) all other numeric feautures indicate weak negative
correlations against survived suggests a lot of noise in observed trends in
features against survival. Weak though they are,the largest coefficients
observed (+ or -) against survived are Pclass & Fare. Within the matrix as a
whole, Pclass & Fare has the strongest observed coefficient of -0.549500
Not a strong correlation though (+/-0.7 threshold). A moderate negative
correlation. Next up would be SibSp & Parch with a weak positive correlation
of 0.414838.'''

########PassengerId###########
# is it unique?
TITANIC_TRAIN.PassengerId.value_counts(dropna=False)
# appears so.

########Survived###########
# distribution?
TITANIC_TRAIN.Survived.value_counts(normalize=True)
# 62% didn't survive. Will chek this against the test set.

########Pclass###########
# distribution?
plt.hist(TITANIC_TRAIN.Pclass, bins=3)
# 3rd class by far the most common pclass. Interesting that there appears to
# be more 1st than 2nd class.
TITANIC_TRAIN.Pclass.value_counts(normalize=True)
# =============================================================================
# 3    0.551066
# 1    0.242424
# 2    0.206510
# =============================================================================

########Age###########
TITANIC_TRAIN.Age.value_counts(dropna=False, normalize=True)
# NaN      0.198653. Nearly 20% missing data. Parch / Sibsp won't offer the
# required directionality to help inform imputation. what prop are estimated
# and equal to 1 or older
EST_AGE_PCT = len(TITANIC_TRAIN.query('Age >=1.0 & Age % 1 ==0.5')) / len(TITANIC_TRAIN)
EST_AGE_PCT
# 2 percent of data are estimated age. Combined with #NAs, this makes for a
# column with significant quality issues. Create a feature that records
# estimated ages - age_estimated
TITANIC_TRAIN.insert(
    (TITANIC_TRAIN.columns.get_loc("Age") + 1),
    "age_estimated",
    np.where((TITANIC_TRAIN.Age >= 1.0) & (TITANIC_TRAIN.Age % 1 == 0.5),
             True, False))
# introduce age bins in order to explore diffreential survival rates
# school age from:
# https://en.wikipedia.org/wiki/Raising_of_school_leaving_age_in_England_and_Wales#19th_century
# define labels for the age categories
CUT_LABELS = ['pre_school',
              'school',
              'adolescent',
              'age_of_majority',
              'pensioners']
# define boundary limits for the boundaries
CUT_BINS = [0, 4.5, 13.5, 20.5, 69.5, 80]
# insert an age quantile column  to investigate differential survival
TITANIC_TRAIN.insert(
    (TITANIC_TRAIN.columns.get_loc("Age") + 1),
    "age_discrete",
    value=pd.cut(TITANIC_TRAIN.Age, bins=CUT_BINS, labels=CUT_LABELS))

# tidy up
del CUT_BINS, CUT_LABELS

#need to investigate differential survival among the age categories,
# interesting to look at pre school survival rates
PRESCHOOL = TITANIC_TRAIN.query('age_discrete == "pre_school"')
# eda on this dataset
PRESCHOOL.corr(method='pearson')
# interesting that a strong negative correlation is returned for SibSp x
# Survived -0.714502 . Could this indicate that passengers of smaller families
# (with PRESCHOOLers) were more likely to survive than individuals with larger families?
PRESCHOOL.Survived.value_counts(normalize=True)
# =============================================================================
# 1    0.675
# 0    0.325
# =============================================================================
len(PRESCHOOL)
# small number of observations, high correlation observed could be a relic of
# the data. Not observed correlation within total df. Perhaps school age?
SCHOOLCHILDREN = TITANIC_TRAIN.query('age_discrete == "school"')
SCHOOLCHILDREN.corr(method='pearson')
len(SCHOOLCHILDREN)
# again, a small subset. Moderate -ve correlation between SibSp & survived,
# -0.668927 more pronounced influence of pclass x survived here -0.526789 than
# in PRESCHOOL -0.325941. Both small subsets however.
SCHOOLCHILDREN.Survived.value_counts(normalize=True)
# =============================================================================
# 0    0.516129
# 1    0.483871
# =============================================================================
# survival in this age group appears to have declined in comparison to
# PRESCHOOL.

# school to adolescence?
ADOLESCENTS = TITANIC_TRAIN.query('age_discrete == "adolescent"')
ADOLESCENTS.corr(method='pearson')
# negligible correlation with SibSp x survived in this subset. -0.054019
# moderate correlation between PClass x survived -0.439298, appears to be
# obeying law of central limit theorem
len(ADOLESCENTS)
# more observations. Could be that a moderate relationship detected between
# PClass & survival could hold more weight adolescent survival by gender
ADOLESCENTS.groupby('Sex').Survived.value_counts().plot.bar()
ADOLESCENTS.groupby('Sex').Survived.value_counts(normalize=True)
# =============================================================================
# Sex     Survived
# female  1           0.744186
#         0           0.255814
# male    0           0.878788
#         1           0.121212
# =============================================================================
# clear distinction in survival rate by sex
# adolescent survival by class?
ADOLESCENTS.groupby('Pclass').Survived.value_counts().plot.bar()
# visually, the survival differential is in favour of the 1st class ADOLESCENTS
ADOLESCENTS.groupby('Pclass').Survived.value_counts(normalize=True)
# =============================================================================
# Pclass  Survived
# 1       1           0.823529
#         0           0.176471
# 2       0           0.529412
#         1           0.470588
# 3       0           0.760000
#         1           0.240000
# =============================================================================
# could be an interesting candidate for chi2 test. Difference between the
# classes appears worhwhile investigating.

# create a crosstab to analyse
TAB = pd.crosstab(ADOLESCENTS.Pclass, ADOLESCENTS.Survived)
# unable to run chi2 as there is an observed frequency in the contingency
# TABle of below 5. Assumption would be violated.
# tidy up
del ADOLESCENTS, PRESCHOOL, SCHOOLCHILDREN, TAB



########Sibsp###########
# reset the plot canvass
plt.clf()
# distribution?
plt.hist(TITANIC_TRAIN.SibSp)
#small number of observations at 8 sibsp skewing the mean over the median.
# Frequency for these?
TITANIC_TRAIN.SibSp.value_counts(dropna=False)
# Interesting that sibsp == 8 returns a count of 7. The missing 1 must be in
# the test df?
# reset the plot canvass
plt.clf()
# outlier analysis
plt.boxplot(TITANIC_TRAIN.SibSp)
# boxplot is flagging sibsp values > 2 as outliers. This coincides with 1.5 *
# inter quartile range (> 2.5). Would see no valid reason to omit these
# outliers, but to be aware that this is a column with significant skew.

########Parch###########
# reset the plot canvass
plt.clf()
# distribution?
plt.hist(TITANIC_TRAIN.Parch)
# some large values with low frequencies here.
TITANIC_TRAIN.Parch.value_counts(dropna=False)
# =============================================================================
# 0    678
# 1    118
# 2     80
# 5      5
# 3      5
# 4      4
# 6      1
# this result indicates an imbalance in train test split. 1 observation for
# Parch for a value of 6. The remaining 5 must be in the test group.
# these are all likely to qualify as outliers. Implications for logistic
# regression model training vs test. Possibly a candidate for outlier removal,
# but see no legitimate reason for doing so apart from improving model accuracy
# for parch rows < 6
# =============================================================================
# reset the plot canvass
plt.clf()
# outlier analysis
plt.boxplot(TITANIC_TRAIN.Parch)
'''boxplot is flagging sibsp values > 2 as outliers. This coincides with 1.5 *
inter quartile range (> 2.5). Would see no valid reason to omit these outliers,
but to be aware that this is a column with significant skew.
f lags everything above a 0 as outlier. Coincides with 1.5 * IQR. Families
were in the minority and most of the families on board were small ones.'''

########Fare###########
# reset the plot canvass
plt.clf()
#distribution
plt.hist(TITANIC_TRAIN.Fare, bins=20)
# previously esTABlished minimum fare is 0. Idea being that these were crew.
# From this histogram it is clear a significant prop paid little or nothing.
TITANIC_TRAIN.Fare.value_counts(dropna=False)
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
ZERO_FARES = TITANIC_TRAIN.query('Fare == 0')
'''visual inspection of the df, all male, all emarked at Southampton, 1
survived, a mixture of classes. 4 have 'LINE' as ticket reference.  All have 0
family on board. all but 3 have no cabin number recorded. I wonder how often
'LINE' occurs in the total training set?'''
TITANIC_TRAIN.query("Ticket == 'LINE'")
# no, this is not more widely recorded than within this group.
#clean up
del ZERO_FARES

########Sex###########
# distribution of gender
TITANIC_TRAIN.Sex.value_counts(normalize=True)
#overall 65% male. Did women and children go first make a difference here?
TITANIC_TRAIN.groupby(TITANIC_TRAIN.Sex).Survived.value_counts(normalize=True)
#although the gender imbalance is evident, ~3/4 females survived, whereas ~1/5
#males survived.
TITANIC_TRAIN.groupby(TITANIC_TRAIN.Sex).Survived.value_counts().plot.bar()
# so gender would be an important feature for log regression.

########Ticket###########
# add a duplicated tickets column
TITANIC_TRAIN.insert(
    (TITANIC_TRAIN.columns.get_loc("Ticket")+1),
    "DUPLICATED_TICKETS",
    TITANIC_TRAIN.groupby('Ticket')['Ticket'].transform('count'))
# isolate duplicate tickets
DUPLICATED_TICKETS = TITANIC_TRAIN.query("DUPLICATED_TICKETS > 1")
DUPLICATED_TICKETS.Ticket.value_counts()
DUPLICATED_TICKETS.describe()
# Interesting here that there are 0s for SibSp & also for Parch separately. Are
# there any for both columns?
len(DUPLICATED_TICKETS.query('SibSp == 0 & Parch == 0'))
# yes, 75 of the duplicated tickets were 0 for both family indicators. Metadata
# indicates that these could be mistresses, fiancees and nannies.
# 20% of passengers with duplicated tickets are 0 for both family indicators
DUPLICATED_TICKETS.Sex.value_counts(normalize=True)
# =============================================================================
# female    0.523256
# male      0.476744
# =============================================================================
# Interesting that in this subset the gender balance is slightly in favour of
# females. This contrasts the training df as a whole.
DUPLICATED_TICKETS.groupby('Sex').Survived.value_counts(normalize=True)
# Closer to 20% males survived here and female survival rate looks the same as
# larger group. No apparent radical departure. No further question springs to
# mind on this with the data to hand.
# Clean up
del DUPLICATED_TICKETS

########Cabin number###########
# value counts with NA
# =============================================================================
TITANIC_TRAIN.Cabin.value_counts(dropna=False)
# NaN                687
# G6                   4
# B96 B98              4
# C23 C25 C27          4
# Interesting that there are a number of observations with multiple rooms
# recorded. Possible reasons - families, group bookings and so on.
# Isolate and observe the SibSp & Parch values for that group.
# =============================================================================
# insert a cabin string length column
TITANIC_TRAIN.insert(
    (TITANIC_TRAIN.columns.get_loc("Cabin")+1),
    "Cabin_str_len",
    TITANIC_TRAIN.Cabin.str.len(),
    True)
# filter on that column
MULTI_CABINS = TITANIC_TRAIN.query("Cabin_str_len>5")
# observe some summary stats
MULTI_CABINS.describe()
# all first class. SibSp and Parch both have minimum of 0.
MULTI_CABINS.SibSp.value_counts(dropna=False)
# =============================================================================
# 1    9
# 0    6
# 3    3
# 2    2
# 6 rows with 0 siblings / spouses on board
# =============================================================================
MULTI_CABINS.Parch.value_counts(dropna=False)
# =============================================================================
# 2    12
# 1     4
# 0     3
# 4     1
# 3 rows with 0 parents / children on board
# =============================================================================
# Are there any passengers with 0 for both Parch & SibSp?
MULTI_CABINS.query("SibSp == 0 and Parch == 0")
# =============================================================================
#      PassengerId  Survived  Pclass                      Name   Sex   Age
# 789          790         0       1  Guggenheim, Mr. Benjamin  male  46.0
# 872          873         0       1  Carlsson, Mr. Frans Olof  male  33.0
# 2 rows of 20.
# Guggenheim boarded the Titanic at Cherbourg with his valet Victor Giglio and
# his "mistress" Mrs Aubart. Guggenheim and Giglio's ticket was 17593 and cost
# £79 4s1. Mr Guggenheim's chauffeur René Pernot travelled in second class.
# Despite Etches best efforts Guggenheim soon returned to his room (B-82) and
# changed into his finest evening wear, his valet, Mr Giglio did likewise. He
# was later heard to remark 'We've dressed up in our best and are prepared to
# go down like gentlemen.
# https://www.encyclopedia-titanica.org/titanic-victim/benjamin-guggenheim.html
# There doesn't appear to be any supporting info as to why Frans Carlsson would
# have had multiple cabins associated, unlike Guggenheim who had an entourage.
# =============================================================================
# 10% of passengers with multiple rooms are 0 for both family indicators.
# I wonder how widespread the 'down with the ship' ethos was among the male
# passengers. Could look at survival rates among males by age categories?
# Classes?
#tidy up
del MULTI_CABINS

########Embarked###########
# distribution?
TITANIC_TRAIN.Embarked.value_counts(dropna=False, normalize=True)
# Interesting that the vast majority of passengers were boarded at Southampton.
# Let's look at the NaNs.
EMBARKED_NAN = TITANIC_TRAIN[TITANIC_TRAIN.Embarked.isna()]
EMBARKED_NAN
# =============================================================================
#      PassengerId  Survived  Pclass                                       Name
# 61            62         1       1                        Icard, Miss. Amelie
# 829          830         1       1  Stone, Mrs. George Nelson (Martha Evelyn)
# =============================================================================
# interesting they both had no family on board but shared a ticket and cabin.
# =============================================================================
# Mrs Stone boarded the Titanic in Southampton on 10 April 1912 and was
# travelling in first class with her maid Amelie Icard. She occupied cabin
# B-28.
# https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html
# =============================================================================
# This could be evidence to amend the embarked column for Martha. Several sites
# online indicate this also. Safe to assume Amelie boarded at same port?
# same website indicates Amelie boarded at S too
# https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html
# replace the NaNs in this column with 'S'
TITANIC_TRAIN.Embarked = TITANIC_TRAIN.Embarked.fillna('S')

########Name###########
# By sorting on names it is possible to see patterns observed in the tickets
# that may indicate families. Could be possible to group family units together
# under group size
# insert a column for families with children
TITANIC_TRAIN.insert(
    (TITANIC_TRAIN.columns.get_loc('DUPLICATED_TICKETS')+1),
    'families_w_children',
    (TITANIC_TRAIN.DUPLICATED_TICKETS > 1) & (TITANIC_TRAIN.Parch > 0))
# insert a column for families with no children on board ()
TITANIC_TRAIN.insert(
    (TITANIC_TRAIN.columns.get_loc('families_w_children')+1),
    'families_no_children',
    (TITANIC_TRAIN.DUPLICATED_TICKETS > 1) & (TITANIC_TRAIN.SibSp > 0) & (TITANIC_TRAIN.Parch == 0))
# observe pairwise trends
PAIR_CORR = TITANIC_TRAIN.corr(method='pearson')
# engineered columns show trends with family indicators as expected, no sig
# influence on survival
#tidy up
del PAIR_CORR
pd.to_pickle(TITANIC_TRAIN, 'cache/titanic_engineered.pkl')

# comparison with test data distributions
TITANIC_TEST = pd.read_csv('titanic/test.csv')
# cache for future use
pd.to_pickle(TITANIC_TEST, 'cache/TITANIC_TEST.pkl')
TEST_PCT = len(TITANIC_TEST) / len(TITANIC_TRAIN)
TEST_PCT
# 0.4691358024691358, nearly 50% training data. Higher than expected.
# compare Pclass
TITANIC_TRAIN.Pclass.value_counts(normalize=True)
# =============================================================================
# 3    0.551066
# 1    0.242424
# 2    0.206510
# =============================================================================
TITANIC_TEST.Pclass.value_counts(normalize=True)
# =============================================================================
# 3    0.521531
# 1    0.255981
# 2    0.222488
# =============================================================================
# seems to have slightly lower proportion 3rd class in test df

# compare Sex ratios
TITANIC_TRAIN.Sex.value_counts(normalize=True)
# =============================================================================
# male      0.647587
# female    0.352413
# =============================================================================

TITANIC_TEST.Sex.value_counts(normalize=True)
# =============================================================================
# male      0.636364
# female    0.363636
# =============================================================================
# 1.1 % towards female in test df

# reset the plot canvass
plt.clf()
# Compare distributions of Age
plt.hist(TITANIC_TRAIN.Age, alpha=0.5, label='train')
plt.hist(TITANIC_TEST.Age, alpha=0.5, label='test')
plt.legend(loc='upper right')
# distribution appears to be similar. Perhaps reverse proportions in ~20yrs vs
# ~30yrs. But overall likely to be of same distribution.
TITANIC_TRAIN.Age.describe()
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
TITANIC_TEST.Age.describe()
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
plt.hist(TITANIC_TRAIN.SibSp, alpha=0.5, label='train')
plt.hist(TITANIC_TEST.SibSp, alpha=0.5, label='test')
plt.legend(loc='upper right')
# visually analagous
TITANIC_TRAIN.SibSp.describe()
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
TITANIC_TEST.SibSp.describe()
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
plt.hist(TITANIC_TRAIN.Parch, alpha=0.5, label='train')
plt.hist(TITANIC_TEST.Parch, alpha=0.5, label='test')
plt.legend(loc='upper right')
# visually analagous
# Perhaps the test set here shows consistent skew to the right of training set,
# though analagous in proportion
TITANIC_TRAIN.Parch.describe()
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
TITANIC_TEST.Parch.describe()
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
plt.hist(TITANIC_TRAIN.Fare, alpha=0.5, label='train')
plt.hist(TITANIC_TEST.Fare, alpha=0.5, label='test')
plt.legend(loc='upper right')
# looks to be a few outlier high values in the test set, have they affected the
# central tendency?
TITANIC_TRAIN.Fare.describe()
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
TITANIC_TEST.Fare.describe()
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
# the quartiles are almost identical but the mean appears to have been
# influenced upwards by the high outliers within the test set

# Embarked
TITANIC_TRAIN.Embarked.value_counts(normalize=True)
# =============================================================================
# S    0.725028
# C    0.188552
# Q    0.086420
# =============================================================================

TITANIC_TEST.Embarked.value_counts(normalize=True)
# =============================================================================
# S    0.645933
# C    0.244019
# Q    0.110048
# =============================================================================
# rel. proportions of Sothhampton are 8% lower in test set. Could this be
# significant? Could be candidate for Chi2 test.
# tidy up
del TITANIC_TEST, TITANIC_TRAIN
