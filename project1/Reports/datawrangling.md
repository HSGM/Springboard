# Lending Club Data Wrangling Steps:


1. The data set was downloaded from Kaggle website (https://www.kaggle.com/wendykan/lending-club-loan-data). 

2. It consisted of a csv file with the data and an excel spreadsheet with the data dictionary.

3. The data were already in a tidy format that made it easy to be read into a pandas dataframe.

4. However as the objective of the project is to build a credit scoring model,firstly it was necessary to classify the loans into two groups, namely those that had defaulted or not. The column loan status had 10 categories. 




|  Loan Status| Number      |
| ----------- | ----------- |
|Current      |	601779      |
|Fully Paid |	207723
Charged Off	|45248
Late (31-120 days) |	11591
Issued |	8460
In Grace Period |	6253
Late (16-30 days)|	2357
Does not meet the credit policy. Status:Fully Paid|	1988
Default	|1219
Does not meet the credit policy. Status:Charged Off	|761


Grouped loan status into a default group and nodefault group as follows by creating a new column called outcome using a dictionary called loan_status_dict
   1. Default Group:  Charged off, Late(31-120 days), In Grace Period, Late(16-30 days), Default.
   2. No Default Group : Fully Paid
   3. Did not include:
    1. Current, Does not meet the credit policy. Issued
    2. Does not meet the credit policy.Status:Charged off, Does not meet the credit policy: Fully Paid
    
```python
loan_status_dict ={"Fully Paid" : 0 , 
                   "Charged Off": 1 , "Late (31-120 days)": 1,  "In Grace Period":1, "Late (16-30 days)":1,
                   "Default":1,
                   "Current" : 2 ,"Issued": 2,
                   "Does not meet the credit policy. Status:Fully Paid": 2,
                   "Does not meet the credit policy. Status:Charged Off": 2}

loanbook['outcome']= loanbook.loan_status.map(loan_status_dict)


#Keep only the rows where outcome is 0 or 1, excluding the rows where outcome =2
loanbook = loanbook[(loanbook['outcome'] < 2) ]

# Take a look at the distribution of the dependent variable outcome: 
sns.countplot(data=loanbook,
                  x ='outcome')

```
The next step was to keep only those columns that were relevant
to the analysis:
1. As of interest are those features that would be available at the time of loan application ,those features that would be available only after the loan has been approved was dropped.
2. Also dropped those features that are unique to each loan applicant ( loan id, url, member_id).
3. The features 'desc', 'title', 'zip_code'  may be useful but they will need to be engineeredfurther to be included in this mod el and taking into consideration the scope of this project they were not be included.
4. Grade was retained for further exploration of the data but will not be included as a predictor variable as the objective    of this project is to illustrate the building of a model that could be used to grade the loans.


```python

col_to_drop =['id','member_id','funded_amnt','funded_amnt_inv', 'pymnt_plan', 'url','desc','title','initial_list_status',
               'out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int',
               'total_rec_late_fee','recoveries', 'collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','next_pymnt_d',
               'last_credit_pull_d','policy_code', 'zip_code' ,  'sub_grade', 'installment']

loanbook.drop(col_to_drop, axis=1, inplace=True)


```
This left 49 columns to work with. One outcome and 48 other columns.

Categorical Features:

If there are two many levels in a categorical feature it will affect the performance of the model. 
By looking at the number of uniques categories for each feature decided to combine/bin categories so there will not be too many categories per feature.
 
>Binning in the case of numeric variables will help with ameliorating the effect of long extreme values whose presence will be evident to us by looking at the histograms of the distibutions of these features. 

```python
# look at the number of unique categories for each categorical feature
for col_name in loanbook.columns:
    if loanbook[col_name].dtypes == 'object':
        unique_cat = len(loanbook[col_name].unique())
        print(" '{col_name}' : {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
        
```
|column | unique categories |
| ----------- | ----------- |
|term  | 2 |
grade | 7 |
emp_title | 139903 |
emp_length| 12 |
home_ownership |6 
verification_status | 3
loan_status | 6 
purpose | 14 
addr_state |51 
earliest_cr_line | 649 
application_type | 2 
verification_status_joint | 4 

Looking at the number of unique categories for each feature:

**emp_title**, **addr_state** and **earliest_cr _line** have a high number of categories.
**Emp_title** refers to the job title which appears to be very specific to the applicant. As regrouping this into smaller categories would require a lot more extensive exploration and for the sake of time this feature was dropped.

**Earliest_cr_line** refers to the month the borrowers earliest credit line was opened. From this feature the length of credit history for the borrower was calculated for use in the model.

**addr_state** refers to the state in which the loan was issued. This was kept as by looking a the loan defaults for each state we may be able to group the states as highly defaulting states and low default states. 

The other features were re-binned  into a fewer number of sub-categories. 


Feature |  Original Categories | New Catgories
--------|----------------------|---------------
**purpose** |credit_card, car, small_business, other, wedding, debt_consolidation, home_improvement, major_purchase, medical, moving, vacation, house, renewable_energy, educational|debt, consumer, home, business |
**home_ownership**|RENT, OWN, MORTGAGE, OTHER, NONE, ANY| MORTGAGE, RENT, OWN, OTHER|
**emp_length**|10+years, <1 year,1 year,2 years, 3 years, 4 years, 5 years,  6 year,  7 yeras, 8 years,  9 years, n/a|0-2, 3-10, 10+, n/a|
**verification_status**| Source Verified, Verified, Not Verified |Verified, Not Verified|
**verification_status_joint**|  Source Verified, Verified, Not Verified|Verified, Not Verified|


Using the date of issue of the loan a feature called **year** corresponding to the year the loan was issued was created using the following code.

```python


    #create column 'date_issued' with date of issue as year-month-date.
    loanbook['date_issued'] =  pd.to_datetime(loanbook['issue_d'], format='%b-%Y')

    #create column 'length_credithist' with no. of days of credit history.

    loanbook['earliest_credit_date'] =  pd.to_datetime(loanbook['earliest_cr_line'], format='%b-%Y')
    loanbook['td_credithist'] = loanbook['date_issued']-loanbook['earliest_credit_date']
    loanbook['length_credithist'] = loanbook['td_credithist'].dt.total_seconds() / (24 * 60 * 60)
    loanbook= loanbook.drop('td_credithist', axis=1)

    #create year variable
    loanbook.loc[:, 'year'] = loanbook.loc[:,'date_issued'].dt.year
    loanbook.loc[:,'year'] =loanbook.loc[:,'year'].astype('category')

```

#### Numeric Features. 

**Outliers**: By looking at histogram plots for each numeric feature, long tails with skewed distributions indicated presence of outliers. As these outliers could be legitimate or erroneous the features were binned which reduces the influence of these outliers.

First it was necessary to figure out whoch features were continous and which ones were discreet. 
This was done by looking at the number of unique values for each features as the continous would have a high count of unique values. 

```python

# Which features are continous and which ones are discreet?
float_features =[]
        
float_features=[(x, len(loanbook[x].unique())) for x in loanbook.columns if loanbook[x].dtypes == 'float64'] 

for x,y in sorted(float_features, key=lambda p:p[1]):
    print(" {x}: {y}" .format (x=x, y=y))



```
Then following features with a high number of unique values was binned into  6 quantiles using qcut: 

length_credithist, annual_inc, dti, revol_bal, revol_util, total_acc, open_acc, loan_amnt, mths_since_last_major_derog, open_il_6m, mths_since_rcnt_il, total_bal_il,il_util, max_bal_bc,  all_util,  total_rev_hi_lim

The following features which had a low number of uniques values was custom binned as shown in the table. A new column was created which had the original name+ _edit. eg: inq_last_6mths_edit 

Feature | Bins |
--------|------|
inq_last_6mths, open_acc_6m, open_il_12m, open_il_24m, open_rv_12m,open_rv_24m, inq_fi, total_cu_tl, inq_last_12m, acc_now_delinq| None, 1-2, 3-4, 5-6, 6-7, 7+|
delinq_2yrs|None, 1-5, 6-10. 11-15, 16-20, 20+ |
tot_coll_amt_quant, tot_cur_bal_quant| None, 1000, 10000, 50000+ |
collections_12_mths_ex_med_quant | None, 1, 2+ |
mths_since_last_record | missing, 0-6, 6+ |
pub_rec| None, 1-5, 6-10, 11+ |

                 


    

    






