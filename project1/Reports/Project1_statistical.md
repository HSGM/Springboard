
#### Predictive Strength of each feature on the rate of default

As the objective of the project is to illustrate the building of a credit scoring model, how each feature related to the default rate was looked at. By doing so it helped in picking out the features that are likely to be useful in the model vs. those that are not. 

Bar graphs were plotted that showed the rate of default for each feature by year  at the different levels of the feature. 


From the bar charts looking at rate of default vs each feature by year the following decisions were made:
1. Use only the data from 2010-1015 as the data from 2007 -2009 has a lot of variabilty.
2. Drop the attributes present only in the year 2015.
3. Drop the attributes only available from 2012( total_rev_hi_lim, tot_cur_bal, tot_coll_amnt, mths_since_last_major_derog, acc_now_delinq, collections_12mths_ex_med)as they are available only form 2012 on and does not show high predictive strength.
4. Drop acc_now_delinq: up until 2011 no delinquent accounts and in 2012 only one loan grade had some. So deliquent account activity present only for 2013-2015. So will not keep this feature. Note: total_rev_hi_lim is present only from 2012 but shows potential predictive ability but is not included.
In the code that follows we will re-bin the following attributes based on similarity in default rates.
5. For the attributes mths_since_last_delinq_quant and mths_since_last_record_quant included the category "missing" with the 6+ category.
6. For home_regrp include the category "OTHER" with the "Rent" group.
7. length_credithist_quant: group quantiles 3 and 4.
8. Emp_length_regrp as "na", "0-10" and "10+"
9. loan_amount group quantiles 0 +1 and bins 2+3+4
10. delinq_2yrs_none_any ( two categories: no delinquencies vs atleast one)

Influence of State on default rate :
The attribute "addr_state" gives the state where the loan was issued. The default rate in relation to state by plotting the default rate per state in a choropleth heat map was looked at.

The states IN, TN, MS, NE and ND have the highest default rate. Greater than 30%.  group These were grouped as High and the rest as low by creatig a new feature called 'state_high_low'

#### Relevant Years and Features for Model Building

Looking at the bar graphs, evident was the variability present in the data for the years 2007 and 2008. The data collected in 2007 and 2008 reflect the fact that lending club was in its early stages and its loan approval strategy had room for improvement. The trends from 2009 onwards appeared to be more stable. For model building only the data from 2009 and after was kept  and dropped 2007 and 2008.
 
Correlations between columns :
Machine learning methods are usually robust in the presence of correlated predictors. However understanding the degree to which predictors are correlated is useful to better understand the relationship between the features with each other and with the outcome. Hence  the correlations between the numeric features was obtained.


total_acc and open_acc_quant had a correlation of 0.62. A decision had to be made as to whether we kept one or both? The variable, **total_acc** is the total number of credit lines currently in the borrower's credit file. The variable **open_acc** is the the number of open credit lines in the borrower's credit file. From the bar graphs that was plotted total_acc showed a stronger trend with default rate. So decided to  retain total_acc and drop open_acc.

The correlation of outcome with the features does not seem strong but as expected the **annual_inc quantile** is negatively correlated indicating higher the annual income lower the rate of default and **dti** , the ratio of debt to income is one of the higher correlated features. We would expect to see this feature being significant in 

