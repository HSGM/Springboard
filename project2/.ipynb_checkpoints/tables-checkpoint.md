
**Logistic Regression**

|Room     |  precision_score	|recall_score|	f1_score|	support|
|---------|---------------------|------------|----------|----------|
|bathroom |		0.90|	0.90|	0.90|	20.0|
bedroom	|0.83	|0.75	|0.79	|20.0|
corridor|		1.00|	1.00|	1.00|	20.0|
dining_room|	0.82|	0.90|	0.86|	20.0|
kitchen |0.94	|0.75	|0.83	|20.0|
living room|0.71|	0.75|	0.73|	20.0|
pantry|		0.91|	1.00|	0.95|	20.0|
staircase|		0.95|	1.00|	0.98|	20.0|
**avg/total**|	0.88|	0.88|	0.88|	160.0|



**Random Forest**

|Room     |  precision_score	|recall_score|	f1_score|	support|
|---------|---------------------|------------|----------|----------|
|bathroom |	       	0.90|	0.95|	0.93|	20.0|
|bedroom|		0.80|	0.80|	0.80|	20.0|
|corridor	|	0.91|	1.00|	0.95|	20.0|
|dining_room	|	0.73|	0.95|	0.83|	20.0|
|kitchen	|	1.00	|0.70|	0.82|	20.0|
|living room|	0.81|	0.65|	0.72|	20.0|
|pantry	|	0.95|	1.00	|0.98	|20.0|
|staircase|		0.95|	0.95|	0.95|	20.0|
|**avg/total**	|	0.88	|0.88|	0.87|	160.0 |



**Support Vector**

|Room     |  precision_score	|recall_score|	f1_score|	support|
|---------|---------------------|------------|----------|----------|
bathroom |		0.89|	0.85|	0.87|	20.0|
bedroom	|	0.82|	0.90|	0.86|	20.0|
corridor|		1.00|	0.95|	0.97|	20.0|
dining_room	|	0.75|	0.90|	0.82|	20.0|
kitchen	|1.00|	0.65|	0.79|	20.0|
living |	0.75|	0.75|	0.75|	20.0|
pantry	|	0.91|	1.00	|0.95	|20.0|
staircase	|0.95|	1.00|	0.98|	20.0|
**avg/total**	|	0.88|	0.88|	0.87|	160.0|


**XGBoost**

|Room     |  precision_score	|recall_score|	f1_score|	support|
|---------|---------------------|------------|----------|----------|
bathroom |	0.89 |	0.80 |	0.84 |	20.0 |
bedroom	 |	0.76 |	0.80	 |0.78	 |20.0 |
corridor |	0.95 |	0.95 |	0.95 |	20.0 |
dining_room	 |0.73	 |0.95 |	0.83 |	20.0 |
kitchen	 |0.88 |	0.70 |	0.78 |	20.0 |
living room	 |0.72	 |0.65	 |0.68	 |20.0 |
pantry	 |0.95 |	1.00 |	0.98	 |20.0 |
staircase	 |	0.95 |	0.95 |	0.95 |	20.0 |
**avg/total**|	0.85|	0.85|	0.85|	160.0|


The Confusion Matrices for each of the models :

**Logistic Regression**

|Predicted	True|Bathroom|bedroom|corridor|dining room	|kitchen|livingroom|pantry|staircase|
|-----------|--------|-------|--------|-------------|--------|---------|-----|----------|								
Bathroom|	18|	1|	0|	0|	0|	0|	1|	0|
bedroom|1|	1	|15	|0	|1	|0	|3	|0	|0|
corridor|2	|0	|0	|20	|0	|0	|0|	0|	0|
dining room|3|	0	|0	|0|	18	|0	|1	|1	|0|
kitchen|	1|	0|	0|	2	|15	|2|0|	0|
living room|	0	|2	|0	|1	|1|	15	|0	|1|
pantry|	0	|0	|0	|0|	0	|0|	20|	0|
staircase|	0|	0|	0|	0|	0|	0|	0|20
