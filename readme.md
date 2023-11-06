# Machine Learning Zoomcamp Midterm Project
---
## German Credit Risk Evaluation
--- 
Course: MLZoomcamp
Institute: DataTalksClub
---
This project seeks to evaluate the risk assessment of providing financial credit to German citizens based on a sample of 1000 users. Here, we sought to carry out an exploratory analysis of the data, as well as modeling for a binary prediction of the risk associated with credit in good or bad risk. The German Data can be obtained at https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data.
The German Data can be obtained at [https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data](German Data).
---
# Index
- Objectives
- Data Description 
- Installation instructions
---
# Objectives
- Find a dataset;
- Explore the data;
- Train a model; 
- Tuning the model;
- Put notebook in a script;
- Put model in a web service; 
- Deploy model localy with docker.
---
# Data Description
Total Samples: 1000
Total variables: 20

- Attribute 1:  (qualitative)
	- Status of existing checking account
		- A11 :      ... <    0 DM
		- A12 : 0 <= ... <  200 DM
		- A13 :      ... >= 200 DM / salary assignments for at least 1 year
		- A14 : no checking account

- Attribute 2:  (numerical)
	- Duration in month

- Attribute 3:  (qualitative)
	- Credit history
		- A30 : no credits taken / all credits paid back duly
        - A31 : all credits at this bank paid back duly
	    - A32 : existing credits paid back duly till now
        - A33 : delay in paying off in the past
	    - A34 : critical account / other credits existing (not at this bank)

- Attribute 4:  (qualitative)
	- Purpose
	    -  A40 : car (new)
	    -  A41 : car (used)
	    -  A42 : furniture/equipment
	    -  A43 : radio/television
	    -  A44 : domestic appliances
	    -  A45 : repairs
	    -  A46 : education
	    -  A47 : (vacation - does not exist?)
	    -  A48 : retraining
	    -  A49 : business
	    -  A410 : others

- Attribute 5:  (numerical)
	- Credit amount

- Attibute 6:  (qualitative)
	- Savings account/bonds
	    -  A61 :          ... <  100 DM
	    -  A62 :   100 <= ... <  500 DM
	    -  A63 :   500 <= ... < 1000 DM
	    -  A64 :          .. >= 1000 DM
        -  A65 :   unknown/ no savings account

- Attribute 7:  (qualitative)
	- Present employment since
	    -  A71 : unemployed
	    -  A72 :       ... < 1 year
	    -  A73 : 1  <= ... < 4 years  
	    -  A74 : 4  <= ... < 7 years
	    -  A75 :       .. >= 7 years

- Attribute 8:  (numerical)
	-  Installment rate in percentage of disposable income

- Attribute 9:  (qualitative)
	- Personal status and sex
	    -  A91 : male   : divorced/separated
	    -  A92 : female : divorced/separated/married
        -  A93 : male   : single
	    -  A94 : male   : married/widowed
	    -  A95 : female : single

- Attribute 10: (qualitative)
	- Other debtors / guarantors
	    -  A101 : none
	    -  A102 : co-applicant
	    -  A103 : guarantor

- Attribute 11: (numerical)
	- Present residence since

- Attribute 12: (qualitative)
	- Property
	    -  A121 : real estate
	    -  A122 : if not A121 : building society savings agreement / life insurance
	    -  A123 : if not A121/A122 : car or other, not in attribute 6
	    -  A124 : unknown / no property

- Attribute 13: (numerical)
	-  Age in years

- Attribute 14: (qualitative)
	- Other installment plans 
	    -  A141 : bank
	    -  A142 : stores
	    -  A143 : none

- Attribute 15: (qualitative)
	- Housing
	    -  A151 : rent
	    -  A152 : own
	    -  A153 : for free

- Attribute 16: (numerical)
	- Number of existing credits at this bank

- Attribute 17: (qualitative)
	- Job
	    -  A171 : unemployed/ unskilled  - non-resident
	    -  A172 : unskilled - resident
	    -  A173 : skilled employee / official
	    -  A174 : management / self-employed / highly qualified employee / officer

- Attribute 18: (numerical)
	- Number of people being liable to provide maintenance for

- Attribute 19: (qualitative)
	- Telephone
	    -  A191 : none
	    -  A192 : yes, registered under the customers name

- Attribute 20: (qualitative)
	- foreign worker
	    - A201 : yes
	    - A202 : no

- Attribute 21: (numerical)
	- class (**target**)
		- 0 : good
		- 1 : bad

---
# Installation Instructions

### Cloning the Repository

To get started with this project, clone the repository to your local machine:

git clone https://github.com/felpscunha/german-credit-risk.git
cd german-credit-risk

--- 

### Virtual Environment

To run the Jupyter notebooks in this project or a web service, is required to set up a virtual environment and install the dependencies.

1. Clone the repository to your local machine and navigate to the project directory (root):

git clone https://github.com/felpscunha/german-credit-risk.git
cd path/german-credit-risk


2. install pipenv if it is not already installed 

pip install pipenv


3. In the root directory of the project, where the `Pipfile` is located, set up the virtual environment and install the dependencies:

pipenv install

### Docker 

1. You can also download the docker image and run a container using the following code:

$ docker build -t german-credit-risk .

or 

$ docker pull marcosbenicio/german-credit-risk:latest

2. After the image is built, you can run this code: 

docker run -p 9696:9696 german-credit-risk

### Cloud Deployment

No cloud deployment yet