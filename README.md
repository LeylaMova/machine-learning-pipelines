# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 5: Feature Selection + Classification

### Domain and Data

This week, you've learned about access and utilizing remote databases, and more advanced topics for conducting logistic regression, selecting features, and building machine learning pipelines. Now, let's put these skills to the test!

You're working as a data scientist with a research firm. You're firm is bidding on a big project that will involve working with thousands or possible tens of thousands of features. You know it will be impossible to use conventional feature selection techniques. You propose that a way to win the contract is to demonstrate a capacity to identify relevant features using machine learning. Your boss says, "Great idea. Write it up." You figure that working with the [Madelon](https://archive.ics.uci.edu/ml/datasets/Madelon) synthetic dataset is an excellent way to demonstrate your abilities. 

A data engineer colleague sets up a remote PostgreSQL database for you to work with. You can connect to that database at `joshuacook.me:5432` with user `dsi` and password "`correct horse battery staple`". You can connect via command line using

	$ psql -h joshuacook.me -p 5432 -d dsi -U dsi_student
	
and entering the password when prompted

(Optional) You tell your colleague, thanks, but you prefer to run your database locally using docker. 

Regardless of whether you use the remote database or Docker, your colleague encourages you to use `sqlalchemy` to connect postgres to pandas. He suggests that the following code might be useful but seems distracted and rushed and tells you to check stack when you push for more:

    engine = create_engine("postgresql://{}:{}@{}:{}/{}".format(user, password, url, port, database))

### Problem Statement

Your challenge here is to implement three machine learning pipelines designed to demonstrate your ability to select salient features programatically. 

### Solution Statement

Your final product will consist of:

1. A prepared report
2. Three Jupyter notebooks to be used to control your pipelines
3. A library of python code you will use to build your pipelines

##### Pipeline 1: Benchmarking

<img src="assets/benchmarking.png" width="600px">

##### Pipeline 2: Select Features with $\ell1$-Penalty

<img src="assets/identify_features.png" width="600px">

##### Pipeline 3: Build Model with Grid Search

<img src="assets/build_model.png" width="800px">

### Tasks

##### Prepared Report

Your report should
1. be a pdf
2. include a well-posed problem statement with Domain, Data, Problem, Solution, Metric, Benchmark
3. optionally include EDA & Data Description
4. present results from Step 1 - Benchmarking
5. present results from Step 2 - Identify Salient Features
6. present results from Step 3 - Build Model
7. compare results obtained by LogisticRegression and KNearestNeighbors in Step 3
8. present compare features identified as important by Step 2 and Step 3
9. recommend feature engineering steps for a potential next phase in project

##### Jupyter Notebook 1, Step 1 - Benchmarking
1. build pipeline to perform a naive logistic regression as a baseline model
	- in order to do this, you will need to set a high `C` value in order to perform minimal regularization

##### Jupyter Notebook, Step 2 - Identify Features
1. build pipeline with `LogisticRegression` using l1 penalty
2. use constructed model to identify important features

##### Jupyter Notebook, Step 3 - Build Model
1. construct a Pipeline that uses `SelectKBest` to transform data
2. construct a Pipeline that uses `LogisticRegression` to model data
3. construct a Pipeline that uses `KNearestNeighbors` to model data
4. Gridsearch optimal parameters for logistic regression and KNN

##### Library of Python Code
1. write docstrings for all wrapper function that describe inputs and outputs
1. write wrapper function to connect remote datasource to pandas
    - receives database information
	- queries and sorts data
	- returns a dataframe
2. write wrapper function to split data into a dictionary object to be passed through pipeline
	- receives a dataframe and a random state
	- performs train test split
	- returns a data dictionary containing all necessary data objects
4. Write wrapper function to perform a general transformation on data
	- receives a data dictionary
	- fits on train data
	- transforms train and test data
	- return a data dictionary with updated train and test data and transformer
5. Write wrapper function to build a general model using data
	- receives a data dictionary
	- fits on train data
	- scores on train and test data
	- return a data dictionary adding model and scores 	

---

### Requirements

- A local PostgreSQL database housing your remote data.
- A Jupyter Notebook with the required problem statement, goals, and technical data.
- A written report of your findings that detail the accuracy and assumptions of your model.

- ***Bonus:***
- Create a blog post of at least 500 words (and 1-2 graphics!) describing your data, analysis, and approach. Link to it in your Jupyter notebook.

---

### Necessary Deliverables / Submission

- Materials must be in a clearly labeled Jupyter notebook.
- Materials must be submitted via a Github PR to the instructor's repo.
- Materials must be submitted by the end of week 5.

---

### Starter code

DSI_SM_3/projects/project-05 (master)$ tree
├── README.md
├── assets
│   ├── benchmarking.png
│   ├── build_model.png
│   └── identify_features.png
├── lib
│   ├── __init__.py
│   ├── project_5.py
├── project-05-rubric.md
├── step_1-benchmarking.ipynb
├── step_2-identify_features_l1_penalty.ipynb
└── step_3-build_model.ipynb

---

### Suggested Ways to Get Started

- Read in your dataset
- Write pseudocode before you write actual code. Thinking through the logic of something helps.  
- Read the docs for whatever technologies you use. Most of the time, there is a tutorial that you can follow, but not always, and learning to read documentation is crucial to your success!
- Document **everything**.
- Look up sample executive summaries online.

---

#### Project Feedback + Evaluation

[Attached here is a complete rubric for this project.](./project-05-rubric.md)

Your instructors will score each of your technical requirements using the scale below:

    Score | Expectations
    ----- | ------------
    **0** | _Incomplete._
    **1** | _Does not meet expectations._
    **2** | _Meets expectations, good job!_
    **3** | _Exceeds expectations, you wonderful creature, you!_

 This will serve as a helpful overall gauge of whether you met the project goals, but __the more important scores are the individual ones__ above, which can help you identify where to focus your efforts for the next project!
