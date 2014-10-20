#UnBIASeD (Unified Bayesian Iterative And Systematically Determined) Pressure Scale Project

Welcome to the home of the UnBIASeD pressure scale project!
Our goal here is to continuously provide well calibrated pressure markers for high pressure and temperature experiments.
These experiments are crucial to our understanding of material properties at extreme conditions, with major applications in both materials science and Earth Science. 

We approach this goal using a few major strategies: 
* providing an open database of published experiments
* maintaining a Bayesian framework for interpreting those measurements
* reviewing and comparing with other pressure scales

##An Open Database of Mineral Physics Pressure-Scale Experiments
In order to uniformly treat and analyze data across many different sources, it is crucial to convert it from its initial published state into a uniform and highly organized database.
This is accomplished in two steps:
1. Data is extracted from tables, figures, and text descriptions within a publication and put into easily analyzed csv text files
2. The data is read into a python script that converts it to the required format and writes it to the project database

The database is stored as a set of SQLite tables, chosen for their robust search and storage features, while remaining relatively simple (if not initially familiar to the scientific community).
The database is most easily edited using the free and easy [DB Browser for SQLite](http://sqlitebrowser.org/).
Reading data from the database into the statistical analysis code is performed by the [HTSQL](http://htsql.org/) library for python, providing a **much** simpler and more intuitive interface to database reading and filtering than SQL itself.

###*Important Note*
The SQLite database is *not* kept under version control directly (since it is a binary file), but rather is written to a text file using sqlite's *dump* command.
This is accomplished simply by issuing the following command within the database directory after saving changes:

```
sqlite3 target_database_name.db .dump > target_database_name.bak
```

##Database Contributions are Welcome
As this is an entirely open project, we welcome contributions to the database.
The best way to contribute is to fork this repo, make changes to your forked repo and then make a pull request.
More detailed instructions for how to do this will be added soon.

##Project Members:
* Aaron S. Wolf (project lead)
* Rong Zhou (undergraduate assistant)
* Wardah M. Fadil (undergraduate assistant)
