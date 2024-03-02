# Cafe Sales Analytics

**ATTENTION**: Never push any credentials or sensitive information to the repository. Always use environment variables or other secure methods to store sensitive information. When using Jupyter Notebooks, make sure to clear the output before pushing the notebook to the repository.

## Project Description

This project aims to conduct sales analysis for a cafe. The data is provided in Google BigQuery and the analysis is performed using SQL and Python.

Expected outcomes of the project include:

- A blog post or report detailing the findings of the analysis.
- A presentation summarizing the findings of the analysis to the stakeholders.
- A updated resume section detailing the project outcomes and the skills used.

## Environemtn Setup

### Prerequisites

1. Make sure Python is installed on your computer. You can download and install Python from the official website: [Python Downloads](https://www.python.org/downloads/).

### Install Required Libraries

1. Open a command prompt or terminal on your computer.
2. Use pip, Python's package installer, to install the required libraries by running the following command:

    ```bash
    pip install pandas pandas-gbq matplotlib seaborn
    ```

### Set Up Google Cloud Platform (GCP) Credentials

1. If you don't already have a Google Cloud Platform (GCP) account, you'll need to create one. Visit [Google Cloud Platform](https://cloud.google.com/) to sign up.
2. Create a new project or use an existing one. (Please note that you may incur charges for using GCP services, so make sure to review the pricing information.)
3. Enable the BigQuery API for your project. You can do this from the GCP Console under APIs & Services > Library.
4. (Recommended) Set up authentication by creating a service account key. Go to IAM & Admin > Service accounts, select your service account or create a new one, and then create a new key in JSON format. Save this JSON file securely on your computer.
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to the location of your service account key JSON file. You can do this by running the following command in the terminal or command prompt:

       ```bash
       export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
       ```

       Replace "/path/to/your/service-account-key.json" with the actual path to your service account key JSON file.
5. (Easier Solution, not recommended for production) Use the `gcloud` command-line tool to authenticate. Run the following command in the terminal or command prompt:

    ```bash
    gcloud auth application-default login
    ```

    This will open a browser window asking you to log in to your GCP account. After logging in, you will be authenticated and the credentials will be stored on your computer.

### Run the demo Python notebook

1. Open the `demo.ipynb` notebook in Jupyter Notebook.
2. Run the cells in the notebook to see the demo of the cafe sales analytics.
3. Make sure to clear the output before pushing the notebook to the repository.

Congratulations! You have successfully executed the Python code to query data from Google BigQuery on your new computer.
