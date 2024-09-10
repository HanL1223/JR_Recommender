# Cafe Sales Analytics

**Data Security**: Remember to avoid pushing sensitive information to public repositories. Use environment variables or secure methods for storing credentials. Clear notebook outputs before sharing.

## Project Description

This project entails analyzing cafe sales data stored in Google BigQuery using SQL and Python. The analysis aims to:

- **Understand sales trends**: Analyze revenue, order value, peak hours, and popular days.
- **Examine product performance**: Identify best-selling items, average order size, and pricing trends.
- **Gain customer insights**: Track customer acquisition, retention, lifetime value, and order patterns.
- **Develop actionable recommendations**: Suggest strategies to boost revenue, customer retention, and optimize operations.

**Expected Deliverables:**

- Jupyter notebook containing analysis code.
- Presentation for stakeholders summarizing findings.
- Public blog post or report detailing insights and value.
- Updated resume section showcasing project outcomes and skills.

## Environment Setup

### Prerequisites

Ensure Python is installed on your computer. You can download and install Python from the [official website](https://www.python.org/downloads/).

Google Cloud SDK is required to authenticate and access Google BigQuery. Follow the instructions on [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) to install it on your computer.

### Install Required Libraries

1. Open a command prompt or terminal on your computer.
2. Use pip, Python's package installer, to install the required libraries by running the following command:

    ```bash
    pip install pandas pandas-gbq matplotlib seaborn google-auth
    ```

### Set Up Google Cloud Platform (GCP) Credentials

1. If you don't already have a Google Cloud Platform (GCP) account, you'll need to create one. Visit [Google Cloud Platform](https://cloud.google.com/) to sign up.
2. Provide your account details to the project administrator to grant you access to the cafe sales data in Google BigQuery.
3. Use the `gcloud` command-line tool to authenticate. Execute the following command in the terminal or command prompt:

    ```bash
    gcloud auth application-default login
    ```

    This action will prompt a browser window to log in to your GCP account. After logging in, your credentials will be authenticated and stored on your computer.

### How to Install GCloud and Ensure It Always Works After Restart on Mac OS High Sierra

If you're using Visual Studio Code as your IDE and need GCP authentication, you might encounter the "gcloud not found" command. Follow these steps to install GCloud and ensure it always works after restarting your Mac OS High Sierra:

1. **Download the install package**:
   - [Download the Google Cloud SDK install package](<https://cloud.google.com/sdk/docs/install-sdk>).

2. **Extract the package**:
   - After downloading the package, extract the contents and drop them into a folder of your choice.

3. **Open Terminal**:
   - Open Terminal and navigate to the folder where you placed the extracted files.

4. **Run the installation script**:
   - In the Terminal, execute the following command:

     ```bash
     ./google-cloud-sdk/install.sh
     ```

5. **Modify profile**:
   - When prompted, select "Yes" to modify your profile and update your `$PATH` to enable bash completion.

6. **Enter the path to modify**:
   - Enter the path to modify, typically:

     ```
     /Users/USERNAME_COMPUTER/.bashrc
     ```

7. **Source the profile**:
   - After the installation completes, enter the following command:

     ```bash
     source ~/.bashrc
     ```

8. **Verify the installation**:
   - Check if GCloud is installed properly by entering:

     ```bash
     gcloud --version
     ```

9. **Open a new Terminal window**:
   - Do not close the old Terminal window. Open a new one (Cmd + N) and enter:

     ```bash
     gcloud --version
     ```

10. **Verify GCloud is working**:
    - If you still see "command not found," proceed to step 11. Otherwise, congratulations, GCloud is working in Terminal.

11. **Update your PATH in BASH_PROFILE**:
    - Open the BASH_PROFILE file by entering the following command:

      ```bash
      open ~/.bash_profile
      ```

    - Add the following line to the file:

      ```
      export PATH="/Users/USERNAME_COMPUTER/google-cloud-sdk/bin:$PATH"
      ```

    - Save the changes and return to step 8.

Follow these steps carefully to install GCloud and ensure it works seamlessly with your Visual Studio Code setup on Mac OS High Sierra.

### Run the Demo Python Notebook

1. Open the `demo.ipynb` notebook in Jupyter Notebook.
2. Execute the cells in the notebook to view the cafe sales analytics demo.
3. Ensure to clear the output before pushing the notebook to the repository.

Congratulations! You have successfully executed the Python code to query data from Google BigQuery on your new computer.

## Problems Statement

The key problems to address in this project are:

- What is values your analytic report can bring to your stakeholders (i.e. the Cafe’s owner)

### Data Description

The cafe sales data is provided in Google BigQuery, containing the following columns available since May 2019:

- `order_id`: Unique identifier for each order.
- `customer_id`: Unique identifier for each customer.
- `ip_addr`: IP address of the customer when placing the order (hashed for privacy).
- `date_created`: Date and time when the order was created.
- `date_paid`: Date and time when the order was paid.
- `total`: Total amount of the order, in cents.
- `status`: Status of the order (e.g., pending, paid, refunded). `2` indicates a successful order.
- `items`: JSON array containing details of the items in the order.

### Questions

#### Descriptive Analysis

##### Revenue Trend

1. Monthly revenue trend analysis.
2. Average order value examination.
3. Identification of peak revenue hours.
4. Determining the most profitable day of the week.
5. Evaluation of the average revenue per customer.

##### Product Analysis

1. Identification of the top 5 best-selling items.
2. Calculation of the average number of items per order.
3. Determination of the average price per item.
4. Examination of the relationship between average price per item and average quantity sold per item.
5. Comparison of the average price per item between new and existing items

##### Customer Analysis

1. Monthly acquisition of new customers.
2. Calculation of customer retention rate.
3. Assessment of average customer lifetime value.
4. Determination of the average time between orders.
5. Evaluation of the average number of orders per customer.
6. Comparison of average order value for new customers versus returning customers.
7. Analysis of the average time between orders for new customers versus returning customers.

#### Predictive Analysis

1. Predictive modeling for daily revenue.
2. Forecasting the number of new customers each month.
3. Prediction of customer retention rate based on the first order.
4. Forecasting average customer lifetime value based on purchase history.

#### Prescriptive Analysis

1. Recommendations to enhance the cafe's revenue.
2. Strategies to improve customer retention at the cafe.
3. Measures to increase the average customer lifetime value.
4. Suggestions for optimizing the cafe's opening hours.
5. Recommendations for enhancing the cafe's menu offerings.

### Deliverables

0. A Jupyter notebook containing SQL and Python code for the analysis.
1. A presentation summarizing the analysis findings to stakeholders.
2. A blog post or report detailing the analysis findings and value to the public.
3. An updated resume section detailing project outcomes and skills used.

## Recommended Workflow

1. **Data Collection**: Extract data from Google BigQuery using SQL.
2. **Data Exploration**: Explore data using Python and Pandas, understand data types and structure, and identify any missing or inconsistent data.
3. **Data Cleaning**: Clean data by handling missing or inconsistent data, and convert data types if necessary.
4. **Data Preprocessing**: Preprocess data by transforming data types, normalizing data, and creating new features if necessary. For example, build a product table from the `items` column, and a customer table from the `customer_id` column.
5. **Descriptive Analysis**: Conduct descriptive analysis to answer questions about revenue trend, product analysis, and customer analysis.
6. **Predictive Analysis**: Conduct predictive analysis to predict daily revenue, number of new customers, customer retention rate, and average customer lifetime value.
7. **Prescriptive Analysis**: Conduct prescriptive analysis to provide recommendations for increasing revenue, customer retention rate, and average customer lifetime value, and for optimizing opening hours and menu.
8. **Code Review**: Review SQL and Python code with peers and provide feedback.
9. **Blog Post or Report**: Detail analysis findings and value in a blog post or report to the public.
10. **Wrap up**
    1. **Resume Update**: Update resume section detailing project outcomes and skills used.
    2. **Project Review**: Review project with peers and provide feedback.
    3. **Project Showcase**: Showcase project to the public and potential employers.
    4. **Project Retrospective**: Reflect on the project and identify areas for improvement.

## Resources

- [Growth Share Matrix](https://www.bcg.com/about/overview/our-history/growth-share-matrix)
- [Customer Churn Rate](https://www.zendesk.com/au/blog/customer-churn-rate/)
- [RFM Analysis](https://www.optimove.com/resources/learning-center/rfm-segmentation)
- [Customer Lifetime Value](https://www.shopify.com/encyclopedia/customer-lifetime-value-clv)
- [Predictive Analytics](https://www.ibm.com/cloud/learn/predictive-analytics)
- [Prescriptive Analytics](https://www.ibm.com/cloud/learn/prescriptive-analytics)
- [Prophet](https://facebook.github.io/prophet/)
