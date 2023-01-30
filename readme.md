#### Project goals:

To determine features that contribute to higher home values in the Californian counties of  Orange, Ventura, and Los Angeles.


#### Project description:

This is a regression project on the 2017 Zillow dataset.  Data was acquired from the Codeup SQL server and imported into my python environment.


#### Project planning:

- Acquire
I acquired the data from Codeup's SQL server then saved it locally in a csv file.  
Properties selected were filtered by having a transaction in 2017 and being a single family home.  Each row represents a home in one of the three counties studied (Los Angeles, Orange, Ventura).  

- Prepare
To prepare the data I created a function to rename columns, set upper and lower limits to remove outliers, converted data types and replaced the fips 4 digit code with the name of the county.

I dropped columns with too many missing values.

The data was then split into Train, Validate, and Test dataframes.

- Explore
To plot various features against the target variable (tax_value) and look for visual signs of correlation.  After plotting
I performed various statistical tests to calculate relationship between variables.


- Model
Finally I scaled the data with MinMax scaler and fit and GLM model to make predictions


- Recommendations/Next Steps

I recommend gathering data for additional features such as:
    How many feet of Water Frontage on property
    Miles to nearest schools and their rating on greatschools.org
    Ratio of owners to renters within the neighborhood or within 5 mile radius


#### Some initial questions I asked were:

- Is county related to tax_value?
- Does total_sqft relate to tax_value?
- Does number of bedrooms relate to tax_value?
- Does number of bathrooms relate to tax_value?
- Does year_built relate to value?

#### Data dictionary

Feature | Description
------------- | -------------
bedrooms | number of bedrooms
bathrooms | number of bathrooms
year_built | Construction date
zip_code | location descriptor of property
total_sqft | measure of total living area
county | Orange, Ventura, or Los Angeles
tax_value | total tax value in 2017



To recreate this project in your IDE, download the zillow database from kaggle,
clones this repository and execute the files.  


#### Key Findings:

I found that total square footage, number of bedrooms, and number of bathrooms greatly contributed to higher tax values.  County had
a weak positive correlation but that may be due to the fact that these three counties are next to eachother.  If the dataset covered the entire state of
California then county would have much more significance in the model.

#### Takeaways:

This model improves upon the baseline for home value predicitions and is recommended for use on future unseen data.  Continued model fitting and feature engineering would improve the performance.