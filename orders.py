import pandas as pd
from sqlalchemy import create_engine,text

# Assuming you're using the 'pyodbc' driver for MS SQL Server
db_url = 'mysql+pymysql://root:root@localhost:3306/client_server'
engine = create_engine(db_url)

csv_file_path = 'Orders.csv'
df = pd.read_csv(csv_file_path, encoding='latin-1')

# Extract columns for Category
table1_columns = ['Product Category']
table1_df = df[table1_columns]
print (table1_df)
# table_name = 'your_table_name'
# df.to_sql(table_name, engine, index=False, if_exists='replace')
# Extract columns for Subategory
table2_columns = ['Product Sub-Category']
table2_df = df[table2_columns]
print (table2_df)
# Extract columns for Regions
# Extract distinct values from the 'Region' column
distinct_regions = df['Region'].unique()

# Create a DataFrame with distinct regions
region_df = pd.DataFrame({'Region': distinct_regions})
print (region_df)
# Use to_sql to insert data into tables
table1_name = 'Category'
table2_name = 'Subategory'
table3_name = 'Regions'
table4_name = 'Customers'
table5_name = 'Products'
table6_name = 'OrderItems'
table7_name = 'Orders'

region_table_name = 'regions'
# Use to_sql to insert distinct regions into the 'Regions' table
# Check if regions already exist in the database
existing_regions_query = "SELECT DISTINCT Region FROM regions"
#existing_regions = pd.read_sql((text(existing_regions_query)), engine)
existing_regions = pd.DataFrame(engine.connect().execute(text(existing_regions_query)))
print(existing_regions)
#s_settings_df = pd.DataFrame(engine_cloud.connect().execute(text(query)))
# Filter out regions that already exist
new_regions_df = region_df[~region_df['Region'].isin(existing_regions['Region'])]
print(new_regions_df)
# If there are new regions, append them to the 'Regions' table
if not new_regions_df.empty:
    new_regions_df.to_sql(region_table_name, index=False, if_exists='append')
    print("New regions added to the 'Regions' table.")
else:
    print("No new regions to add.")
'''
table1_df.to_sql(table1_name, engine, index=False, if_exists='replace')
table2_df.to_sql(table2_name, engine, index=False, if_exists='replace')
table3_df.to_sql(table3_name, engine, index=False, if_exists='replace')
'''