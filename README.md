import kagglehub

# Download latest version
path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")

import pandas as pd
# View the first five rows
df.head()
# Data description
df.info()
# List unique values ​​for the hotel column
df['hotel'].unique()

# Check for missing values
print(df.isna().sum())

# Fill in missing values
df['children'] = df['children'].fillna(0)

# Drop duplicates
df = df.drop_duplicates()

# Create a new feature
df['total_guests'] = df['adults'] + df['children'] + df['babies']

# Rename the adr column
df = df.rename(columns={'adr': 'price_per_night'})

# Filtering and selecting data
hotel_price = df[(df['hotel'] == 'City Hotel') & (df['price_per_night'] > 150)]
print(hotel_price)

print(df.iloc[:10, :5])

prt_bookings = df.loc[df['country'] == 'PRT']
print(prt_bookings)

# Groupings and aggregations
print(df.groupby('hotel', as_index = False).agg({'price_per_night':'mean'}))

country = df['country'].value_counts().index[0]
count = df['country'].value_counts().values[0]
print(f"Country from which guests most often come: {country} ({count} bookings)")

pivot_table = df.pivot_table(
values='price_per_night',
index='hotel',
columns='arrival_date_month',
aggfunc='mean'
)
print(pivot_table)

import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(data=df, x='hotel', y='price_per_night')
plt.xlabel('Hotel')
plt.ylabel('Price per night')
plt.title('Price distribution by hotel')

df['total_guests'] = df['adults'] + df['children'] + df['babies']
avg_price_by_guests = df.groupby('total_guests')['price_per_night'].mean()
sns.barplot(x=avg_price_by_guests.index, y=avg_price_by_guests.values)
plt.xlabel('Number of guests')
plt.ylabel('Average price per night')
plt.title('Relationship between the number of guests and the average price')

df['revenue'] = df['price_per_night'] * df['total_guests']
monthly_revenue = df.groupby('arrival_date_month', as_index = False).agg({'revenue':'sum'}).sort_values(by=['revenue'], ascending = False)
print(monthly_revenue.head(5))

# Task: Segment guests by behavior (Clustering)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

categorical_features = ['hotel', 'arrival_date_month', 'meal', 'market_segment', 'distribution_channel', 'deposit_type', 'customer_type']
numeric_features = ['lead_time', 'total_guests', 'price_per_night', 'stays_in_weekend_nights', 'stays_in_week_nights']

X = df[categorical_features + numeric_features].copy()

preprocessor = ColumnTransformer( 
transformers=[ 
('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), 
('num', StandardScaler(), numeric_features) 
])

X_all_processed = preprocessor.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_all_processed)

df['cluster'] = clusters

print("Dimensions of clusters:")
print(df['cluster'].value_counts())

print("\nAverage characteristics by clusters:")
print(df.groupby('cluster')[numeric_features].mean().round(2))

plt.scatter(df['price_per_night'], df['lead_time'], c=df['cluster'])
plt.xlabel('Price per night')
plt.ylabel('Lead time')
plt.title('Price per night vs Lead Time')

# Cluster 0 - customers who book in advance, pay a high nightly rate, often choose resort hotels, and rarely cancel.
# Cluster 1 - families or groups who travel with large groups, stay long-term, and have a medium budget.
# Cluster 2 - short-term or business guests who book at the last minute, stay a few nights, pay a low rate, prefer city hotels, and frequently cancel.
