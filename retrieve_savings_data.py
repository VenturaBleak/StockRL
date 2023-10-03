import quandl

def fetch_fed_funds_rate(api_key, start_date, end_date):
    quandl.ApiConfig.api_key = api_key
    data = quandl.get("FRED/DFF", start_date=start_date, end_date=end_date)
    return data

# Replace 'YOUR_API_KEY' with your actual API key
fed_funds_rate = fetch_fed_funds_rate("xjs2eaks3tXCRaQqzoF5", "2000-01-01", "2023-01-01")
print(fed_funds_rate)

# save the data to a CSV file
fed_funds_rate.to_csv("data/fed_funds_rate.csv")
print("Saved data to data/fed_funds_rate.csv")