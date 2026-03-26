# predict_today.py

import datetime

# Get today's date
current_date = datetime.datetime.utcnow()

print(f"Today's date is: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
