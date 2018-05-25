
import Q0
import Q1
import Q2

# Q0
rawData = Q0.get_raw_data()
print ("Data loaded")
# Q1
clean_data = Q1.clean(rawData)
print ("Data cleaned")

print("starting extracting features..")
Q2.extract_features(clean_data)

