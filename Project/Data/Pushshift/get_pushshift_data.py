#!/bin/python
# -*- coding: utf-8 -*-
# Filippo Libardi & Otto Mättas
# This script helps with getting data from the Pushshift Reddit API
# See more: https://github.com/pushshift/api

# Versioning
# v0.1 Defining base elements

# Import the requests library
import requests

###############################
### Define query parameters ###
###############################
# Domain of the query
## accepts "comment", "submission"
### The "submission" domain is not fully developed into our script yet
domain = "comment"

# Search term;
## accepts String or Quoted String for phrases
q = "science"

# Get specific comments via their IDs;
## accepts Comma-delimited Base36 IDs
#ids = "f9jyz1r"

# Number of results to return;
## accepts Integer <= 500
size = 10

# Only return specific fields (comma delimited); all fields returned by default;
## accepts String or comma-delimited String
fields = "author","body"

# Sort results in a specific order; "desc" by default;
## accepts "asc", "desc"
sort = "desc"

# Sort by a specific attribute; "created_utc" by default;
## accepts "score", "num_comments", "created_utc"
sort_type = "created_utc"

# Return aggregation summary;
## accepts ["author", "link_id", "created_utc", "subreddit"]
aggs = "created_utc"

# Used with the aggs parameter when set to created_utc;
## accepts "second", "minute", "hour", "day"
frequency = "second"

# Restrict to a specific author;
# accepts String
author = "MockDeath"

# Restrict to a specific subreddit;
## accepts String
subreddit = "askscience"

# Return results after this date;
## accepts Epoch value or Integer + "s,m,h,d" (i.e. 30d for 30 days)
after = "1217028326"

# Return results before this date;
# accepts Epoch value or Integer + "s,m,h,d" (i.e. 30d for 30 days)
before = "1317028326"

# Display metadata about the query; "false" by default;
## accepts "true", "false"
metadata = "false"

#domain = "submission" # comment/submission

# Define the API endpoint 
URL = "https://api.pushshift.io/reddit/search/" + domain

# Define a headers dict for the headers to be sent to the API 
HEADERS = {
    "Content-Type": "application/json",
#    "Authorization": "Bearer xXxXXXxxXXX"
}

# Define a params dict for the parameters to be sent to the API 
PARAMS = {
     "q": q,
#     "ids": ids,
     "size": size,
     "fields": fields,
     "sort": sort,
     "sort_type": sort_type,
     "aggs": aggs,
     "frequncy": frequency,
     "author": author,
     "subreddit": subreddit,
     "after": after,
     "before": before,
     "metadata": metadata,
}

# Send a GET request and save the response as response object
r = requests.get(url = URL, headers = HEADERS, params = PARAMS)

# DEBUG for status code
#print(r.status_code)

# Extract data from the response object in JSON format 
output = r.json()

# Print output
print(output)
