import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import requests 
import json 

# Read image
image_path = 'data/booktest2.png'

img = cv2.imread(image_path)

# instance text detector
reader = easyocr.Reader(['en'], gpu=True)

# detect text on image
text_ = reader.readtext(img)

threshold = 0.25

# draw bbox and text
for t_, t in enumerate(text_):
    print(t)

    bbox, text, score = t

    if score > threshold:
        cv2.rectangle(img, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
        cv2.putText(img, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

# Concatenate detected texts into a single string
result = " ".join([t[1] for t in text_ if t[2] > threshold])

# Print the concatenated text
print("Detected Text: ", result)

# Use the Google Books API to search for the top match
google_books_api_url = "https://www.googleapis.com/books/v1/volumes"
params = {"q": result, "maxResults": 1}
response = requests.get(google_books_api_url, params=params)

# Print the JSON format with book details
if response.status_code == 200:
    book_data = response.json()

    if 'items' in book_data and len(book_data['items']) > 0:
        book_info = book_data['items'][0]['volumeInfo']

        # Extract and print the desired fields
        title = book_info.get('title', 'N/A')
        authors = ', '.join(book_info.get('authors', ['N/A']))
        categories = ', '.join(book_info.get('categories', ['N/A']))
        maturityRating = book_info.get('maturityRating', 'N/A')
        description = book_info.get('description', 'N/A')

        print(f"\nBook Details:")
        print(f"Title: {title}")
        print(f"Author: {authors}")
        print(f"Category: {categories}")
        print(f"Mature Rating: {maturityRating}")
        print(f"Description: {description}")
    else:
        print("No book details found for the given query.")
else:
    print(f"\nFailed to retrieve book details. Status Code: {response.status_code}")

    
# Display the image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()