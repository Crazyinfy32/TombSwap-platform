import cv2
import easyocr
import matplotlib.pyplot as plt
import requests
import time

def detect_text(image_path):
    # Start timer 
    start_ocr_time = time.time()
    
    # Read image
    img = cv2.imread(image_path)
    reader = easyocr.Reader(['en', 'ko'], gpu=True)
    text_ = reader.readtext(img)

    # End timer 
    end_ocr_time = time.time()
    ocr_time = end_ocr_time - start_ocr_time
    print(f"Time taken for text detection with EasyOCR: {ocr_time:.2f} seconds")
    
    return img, text_

def draw_text_boxes(image, text_data, threshold=0.25):
    # Draw bbox and text
    for t_, t in enumerate(text_data):
        bbox, text, score = t

        if score > threshold:
            cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
            cv2.putText(image, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

    return image

def concat_text(text_data, threshold=0.25):
    # Concatenate detected texts into a single string
    result = " ".join([t[1] for t in text_data if t[2] > threshold])
    return result

def search_book_google_api(query):
    # Start timer
    start_api_time = time.time()

    # Search for the top match
    google_books_api_url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": query, "maxResults": 1}  
    response = requests.get(google_books_api_url, params=params)

    # End timer 
    end_api_time = time.time()
    api_time = end_api_time - start_api_time
    print(f"Time taken for Google Books API request: {api_time:.2f} seconds")

    return response

def extract_book_details(response):
    if response.status_code == 200:
        book_data = response.json()

        if 'items' in book_data and len(book_data['items']) > 0:
            book_info = book_data['items'][0]['volumeInfo']

            # Extracting relevant info
            title = book_info.get('title', 'N/A')
            authors = ', '.join(book_info.get('authors', ['N/A']))
            categories = ', '.join(book_info.get('categories', ['N/A']))
            maturityRating = book_info.get('maturityRating', 'N/A')
            description = book_info.get('description', 'N/A')

            return title, authors, categories, maturityRating, description
        else:
            return None
    else:
        print(f"\nFailed to retrieve book details. Status Code: {response.status_code}")
        return None

def display_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def main():
    image_path = 'data/book7.png'

    # Detect text on the image
    image, text_data = detect_text(image_path)
    # Draw text boxes on the image
    image_with_boxes = draw_text_boxes(image, text_data)

    # Concatenate detected text
    detected_text = concat_text(text_data)
    print(detected_text)

    # Search book using Google Books API
    response = search_book_google_api(detected_text)

    # Extract book details
    if response:
        book_details = extract_book_details(response)

        if book_details:
            title, authors, categories, maturityRating, description = book_details
            print(f"\nBook Details:")
            print(f"Title: {title}")
            print(f"Author: {authors}")
            print(f"Category: {categories}")
            print(f"Mature Rating: {maturityRating}")
            print(f"Description: {description}")

    # Display the image with text boxes
    display_image(image_with_boxes)

if __name__ == "__main__":
    main()
