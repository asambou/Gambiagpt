import requests
from bs4 import BeautifulSoup
import os

urls = [
    "https://en.wikipedia.org/wiki/The_Gambia",
    "https://en.wikipedia.org/wiki/Banjul",
    "https://en.wikipedia.org/wiki/History_of_the_Gambia",
    "https://en.wikipedia.org/wiki/Economy_of_the_Gambia",
    "https://en.wikipedia.org/wiki/Culture_of_the_Gambia"
]

save_folder = "data/documents"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for i, url in enumerate(urls):

    print(f"Downloading: {url}")

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = soup.find_all("p")

    text = ""

    for p in paragraphs:
        text += p.get_text()

    filename = f"{save_folder}/page_{i}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved {filename}")

print("Scraping complete.")
