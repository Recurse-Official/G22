import requests
from bs4 import BeautifulSoup
import os
import pdfplumber
def download_pdfs_from_url(url, download_folder='downloads'):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to access the URL: {url}")
        print(f"Error: {response.status_code} {response.reason}")
        return []
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find links that contain 'pdf'
    pdf_links = [
        link.get('href') for link in soup.find_all('a', href=True) 
        if 'pdf' in link.get('href')
    ]
    
    if not pdf_links:
        print("No PDF links found on the page.")
        return []

    downloaded_pdfs = []
    for link in pdf_links:
        file_url = link if link.startswith('http') else 'https://arxiv.org' + link
        file_name = os.path.join(download_folder, file_url.split('/')[-1] + ".pdf")
        try:
            pdf_response = requests.get(file_url)
            if pdf_response.status_code == 200:
                with open(file_name, 'wb') as f:
                    f.write(pdf_response.content)
                downloaded_pdfs.append(file_name)
                print(f"Downloaded: {file_name}")
            else:
                print(f"Failed to download PDF: {file_url} - {pdf_response.status_code}")
        except Exception as e:
            print(f"Error downloading {file_url}: {e}")
    
    return downloaded_pdfs

def parse_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text.strip()


def main(url):
    # Download PDFs
    pdfs = download_pdfs_from_url(url)
    if not pdfs:
        print("No PDFs downloaded. Exiting.")
        return
    
    # Extract text
    all_text = ""
    for pdf in pdfs:
        text = parse_pdf(pdf)
        if text:
            all_text += text + "\n"
    
    if not all_text:
        print("No text extracted from the PDFs.")
    else:
        print(f"Extracted Text:\n{all_text[:500]}") 
url = 'https://arxiv.org/list/cs.AI/recent'
main(url)
