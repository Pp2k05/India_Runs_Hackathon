import os
import sys
import urllib.request
import urllib.parse
import http.cookiejar
import re
import zipfile

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    
    # Set up cookie jar to handle session cookies
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
    
    print(f"Requesting download link for file ID: {file_id}...")
    # Make the initial request
    params = {'export': 'download', 'id': file_id}
    query_string = urllib.parse.urlencode(params)
    full_url = f"{URL}&{query_string}"
    
    try:
        response = opener.open(full_url)
        content = response.read()
    except Exception as e:
        print(f"Error during initial request: {e}")
        sys.exit(1)
        
    # Check if we got the warning page or the file itself
    # If it's the file, the header Content-Disposition will usually contain 'attachment'
    content_disposition = response.info().get('Content-Disposition', '')
    if 'attachment' in content_disposition:
        print("Direct download started...")
        save_file(response, content, destination)
        return
        
    # If not direct, look for confirmation token in the HTML content
    html_text = content.decode('utf-8', errors='ignore')
    confirm_token = None
    
    # Try different regex patterns to extract the token
    # pattern 1: look for confirm=XXXX in links or forms
    match = re.search(r'confirm=([a-zA-Z0-9_-]+)', html_text)
    if match:
        confirm_token = match.group(1)
    else:
        # pattern 2: check if there is a form value/input named "confirm"
        match = re.search(r'name="confirm" value="([a-zA-Z0-9_-]+)"', html_text)
        if match:
            confirm_token = match.group(1)
            
    if confirm_token:
        print(f"Found confirmation token: {confirm_token}. Requesting download...")
        params['confirm'] = confirm_token
        query_string = urllib.parse.urlencode(params)
        download_url = f"{URL}&{query_string}"
        
        try:
            download_response = opener.open(download_url)
            # Write to destination
            with open(destination, 'wb') as f:
                # Read in chunks to handle large files
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = download_response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"Download complete! Saved to {destination}")
        except Exception as e:
            print(f"Error during token-based download: {e}")
            sys.exit(1)
    else:
        # If no token, maybe it's just small or warning page didn't show up but was returned
        print("No confirmation token found in HTML. Saving initial response content...")
        with open(destination, 'wb') as f:
            f.write(content)
        print(f"Download complete! Saved to {destination}")

def save_file(response, content, destination):
    with open(destination, 'wb') as f:
        f.write(content)
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    print(f"Download complete! Saved to {destination}")

if __name__ == "__main__":
    file_id = "1MfD47XvVdRKBGRAyzGOxDCEf2ve96Jjo"
    
    # Ensure data directory exists
    project_dir = "C:/Users/parth/OneDrive/Documents/India-runs/ai_candidate_ranker"
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, "dataset.zip")
    
    download_file_from_google_drive(file_id, zip_path)
    
    # Extract
    print(f"Extracting {zip_path} to {data_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete!")
    except Exception as e:
        print(f"Error extracting ZIP: {e}")
        sys.exit(1)
