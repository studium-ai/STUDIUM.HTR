
import requests, os, argparse, re
from xml.etree import ElementTree
from tqdm import tqdm
from pathlib import Path

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]
a4_size = (2479, 3508)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str)
    parser.add_argument("--username", type=str, help= 'Transkribus username')
    parser.add_argument("--password", type=str,  help='Transkribus password')
    parser.add_argument("--collection_id", type=str, help='collection id (Transkribus: https://app.transkribus.org/collection/collectsion_ids/doc/document_id)')
    parser.add_argument("--document_id", type=str,  help='document id (see above)')
    args = parser.parse_args()

    xmls = os.listdir(args.xml_path)
    xmls = sorted(xmls, key=natural_key)
    login_url = 'https://transkribus.eu/TrpServer/rest/auth/login'
    login_response = requests.post(login_url, data={'user': args.username, 'pw': args.password})

    if login_response.status_code == 200:
        root = ElementTree.fromstring(login_response.content)
        session_id = root.findtext('.//sessionId')
        print(f"Logged in! Session ID: {session_id}")
    else:
        print("Login failed! Check username/password.")
        exit()

    for idx, xml_name in tqdm(enumerate(xmls)):
        # image_name_split = image_name[:-4]
        XML_FILE_PATH = os.path.join(args.xml_path, xml_name)
        # generate_page_xml(image_name, a4_size[0], a4_size[1], lines_data, XML_FILE_PATH)
        PAGE_NR = str(idx+1)
        upload_url = f"https://transkribus.eu/TrpServer/rest/collections/{args.collection_id}/{args.document_id}/{PAGE_NR}/text"
        cookies = {'JSESSIONID': session_id}
        headers = {
            'Content-Type': 'application/xml',
            'Accept-Charset': 'UTF-8' }
        with open(XML_FILE_PATH, "r", encoding="utf-8") as file:
            xml_data = file.read().encode('utf-8')
        upload_response = requests.post(upload_url, data=xml_data, headers=headers, cookies=cookies)

        if upload_response.status_code == 200:
            print("XML uploaded successfully!")
        else:
            print(f"Upload failed! Status Code: {upload_response.status_code}")
            print("Response:", upload_response.text)
