import os
import requests

# ======================================================================================================================
#
# ======================================================================================================================
def NEON_DownloadFieldSiteTableFiles(csv_links):
    link = csv_links

    # obtain filename by splitting url and getting last string
    file_name = link.split('/')[-1]

    # create response object
    r = requests.get(link, stream=True)

    configFilePath = os.path.expanduser('~')
    configFilePath = os.path.join(configFilePath, 'Documents')
    configFilePath = os.path.join(configFilePath, 'GRIMe-AI')
    if not os.path.exists(configFilePath):
        os.mkdir(configFilePath)
    filename_with_path = os.path.join(configFilePath, file_name)

    # download started
    with open(filename_with_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return filename_with_path
