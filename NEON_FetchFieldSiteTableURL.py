from urllib.request import urlopen
from bs4 import BeautifulSoup

# ======================================================================================================================
# THIS FUNCTION FETCHES THE FIELD SITE TABLE FROM THE NEON SITE AND PARSES ITS INFORMATION.
# ======================================================================================================================
def NEON_FetchFieldSiteTableURL(my_url):
    csv_links = []

    # r = requests.get(my_url)
    #ssl._create_default_https_context = ssl._create_unverified_context
    r = urlopen(my_url)
    # context = ssl._create_unverified_context()
    # r = urlopen(my_url, context=context)

    if 1:
        # if r.status_code == 200:
        # create beautiful-soup object
        # soup = BeautifulSoup(r.content, 'html5lib')
        soup = BeautifulSoup(r, 'html5lib')

        # FIND ALL CSV LINKS ON THE WEB-PAGE. CURRENTLY THERE IS ONLY ONE. HOWEVER, THERE COULD BE MULTIPLES IN THE FUTURE
        links = soup.findAll("a", href=lambda href: href and "csv" in href)

        # CREATE COMPLETE URL FOR LINK TO CSV FILE. ASSUME THERE IS ONLY ONE FOR NOW BUT LOOP FOR FUTURE USE-CASES
        for link in links:
            #csvLink = root_url + link['href']
            csvLink = link['href']

        csv_links = csvLink

    return csv_links


