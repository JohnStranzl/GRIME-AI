import requests

# ======================================================================================================================
#
# ======================================================================================================================
def NEON_FetchSiteInfoFromNEON(server, siteCode):

    # Make request, using the sites endpoint
    site_request = requests.get(server + 'sites/' + siteCode)

    # Convert to Python JSON object
    site_json = site_request.json()

    return (site_json)


