import requests

SERVER = 'http://data.neonscience.org/api/v0/'

# ======================================================================================================================
# The purpose of this function is to query information above a specific product. The information
# contains the product description, sites for which the product is available among other information.
# ======================================================================================================================
def NEON_QueryProductInfo(productCode):
    product_request = requests.get(SERVER + 'products/' + productCode)
    product_json = product_request.json()

    return product_json


