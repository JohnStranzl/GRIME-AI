#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# ======================================================================================================================
#
# ======================================================================================================================
def googleMap(self):
    # Enter your api key here
    api_key = "_your_api_key_"

    # url variable store url
    url = "https://maps.googleapis.com/maps/api/staticmap?"

    # center defines the center of the map,
    # equidistant from all edges of the map.
    center = "Dehradun"

    # zoom defines the zoom
    # zoom defines the zoom
    # level of the map
    zoom = 10

    # get method of requests module
    # return response object
    r = requests.get(
        url + "center =" + center + "&zoom =" + str(zoom) + "&size = 400x400&key =" + api_key + "sensor = false")

    # wb mode is stand for write binary mode
    f = open('address of the file location ', 'wb')

    # r.content gives content,
    # in this case gives image
    f.write(r.content)

    # close method of file object
    # save and close the file
    f.close()

