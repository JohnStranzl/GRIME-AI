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
class GRIME_AI_ImageStats:
    def __init__(self, blurValue=0.0, brightnessValue=0.0, label=''):
        self.blurValue = blurValue
        self.brightnessValue = brightnessValue
        self.label = label

    def setBlurValue(self, blurValue):
        self.blurValue = blurValue

    def getBlurValue(self):
        return self.blurValue

    def setBrightnessValue(self, brightnessValue):
        self.brightnessValue = brightnessValue

    def getBrightnessValue(self):
        return self.brightnessValue

    def setLabel(self, label):
        self.label = label

    def getLabel(self):
        return self.label