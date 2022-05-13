import datetime

class GRIMe_ProductTable():
    def __init__(self):
        self.nStartYear = 0
        self.nStartMonth = 0
        self.nStartDay = 0
        self.strStartDate = ''
        self.start_date = datetime.date(1970, 1, 1)
        self.startTime = 0

        self.nEndYear = 0
        self.nEndMonth = 0
        self.nEndDay = 0
        self.strEndDate = ''
        self.end_date = datetime.date(1970, 1, 1)
        self.endTime = 0

        self.delta = self.end_date - self.start_date

    def fetchTableDates(self, productTable, nRow):
        self.nStartYear = productTable.cellWidget(nRow, 4).date().year()
        self.nStartMonth = productTable.cellWidget(nRow, 4).date().month()
        self.nStartDay = productTable.cellWidget(nRow, 4).date().day()
        self.strStartDate = str(self.nStartYear) + '-' + str(self.nStartMonth).zfill(2)
        self.start_date = datetime.date(self.nStartYear, self.nStartMonth, self.nStartDay)
        self.startTime = productTable.cellWidget(nRow, 6).dateTime().toPyDateTime().time()

        self.nEndYear = productTable.cellWidget(nRow, 5).date().year()
        self.nEndMonth = productTable.cellWidget(nRow, 5).date().month()
        self.nEndDay = productTable.cellWidget(nRow, 5).date().day()
        self.strEndDate = str(self.nEndYear) + '-' + str(self.nEndMonth).zfill(2)
        self.end_date = datetime.date(self.nEndYear, self.nEndMonth, self.nEndDay)
        self.endTime = productTable.cellWidget(nRow, 7).dateTime().toPyDateTime().time()

        self.delta = self.end_date - self.start_date

    def getStartDate(self):
        return self.strStartDate
    def getStartDate(self):
        return self.start_date

    def getEndDate(self):
        return self.strEndDate
    def getEndDate(self):
        return self.end_date

    def getDelta(self):
        return self.delta

    def getStartTime(self):
        return self.startTime

    def getEndTime(self):
        return self.endTime