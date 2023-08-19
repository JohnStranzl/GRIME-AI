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
        startDateCol = 4
        self.nStartYear = productTable.cellWidget(nRow, startDateCol).date().year()
        self.nStartMonth = productTable.cellWidget(nRow, startDateCol).date().month()
        self.nStartDay = productTable.cellWidget(nRow, startDateCol).date().day()
        self.strStartDate = str(self.nStartYear) + '-' + str(self.nStartMonth).zfill(2)
        self.start_date = datetime.date(self.nStartYear, self.nStartMonth, self.nStartDay)

        startTimeCol = 6
        self.startTime = productTable.cellWidget(nRow, startTimeCol).dateTime().toPyDateTime().time()

        endDateCol = 5
        self.nEndYear = productTable.cellWidget(nRow, endDateCol).date().year()
        self.nEndMonth = productTable.cellWidget(nRow, endDateCol).date().month()
        self.nEndDay = productTable.cellWidget(nRow, endDateCol).date().day()
        self.strEndDate = str(self.nEndYear) + '-' + str(self.nEndMonth).zfill(2)
        self.end_date = datetime.date(self.nEndYear, self.nEndMonth, self.nEndDay)

        endTimeCol = 7
        self.endTime = productTable.cellWidget(nRow, endTimeCol).dateTime().toPyDateTime().time()

        self.delta = self.end_date - self.start_date

        return self.start_date, self.startTime, self.end_date, self.endTime

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