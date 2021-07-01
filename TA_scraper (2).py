from bs4 import BeautifulSoup
from urllib import request
import pandas as pd 
import pymongo
from statistics import mean

#define function 
def scrapeTripAdvisor(baseLink, nextLink, linkEnd, title):

    #connect to the database 
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["hotels"]
    currentHotels = mydb.list_collection_names()

    #check for existance of a hotel in the collection and exit function if it exists
    if title in currentHotels:
        print(title, 'is already in the database. please drop.')
        return


    response = request.urlopen(baseLink).read().decode('utf8')
    parsed = BeautifulSoup(response, 'html.parser')

    #use classes in html to retrieve the correct text 
    rawLocation = str(parsed.find(class_ = '_3ErVArsu').get_text())




    rawPrice = str(parsed.find(class_ = '_1NHwuRzF').get_text())
    rawPrice = rawPrice.replace('(Based on Average Rates for a Standard Room)', '').replace(',','').replace('$','').replace('-',',').strip().split(',')

    #parse the price text and take the mean of lower and upper bound 
    for i in range(0,2,1):
        rawPrice[i] = int(rawPrice[i])
    avgPrice = mean(rawPrice)

    #catch hotel name 
    hotelName = str(parsed.find(class_ = '_1mTlpMC3').get_text())


    #loop through the links by appending them to a list 
    urlList = []
    urlList.append(baseLink)
    for site in range(5,linkEnd ,5):
        urlList.append(nextLink.format(number = site))

    tempReviews = []
    tempRatings = []

    print(len(urlList))

#needs outer container and inner container to sort income points....str looks like [[R1,R2,R3],[R1,R2,R3]]
    for links in urlList:
        response = request.urlopen(links).read().decode('utf8')
        parsed = BeautifulSoup(response, 'html.parser')
        tempReviews.append(parsed.find_all(class_ = 'IRsGHoPm'))
        tempRatings.append(parsed.find_all(class_ = 'nf9vGX55'))


    unpackReviews = []
    unpackRatings = []

#removes the shells around each page and puts them into a single list for reviews and ratings 
    for outerShell in tempReviews:
        for innerShell in outerShell:
            unpackReviews.append((innerShell))

    for outerShell in tempRatings:
        for innerShell in outerShell:
            unpackRatings.append((innerShell))

    Reviews = []
    Ratings = []

#strip the uneeded html code 
    for i in range(len(unpackReviews)):
        unpackReviews[i] = str(unpackReviews[i])
        Reviews.append(unpackReviews[i].replace('<q class="IRsGHoPm"><span>', '').replace('</span><span>', '').replace('</span></q>', '').replace('</span><span class="_1M-1YYJt">', ''))

    for i in range(len(unpackRatings)):
        unpackRatings[i] = str(unpackRatings[i])
        Ratings.append(int(unpackRatings[i][92:93])) 

#save into a dictionairy and store in database 
    rating = mean(Ratings)
    individual_Reviews = {'Reviews': Reviews, 'Ratings': Ratings }
    data = {'Name': hotelName ,'Address': rawLocation, 'Rating': rating, 'Price': avgPrice,'individual_Reviews': individual_Reviews}
    mycol = mydb[title]
    mycol.insert_one(data)
    print('succesfully parsed and loaded {} data in the mongodb'.format(title))



