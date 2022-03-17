#M-AI-AN Glyph Scraper
#Scrapes data from the Virginia Ed Mayan Epigraphic Database Project

#Imports
import os
import requests
from bs4 import BeautifulSoup

ri = requests.get('http://www2.iath.virginia.edu/med/docs/_catalog.html')
baselink = 'http://www2.iath.virginia.edu/med/'
#print(ri)

#print(ri.content)

soup = BeautifulSoup(ri.content, 'html.parser')
#print(soup.prettify())

#Finds Hyperlinks
links = soup.find("body").find_all("a")

#Start of the main content loop
for i in links:
    #print(str(i))
    if str(i).find("/docs/") > 0:
        #Creates the linking suffix
        linker = str(i)[12:29]

        print(linker)
   
        try:
            #Creates request for individual link
            rs = requests.get(baselink + linker)
            rsbowl = BeautifulSoup(rs.content, 'html.parser')
            
            #Image parser
            #Collects all glyphs present on the website
            rsimages = rsbowl.select('img')
            for image in rsimages:
                src = image.get('src')
                print(baselink + src[3:])
                rimage = requests.get(baselink + src[3:], stream=True)
                
                try:
                    #Saves files as jpgs
                    with open(src[17:23]+".jpg", 'wb') as f: 
                        rimage.raw.decode_content = True
                        f.write(rimage.content)
                except:
                    print("Error handling image: " + str(image))
            
        except:
            print("Error handling link: " + str(i))