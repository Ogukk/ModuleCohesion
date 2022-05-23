from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    #return "Hello, World!"
    import pandas as pd
    from bs4 import BeautifulSoup
    import requests

    courses = [
    "csc-10024",
    "csc-10025",
    "csc-10033",
    "csc-10026",
    "csc-10035",
    "csc-10056",
    "csc-10046",
    "csc-10044",
    "csc-10042",
    "csc-10054",
    "csc-10050",
    "csc-10052",
    "csc-10060",
    "csc-10056",
    "csc-10058",
    "csc-10048",
    "csc-10040",
    "man-10015",
    "mat-10055",
    "mat-10057",
    "mat-10053",
    "csc-20037",
    "csc-20043",
    "csc-20065",
    "csc-20059",
    "csc-20067",
    "csc-20055",
    "csc-20061",
    "csc-20071",
    "csc-20057",
    "csc-20069",
    "csc-20063",
    "csc-20021",
    "csc-20038",
    "csc-20004",
    "csc-20002",
    "csc-20041",
    "che-20042",
    "csc-20047",
    "mat-20023",
    "mat-20039",
    "mat-20027",
    "mat-20037",
    "eco-20042",
    "eco-20007",
    "csc-30016",
    "csc-30019",
    "csc-30023",
    "csy-30001",
    "csc-30022",
    "csc-30025",
    "csc-30035",
    "csc-30031",
    "csc-30049",
    "csc-30033",
    "csc-30027",
    "csc-30002",
    "csc-30012",
    "csc-30014",
    "csc-30041",
    "csc-30043",
    "csc-30021",
    "csc-30045",
    "csc-30027",
    "mat-30049",
    "mat-30014",
    "eco-30037",
    "eco-30045",
    "eco-30053",
    "eco-30038",
    "csc-40041",
    "csc-40043",
    "csc-40042",
    "csc-40045",
    "csc-40039",
    "csc-40052",
    "csc-40104",
    "csc-40044",
    "csc-40054",
    "csc-40064",
    "csc-40122",
    "csc-40039",
    "csc-40046",
    "csc-40120",
    "csc-40040",
    "csc-40035",
    "csc-40056",
    "csc-40072",
    "csc-40045",
    "csc-40048",
    "csc-40070",
    "csc-40038",
    "csc-40050",
    "csc-40102",
    "csc-40062",
    "csc-40040",
    "csc-40068",
    "csc-40066",
    "csc-40060",
    "csc-40062",
    "csc-40058",
    "geg-40058",
    "pha-10026",
    "pha-xxxxx",
    "pcs-2001 ",
    "pcs-20005",
    "pha-xxxx ",
    "pcs-30001",
    "lsc-30055",
    "hlt-40003",
    "mte-40040",
    "mte-40041",
    "mte-40031",
    "fin-40037",
    "fin-40039",
    "fin-40051",
    "fin-40053",
    "fin-40047",
    "fin-40041"
                ]

    ModuleDetails = pd.DataFrame( columns=['Module','Details'] )

    for x in courses:
        print(x)
        URL = "https://www.keele.ac.uk/catalogue/2021-22/"+x+".htm"
        print(URL)
        Modulepage = requests.get(URL)

        ModuleSoup = BeautifulSoup(Modulepage.content, "html.parser")
        ModuleName = ModuleSoup.find("div", class_= "panel-heading").text.strip()
        ModuleAims = ModuleSoup.find_all("div", class_= "col-sm-12")
        ModuleText = ""
        check = 0

        for element in ModuleAims:
            if check == 1:
                ModuleText = ModuleText + " " + element.text.strip()
                check = 0    

            if element.text.strip() == "Description for 2021/22":
                check = 1

            if element.text.strip() == "Aims":
                check = 1

            if element.text.strip() == "Intended Learning Outcomes":
                check = 1


        new_row = {'Module':ModuleName, 'Details':ModuleText}
        ModuleDetails = ModuleDetails.append( new_row, ignore_index=True )
    Return ModuleDetails
