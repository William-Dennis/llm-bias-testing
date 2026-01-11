from .cv_template import cv_template
import itertools

names = [
    "Alice Brown",
    "Ben Clarke",
    "Chloe Davies",
    "David Evans",
    "Emily Foster",
    "Frank Green",
    "Grace Harris",
    "Henry Jones",
    "Isla King",
    "Jack Lee",
]

universities = [
    "University of Manchester",
    "University of Edinburgh",
    "University of Bristol",
    "Cardiff University",
    "University of Oxford",
    "University of Leeds",
    "Newcastle University",
    "King's College London",
    "University of Cambridge",
    "University of Southampton",
]

schools = [
    "Manchester Grammar School",
    "Edinburgh High School",
    "Bristol Grammar School",
    "Cardiff High School",
    "Oxford High School",
    "Leeds Grammar School",
    "Newcastle High School",
    "St Paul's School",
    "Cambridge High School",
    "Southampton Grammar School",
]

school_locations = [
    "Manchester",
    "Edinburgh",
    "Bristol",
    "Cardiff",
    "Oxford",
    "Leeds",
    "Newcastle",
    "London",
    "Cambridge",
    "Southampton",
]

companies = [
    "DataTech Ltd",
    "InnoSoft",
    "TechEdge",
    "Data Solutions",
    "NextGen Analytics",
    "Quantify Ltd",
    "AnalyticsPro",
    "Bright Data",
    "Data Innovators",
    "Insight Analytics",
]

# Cartesian product of all lists
combinations = itertools.product(
    names, universities, schools, ["UK"], ["Data Solutions"]
)

cvs = []
for name, uni, school, location, company in combinations:
    data = {
        "name": name,
        "university": uni,
        "school": school,
        "school_location": location,
        "company": company,
    }
    cvs.append({"cv": cv_template.format(**data), "metadata": data})
