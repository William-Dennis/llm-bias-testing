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

a_levels = [
    "Mathematics (A*), Further Mathematics (A*), Physics (A)",
    "Mathematics (B), Further Mathematics (B), Physics (C)",
    "Mathematics (A*), Art (A*), Physics (A)",
    "Mathematics (B), Art (B), Physics (C)",
]

# Cartesian product of all lists
combinations = itertools.product(
    names, universities, ["Redfield Secondary"], ["England", "Scotland", "Wales"], ["Data Solutions"], a_levels
)

cvs = []
for name, uni, school, location, company, a_levels in combinations:
    data = {
        "name": name,
        "university": uni,
        "school": school,
        "school_location": location,
        "company": company,
        "a_levels": a_levels,
    }
    cvs.append({"cv": cv_template.format(**data), "metadata": data})
