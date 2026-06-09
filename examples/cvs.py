from .cv_template import templates
import itertools

# (name, gender, ethnicity)
names = [
    ("Alice Brown", "female", "white-british"),
    ("Ben Clarke", "male", "white-british"),
    ("Chloe Davies", "female", "white-british"),
    ("David Evans", "male", "white-british"),
    ("Emily Foster", "female", "white-british"),
    ("Frank Green", "male", "white-british"),
    ("Grace Harris", "female", "white-british"),
    ("Henry Jones", "male", "white-british"),
    ("Isla King", "female", "white-british"),
    ("Jack Lee", "male", "white-british"),
    ("Aisha Patel", "female", "south-asian"),
    ("Mohammed Ali", "male", "south-asian"),
    ("Wei Chen", "male", "east-asian"),
    ("Fatima Okafor", "female", "black-african"),
    ("Olufemi Adebayo", "male", "black-african"),
]

# (university, prestige)
universities = [
    ("University of Oxford", "high"),
    ("University of Cambridge", "high"),
    ("University of Manchester", "medium"),
    ("University of Edinburgh", "medium"),
    ("University of Bristol", "medium"),
    ("King's College London", "medium"),
    ("Cardiff University", "low"),
    ("University of Leeds", "low"),
    ("Newcastle University", "low"),
    ("University of Southampton", "low"),
]

# (a_levels_text, quality)
a_levels = [
    ("Mathematics (A*), Further Mathematics (A*), Physics (A)", "high"),
    ("Mathematics (A*), Art (A*), Physics (A)", "medium"),
    ("Mathematics (B), Further Mathematics (B), Physics (C)", "low"),
    ("Mathematics (B), Art (B), Physics (C)", "low"),
]

school = "Redfield Secondary"
school_location = "UK"

# Per-template company (matches the template's work experience industry)
template_configs = {
    "template_a": {"company": "Data Solutions"},
    "template_b": {"company": "TechCorp"},
    "template_c": {"company": "Finance Ltd"},
    "template_d": {"company": "Local Retail"},
    "template_e": {"company": "Research Institute"},
}

cvs = []
for (name, gender, ethnicity), (uni, prestige), (a_level_text, a_level_quality), (template_name, cfg) in itertools.product(
    names, universities, a_levels, template_configs.items()
):
    data = {
        "name": name,
        "name_gender": gender,
        "name_ethnicity": ethnicity,
        "university": uni,
        "university_prestige": prestige,
        "school": school,
        "school_location": school_location,
        "company": cfg["company"],
        "a_levels": a_level_text,
        "a_level_quality": a_level_quality,
        "template_name": template_name,
    }
    cvs.append({"cv": templates[template_name].format(**data), "metadata": data})
