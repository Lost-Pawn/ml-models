"""
This file builds a synthetic resume dataset since we don't have a real
company database to pull from. It mixes skills, tools, and phrases that
usually show up in resumes for six common job categories, then saves
everything into data/resumes.csv for the rest of the project to use.

Run this file first before training the model.
"""

import random
import csv
import os

random.seed(42)

# Each category has its own pool of skills, tools and phrases.
# Some skills are shared between categories on purpose, real resumes
# overlap too, and that overlap is what makes the classification
# problem realistic instead of trivially easy.

CATEGORIES = {
    "Data Science": {
        "skills": ["python", "pandas", "numpy", "scikit learn", "machine learning",
                   "deep learning", "sql", "data visualization", "statistics",
                   "tensorflow", "pytorch", "feature engineering", "nlp"],
        "tools": ["jupyter notebook", "tableau", "power bi", "git", "docker"],
        "titles": ["Data Scientist", "Machine Learning Engineer", "Data Analyst Intern"],
        "summary": [
            "Built predictive models to solve business problems using structured and unstructured data.",
            "Experienced in cleaning large datasets and turning them into useful insights.",
            "Worked on classification and regression problems for real world applications."
        ]
    },
    "Software Development": {
        "skills": ["python", "java", "c++", "javascript", "react", "node js",
                   "rest api", "system design", "object oriented programming",
                   "data structures", "algorithms", "microservices"],
        "tools": ["git", "docker", "jenkins", "postman", "aws"],
        "titles": ["Software Engineer", "Backend Developer", "Full Stack Developer"],
        "summary": [
            "Developed and maintained scalable web applications for enterprise clients.",
            "Wrote clean and testable code while collaborating with cross functional teams.",
            "Built REST APIs and integrated them with front end applications."
        ]
    },
    "Human Resources": {
        "skills": ["recruitment", "onboarding", "employee relations", "payroll",
                   "performance management", "hr policies", "conflict resolution",
                   "talent acquisition", "labor law", "employee engagement"],
        "tools": ["workday", "sap successfactors", "excel", "linkedin recruiter"],
        "titles": ["HR Executive", "Talent Acquisition Specialist", "HR Generalist"],
        "summary": [
            "Managed end to end recruitment cycles for multiple departments.",
            "Handled employee onboarding, engagement activities and grievance redressal.",
            "Coordinated with department heads to plan hiring and workforce needs."
        ]
    },
    "Sales": {
        "skills": ["lead generation", "client relationship management", "negotiation",
                   "cold calling", "sales forecasting", "crm", "target achievement",
                   "b2b sales", "customer retention", "market research"],
        "tools": ["salesforce", "hubspot", "excel", "zoho crm"],
        "titles": ["Sales Executive", "Business Development Associate", "Account Manager"],
        "summary": [
            "Consistently exceeded monthly sales targets through strong client relationships.",
            "Generated new business leads and converted them into long term clients.",
            "Managed a portfolio of key accounts and handled contract renewals."
        ]
    },
    "Web Designing": {
        "skills": ["html", "css", "javascript", "figma", "adobe xd", "ui design",
                   "ux research", "responsive design", "wireframing", "photoshop",
                   "react", "bootstrap"],
        "tools": ["figma", "adobe xd", "git", "vs code"],
        "titles": ["Web Designer", "UI UX Designer", "Front End Developer"],
        "summary": [
            "Designed clean and responsive websites focused on good user experience.",
            "Created wireframes and prototypes before moving into visual design.",
            "Worked closely with developers to turn design mockups into working pages."
        ]
    },
    "Mechanical Engineering": {
        "skills": ["autocad", "solidworks", "product design", "thermodynamics",
                   "manufacturing processes", "quality control", "cad modeling",
                   "project planning", "root cause analysis", "six sigma"],
        "tools": ["autocad", "solidworks", "ansys", "ms project"],
        "titles": ["Mechanical Engineer", "Design Engineer", "Production Engineer"],
        "summary": [
            "Designed mechanical components and validated them using simulation tools.",
            "Worked on the production floor to improve process efficiency and reduce waste.",
            "Prepared detailed technical drawings for manufacturing teams."
        ]
    }
}

EDUCATION = [
    "Bachelor of Technology in relevant field",
    "Bachelor of Science, graduated with distinction",
    "Master of Business Administration",
    "Diploma in relevant discipline",
    "Bachelor of Engineering"
]


def build_resume_text(category_data):
    skills = random.sample(category_data["skills"], k=random.randint(5, 8))
    tools = random.sample(category_data["tools"], k=random.randint(2, 4))
    title = random.choice(category_data["titles"])
    summary = random.choice(category_data["summary"])
    education = random.choice(EDUCATION)
    years = random.randint(1, 8)

    resume_text = (
        f"{title} with {years} years of experience. {summary} "
        f"Key skills include {', '.join(skills)}. "
        f"Familiar with tools such as {', '.join(tools)}. "
        f"Education, {education}."
    )
    return resume_text


def main():
    rows = []
    samples_per_category = 60

    for category, data in CATEGORIES.items():
        for _ in range(samples_per_category):
            text = build_resume_text(data)
            rows.append({"resume_text": text, "category": category})

    random.shuffle(rows)

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "resumes.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resume_text", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset created with {len(rows)} resumes across {len(CATEGORIES)} categories.")
    print(f"Saved to {output_path}")

    # Result after running this file, 360 rows were generated in total,
    # that is 60 resumes for each of the 6 categories, and the file
    # was saved at data/resumes.csv


if __name__ == "__main__":
    main()
