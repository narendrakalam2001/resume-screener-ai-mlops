# ============================================================
# RESUME SIMULATOR — Resume Screener AI
# ============================================================
# Generates synthetic resume submissions and sends to API.
# 3 scenarios:
#   strong  — highly qualified, keyword-rich resumes
#   weak    — underqualified, sparse resumes
#   random  — mixed realistic applicant pool
#
# Usage:
#   python scripts/run_simulation.py
#   python scripts/run_simulation.py --scenario strong --n 30
# ============================================================

import requests
import random
import time
import os

# ── API URL — local dev vs deployed ──────────────────────────
API_URL = os.getenv(
    "RESUME_SCREENER_API_URL",
    "http://localhost:8000"
) + "/predict"

# ============================================================
# RESUME TEMPLATES
# ============================================================

TECH_SKILLS_POOL = [
    "Python", "Machine Learning", "Deep Learning", "NLP", "TensorFlow",
    "PyTorch", "scikit-learn", "pandas", "NumPy", "SQL", "Spark", "Hadoop",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Git", "FastAPI", "Django",
    "React", "Node.js", "Java", "Scala", "Tableau", "Power BI",
]

SOFT_SKILLS_POOL = [
    "strong communication skills", "team leadership", "analytical thinking",
    "problem solving", "collaborative mindset", "adaptable and driven",
    "excellent presentation skills", "project management",
]

COMPANIES = [
    "Infosys", "TCS", "Wipro", "HCL Technologies", "Cognizant",
    "Accenture", "IBM", "Capgemini", "Tech Mahindra", "L&T Infotech",
]

JOB_CATEGORIES = [
    "Data Science", "Java Developer", "Python Developer", "DevOps Engineer",
    "HR", "Testing", "Sales", "Blockchain", "ETL Developer",
    "Network Security Engineer",
]

JD_TEMPLATES = {
    "Data Science": (
        "We are looking for a Data Scientist with experience in machine learning, "
        "Python, TensorFlow, pandas, SQL, and data visualization. The candidate "
        "should have strong analytical and communication skills. 3+ years experience required."
    ),
    "Python Developer": (
        "Hiring a Python Developer with expertise in Django, FastAPI, REST APIs, "
        "SQL databases, Docker, and Git. Agile experience preferred. 2+ years required."
    ),
    "DevOps Engineer": (
        "Seeking a DevOps Engineer skilled in Docker, Kubernetes, AWS, CI/CD pipelines, "
        "Linux, and Terraform. Strong problem-solving and collaboration skills. 3+ years required."
    ),
}


# ============================================================
# GENERATE SYNTHETIC RESUME
# ============================================================

def generate_resume(scenario: str = "random") -> dict:
    """
    Generates a synthetic resume + JD payload.

    Scenarios:
        strong  — rich, keyword-heavy, well-structured resumes
        weak    — sparse, minimal content, low keyword overlap
        random  — mixed realistic applications
    """

    category = random.choice(JOB_CATEGORIES)
    company  = random.choice(COMPANIES)
    years    = random.randint(1, 12)

    if scenario == "strong":
        n_skills     = random.randint(8, 14)
        n_soft       = random.randint(3, 5)
        has_edu      = True
        has_proj     = True
        resume_years = random.randint(3, 10)

    elif scenario == "weak":
        n_skills     = random.randint(1, 3)
        n_soft       = random.randint(0, 1)
        has_edu      = random.random() > 0.5
        has_proj     = False
        resume_years = random.randint(0, 2)

    else:  # random
        n_skills     = random.randint(2, 10)
        n_soft       = random.randint(0, 4)
        has_edu      = random.random() > 0.3
        has_proj     = random.random() > 0.5
        resume_years = random.randint(1, 8)

    skills = random.sample(TECH_SKILLS_POOL, min(n_skills, len(TECH_SKILLS_POOL)))
    softs  = random.sample(SOFT_SKILLS_POOL, min(n_soft,  len(SOFT_SKILLS_POOL)))

    # Build resume text
    parts = []

    if has_edu:
        degree = random.choice(["B.Tech Computer Science", "M.Tech Data Science",
                                "BSc Information Technology", "MCA", "B.E. Electronics"])
        univ   = random.choice(["IIT Bombay", "NIT Surathkal", "BITS Pilani",
                                "Pune University", "Anna University"])
        parts.append(
            f"EDUCATION\n{degree} — {univ} ({random.randint(2010, 2023)})"
        )

    parts.append(
        f"EXPERIENCE\n{resume_years}+ years of experience as a "
        f"{category.replace('_', ' ')} at {company}.\n"
        f"Worked extensively with {', '.join(skills[:4])} in a fast-paced environment."
    )

    parts.append(f"SKILLS\n{', '.join(skills)}")

    if softs:
        parts.append(f"SUMMARY\nProfessional with {', '.join(softs)}.")

    if has_proj:
        proj_skill = random.choice(skills) if skills else "Python"
        parts.append(
            f"PROJECTS\n"
            f"Project: {category} Automation System\n"
            f"Built an end-to-end {proj_skill} solution to automate {category.lower()} workflows.\n"
            f"Tools: {', '.join(random.sample(skills, min(3, len(skills))))}"
        )

    if softs:
        parts.append(f"CERTIFICATIONS\nCertified in {random.choice(skills)} (Coursera, 2023)")

    resume_text = "\n\n".join(parts)

    # Pick JD
    jd_text = JD_TEMPLATES.get(category, JD_TEMPLATES["Data Science"])

    return {
        "resume_text":     resume_text,
        "job_description": jd_text,
        "years_required":  random.randint(0, 3),
        "_meta": {
            "scenario": scenario,
            "category": category,
            "skills":   skills,
        }
    }


# ============================================================
# SEND TO API + PRINT RESULT
# ============================================================

def send_resume(payload: dict, idx: int):
    meta     = payload.pop("_meta", {})
    scenario = meta.get("scenario", "?")
    category = meta.get("category", "?")

    try:
        response = requests.post(API_URL, json=payload, timeout=15)

        if response.status_code == 200:
            result = response.json()
            rec    = result.get("recommendation", "?")
            score  = result.get("final_score",    "?")
            band   = result.get("match_band",     "?")
            pred   = result.get("predicted_category", "?")

            status_emoji = {"SHORTLIST": "✅", "REVIEW": "⚠️", "REJECT": "❌"}.get(rec, "📌")

            print(
                f"[{idx+1:3d}] {scenario:6s} | true={category:25s} | "
                f"pred={pred:25s} | score={score:.3f} | {band:8s} | {status_emoji} {rec}"
            )
        else:
            print(f"[{idx+1:3d}] API error: HTTP {response.status_code}")

    except requests.exceptions.Timeout:
        print(f"[{idx+1:3d}] Timeout — Render cold start? Retry in 30s.")
    except Exception as e:
        print(f"[{idx+1:3d}] Connection error: {e}")


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_applications(n: int = 20, scenario: str = "random"):
    print(f"\nSimulating {n} resume applications  |  scenario={scenario}")
    print("-" * 90)
    print(
        f"{'#':>4}  {'Scen':6}  {'True Category':25}  "
        f"{'Predicted':25}  {'Score':5}  {'Band':8}  Decision"
    )
    print("-" * 90)

    for i in range(n):
        payload = generate_resume(scenario)
        send_resume(payload, i)
        time.sleep(0.4)   # rate limit

    print("-" * 90)
    print(f"Simulation complete — {n} resumes screened\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resume Screener Simulator")
    parser.add_argument("--n",        type=int, default=20,     help="Number of resumes")
    parser.add_argument("--scenario", type=str, default="random",
                        choices=["random", "strong", "weak"],   help="Simulation scenario")
    args = parser.parse_args()

    simulate_applications(n=args.n, scenario=args.scenario)
