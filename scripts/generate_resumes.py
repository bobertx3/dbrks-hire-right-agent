"""
Generate 10 synthetic PDF resumes for Director of HR candidates using reportlab.
Run: pip install reportlab && python generate_resumes.py
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "unstructured")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACCENT = colors.HexColor("#1B3139")
RED    = colors.HexColor("#FF3621")

def make_styles():
    styles = getSampleStyleSheet()
    name_style = ParagraphStyle("name", fontName="Helvetica-Bold", fontSize=18, textColor=ACCENT, spaceAfter=2)
    title_style = ParagraphStyle("title", fontName="Helvetica", fontSize=11, textColor=RED, spaceAfter=4)
    contact_style = ParagraphStyle("contact", fontName="Helvetica", fontSize=9, textColor=colors.grey, spaceAfter=6)
    section_style = ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=11, textColor=ACCENT,
                                   spaceBefore=10, spaceAfter=4, borderPad=2)
    body_style = ParagraphStyle("body", fontName="Helvetica", fontSize=9, spaceAfter=3, leading=13)
    bullet_style = ParagraphStyle("bullet", fontName="Helvetica", fontSize=9, spaceAfter=2, leading=12,
                                  leftIndent=12, bulletIndent=0)
    return name_style, title_style, contact_style, section_style, body_style, bullet_style


def section_header(text, section_style):
    return [
        Paragraph(text.upper(), section_style),
        HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=4),
    ]


def build_resume(filename, data):
    doc = SimpleDocTemplate(
        os.path.join(OUTPUT_DIR, filename),
        pagesize=letter,
        rightMargin=0.65*inch, leftMargin=0.65*inch,
        topMargin=0.6*inch, bottomMargin=0.6*inch,
    )
    name_s, title_s, contact_s, section_s, body_s, bullet_s = make_styles()
    story = []

    # Header
    story.append(Paragraph(data["name"], name_s))
    story.append(Paragraph(data["current_title"], title_s))
    story.append(Paragraph(data["contact"], contact_s))
    story.append(HRFlowable(width="100%", thickness=2, color=RED, spaceAfter=8))

    # Summary
    story += section_header("Professional Summary", section_s)
    story.append(Paragraph(data["summary"], body_s))

    # Experience
    story += section_header("Professional Experience", section_s)
    for exp in data["experience"]:
        story.append(Paragraph(f"<b>{exp['title']}</b> — {exp['company']} | {exp['dates']}", body_s))
        story.append(Paragraph(f"<i>{exp['location']}</i>", contact_s))
        for bullet in exp["bullets"]:
            story.append(Paragraph(f"• {bullet}", bullet_s))
        story.append(Spacer(1, 4))

    # Education
    story += section_header("Education", section_s)
    for edu in data["education"]:
        story.append(Paragraph(f"<b>{edu['degree']}</b>, {edu['school']} ({edu['year']})", body_s))

    # Certifications
    story += section_header("Certifications & Credentials", section_s)
    story.append(Paragraph(" | ".join(data["certifications"]) if data["certifications"] else "None listed", body_s))

    # Skills
    story += section_header("Core Competencies", section_s)
    skills_text = " • ".join(data["skills"])
    story.append(Paragraph(skills_text, body_s))

    doc.build(story)
    print(f"  Created: {filename}")


RESUMES = [
    {
        "filename": "resume_01.pdf",
        "name": "Sarah Chen",
        "current_title": "Vice President of Human Resources | Novartis Pharmaceuticals",
        "contact": "sarah.chen@email.com  |  +1 (312) 555-0101  |  Chicago, IL  |  linkedin.com/in/sarahchen-hr",
        "summary": (
            "Strategic HR executive with 15 years of progressive experience in pharmaceutical and healthcare "
            "environments. Proven ability to drive enterprise-wide talent initiatives, lead organizational "
            "transformations, and build high-performing HR functions. SPHR-certified leader with deep expertise "
            "in executive compensation, M&A integration, and global talent acquisition. Managed HR programs "
            "spanning 45+ employees and $18.5M in budget. Passionate about leveraging data and technology "
            "to create people strategies that align with business outcomes."
        ),
        "experience": [
            {
                "title": "Vice President of Human Resources",
                "company": "Novartis Pharmaceuticals",
                "dates": "2019 – Present",
                "location": "Chicago, IL",
                "bullets": [
                    "Lead HR strategy for 3,200-employee North American commercial division; oversee team of 45 HR professionals",
                    "Designed and implemented competency-based performance management system, increasing employee engagement scores by 22%",
                    "Spearheaded M&A due diligence and integration for two acquisitions totaling $2.1B in enterprise value",
                    "Partnered with C-suite to reduce voluntary turnover by 18% through enhanced retention and development programs",
                    "Managed $18.5M HR budget including compensation, benefits, L&D, and HRIS technology",
                ],
            },
            {
                "title": "Senior Director, Talent Management",
                "company": "AbbVie Inc.",
                "dates": "2015 – 2019",
                "location": "North Chicago, IL",
                "bullets": [
                    "Built succession planning program covering 200+ critical roles across global R&D organization",
                    "Launched enterprise leadership development curriculum with 94% completion rate and measurable business impact",
                    "Deployed Workday HCM for 8,000+ employees, managing change management and training rollout",
                ],
            },
            {
                "title": "HR Business Partner – Commercial",
                "company": "Baxter International",
                "dates": "2011 – 2015",
                "location": "Deerfield, IL",
                "bullets": [
                    "Served as strategic partner to VP-level Commercial leaders across 4 business units",
                    "Led workforce planning, organizational design, and talent calibration for 600-employee division",
                ],
            },
        ],
        "education": [
            {"degree": "Master of Business Administration (MBA), Human Resources Concentration", "school": "University of Chicago Booth School of Business", "year": "2010"},
            {"degree": "Bachelor of Science, Psychology", "school": "University of Illinois Urbana-Champaign", "year": "2008"},
        ],
        "certifications": ["Senior Professional in Human Resources (SPHR)", "Certified Executive Coach (ICF-ACC)", "Prosci Change Management Practitioner"],
        "skills": ["Strategic Workforce Planning", "Executive Compensation", "M&A Integration", "HRIS (Workday, SAP SuccessFactors)", "Talent Acquisition", "Organizational Development", "Employment Law & Compliance", "Diversity, Equity & Inclusion"],
    },
    {
        "filename": "resume_02.pdf",
        "name": "Michael Torres",
        "current_title": "Director of People & Culture | Boston Scientific",
        "contact": "michael.torres@email.com  |  +1 (617) 555-0202  |  Boston, MA  |  linkedin.com/in/michaeltorres-hr",
        "summary": (
            "Results-driven HR Director with 13 years of experience in medical device and healthcare sectors. "
            "SHRM-SCP certified with a strong track record in building scalable HR infrastructure, designing "
            "compensation frameworks, and developing people strategies for regulated environments. Led HR "
            "functions for divisions up to 35 HR professionals and $12M in budget. Known for translating "
            "business strategy into actionable people programs with measurable ROI."
        ),
        "experience": [
            {
                "title": "Director of People & Culture",
                "company": "Boston Scientific",
                "dates": "2018 – Present",
                "location": "Marlborough, MA",
                "bullets": [
                    "Lead people strategy for 2,800-person cardiac rhythm management division; manage 35-person HR team",
                    "Redesigned total rewards framework, achieving 96th-percentile market positioning for critical STEM roles",
                    "Implemented data-driven talent analytics platform, reducing time-to-hire by 31% and improving quality-of-hire metrics",
                    "Championed global DEI initiative resulting in 28% increase in underrepresented talent at director+ levels",
                    "Managed $12M HR budget with zero year-over-year budget overruns for 5 consecutive years",
                ],
            },
            {
                "title": "Senior HR Business Partner",
                "company": "Medtronic",
                "dates": "2014 – 2018",
                "location": "Minneapolis, MN",
                "bullets": [
                    "Strategic HR partner to R&D and Quality Assurance leadership across 1,200-person business unit",
                    "Navigated two FDA regulatory inspections with zero HR compliance findings",
                    "Developed HR integration playbook adopted company-wide for post-acquisition onboarding",
                ],
            },
            {
                "title": "HR Manager",
                "company": "Fresenius Medical Care",
                "dates": "2011 – 2014",
                "location": "Waltham, MA",
                "bullets": [
                    "Managed full-cycle HR operations for 400-person dialysis services division",
                    "Reduced workers' compensation costs by $800K through proactive safety and wellness programs",
                ],
            },
        ],
        "education": [
            {"degree": "Master of Arts, Human Resources Management", "school": "Boston University", "year": "2011"},
            {"degree": "Bachelor of Science, Business Administration", "school": "University of Massachusetts Amherst", "year": "2009"},
        ],
        "certifications": ["SHRM Senior Certified Professional (SHRM-SCP)", "Certified Compensation Professional (CCP)", "Lean Six Sigma Green Belt"],
        "skills": ["HR Strategy", "Compensation Design", "Talent Analytics", "Regulatory Compliance (FDA/GxP)", "Organizational Effectiveness", "DEI Program Leadership", "HRIS (Workday)", "Employee Relations"],
    },
    {
        "filename": "resume_03.pdf",
        "name": "Jennifer Williams",
        "current_title": "HR Business Partner | Target Corporation",
        "contact": "jennifer.williams@email.com  |  +1 (404) 555-0303  |  Atlanta, GA  |  linkedin.com/in/jenniferwilliams-hr",
        "summary": (
            "Dedicated HR professional with 8 years of experience in retail and consumer goods environments. "
            "Skilled in employee relations, staffing coordination, and HR administration for high-volume, "
            "distributed workforces. Collaborative team player with strong communication skills and a "
            "commitment to building positive employee experiences. Currently growing into strategic HR "
            "leadership responsibilities."
        ),
        "experience": [
            {
                "title": "HR Business Partner",
                "company": "Target Corporation",
                "dates": "2020 – Present",
                "location": "Atlanta, GA",
                "bullets": [
                    "Support HR needs for 5 retail store locations with combined 800+ employees",
                    "Coordinate seasonal hiring campaigns, onboarding 200+ associates per quarter",
                    "Manage employee relations cases and conduct investigations for policy violations",
                    "Facilitate new manager orientation and basic leadership training sessions",
                ],
            },
            {
                "title": "HR Coordinator",
                "company": "Home Depot",
                "dates": "2017 – 2020",
                "location": "Atlanta, GA",
                "bullets": [
                    "Processed FMLA, ADA accommodations, and benefits enrollment for store employees",
                    "Maintained HRIS records and generated weekly workforce reports for district manager",
                    "Assisted in implementation of new applicant tracking system across 12 stores",
                ],
            },
            {
                "title": "HR Administrative Assistant",
                "company": "Macy's Inc.",
                "dates": "2016 – 2017",
                "location": "Atlanta, GA",
                "bullets": [
                    "Provided administrative support for HR department serving 300-person regional office",
                    "Scheduled interviews, coordinated onboarding logistics, and maintained personnel files",
                ],
            },
        ],
        "education": [
            {"degree": "Bachelor of Arts, Business Administration", "school": "Georgia State University", "year": "2016"},
        ],
        "certifications": [],
        "skills": ["Employee Relations", "Staffing & Recruitment", "Benefits Administration", "HRIS Data Entry", "New Employee Onboarding", "Policy Administration", "Microsoft Office Suite", "Workday"],
    },
    {
        "filename": "resume_04.pdf",
        "name": "David Kim",
        "current_title": "Chief People Officer | Merck & Co.",
        "contact": "david.kim@email.com  |  +1 (215) 555-0404  |  Philadelphia, PA  |  linkedin.com/in/davidkim-chro",
        "summary": (
            "Transformational people leader with 18 years of HR experience in global pharmaceutical organizations. "
            "Dual-certified (SPHR, PHR) HR executive with a proven record of building world-class talent functions, "
            "driving cultural transformation, and delivering measurable business results. Led HR for organizations "
            "of 60+ professionals and managed budgets exceeding $25M. Deep expertise in global talent strategy, "
            "executive leadership development, and organizational design for complex matrix structures."
        ),
        "experience": [
            {
                "title": "Chief People Officer",
                "company": "Merck & Co.",
                "dates": "2017 – Present",
                "location": "Kenilworth, NJ",
                "bullets": [
                    "Executive HR leadership for 20,000+ global workforce; oversee 60-person HR organization with $25M budget",
                    "Architected 5-year HR transformation strategy adopted by Board of Directors as enterprise priority",
                    "Led organizational restructuring saving $45M annually while improving HR service delivery satisfaction by 34%",
                    "Built first global people analytics team, enabling data-driven decisions for executive talent planning",
                    "Championed gender pay equity audit and remediation across 50+ countries, achieving 99.2% pay parity",
                    "Negotiated three labor agreements with UFCW and Teamsters representing 4,200 manufacturing employees",
                ],
            },
            {
                "title": "VP, Global Human Resources",
                "company": "Pfizer Inc.",
                "dates": "2012 – 2017",
                "location": "New York, NY",
                "bullets": [
                    "Led HR for $8B Oncology business unit, partnering with President and EVP leadership team",
                    "Designed global talent acquisition center of excellence reducing cost-per-hire by 28%",
                    "Managed post-merger integration of Wyeth HR organization (12,000 employees, 40 countries)",
                ],
            },
            {
                "title": "Senior Director, Organizational Effectiveness",
                "company": "Johnson & Johnson",
                "dates": "2008 – 2012",
                "location": "New Brunswick, NJ",
                "bullets": [
                    "Designed and facilitated executive team effectiveness engagements for operating company presidents",
                    "Led global HR technology strategy, overseeing implementation of SAP SuccessFactors for 130,000 employees",
                ],
            },
            {
                "title": "HR Director, Pharmaceutical Sector",
                "company": "Bristol-Myers Squibb",
                "dates": "2006 – 2008",
                "location": "Princeton, NJ",
                "bullets": [
                    "Directed HR operations for 1,800-person commercial pharmaceutical organization",
                ],
            },
        ],
        "education": [
            {"degree": "Master of Business Administration (MBA), Strategy & Organizational Behavior", "school": "Wharton School, University of Pennsylvania", "year": "2006"},
            {"degree": "Bachelor of Science, Industrial & Labor Relations", "school": "Cornell University", "year": "2004"},
        ],
        "certifications": ["Senior Professional in Human Resources (SPHR)", "Professional in Human Resources (PHR)", "Board Certified Coach (BCC)", "Hogan Assessment Certified Practitioner"],
        "skills": ["Global HR Strategy", "C-Suite Partnership", "M&A Integration", "Labor Relations", "People Analytics", "Executive Development", "Organizational Design", "Pay Equity", "Change Management"],
    },
    {
        "filename": "resume_05.pdf",
        "name": "Amanda Rodriguez",
        "current_title": "Head of Human Resources | Gilead Sciences",
        "contact": "amanda.rodriguez@email.com  |  +1 (858) 555-0505  |  San Diego, CA  |  linkedin.com/in/amandarodriguez-hr",
        "summary": (
            "Scientist-turned-HR-executive with a PhD in Industrial-Organizational Psychology and 11 years "
            "of applied experience in biotech and pharmaceutical environments. SHRM-SCP certified with "
            "exceptional expertise in evidence-based talent assessment, predictive workforce analytics, and "
            "building high-performance cultures in fast-paced, research-driven organizations. Led HR teams "
            "of 25 professionals with $10M budget oversight."
        ),
        "experience": [
            {
                "title": "Head of Human Resources – Commercial & Medical Affairs",
                "company": "Gilead Sciences",
                "dates": "2019 – Present",
                "location": "Foster City, CA",
                "bullets": [
                    "Lead HR strategy for 2,100-person commercial organization; manage 25-member HR team with $10M budget",
                    "Developed predictive attrition model identifying at-risk talent 90 days in advance, reducing turnover by 24%",
                    "Designed competency framework and selection tools validated against 3-year performance outcomes",
                    "Built internal talent marketplace enabling 180 cross-functional moves, improving internal mobility by 47%",
                    "Partnered with DEI council to increase representation of women in senior leadership from 31% to 48%",
                ],
            },
            {
                "title": "Senior Manager, Talent & Organizational Effectiveness",
                "company": "Amgen",
                "dates": "2015 – 2019",
                "location": "Thousand Oaks, CA",
                "bullets": [
                    "Deployed psychometric selection battery for director-level hiring, improving predictive validity by 38%",
                    "Facilitated executive team diagnostics and leadership effectiveness coaching programs",
                    "Led organizational network analysis identifying collaboration bottlenecks in R&D pipeline teams",
                ],
            },
            {
                "title": "HR Business Partner",
                "company": "Biogen",
                "dates": "2013 – 2015",
                "location": "Cambridge, MA",
                "bullets": [
                    "Supported HR operations for 400-person neuroscience research division",
                    "Designed onboarding experience for scientific talent that reduced 90-day attrition by 40%",
                ],
            },
        ],
        "education": [
            {"degree": "Doctor of Philosophy, Industrial-Organizational Psychology", "school": "University of Minnesota", "year": "2013"},
            {"degree": "Bachelor of Science, Psychology (Honors)", "school": "University of California San Diego", "year": "2009"},
        ],
        "certifications": ["SHRM Senior Certified Professional (SHRM-SCP)", "Evidence-Based Coaching Certification (EMCC)", "Predictive Index Certified Analyst"],
        "skills": ["I-O Psychology & Psychometrics", "Predictive Talent Analytics", "Selection & Assessment Design", "Workforce Planning", "DEI Strategy", "Executive Coaching", "Organizational Network Analysis", "R & Python (people analytics)"],
    },
    {
        "filename": "resume_06.pdf",
        "name": "Robert Johnson",
        "current_title": "HR Generalist | Amazon Web Services",
        "contact": "robert.johnson@email.com  |  +1 (206) 555-0606  |  Seattle, WA  |  linkedin.com/in/robertjohnson-hr",
        "summary": (
            "Motivated HR professional with 6 years of experience supporting technology and e-commerce "
            "organizations. Skilled in benefits administration, employee onboarding, and HR system maintenance. "
            "SHRM-CP certified with working knowledge of employment law and HR compliance. Eager to grow "
            "into an HR management role with increased strategic responsibility."
        ),
        "experience": [
            {
                "title": "HR Generalist II",
                "company": "Amazon Web Services",
                "dates": "2021 – Present",
                "location": "Seattle, WA",
                "bullets": [
                    "Support HR operations for 250-person software engineering division",
                    "Process employee lifecycle transactions including new hires, transfers, and separations in Workday",
                    "Coordinate performance review cycles and assist managers with calibration documentation",
                    "Respond to employee inquiries on policy, benefits, and leave of absence requests",
                ],
            },
            {
                "title": "HR Coordinator",
                "company": "Microsoft",
                "dates": "2019 – 2021",
                "location": "Redmond, WA",
                "bullets": [
                    "Supported full-cycle recruiting coordination for cloud computing business group",
                    "Managed intern program logistics for 60-person summer cohort",
                ],
            },
            {
                "title": "Recruiting Coordinator",
                "company": "Zillow Group",
                "dates": "2018 – 2019",
                "location": "Seattle, WA",
                "bullets": [
                    "Scheduled 150+ interviews per month across multiple hiring teams",
                    "Maintained candidate records in Greenhouse ATS with 99% data accuracy",
                ],
            },
        ],
        "education": [
            {"degree": "Bachelor of Arts, Communications", "school": "University of Washington", "year": "2018"},
        ],
        "certifications": ["SHRM Certified Professional (SHRM-CP)"],
        "skills": ["HRIS Administration (Workday)", "Benefits Coordination", "Recruiting Support", "Onboarding", "Leave of Absence Processing", "Employment Law Basics", "Microsoft Office", "Greenhouse ATS"],
    },
    {
        "filename": "resume_07.pdf",
        "name": "Lisa Park",
        "current_title": "HR Manager | Ford Motor Company",
        "contact": "lisa.park@email.com  |  +1 (313) 555-0707  |  Detroit, MI  |  linkedin.com/in/lisapark-hr",
        "summary": (
            "HR Manager with 10 years of experience in manufacturing and automotive industries. "
            "Strong background in labor relations, safety compliance, and HR operations for hourly "
            "workforces. SHRM-CP certified with solid knowledge of union contract administration "
            "and OSHA regulations. Effective communicator and team player with a pragmatic, "
            "process-oriented approach to HR management."
        ),
        "experience": [
            {
                "title": "HR Manager – Assembly Plant",
                "company": "Ford Motor Company",
                "dates": "2018 – Present",
                "location": "Dearborn, MI",
                "bullets": [
                    "Manage HR operations for 900-person assembly facility; supervise team of 8 HR professionals",
                    "Administer UAW collective bargaining agreement covering 750 hourly employees",
                    "Manage workers' compensation program with $2.1M annual cost reduction through early intervention",
                    "Coordinate annual merit increase process and hourly wage administration for 3 contract years",
                ],
            },
            {
                "title": "HR Business Partner",
                "company": "General Motors",
                "dates": "2014 – 2018",
                "location": "Warren, MI",
                "bullets": [
                    "Supported HR for 600-person powertrain engineering center",
                    "Coordinated staffing for 3 skilled trades classifications and 12 salaried job families",
                    "Led OSHA VPP Star re-certification process with zero compliance findings",
                ],
            },
            {
                "title": "HR Coordinator",
                "company": "Magna International",
                "dates": "2014",
                "location": "Troy, MI",
                "bullets": [
                    "Supported HR operations for two tier-1 automotive supply facilities",
                ],
            },
        ],
        "education": [
            {"degree": "Master of Arts, Business Administration", "school": "Wayne State University", "year": "2014"},
            {"degree": "Bachelor of Arts, Sociology", "school": "Michigan State University", "year": "2012"},
        ],
        "certifications": ["SHRM Certified Professional (SHRM-CP)", "OSHA 30-Hour General Industry", "Certified Labor Relations Professional (CLRP)"],
        "skills": ["Union Contract Administration", "Labor Relations", "OSHA Compliance", "Workers Compensation", "Hourly Workforce Management", "Grievance Handling", "HR Operations", "SAP HR"],
    },
    {
        "filename": "resume_08.pdf",
        "name": "James Wilson",
        "current_title": "Senior HR Business Partner | Goldman Sachs",
        "contact": "james.wilson@email.com  |  +1 (212) 555-0808  |  New York, NY  |  linkedin.com/in/jameswilson-hr",
        "summary": (
            "HR professional with 12 years of experience primarily in financial services and professional "
            "services environments. PHR certified with expertise in executive compensation, talent acquisition "
            "for highly competitive markets, and managing HR for demanding, performance-driven cultures. "
            "Strong analytical skills and business acumen with a track record of aligning HR initiatives "
            "with revenue and profitability goals. Looking to transition from financial services into "
            "a corporate HR Director role in a healthcare or life sciences environment."
        ),
        "experience": [
            {
                "title": "Senior HR Business Partner – Investment Banking",
                "company": "Goldman Sachs",
                "dates": "2017 – Present",
                "location": "New York, NY",
                "bullets": [
                    "Strategic HR partner to Managing Director and Partner cohort in Global M&A and Leveraged Finance divisions",
                    "Manage compensation review for 350 professionals including complex deferred compensation structures",
                    "Lead performance differentiation calibration for analyst through partner populations",
                    "Partner with talent acquisition on lateral hiring of MD-level revenue producers",
                    "Coach and develop 12 junior HR colleagues as informal team lead",
                ],
            },
            {
                "title": "HR Business Partner",
                "company": "JPMorgan Chase",
                "dates": "2013 – 2017",
                "location": "New York, NY",
                "bullets": [
                    "Supported HR for 800-person commercial banking division across Northeast region",
                    "Managed workforce reduction affecting 220 positions with zero employment litigation",
                ],
            },
            {
                "title": "HR Associate",
                "company": "Deloitte",
                "dates": "2012 – 2013",
                "location": "New York, NY",
                "bullets": [
                    "Supported HR operations for 500-person audit and advisory practice",
                ],
            },
        ],
        "education": [
            {"degree": "Master of Business Administration (MBA), Finance", "school": "NYU Stern School of Business", "year": "2012"},
            {"degree": "Bachelor of Science, Economics", "school": "Georgetown University", "year": "2010"},
        ],
        "certifications": ["Professional in Human Resources (PHR)", "Certified Equity Professional (CEP)"],
        "skills": ["Executive Compensation", "Talent Acquisition – Senior Levels", "Performance Management", "Workforce Reduction Planning", "Financial Services HR", "Analytical Modeling", "Compensation Analytics", "SuccessFactors"],
    },
    {
        "filename": "resume_09.pdf",
        "name": "Maria Gonzalez",
        "current_title": "HR Supervisor | Publix Super Markets",
        "contact": "maria.gonzalez@email.com  |  +1 (305) 555-0909  |  Miami, FL  |  linkedin.com/in/mariagonzalez-hr",
        "summary": (
            "Bilingual (English/Spanish) HR professional with 9 years of experience in retail and food service "
            "sectors. SHRM-CP certified with strong skills in hourly workforce management, bilingual employee "
            "relations, and high-volume recruiting. Experienced in managing HR for diverse, frontline workforces "
            "with demonstrated commitment to creating inclusive environments. Seeking to advance into a "
            "broader HR management role with increased scope."
        ),
        "experience": [
            {
                "title": "HR Supervisor",
                "company": "Publix Super Markets",
                "dates": "2019 – Present",
                "location": "Miami, FL",
                "bullets": [
                    "Supervise 7 HR associates supporting 12 store locations with 2,800+ total employees",
                    "Lead bilingual recruitment campaigns for full-time and part-time positions, filling 150+ roles quarterly",
                    "Manage employee relations, conduct workplace investigations, and advise store managers on HR matters",
                    "Coordinate benefits open enrollment and wellness program participation for Miami district",
                ],
            },
            {
                "title": "HR Coordinator",
                "company": "Winn-Dixie Stores",
                "dates": "2016 – 2019",
                "location": "Miami, FL",
                "bullets": [
                    "Supported HR for 6 grocery store locations; managed onboarding for 300+ new associates annually",
                    "Maintained I-9 compliance and e-Verify records for 1,200 active employees",
                ],
            },
            {
                "title": "Recruiting Coordinator",
                "company": "Carnival Cruise Line",
                "dates": "2015 – 2016",
                "location": "Miami, FL",
                "bullets": [
                    "Coordinated shipboard crew recruitment for entertainment and hospitality positions",
                    "Processed 500+ applications per month in Taleo ATS",
                ],
            },
        ],
        "education": [
            {"degree": "Bachelor of Arts, Human Resources Management", "school": "Florida International University", "year": "2015"},
        ],
        "certifications": ["SHRM Certified Professional (SHRM-CP)"],
        "skills": ["Bilingual HR (English/Spanish)", "High-Volume Recruiting", "Employee Relations", "Benefits Administration", "I-9 / E-Verify Compliance", "Frontline Workforce Management", "Taleo ATS", "ADP Workforce Now"],
    },
    {
        "filename": "resume_10.pdf",
        "name": "Thomas Brown",
        "current_title": "HR Consultant | Deloitte Consulting",
        "contact": "thomas.brown@email.com  |  +1 (312) 555-1010  |  Chicago, IL  |  linkedin.com/in/thomasbrown-hr",
        "summary": (
            "SPHR-certified HR consultant with 10 years of experience advising Fortune 500 clients on HR "
            "transformation, technology implementation, and talent strategy. Deep technical expertise in HRIS "
            "platforms (Workday, SAP SuccessFactors, Oracle HCM) and strong project management capabilities. "
            "Experience spans pharmaceutical, technology, and financial services clients. Seeking transition "
            "from consulting into an in-house HR Director role to apply strategic expertise in a single "
            "enterprise context. Limited direct people management experience given consulting model."
        ),
        "experience": [
            {
                "title": "Senior HR Consultant – Human Capital Practice",
                "company": "Deloitte Consulting",
                "dates": "2018 – Present",
                "location": "Chicago, IL",
                "bullets": [
                    "Lead HR transformation engagements for pharmaceutical and life sciences clients (avg. $2M project scope)",
                    "Serve as workstream lead for HRIS implementations; manage project team of 6 analysts",
                    "Designed enterprise talent acquisition process re-engineering for 3 Fortune 500 clients",
                    "Facilitated HR shared services design and transition for global pharma client (12,000 employees, 30 countries)",
                ],
            },
            {
                "title": "HR Consultant",
                "company": "Mercer LLC",
                "dates": "2014 – 2018",
                "location": "Chicago, IL",
                "bullets": [
                    "Delivered compensation benchmarking, job architecture, and pay equity analyses for 20+ clients",
                    "Authored HR function effectiveness assessments and presented findings to CHRO-level stakeholders",
                ],
            },
            {
                "title": "HR Analyst",
                "company": "Hewitt Associates",
                "dates": "2014",
                "location": "Lincolnshire, IL",
                "bullets": [
                    "Supported benefits outsourcing delivery for large employer clients",
                ],
            },
        ],
        "education": [
            {"degree": "Master of Arts, Human Resources Management", "school": "Loyola University Chicago", "year": "2014"},
            {"degree": "Bachelor of Science, Business Administration", "school": "Indiana University Kelley School of Business", "year": "2012"},
        ],
        "certifications": ["Senior Professional in Human Resources (SPHR)", "Workday HCM Certified Implementer", "SAP SuccessFactors Associate Certification", "Project Management Professional (PMP)"],
        "skills": ["HRIS Implementation (Workday, SAP SF, Oracle)", "HR Process Design", "Change Management", "Project Management", "Compensation Analysis", "HR Shared Services Design", "Talent Strategy", "Data Analytics"],
    },
]


if __name__ == "__main__":
    print(f"Generating {len(RESUMES)} resume PDFs in {OUTPUT_DIR}/")
    for resume_data in RESUMES:
        build_resume(resume_data["filename"], resume_data)
    print(f"\nDone! {len(RESUMES)} PDFs created in {OUTPUT_DIR}")
