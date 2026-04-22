"""
Generate a BEGINNER-FRIENDLY AWS deployment PDF for the Agentic Compliance Auditor.
This guide assumes:
  - You have NEVER used AWS before
  - You want the CHEAPEST working setup
  - You need literal click-by-click instructions
Output: AWS_Beginner_Guide.pdf
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    ListFlowable, ListItem, Preformatted,
)

OUTPUT = "AWS_Beginner_Guide_OpenAI.pdf"

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=styles["Title"], fontSize=24,
                       textColor=HexColor("#0B3D91"))
SUB = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=12,
                     textColor=HexColor("#555555"), alignment=TA_CENTER)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, spaceAfter=10,
                    textColor=HexColor("#0B3D91"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=10,
                    spaceAfter=6, textColor=HexColor("#0B3D91"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceBefore=8,
                    spaceAfter=4, textColor=HexColor("#333333"))
BODY = ParagraphStyle("B", parent=styles["BodyText"], fontSize=10.5,
                      leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
STEP = ParagraphStyle("Step", parent=BODY, fontSize=10.5, leading=14,
                      leftIndent=6, spaceAfter=4)
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=9.5, leading=13,
                      textColor=HexColor("#333"), leftIndent=10, rightIndent=10,
                      backColor=HexColor("#FFF8E1"), borderPadding=6, spaceAfter=8)
WARN = ParagraphStyle("Warn", parent=BODY, fontSize=9.5, leading=13,
                      textColor=HexColor("#7A1F1F"), leftIndent=10, rightIndent=10,
                      backColor=HexColor("#FDECEC"), borderPadding=6, spaceAfter=8)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=8.5, leading=11,
                      textColor=HexColor("#111"), backColor=HexColor("#F2F2F2"),
                      leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=8,
                      borderPadding=6)


def p(t): return Paragraph(t, BODY)
def h1(t): return Paragraph(t, H1)
def h2(t): return Paragraph(t, H2)
def h3(t): return Paragraph(t, H3)
def note(t): return Paragraph("<b>Note:</b> " + t, NOTE)
def warn(t): return Paragraph("<b>Warning:</b> " + t, WARN)
def code(t): return Preformatted(t.strip("\n"), CODE)


def steps(items):
    return ListFlowable(
        [ListItem(Paragraph(i, STEP), leftIndent=10) for i in items],
        bulletType="1", start="1", leftIndent=20,
    )


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(i, BODY), leftIndent=10) for i in items],
        bulletType="bullet", start="circle", leftIndent=16,
    )


def kv(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths or [2.2 * inch, 4.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0B3D91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, HexColor("#BBBBBB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [HexColor("#FFFFFF"), HexColor("#F7F9FC")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


story = []

# ===== Cover =====
story += [
    Spacer(1, 1.0 * inch),
    Paragraph("AWS Deployment — Beginner Guide", TITLE),
    Spacer(1, 0.15 * inch),
    Paragraph("Agentic Compliance Auditor", SUB),
    Spacer(1, 0.08 * inch),
    Paragraph("Step-by-step, cheapest-option, zero AWS experience required",
              SUB),
    Spacer(1, 0.6 * inch),
]
story.append(kv([
    ["What this guide gives you", "Details"],
    ["Total AWS cost", "~$0 for first 12 months (free tier), ~$8/month after"],
    ["LLM cost (OpenAI gpt-4o-mini)", "~$0.0006 per query (~$18/month @ 1k queries/day)"],
    ["AWS services used", "EC2 (1 instance, t3.micro), Elastic IP, Security Group"],
    ["Services we SKIP to save money",
     "ALB ($16/mo), RDS, ElastiCache, OpenSearch, Bedrock, ECS"],
    ["Deployment style",
     "Single EC2 server running FastAPI + Streamlit + ChromaDB. LLM is OpenAI API (no GPU needed)."],
    ["Time to complete", "About 60-90 minutes the first time"],
    ["What you need before starting",
     "Credit card, email, phone, project code, OpenAI API key (sk-...)"],
]))
story.append(PageBreak())

# ===== Table of Contents =====
story += [h1("What's Inside")]
story += [bullets([
    "Part 1 — What AWS will cost you (read first)",
    "Part 2 — Create your AWS account and set safety alarms",
    "Part 3 — Create an IAM user (never use the root account)",
    "Part 4 — Launch your EC2 server (the cheapest sizing)",
    "Part 5 — Connect to your server using SSH",
    "Part 6 — Install Python and project dependencies",
    "Part 7 — Upload your project code to the server",
    "Part 8 — Configure the project for production",
    "Part 9 — Run the API and UI as background services (systemd)",
    "Part 10 — Ingest your compliance documents on the server",
    "Part 11 — Open the app to the internet (Security Group rules)",
    "Part 12 — (Optional) Add a domain name and HTTPS",
    "Part 13 — Daily operations: stop, start, update, check logs",
    "Part 14 — How to save even more money",
    "Part 15 — Troubleshooting",
])]
story.append(PageBreak())

# ===== Part 1 =====
story += [h1("Part 1 — What AWS will cost you")]
story += [p(
    "Before you touch AWS, understand what you'll pay. AWS charges per hour "
    "for running servers and per GB for storage and data transfer. We're "
    "picking the smallest pieces that still work.")]
story += [h2("Free Tier (first 12 months from account creation)")]
story += [bullets([
    "<b>EC2 t2.micro or t3.micro</b>: 750 hours/month FREE. That is enough to run one server 24/7.",
    "<b>EBS storage</b>: 30 GB FREE.",
    "<b>Data transfer OUT</b>: 100 GB/month FREE.",
    "<b>Elastic IP</b>: FREE while attached to a running instance.",
])]
story += [h2("After the free tier ends (or if you pick a larger size)")]
story.append(kv([
    ["Item", "Monthly cost (us-east-1, approx)"],
    ["t3.micro (1 GB RAM) — recommended, fits free tier", "~$7.50"],
    ["t3.small (2 GB RAM) — if you want more headroom", "~$15.00"],
    ["30 GB EBS gp3 storage", "~$2.40"],
    ["Elastic IP (unattached)", "~$3.60 — keep it attached to avoid this"],
    ["Data transfer OUT (typical personal use)", "$0-2"],
]))
story += [note(
    "Because the LLM now runs on OpenAI's servers (not on your EC2), you no "
    "longer need a big instance. <b>t3.micro</b> is enough and is free for "
    "12 months. The only heavy thing on the box is the embedding model "
    "(~400 MB RAM), which t3.micro handles fine.")]
story += [h2("OpenAI API cost")]
story.append(kv([
    ["Metric", "gpt-4o-mini"],
    ["Input tokens", "$0.15 per 1 million"],
    ["Output tokens", "$0.60 per 1 million"],
    ["Typical query (2k in, 500 out)", "~$0.0006"],
    ["1,000 queries/day", "~$18/month"],
    ["100 queries/day (testing)", "~$1.80/month"],
]))
story += [warn(
    "Always set a BILLING ALARM before you do anything else. A $5 alarm will "
    "email you the moment AWS charges start — you'll set this up in Part 2.")]
story.append(PageBreak())

# ===== Part 2 =====
story += [h1("Part 2 — Create AWS account and billing alarm")]
story += [h2("2.1 Create the account")]
story += [steps([
    "Open <b>https://aws.amazon.com</b> in your browser.",
    "Click the orange <b>Create an AWS Account</b> button in the top-right.",
    "Enter your email, a password, and an account name (e.g., \"My Compliance App\").",
    "Choose <b>Personal</b> account type. Fill in name, address, phone.",
    "Enter a credit/debit card. AWS will charge $1 temporarily to verify.",
    "Verify your phone via SMS or voice call — enter the 4-digit code they read.",
    "Choose the <b>Basic support - Free</b> plan.",
    "You'll get an email \"Welcome to Amazon Web Services\". Click the link to sign in.",
])]
story += [h2("2.2 Pick your region")]
story += [p(
    "In the top-right corner of the AWS Console, click the region name. "
    "Pick the region closest to you. Cheapest two options:")]
story += [bullets([
    "<b>us-east-1 (N. Virginia)</b> — cheapest region, pick this if you're in the US east coast.",
    "<b>us-east-2 (Ohio)</b> — same price as us-east-1, pick if you prefer.",
    "<b>ap-south-1 (Mumbai)</b> — pick if you're in India (lower latency).",
])]
story += [warn(
    "Whatever region you pick, STAY IN IT. Resources in different regions "
    "cannot talk to each other cheaply. This guide assumes us-east-1 but "
    "every step works in any region.")]

story += [h2("2.3 Set a billing alarm ($5 safety net)")]
story += [steps([
    "In the top search bar, type <b>Billing</b> and click \"Billing and Cost Management\".",
    "On the left sidebar, click <b>Billing preferences</b>.",
    "Check the box <b>Receive Free Tier Usage Alerts</b>. Enter your email. Click Save.",
    "In the same left sidebar, click <b>Budgets</b>.",
    "Click <b>Create budget</b> → choose <b>Use a template</b> → <b>Zero spend budget</b>.",
    "Enter your email address. Click <b>Create budget</b>.",
    "You'll now get an email the moment your bill goes above $0.01.",
])]
story += [note(
    "The Zero Spend budget is the safest option — you'll be notified "
    "immediately if anything starts costing money.")]
story.append(PageBreak())

# ===== Part 3 =====
story += [h1("Part 3 — Create an IAM user (don't use root)")]
story += [p(
    "Your root account (the email you signed up with) has unlimited power "
    "and is a big target for hackers. AWS best practice is to create a "
    "normal \"IAM user\" and use that for everything.")]
story += [steps([
    "In the AWS Console search bar, type <b>IAM</b> and click the IAM service.",
    "On the left sidebar, click <b>Users</b>, then click <b>Create user</b> (top-right).",
    "User name: <b>admin-yourname</b>. Check <b>Provide user access to the AWS Management Console</b>. Pick <b>I want to create an IAM user</b>. Set a password. Uncheck \"User must create a new password\".",
    "Click <b>Next</b>. For permissions, pick <b>Attach policies directly</b>, search for <b>AdministratorAccess</b> and check it.",
    "Click <b>Next</b>, then <b>Create user</b>.",
    "On the success page, click <b>Return to users list</b>. Click your new user. Note the <b>Console sign-in URL</b> (looks like https://123456789012.signin.aws.amazon.com/console).",
    "Sign out of the root account. Sign back in using the IAM user URL, your user name, and password.",
])]
story += [warn(
    "From now on, NEVER sign in with your root email. Lock the root credentials "
    "in a password manager and only use them for emergencies (like closing "
    "the account).")]
story.append(PageBreak())

# ===== Part 4 =====
story += [h1("Part 4 — Launch your EC2 server")]
story += [p(
    "EC2 is the service that gives you a virtual computer in the cloud. "
    "We'll launch one small Linux server that runs everything: the API, the "
    "UI, Ollama (the LLM), and ChromaDB (the vector store).")]

story += [h2("4.1 Open the EC2 console")]
story += [steps([
    "In the search bar type <b>EC2</b> and click the EC2 service.",
    "On the left sidebar, click <b>Instances</b>.",
    "Click the orange <b>Launch instances</b> button (top-right).",
])]

story += [h2("4.2 Fill the launch form (exact values to pick)")]
story.append(kv([
    ["Field", "What to pick"],
    ["Name", "compliance-auditor"],
    ["AMI (Operating System)",
     "<b>Amazon Linux 2023</b> (free tier eligible). Leave default."],
    ["Architecture", "64-bit (x86)"],
    ["Instance type", "<b>t3.micro</b> (recommended, free tier). Upgrade to t3.small later if needed."],
    ["Key pair",
     "Click <b>Create new key pair</b>. Name: <b>compliance-key</b>. Type: RSA. Format: <b>.pem</b>. Click Create → it downloads compliance-key.pem. <b>Save this file.</b>"],
    ["Network settings",
     "Click <b>Edit</b>. Check all three: <b>Allow SSH from My IP</b>, <b>Allow HTTP from anywhere</b>, <b>Allow HTTPS from anywhere</b>."],
    ["Add custom rule",
     "Click <b>Add security group rule</b>. Port: 8501 (Streamlit UI). Source: Anywhere. Click again: Port 8000 (API). Source: Anywhere."],
    ["Configure storage", "30 GB gp3 (free tier limit)"],
    ["Advanced details", "Leave everything default"],
]))
story += [warn(
    "Keep the <b>.pem key file safe</b>. You CANNOT download it again. If you "
    "lose it, you lose access to the server and must terminate it and start "
    "over.")]
story += [steps([
    "Scroll to the bottom. Click <b>Launch instance</b>.",
    "Wait 30 seconds. Click <b>View all instances</b>.",
    "You should see your instance. Wait for <b>Instance state: Running</b> and <b>Status checks: 2/2 checks passed</b> (takes 1-2 minutes).",
])]

story += [h2("4.3 Attach an Elastic IP (so your IP doesn't change)")]
story += [p(
    "Without this, your server's public IP will change every time you "
    "stop/start it. Elastic IP is free while attached.")]
story += [steps([
    "In the EC2 left sidebar, click <b>Elastic IPs</b>.",
    "Click <b>Allocate Elastic IP address</b> → <b>Allocate</b>.",
    "Select the new IP, click <b>Actions</b> → <b>Associate Elastic IP address</b>.",
    "Instance: choose your compliance-auditor instance. Click <b>Associate</b>.",
    "Note the IP address (like <b>54.123.45.67</b>). You'll use it everywhere.",
])]
story.append(PageBreak())

# ===== Part 5 =====
story += [h1("Part 5 — Connect to your server (SSH)")]
story += [h2("5.1 On Windows (using built-in OpenSSH)")]
story += [steps([
    "Open PowerShell.",
    "Move your key file somewhere permanent, e.g. <b>C:\\aws\\compliance-key.pem</b>.",
    "Run the command below to fix permissions (Windows requires strict permissions on key files).",
])]
story += [code(
    'icacls "C:\\aws\\compliance-key.pem" /inheritance:r\n'
    'icacls "C:\\aws\\compliance-key.pem" /grant:r "$($env:USERNAME):R"'
)]
story += [steps([
    "Now SSH in (replace 54.123.45.67 with YOUR Elastic IP):",
])]
story += [code('ssh -i "C:\\aws\\compliance-key.pem" ec2-user@54.123.45.67')]
story += [steps([
    "The first time, it'll ask \"Are you sure you want to continue connecting?\". Type <b>yes</b>.",
    "You should see a prompt like <b>[ec2-user@ip-172-31-x-x ~]$</b>. You're inside the server.",
])]

story += [h2("5.2 On Mac/Linux")]
story += [code(
    "chmod 400 ~/Downloads/compliance-key.pem\n"
    "ssh -i ~/Downloads/compliance-key.pem ec2-user@54.123.45.67"
)]
story += [note(
    "From now on, every command starting with a <b>$</b> means you run it "
    "on the server (while SSH'd in). Commands starting with <b>PS&gt;</b> or "
    "without a prefix run on your local laptop.")]
story.append(PageBreak())

# ===== Part 6 =====
story += [h1("Part 6 — Install Python and project dependencies")]
story += [p("Run these commands one block at a time, on the server.")]

story += [h2("6.1 System update and basic tools")]
story += [code(
    "sudo dnf update -y\n"
    "sudo dnf install -y git python3.11 python3.11-pip gcc-c++ make tar gzip"
)]

story += [h2("6.2 Get your OpenAI API key")]
story += [steps([
    "Open <b>https://platform.openai.com/api-keys</b> in your browser (on your laptop, not the server).",
    "Sign in or create an account.",
    "Click <b>Create new secret key</b>. Name it <b>compliance-auditor</b>.",
    "<b>Copy the key immediately</b> (starts with <b>sk-...</b>). You cannot view it again — if you lose it, create a new one.",
    "Optional: go to <b>Billing</b> → add $5-10 credit. Usage limits in <b>Settings</b> → <b>Limits</b> → set a monthly hard cap (e.g., $10) to avoid surprises.",
])]
story += [warn(
    "<b>Never put your key in code, GitHub, Docker images, or chat messages.</b> "
    "It only goes in the <b>.env</b> file on the server (covered in Part 8).")]

story += [h2("6.3 (Optional) Skip Redis — use in-memory mode")]
story += [p(
    "Your app already falls back to in-memory session storage when Redis "
    "isn't available. This saves money (no ElastiCache fee). No action needed.")]
story.append(PageBreak())

# ===== Part 7 =====
story += [h1("Part 7 — Upload your project code")]
story += [h2("7.1 Option A: Use Git (easiest if your code is on GitHub)")]
story += [code(
    "cd ~\n"
    "git clone https://github.com/YOUR_USERNAME/Agentic_Compliance_Auditor.git\n"
    "cd Agentic_Compliance_Auditor"
)]

story += [h2("7.2 Option B: Upload from your laptop with SCP")]
story += [p("From your LOCAL PowerShell (not the server):")]
story += [code(
    'scp -i "C:\\aws\\compliance-key.pem" -r '
    '"C:\\All Files\\Projects\\AI_Projects\\Agentic_Compliance_Auditor" '
    'ec2-user@54.123.45.67:~/'
)]
story += [note(
    "Before uploading, delete <b>venv/</b>, <b>data/chroma_db/</b>, "
    "<b>__pycache__</b> folders from your laptop copy — they're huge and will "
    "be rebuilt on the server. We'll rebuild chroma_db in Part 10.")]

story += [h2("7.3 Create a fresh virtual environment on the server")]
story += [code(
    "cd ~/Agentic_Compliance_Auditor\n"
    "python3.11 -m venv venv\n"
    "source venv/bin/activate\n"
    "pip install --upgrade pip\n"
    "pip install -r requirements.txt"
)]
story += [note(
    "This takes 5-10 minutes. sentence-transformers and torch are big.")]
story.append(PageBreak())

# ===== Part 8 =====
story += [h1("Part 8 — Configure for production")]

story += [h2("8.1 Create .env file on the server")]
story += [code("nano ~/Agentic_Compliance_Auditor/.env")]
story += [p("Paste this — replace <b>sk-YOUR_KEY_HERE</b> with your real OpenAI key from Part 6.2:")]
story += [code(
    "# LLM Configuration (OpenAI)\n"
    "OPENAI_API_KEY=sk-YOUR_KEY_HERE\n"
    "OPENAI_MODEL=gpt-4o-mini\n"
    "OPENAI_BASE_URL=https://api.openai.com/v1\n"
    "OPENAI_TEMPERATURE=0.1\n\n"
    "EMBEDDING_MODEL=all-mpnet-base-v2\n\n"
    "# Use in-memory fallback (no Redis needed)\n"
    "REDIS_HOST=localhost\n"
    "REDIS_PORT=6379\n"
    "REDIS_DB=0\n\n"
    "API_HOST=0.0.0.0\n"
    "API_PORT=8000\n"
    "API_WORKERS=2\n\n"
    "LOG_LEVEL=INFO\n"
    "LOG_FILE=./logs/app.log\n\n"
    "TOP_K_RESULTS=5\n"
    "CHUNK_SIZE=256\n"
    "CHUNK_OVERLAP=20\n"
    "RERANK_ENABLED=false\n\n"
    "MAX_ITERATIONS=3\n"
    "AGENT_TIMEOUT=300\n"
    "ENABLE_SELF_REFLECTION=false"
)]
story += [p("Press <b>Ctrl+O</b>, Enter to save, <b>Ctrl+X</b> to exit nano.")]
story += [h3("8.1.1 Lock down .env permissions")]
story += [p("So only you (ec2-user) can read the key file:")]
story += [code("chmod 600 ~/Agentic_Compliance_Auditor/.env")]

story += [h2("8.2 Allow the UI to reach the API")]
story += [p(
    "Streamlit calls the API at http://localhost:8000 by default — that "
    "works since they're on the same box. No change needed.")]

story += [h2("8.3 Open firewall ports on the instance OS (usually already open)")]
story += [p(
    "Amazon Linux 2023 has no firewall daemon by default. The AWS Security "
    "Group you configured in Part 4.2 is what matters.")]
story.append(PageBreak())

# ===== Part 9 =====
story += [h1("Part 9 — Run API and UI as background services")]
story += [p(
    "<b>systemd</b> is Linux's built-in service manager. It starts your app "
    "at boot and restarts it if it crashes.")]

story += [h2("9.1 Create the API service")]
story += [code("sudo nano /etc/systemd/system/compliance-api.service")]
story += [p("Paste:")]
story += [code(
    "[Unit]\n"
    "Description=Compliance Auditor API\n"
    "After=network.target\n\n"
    "[Service]\n"
    "User=ec2-user\n"
    "WorkingDirectory=/home/ec2-user/Agentic_Compliance_Auditor\n"
    "Environment=\"PATH=/home/ec2-user/Agentic_Compliance_Auditor/venv/bin\"\n"
    "ExecStart=/home/ec2-user/Agentic_Compliance_Auditor/venv/bin/uvicorn "
    "src.api.main:app --host 0.0.0.0 --port 8000\n"
    "Restart=always\n"
    "RestartSec=5\n\n"
    "[Install]\n"
    "WantedBy=multi-user.target"
)]

story += [h2("9.2 Create the UI service")]
story += [code("sudo nano /etc/systemd/system/compliance-ui.service")]
story += [p("Paste:")]
story += [code(
    "[Unit]\n"
    "Description=Compliance Auditor UI\n"
    "After=network.target compliance-api.service\n\n"
    "[Service]\n"
    "User=ec2-user\n"
    "WorkingDirectory=/home/ec2-user/Agentic_Compliance_Auditor\n"
    "Environment=\"PATH=/home/ec2-user/Agentic_Compliance_Auditor/venv/bin\"\n"
    "ExecStart=/home/ec2-user/Agentic_Compliance_Auditor/venv/bin/streamlit "
    "run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 "
    "--server.headless true\n"
    "Restart=always\n"
    "RestartSec=5\n\n"
    "[Install]\n"
    "WantedBy=multi-user.target"
)]

story += [h2("9.3 Enable and start both services")]
story += [code(
    "sudo systemctl daemon-reload\n"
    "sudo systemctl enable compliance-api compliance-ui\n"
    "sudo systemctl start compliance-api\n"
    "sudo systemctl start compliance-ui\n"
    "sudo systemctl status compliance-api\n"
    "sudo systemctl status compliance-ui"
)]
story += [note(
    "Both should say <b>active (running)</b>. Press <b>q</b> to exit the "
    "status screen.")]
story.append(PageBreak())

# ===== Part 10 =====
story += [h1("Part 10 — Ingest your compliance documents")]
story += [h2("10.1 Upload your PDFs/docs to the server")]
story += [p("From your LOCAL PowerShell:")]
story += [code(
    'scp -i "C:\\aws\\compliance-key.pem" -r '
    '"C:\\path\\to\\your\\documents\\*" '
    'ec2-user@54.123.45.67:~/Agentic_Compliance_Auditor/data/sample_docs/'
)]

story += [h2("10.2 Run the ingestion script")]
story += [code(
    "cd ~/Agentic_Compliance_Auditor\n"
    "source venv/bin/activate\n"
    "python ingest_documents.py"
)]
story += [p(
    "Wait until you see a line like <b>\"Added N chunks to vector store\"</b>. "
    "This takes 2-5 minutes depending on document count.")]

story += [h2("10.3 Restart the API so it sees the new vector store")]
story += [code("sudo systemctl restart compliance-api")]
story.append(PageBreak())

# ===== Part 11 =====
story += [h1("Part 11 — Open the app to the internet")]
story += [p(
    "If you followed Part 4.2 exactly, ports 8000 and 8501 are already open. "
    "Test from your laptop browser:")]
story += [bullets([
    "API docs: <b>http://54.123.45.67:8000/docs</b>",
    "Streamlit UI: <b>http://54.123.45.67:8501</b>",
])]
story += [note(
    "Replace 54.123.45.67 with your Elastic IP from Part 4.3.")]

story += [h2("11.1 If it doesn't load")]
story += [steps([
    "Check services are running: <b>sudo systemctl status compliance-api compliance-ui</b>",
    "Check listening ports: <b>sudo ss -tlnp | grep -E '8000|8501'</b>",
    "Check the Security Group: EC2 Console → Instances → your instance → Security tab → click the security group → Inbound rules. Ensure ports 8000 and 8501 have source <b>0.0.0.0/0</b>.",
    "Check logs: <b>sudo journalctl -u compliance-api -n 50 --no-pager</b>",
])]

story += [h2("11.2 Lock down later (recommended)")]
story += [p(
    "Exposing ports 8000 and 8501 to the whole internet is fine for testing. "
    "Once it works, restrict the Security Group to your own IP (change source "
    "from 0.0.0.0/0 to My IP).")]
story.append(PageBreak())

# ===== Part 12 =====
story += [h1("Part 12 — (Optional) Domain name + HTTPS")]
story += [p(
    "Only do this if you want a nice URL like https://audit.mydomain.com "
    "instead of http://54.123.45.67:8501.")]

story += [h2("12.1 Point a domain at your server")]
story += [steps([
    "Buy a domain (Namecheap or Cloudflare, ~$10/year). Cheaper than AWS Route 53.",
    "In your DNS provider, create an <b>A record</b>: name <b>audit</b>, value <b>54.123.45.67</b> (your Elastic IP).",
    "Wait 5 minutes for DNS to propagate.",
])]

story += [h2("12.2 Install nginx and Let's Encrypt on the server")]
story += [code(
    "sudo dnf install -y nginx\n"
    "sudo systemctl enable --now nginx\n"
    "sudo dnf install -y certbot python3-certbot-nginx"
)]

story += [h2("12.3 Configure nginx to reverse-proxy to Streamlit")]
story += [code("sudo nano /etc/nginx/conf.d/compliance.conf")]
story += [code(
    "server {\n"
    "    listen 80;\n"
    "    server_name audit.mydomain.com;\n"
    "    location / {\n"
    "        proxy_pass http://127.0.0.1:8501;\n"
    "        proxy_http_version 1.1;\n"
    "        proxy_set_header Upgrade $http_upgrade;\n"
    "        proxy_set_header Connection \"upgrade\";\n"
    "        proxy_set_header Host $host;\n"
    "    }\n"
    "    location /api/ {\n"
    "        proxy_pass http://127.0.0.1:8000/;\n"
    "    }\n"
    "}"
)]
story += [code(
    "sudo nginx -t\n"
    "sudo systemctl reload nginx"
)]

story += [h2("12.4 Get free HTTPS certificate")]
story += [code("sudo certbot --nginx -d audit.mydomain.com")]
story += [p(
    "Enter your email, agree to terms, pick redirect-to-HTTPS. Certbot "
    "auto-renews every 60 days.")]
story += [note(
    "Once HTTPS works, you can close ports 8000 and 8501 in the Security "
    "Group — only keep 80, 443, and 22 (SSH).")]
story.append(PageBreak())

# ===== Part 13 =====
story += [h1("Part 13 — Daily operations")]
story += [h2("13.1 Useful commands")]
story.append(kv([
    ["Task", "Command (on the server)"],
    ["Check API logs (live)", "sudo journalctl -u compliance-api -f"],
    ["Check UI logs (live)", "sudo journalctl -u compliance-ui -f"],
    ["Restart API", "sudo systemctl restart compliance-api"],
    ["Restart UI", "sudo systemctl restart compliance-ui"],
    ["Stop both", "sudo systemctl stop compliance-api compliance-ui"],
    ["Disk usage", "df -h"],
    ["RAM usage", "free -h"],
    ["Running processes", "top (press q to quit)"],
]))

story += [h2("13.2 Deploy new code")]
story += [code(
    "cd ~/Agentic_Compliance_Auditor\n"
    "git pull              # if you used Git\n"
    "source venv/bin/activate\n"
    "pip install -r requirements.txt\n"
    "sudo systemctl restart compliance-api compliance-ui"
)]

story += [h2("13.3 Stop the server to save money (when not in use)")]
story += [steps([
    "EC2 Console → Instances → select your instance → Instance state → <b>Stop instance</b>.",
    "When stopped, you pay ONLY for storage (~$2.40/month for 30GB) — no compute charge.",
    "Start it again the same way when you want to use it.",
    "Because you attached an Elastic IP, the IP stays the same.",
])]
story.append(PageBreak())

# ===== Part 14 =====
story += [h1("Part 14 — Saving even more money")]
story += [h2("AWS side")]
story += [bullets([
    "<b>Stay on t3.micro</b> for the first 12 months — it's free. After that it's ~$7.50/month.",
    "<b>Stop the instance when idle.</b> Stopping saves ~95% of the compute cost.",
    "<b>Use Savings Plans (after you confirm the project).</b> 1-year no-upfront Savings Plan saves ~30%.",
    "<b>Keep Elastic IP attached.</b> Unattached Elastic IPs cost $3.60/month each.",
    "<b>Reduce embedding dimensions.</b> Switch to all-MiniLM-L6-v2 (384-dim) from all-mpnet-base-v2 (768-dim) to halve disk use and speed up search.",
    "<b>Delete old snapshots.</b> If you enable backups later, stale snapshots accumulate cost.",
    "<b>Monitor with Cost Explorer.</b> AWS Console → Billing → Cost Explorer. Check weekly.",
])]
story += [h2("OpenAI side")]
story += [bullets([
    "<b>Set a monthly hard limit</b> in OpenAI dashboard → Settings → Limits. Example: $10/month. Requests fail closed after that — no bill shock.",
    "<b>Use gpt-4o-mini (already the default).</b> 20x cheaper than gpt-4o, and strong enough for RAG answers.",
    "<b>Lower MAX_ITERATIONS</b> in .env to 2 or 3. Each agent loop = another LLM call.",
    "<b>Disable self-reflection</b> (ENABLE_SELF_REFLECTION=false — already default). Reflection doubles token usage.",
    "<b>Trim retrieved context.</b> TOP_K_RESULTS=3 instead of 5 saves ~40% on input tokens per query.",
    "<b>Cache repeated queries</b> (advanced). If the same question is asked often, store the answer in Redis or a JSON file for 24 hours.",
])]
story += [note(
    "Realistic total monthly cost, free-tier EC2 + 100 queries/day on OpenAI: "
    "<b>~$2</b>. Always-on t3.micro after free tier + 1,000 queries/day: "
    "<b>~$25</b>. The bulk of cost is now API usage, not AWS.")]
story.append(PageBreak())

# ===== Part 15 =====
story += [h1("Part 15 — Troubleshooting")]

story += [h3("SSH says \"Connection refused\" or \"timed out\"")]
story += [bullets([
    "Instance is still booting — wait 2 minutes.",
    "Security Group missing SSH rule — verify port 22, source My IP.",
    "Your public IP changed (home WiFi) — update the SSH rule source to your current IP.",
])]

story += [h3("\"Permissions 0644 for key file are too open\" (Windows)")]
story += [p("Run the two <b>icacls</b> commands in Part 5.1 again.")]

story += [h3("Service fails with \"ModuleNotFoundError\"")]
story += [bullets([
    "Wrong Python path in systemd unit — confirm <b>/home/ec2-user/Agentic_Compliance_Auditor/venv/bin/python</b> exists.",
    "<b>pip install -r requirements.txt</b> wasn't run inside the venv.",
])]

story += [h3("\"Out of memory\" / instance becomes unresponsive")]
story += [bullets([
    "Embedding model too big — switch EMBEDDING_MODEL to <b>all-MiniLM-L6-v2</b> in .env (uses ~200 MB instead of 400 MB).",
    "Resize instance: Stop → Actions → Instance settings → Change instance type → t3.small → Start.",
    "Add swap: <b>sudo dd if=/dev/zero of=/swapfile bs=1M count=2048 &amp;&amp; sudo chmod 600 /swapfile &amp;&amp; sudo mkswap /swapfile &amp;&amp; sudo swapon /swapfile</b>",
])]

story += [h3("Streamlit UI loads but says \"Cannot connect to API\"")]
story += [bullets([
    "API service not running — <b>sudo systemctl status compliance-api</b>.",
    "Check API logs — <b>sudo journalctl -u compliance-api -n 50 --no-pager</b>.",
    "Firewall blocking port 8000 — verify Security Group has port 8000 inbound.",
])]

story += [h3("OpenAI auth error: \"Incorrect API key\"")]
story += [bullets([
    "Check <b>.env</b> — the key must start with <b>sk-</b> and have no quotes or spaces.",
    "Confirm the key is active at <b>https://platform.openai.com/api-keys</b>.",
    "Restart the API after editing .env — <b>sudo systemctl restart compliance-api</b>.",
])]

story += [h3("OpenAI rate limit / \"You exceeded your current quota\"")]
story += [bullets([
    "You hit the spending limit you set. Raise it in OpenAI dashboard → Settings → Limits.",
    "Add billing credit at <b>https://platform.openai.com/account/billing</b>.",
    "For free-trial accounts: add a payment method — free trial tokens expire after 3 months.",
])]

story += [h3("\"Connection timeout\" when calling OpenAI")]
story += [bullets([
    "EC2 needs outbound internet — default VPC and Security Group already allow this. If you changed network settings, verify outbound HTTPS (port 443) to <b>0.0.0.0/0</b> is allowed.",
    "OpenAI itself may have an incident — check <b>https://status.openai.com</b>.",
])]

story += [h3("\"Found 0 results\" for every query")]
story += [bullets([
    "Ingestion never ran or failed. Re-run <b>python ingest_documents.py</b>.",
    "Verify chunks exist: <b>ls ~/Agentic_Compliance_Auditor/data/chroma_db/</b> should show files.",
])]

story += [h3("Browser shows \"ERR_CONNECTION_REFUSED\"")]
story += [bullets([
    "Security Group missing port 8501/8000 inbound rule.",
    "Service not running — check systemctl status.",
    "Streamlit bound to 127.0.0.1 only — ensure <b>--server.address 0.0.0.0</b> in unit file.",
])]

story += [h3("Bill showing unexpected charges")]
story += [bullets([
    "Billing → Cost Explorer → group by Service. Find the culprit.",
    "Most common: unattached Elastic IP, NAT Gateway, oversized EBS, extra regions.",
    "Delete anything you don't recognize. Then email AWS Support — they often refund first-time accidents.",
])]

story += [Spacer(1, 0.3 * inch),
          Paragraph("— End of guide. Good luck! —",
                    ParagraphStyle("End", parent=BODY, alignment=TA_CENTER,
                                   textColor=HexColor("#0B3D91"), fontSize=11))]


# ===== Build =====
doc = SimpleDocTemplate(
    OUTPUT, pagesize=letter,
    leftMargin=0.8 * inch, rightMargin=0.8 * inch,
    topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    title="AWS Beginner Deployment Guide",
    author="Agentic Compliance Auditor",
)

doc.build(story)
print(f"Wrote {OUTPUT}")
