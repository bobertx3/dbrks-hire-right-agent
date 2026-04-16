"""
Tool: send_email
Sends an HTML-formatted email via Mailgun with J&J HRD branding.
Adapted from the weatherwise agent email tool pattern.
"""
import os
import requests
from langchain_core.tools import tool


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email with HR hiring analysis results to a manager, recruiter, or stakeholder.
    Use this when the user asks to email results, send a report, or share a candidate summary.
    The 'to' parameter should be an email address; if not provided or invalid, the default
    recipient from the environment will be used."""
    # Mailgun credentials are sensitive — read from env vars only (set at endpoint deploy time).
    mailgun_url = os.getenv("MAILGUN_API_URL", "")
    mailgun_key = os.getenv("MAILGUN_API_KEY", "")
    sender = os.getenv("SENDER", "")
    default_recipient = os.getenv("RECIPIENT", "")

    if not all([mailgun_url, mailgun_key, sender]):
        return "Error: Email configuration (MAILGUN_API_URL, MAILGUN_API_KEY, SENDER) is not set."

    # Validate or fall back to default recipient
    recipient = to.strip() if to and "@" in to else default_recipient
    if not recipient:
        return "Error: No valid email recipient provided and RECIPIENT env var is not set."

    # Convert plain-text body to HTML with line breaks
    html_body = body.replace("\n", "<br>").replace("**", "")

    html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;font-family:'Segoe UI',Arial,sans-serif;background:#f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f5f5;padding:20px 0;">
    <tr><td align="center">
      <table width="680" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08);">

        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#1B3139 0%,#0f2028 100%);padding:28px 32px;">
            <div style="color:#FF3621;font-size:22px;font-weight:800;letter-spacing:1px;">HIRE RIGHT</div>
            <div style="color:rgba(255,255,255,0.65);font-size:13px;margin-top:4px;">J&amp;J HRD 2030 · AI-Powered Hiring Analysis</div>
          </td>
        </tr>

        <!-- Subject bar -->
        <tr>
          <td style="background:#FF3621;padding:10px 32px;">
            <span style="color:#ffffff;font-size:15px;font-weight:600;">{subject}</span>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="padding:32px;color:#1B1B1B;font-size:14px;line-height:1.7;">
            {html_body}
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="padding:20px 32px;background:#f8f6f4;border-top:1px solid #eee;">
            <div style="font-size:12px;color:#8A8A8A;">
              Sent by <strong>Hire Right Agent</strong> · Powered by Databricks AI &amp; MLflow<br>
              J&amp;J HRD 2030 Predictive Hiring Platform
            </div>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""

    try:
        response = requests.post(
            mailgun_url,
            auth=("api", mailgun_key),
            data={
                "from": f"Hire Right Agent <{sender}>",
                "to": recipient,
                "subject": subject,
                "html": html_content,
            },
            timeout=15,
        )
        if response.status_code in (200, 202):
            return f"✅ Email sent successfully to {recipient} (Mailgun ID: {response.json().get('id', 'unknown')})"
        else:
            return f"Email failed (HTTP {response.status_code}): {response.text[:200]}"
    except Exception as e:
        return f"Error sending email: {str(e)}"
