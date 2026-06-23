from io import BytesIO

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


def create_pdf(title, content):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    normal_style = styles["BodyText"]

    story = []

    story.append(
        Paragraph(title, title_style)
    )

    story.append(
        Spacer(1, 0.3 * inch)
    )

    for line in content.split("\n"):

        if line.strip() != "":

            story.append(
                Paragraph(line, normal_style)
            )

            story.append(
                Spacer(1, 0.1 * inch)
            )

    doc.build(story)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf