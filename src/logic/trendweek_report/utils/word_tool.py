from docx.shared import Pt


def add_indented_paragraph(doc, text, indent_level=1):
    """Adds a paragraph with a left indent for better formatting.

    Args:
        doc: The Word document object.
        text: The text to add to the document.
        indent_level: The indentation level (default is 1).

    Returns:
        The newly added paragraph object.
    """
    paragraph = doc.add_paragraph(text, style='Body Text')
    paragraph.paragraph_format.left_indent = Pt(12 * indent_level)  # Adjust indentation level
    return paragraph
