from textwrap import dedent

system_template = dedent('''
Create a management summary and a to-do list based
on the provided report for the department
leadership.

Below are the tasks and format for each section:

#Steps

1. **Management Summary**:
   - Write a concise 5-bullet point summary.
   - Include only the most important decisions,
risks, and next steps.

2. **To-Do List**:
   - Format each to-do as a clear task.
   - Use the structure: "Who does what by when?"
   - Suggest a priority for each task.

3. **Final Output**:
   - Compile the management summary and to-do list
into a Word document for download.

#Output Format

- The management summary should be brief, with
clear, concise bullet points.
- The to-do list should be tabulated, clearly
outlining "who, what, by when" along with the
prioritization.
- The final document should be a downloadable Word
file.

#Examples

**Management Summary Example**:
- Decision: Approved budget for Q1.
- Risk: Supply chain delays.
- Next Step: Initiate phase 2 of the project.

**To-Do List Example**:
| Task                   | Responsible | Deadline
| Priority     |
|------------------------|-------------|---------------|--------------|
| Complete market analysis | Jane Doe    |
MM/DD/YYYY    | High         |
| Update project plan     | John Smith  |
MM/DD/YYYY    | Medium       |

(Note: Real examples should be specific and
adjusted based on the provided report or content.)

#Notes

- Ensure clarity and conciseness in both sections.
- Use placeholders for names and dates if specific
information is not available.
- Prioritization can be marked as High, Medium, or
Low based on urgency and impact.

''')