import pandas as pd
from pathlib import Path

def df_to_latex_table(df, caption=None, label=None, float_format=".2f"):
    """
    Convert a DataFrame to a LaTeX table string.
    """
    latex = df.to_latex(index=True, float_format=lambda x: f"{x:{float_format}}")
    insert_str = ''
    if caption:
        insert_str += r'\caption{' + caption + '}' + '\n'
    if label:
        insert_str += r'\label{' + label + '}' + '\n'
    # Insert after \begin{tabular}
    latex = latex.replace('\begin{tabular}', insert_str + r'\begin{tabular}')
    return latex

def save_latex_file(content, output_path):
    """
    Save LaTeX content to a file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

def latex_section(title, body):
    """
    Create a LaTeX section with a title and body.
    """
    return f"\\section{{{title}}}\n{body}\n"

def latex_subsection(title, body):
    return f"\\subsection{{{title}}}\n{body}\n"

def latex_list(items):
    """
    Create a LaTeX itemize list from a list of strings.
    """
    return "\\begin{itemize}\n" + "\n".join([f"  \\item {item}" for item in items]) + "\n\\end{itemize}\n"
