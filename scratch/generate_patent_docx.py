"""
Compiles patent/FORM_2_COMPLETE_SPECIFICATION.md into a formal Word Document:
patent/FORM_2_COMPLETE_SPECIFICATION.docx with formal IPO styling.
"""
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def generate_patent_docx():
    md_path = "patent/FORM_2_COMPLETE_SPECIFICATION.md"
    out_docx = "patent/FORM_2_COMPLETE_SPECIFICATION.docx"
    
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    doc = docx.Document()
    
    # 1 inch margins all around as required by Patent Office
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)
        
    # Styles
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(0, 0, 0)
    style_normal.paragraph_format.line_spacing = 1.3
    style_normal.paragraph_format.space_after = Pt(4)
    
    lines = text.split("\n")
    i = 0
    in_table = False
    table_lines = []
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check table
        if line.startswith("|") and line.endswith("|"):
            table_lines.append(line)
            i += 1
            continue
        elif table_lines:
            # Process accumulated table
            rows_data = []
            for t_line in table_lines:
                if re.match(r'^\|[\s\-:]+(\|[\s\-:]+)+\|$', t_line):
                    continue
                cells = [c.strip() for c in t_line.split("|")[1:-1]]
                rows_data.append(cells)
                
            if rows_data:
                num_cols = len(rows_data[0])
                t = doc.add_table(rows=len(rows_data), cols=num_cols)
                t.alignment = WD_TABLE_ALIGNMENT.CENTER
                for r_idx, r_data in enumerate(rows_data):
                    for c_idx, cell_value in enumerate(r_data[:num_cols]):
                        c = t.rows[r_idx].cells[c_idx]
                        c.text = cell_value
                        set_cell_margins(c, 100, 100, 120, 120)
                        if r_idx == 0:
                            set_cell_background(c, "EAEAEA")
                            for p in c.paragraphs:
                                for run in p.runs:
                                    run.font.bold = True
                                    run.font.size = Pt(9.5)
                        else:
                            for p in c.paragraphs:
                                for run in p.runs:
                                    run.font.size = Pt(9.5)
                doc.add_paragraph().paragraph_format.space_after = Pt(4)
            table_lines = []
            
        if not line:
            i += 1
            continue
            
        if line.startswith("# FORM 2"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run("FORM 2")
            run.font.bold = True
            run.font.size = Pt(16)
        elif line.startswith("## THE PATENTS ACT, 1970"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run("THE PATENTS ACT, 1970 (39 of 1970)\n&\nTHE PATENTS RULES, 2003")
            run.font.bold = True
            run.font.size = Pt(13)
        elif line.startswith("## COMPLETE SPECIFICATION"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run("COMPLETE SPECIFICATION\n(See Section 10; Rule 13)")
            run.font.bold = True
            run.font.size = Pt(14)
            p.paragraph_format.space_after = Pt(12)
        elif line.startswith("### "):
            # Heading 1 / Section title
            title_text = line.replace("### ", "").strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(title_text)
            run.font.bold = True
            run.font.size = Pt(12)
        elif line.startswith("#### "):
            # Heading 2 / Sub-section title
            sub_text = line.replace("#### ", "").strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(sub_text)
            run.font.bold = True
            run.font.size = Pt(11)
        elif line == "---":
            # Horizontal rule -> small space
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
        elif line.startswith("WE CLAIM:"):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run("WE CLAIM:")
            run.font.bold = True
            run.font.size = Pt(12)
        else:
            # Normal paragraph or claim item
            p = doc.add_paragraph()
            # Clean markdown bold/italics
            parts = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', line)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    r = p.add_run(part[2:-2])
                    r.font.bold = True
                elif part.startswith("*") and part.endswith("*"):
                    r = p.add_run(part[1:-1])
                    r.font.italic = True
                elif part.startswith("`") and part.endswith("`"):
                    r = p.add_run(part[1:-1])
                    r.font.name = 'Courier New'
                else:
                    p.add_run(part)
        i += 1
        
    doc.save(out_docx)
    print(f"Successfully generated formal IPO Form 2 specification at {out_docx}")

if __name__ == "__main__":
    generate_patent_docx()
