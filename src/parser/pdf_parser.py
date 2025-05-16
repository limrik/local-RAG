import os
import fitz
import logging

logger = logging.getLogger(__name__)

def parse_pdf(pdf_path):
    logger.info(f"Parsing PDF: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        content_by_page = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            page_content = []
            headers = []
            
            for block in blocks:
                if block["type"] == 0:
                    block_text = ""
                    font_sizes = []
                    is_bold = False
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                            font_sizes.append(span["size"])
                            if "bold" in span.get("font", "").lower():
                                is_bold = True
                    
                    block_text = block_text.strip()
                    if not block_text:
                        continue
                    
                    avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                    is_header = (avg_size > 12 or is_bold) and len(block_text) < 100
                    
                    if is_header:
                        headers.append(block_text)
                        page_content.append(f"## {block_text}")
                    else:
                        page_content.append(block_text)
            
            content_by_page.append({
                "content": "\n\n".join(page_content),
                "page_num": page_num + 1,
                "headers": headers
            })
        
        all_text = ""
        for page in content_by_page:
            all_text += f"\n\n--- Page {page['page_num']} ---\n\n"
            all_text += page["content"]
        
        logger.info(f"Successfully parsed PDF: {pdf_path} ({len(doc)} pages)")
        return all_text
    
    except Exception as e:
        logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
        raise