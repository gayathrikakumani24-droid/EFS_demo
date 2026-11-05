import sys
sys.path.insert(0, "hardware_graphrag")
import re

doc_sample = """
3.1 Instruction Format
Each instruction is a 64-bit word with the following fields:
   Opcode    (6 bits): Operation to perform
   Destination Register    (4 bits): Target register (16 matrix registers)
   Source Register 1    (4 bits): First operand register
   Source Register 2    (4 bits): Second operand register
   Dimension Control    (10 bits): Matrix dimension information
Rows Source 1 (5 bits): Number of rows in first source matrix
Cols Source 1/Rows Source 2 (5 bits): Number of columns in first source/rows in second source
   Cols Source 2    (5 bits): Number of columns in second source matrix
   Format    (3 bits): Data format control (fixed-point, floating-point, etc.)
   Flags    (4 bits): Special operation modifiers
   Reserved    (24 bits): For future extensions
"""

field_pattern = re.compile(r'^\s*([A-Za-z0-9_\s/]+?)\s*\(\s*(\d+)\s*bits?\s*\)', re.MULTILINE)
matches = field_pattern.findall(doc_sample)
print("Matches found:", matches)
print("Count:", len(matches))
