import sys
from nbformat import v3, v4

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

with open(input_file_path) as fpin:
    text = fpin.read()

text += """
# <markdowncell>

# If you can read this, reads_py() is no longer broken! 
"""

nbook = v3.reads_py(text)
nbook = v4.upgrade(nbook)  # Upgrade v3 to v4

jsonform = v4.writes(nbook) + "\n"
with open(output_file_path, "w") as fpout:
    fpout.write(jsonform)
