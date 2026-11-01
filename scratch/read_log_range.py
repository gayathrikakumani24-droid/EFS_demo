with open('hardware_graphrag/data/logs/app.log', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

for i in range(25368, 25385):
    clean_line = lines[i].rstrip().encode('ascii', errors='replace').decode('ascii')
    print(f"{i}: {clean_line}")
