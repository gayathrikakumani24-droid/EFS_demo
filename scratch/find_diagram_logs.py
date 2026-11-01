with open("data/logs/app.log", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

recent = lines[-200:]
for l in recent:
    if any(k in l.lower() for k in ["plantuml", "diagram", "puml", "svg"]):
        print(l.strip().encode("ascii", "replace").decode("ascii"))
