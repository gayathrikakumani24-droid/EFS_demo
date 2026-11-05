file_path = "hardware_graphrag/app.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

old_pages = '''PAGES = {
    "Dashboard": "ui.dashboard",
    "Developer Debug View": "ui.debug_view",
    "Protocol Library": "ui.protocol_library",
    "Protocol Graph Explorer": "ui.graph_view",
    "Existing GraphRAG QA": "ui.qa_page",
    "Hardware Design": "ui.hardware_design",
    "Design Plan": "ui.design_plan",
    "EFS IR Review": "ui.efs_ir_review",
    "Generated RTL": "ui.generated_rtl",
    "PlantUML Diagrams": "ui.plantuml_diagrams",
    # Existing features must remain
    "Document Upload": "ui.upload_page",
    "Chunk Explorer": "ui.chunk_explorer",
    "Entity Explorer": "ui.entity_explorer",
    "Vector Search": "ui.vector_search_page",
    "Logs": "ui.logs_page",
}'''

new_pages = '''PAGES = {
    "Dashboard": "ui.dashboard",
    "Developer Debug View": "ui.debug_view",
    "Protocol Library": "ui.protocol_library",
    "Protocol Graph Explorer": "ui.graph_view",
    "Existing GraphRAG QA": "ui.qa_page",
    "Hardware Design": "ui.hardware_design",
    "Design Plan": "ui.design_plan",
    "EFS IR Review": "ui.efs_ir_review",
    "Generated RTL": "ui.generated_rtl",
    "PlantUML Diagrams": "ui.plantuml_diagrams",
    "Testbenches & Assertions": "ui.testbench_assertions",
    "Verification & Compliance": "ui.verification_compliance",
    "Repair History": "ui.repair_history",
    # Existing features must remain
    "Document Upload": "ui.upload_page",
    "Chunk Explorer": "ui.chunk_explorer",
    "Entity Explorer": "ui.entity_explorer",
    "Vector Search": "ui.vector_search_page",
    "Logs": "ui.logs_page",
}'''

old_icons = '''PAGE_ICONS = {
    "Dashboard": "📊",
    "Developer Debug View": "🔍",
    "Protocol Library": "📚",
    "Protocol Graph Explorer": "🕸️",
    "Existing GraphRAG QA": "💬",
    "Hardware Design": "💻",
    "Design Plan": "📋",
    "EFS IR Review": "🖥️",
    "Generated RTL": "⚙️",
    "PlantUML Diagrams": "📊",
    "Document Upload": "📤",
    "Chunk Explorer": "🧩",
    "Entity Explorer": "🏷️",
    "Vector Search": "🔍",
    "Logs": "📜",
}'''

new_icons = '''PAGE_ICONS = {
    "Dashboard": "📊",
    "Developer Debug View": "🔍",
    "Protocol Library": "📚",
    "Protocol Graph Explorer": "🕸️",
    "Existing GraphRAG QA": "💬",
    "Hardware Design": "💻",
    "Design Plan": "📋",
    "EFS IR Review": "🖥️",
    "Generated RTL": "⚙️",
    "PlantUML Diagrams": "📊",
    "Testbenches & Assertions": "🧪",
    "Verification & Compliance": "🛡️",
    "Repair History": "📜",
    "Document Upload": "📤",
    "Chunk Explorer": "🧩",
    "Entity Explorer": "🏷️",
    "Vector Search": "🔍",
    "Logs": "📜",
}'''

content_norm = content.replace("\r\n", "\n")
content_norm = content_norm.replace(old_pages.replace("\r\n", "\n"), new_pages.replace("\r\n", "\n"))
content_norm = content_norm.replace(old_icons.replace("\r\n", "\n"), new_icons.replace("\r\n", "\n"))

final_content = content_norm.replace("\n", "\r\n") if "\r\n" in content else content_norm

with open(file_path, "w", encoding="utf-8") as f:
    f.write(final_content)

print("Successfully added Testbenches & Assertions to app.py")

