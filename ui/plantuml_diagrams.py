"""PlantUML Diagrams Page: display generated architecture, FSM, and sequence diagrams."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("📊 PlantUML Design Diagrams")
    st.caption("View block architectures, state transitions, and bus transfer timelines generated from the Design Context.")

    if "design_results" not in st.session_state:
        st.info("💡 No active design results found in this session. Head to the **Hardware Design** page to start a new generation.")
        return

    results = st.session_state["design_results"]
    
    if "diagrams" not in results or not results["diagrams"]:
        st.warning("⚠️ No diagrams were generated. Go to the **Hardware Design** page and ensure 'PlantUML' is checked during generation.")
        return

    diagrams = results["diagrams"]
    diagram_urls = results.get("diagram_urls", {})

    tabs = st.tabs(["🏗️ Architecture Component Diagram", "🔄 Protocol Sequence Diagram"])

    diagram_keys = [("architecture", "architecture.puml"), ("sequence", "sequence.puml")]

    for idx, (key, filename) in enumerate(diagram_keys):
        with tabs[idx]:
            st.subheader(f"Diagram: {key.capitalize()}")
            
            puml_code = diagrams.get(key, "")
            img_url = diagram_urls.get(key, "")
            
            # Action buttons
            col_dl, col_blank = st.columns([1, 4])
            with col_dl:
                st.download_button(
                    label=f"💾 Download {filename}",
                    data=puml_code,
                    file_name=filename,
                    mime="text/plain",
                    key=f"download_{key}"
                )
                
            # Try to show rendered image from PlantUML web service
            if img_url:
                st.markdown(f"**Web rendering:**")
                st.image(img_url, use_column_width=False, caption=f"Generated {key} layout")
            
            # Expandable source code
            with st.expander("📝 View PlantUML Source Code", expanded=False):
                st.code(puml_code, language="protobuf")
