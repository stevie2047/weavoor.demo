import streamlit as st
import yt_dlp
import openai
import os
import chromadb
from chromadb.utils import embedding_functions
import networkx as nx
from pyvis.network import Network
import tempfile
import shutil

st.set_page_config(page_title="Weavoor", page_icon="🧵👀")
st.title("Weavoor 🧵👀")
st.caption("Summaries decay. Connections compound.")

url = st.text_input("Paste any YouTube/podcast URL:")
if st.button("Weave", type="primary"):
    if not url:
        st.error("Please paste a URL!")
    else:
        with st.spinner("Downloading audio..."):
            temp_dir = tempfile.mkdtemp()
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            audio_path = os.path.join(temp_dir, "audio.wav")

        with st.spinner("Transcribing with Whisper (cloud)..."):
            with open(audio_path, "rb") as f:
                transcript = openai.Audio.transcribe("whisper-1", f)
            text = transcript.text

        with st.spinner("Summarizing..."):
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # cheap & fast — change to claude if you prefer
                messages=[{"role": "user", "content": f"Summarize in 5 short bullets:\n\n{text[:15000]}"}]
            )
            summary = response.choices[0].message.content
            st.markdown("**Summary**")
            st.write(summary)

        # Graph logic
        embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=st.secrets["OPENAI_API_KEY"])
        client = chromadb.PersistentClient(path="db")
        collection = client.get_or_create_collection("weaves", embedding_function=embedding_fn)

        video_id = url.split("v=")[-1][:11] if "v=" in url else "weave"
        collection.add(documents=[summary], metadatas=[{"url": url}], ids=[video_id])

        results = collection.query(query_texts=[summary], n_results=10)
        G = nx.Graph()
        G.add_node("NEW", title=summary.replace("\n", " "), color="#00ff00", size=30)
        connections = 0
        for i, dist in enumerate(results['distances'][0]):
            if dist < 0.25:
                old_id = results['ids'][0][i]
                old_text = results['documents'][0][i][:100] + "..."
                G.add_node(old_id, title=old_text)
                G.add_edge("NEW", old_id)
                connections += 1

        net = Network(height="600px", width="100%", bgcolor="#1a1a1a", font_color="white")
        net.from_nx(G)
        html_path = "graph.html"
        net.save_graph(html_path)

        st.success(f"Weave complete! Found {connections} connections")
        st.components.v1.html(open(html_path, "r", encoding="utf-8").read(), height=700, scrolling=True)

        markdown = f"# {url}\n\n{summary}\n\n![[graph.html]]"
        st.download_button("Download Obsidian note", data=markdown, file_name=f"{video_id}.md")

        # Cleanup temp
        shutil.rmtree(temp_dir)
