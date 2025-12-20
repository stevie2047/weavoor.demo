import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import chromadb
from chromadb.utils import embedding_functions
import networkx as nx
from pyvis.network import Network
import openai

st.set_page_config(page_title="Weavoor", page_icon="🧵👀")
st.title("Weavoor 🧵👀")
st.caption("Summaries decay. Connections compound.")

openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer manual English, fallback to auto-generated
        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(['en'])
        return transcript.fetch()
    except Exception as e:
        raise ValueError(f"No transcript available: {str(e)}")

url = st.text_input("Paste any YouTube/podcast URL:")
if st.button("Weave", type="primary"):
    if not url:
        st.error("Please paste a URL!")
    else:
        try:
            video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]
            
            with st.spinner("Fetching transcript..."):
                transcript_data = get_transcript(video_id)
                text = " ".join([entry['text'] for entry in transcript_data])

            with st.spinner("Summarizing..."):
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": f"Summarize in 5 short bullets:\n\n{text[:15000]}"}]
                )
                summary = response.choices[0].message.content
                st.markdown("**Summary**")
                st.write(summary)

            # Graph logic
            embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=st.secrets["OPENAI_API_KEY"])
            client = chromadb.PersistentClient(path="db")
            collection = client.get_or_create_collection("weaves", embedding_function=embedding_fn)

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

        except ValueError as e:
            st.error(str(e))
            st.info("Tip: Try a video with captions enabled (most podcasts have them!). For no-captions videos, full audio transcription coming in v2.")

