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

# Create the API instance
ytt_api = YouTubeTranscriptApi()

url = st.text_input("Paste any YouTube/podcast URL:")
if st.button("Weave", type="primary"):
    if not url:
        st.error("Please paste a URL!")
    else:
        try:
            # Extract video ID
            if "v=" in url:
                video_id = url.split("v=")[-1].split("&")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[-1].split("?")[0]
            else:
                video_id = url.split("/")[-1]

            with st.spinner("Fetching transcript..."):
                # New way: instance.list(video_id)
                transcript_list = ytt_api.list(video_id)
                try:
                    transcript = transcript_list.find_transcript(['en'])
                except NoTranscriptFound:
                    transcript = transcript_list.find_generated_transcript(['en'])
                transcript_data = transcript.fetch()
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

        except NoTranscriptFound:
            st.error("No English transcript found (manual or generated).")
            st.info("Try a popular podcast video — most have auto-captions!")
        except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.info("Make sure the video has captions enabled.")

