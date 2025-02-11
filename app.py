import streamlit as st
import re
import time  # Nueva importación
from youtube_transcript_api import YouTubeTranscriptApi
from openai import AsyncOpenAI
import asyncio

# Configuración inicial
MAX_WORDS_PER_CHUNK = 1500
MODEL_NAME = "gpt-4-turbo"
SUMMARY_TOKEN_LIMIT = 1500

# Cliente OpenAI configurado con secrets
async_client = AsyncOpenAI(api_key=st.secrets["openai"]["api_key"])

def extraer_video_id(url):
    """Mejorado para manejar más formatos de URL"""
    patterns = [
        r'(?:v=|\/videos\/|embed\/|shorts\/|v\/|e\/|watch\?v=)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_transcript(video_id):
    """Obtiene la transcripción con manejo de errores mejorado"""
    try:
        available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = available_transcripts.find_transcript(['es', 'en'])
        return transcript.fetch()
    except Exception as e:
        st.error(f"Error al obtener subtítulos: {str(e)}")
        st.stop()

def chunk_text(text, max_words=MAX_WORDS_PER_CHUNK):
    """Optimizado usando textwrap para mejor manejo de párrafos"""
    paragraphs = text.split('\n')
    current_chunk = []
    current_word_count = 0
    
    for para in paragraphs:
        words = para.split()
        if current_word_count + len(words) > max_words:
            yield ' '.join(current_chunk)
            current_chunk = []
            current_word_count = 0
        current_chunk.extend(words)
        current_word_count += len(words)
    
    if current_chunk:
        yield ' '.join(current_chunk)

async def summarize_text_async(text, max_tokens=200):
    """Optimizado con mejor prompt engineering y manejo de errores"""
    prompt = f"""Genera un resumen parcial conciso que capture:
- Ideas principales y puntos clave
- Datos numéricos relevantes
- Conclusiones importantes
- Términos técnicos significativos

Texto: {text[:8000]}"""

    try:
        response = await async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Eres un experto en análisis y síntesis de contenido técnico."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error en API: {str(e)}")
        return ""

async def process_chunks_concurrently(chunks):
    """Procesamiento paralelo de chunks para mayor velocidad"""
    tasks = [summarize_text_async(chunk) for chunk in chunks]
    return await asyncio.gather(*tasks)

async def generate_final_summary(partial_summaries):
    """Generación de resumen final con formato mejorado"""
    combined = "\n\n".join(partial_summaries)
    
    prompt = f"""Crea un resumen profesional con:
1. Título principal en **negritas**
2. Introducción contextual
3. Secciones temáticas con subtítulos
4. Puntos clave en viñetas
5. Conclusiones destacadas
6. Palabras clave técnicas

Incluye elementos importantes en **negritas** y usa formato Markdown.

Contenido: {combined[:12000]}"""

    try:
        response = await async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Eres un editor profesional de contenido técnico."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=SUMMARY_TOKEN_LIMIT,
            stream=True
        )
        
        full_response = []
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                full_response.append(content)
                yield content
                
    except Exception as e:
        st.error(f"Error en generación final: {str(e)}")
        yield "❌ Error al generar el resumen final"

# UI Streamlit
st.set_page_config(page_title="GOGA YouTube Summarizer Pro", layout="wide")
st.title("📹 GOGA YouTube Summarizer Pro")

with st.form("input_form"):
    url = st.text_input("Ingrese URL de YouTube:", placeholder="https://youtube.com/watch?v=...")
    submitted = st.form_submit_button("Generar Resumen")

if submitted and url:
    with st.spinner("🔍 Analizando video..."):
        video_id = extraer_video_id(url)
        if not video_id:
            st.error("URL inválida")
            st.stop()
            
        transcript = get_transcript(video_id)
        full_text = " ".join([t['text'] for t in transcript])
        
        chunks = list(chunk_text(full_text))
        n_chunks = len(chunks)
        
        st.info(f"🔢 Procesando {n_chunks} bloques de contenido...")
        progress_bar = st.progress(0)
        
        # Procesamiento asíncrono de chunks
        partial_summaries = asyncio.run(process_chunks_concurrently(chunks))
        
        # Actualizar barra de progreso (versión corregida)
        for i in range(n_chunks):
            progress_bar.progress((i + 1) / n_chunks)
            time.sleep(0.01)  # Delay sincrónico
        
        st.success("✅ Resúmenes parciales completados")
        st.subheader("Resumen Final del Video")
        
        summary_box = st.empty()
        full_summary = []
        
        # Generar resumen final con streaming
        async def display_final_summary():
            async for chunk in generate_final_summary(partial_summaries):
                full_summary.append(chunk)
                summary_box.markdown("".join(full_summary))
        
        asyncio.run(display_final_summary())

    with st.expander("📝 Transcripción Completa"):
        st.text_area("Subtítulos", full_text, height=300, label_visibility="collapsed")

    st.download_button(
        label="⬇️ Descargar Resumen",
        data="".join(full_summary),
        file_name="resumen_video.md",
        mime="text/markdown"
    )
