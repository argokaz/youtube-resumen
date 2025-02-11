import streamlit as st
import re
import asyncio
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

# Configuración inicial
MAX_WORDS_PER_CHUNK = 1500
MODEL_NAME = "gpt-4-turbo"
SUMMARY_TOKEN_LIMIT = 1500

# Cliente OpenAI configurado con secrets
async_client = AsyncOpenAI(api_key=st.secrets["openai"]["api_key"])

# Estilos Apple mejorados
apple_css = """
<style>
    /* (Mantener los estilos CSS previos que te envié) */
</style>
"""
st.markdown(apple_css, unsafe_allow_html=True)

def extraer_video_id(url):
    patterns = [
        r'(?:v=|\/videos\/|embed\/|shorts\/|v\/|e\/|watch\?v=)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        return match.group(1) if match else None

@st.cache_data(ttl=3600, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Prioridad de idiomas y tipos de subtítulos
        try:
            transcript = transcript_list.find_manually_created_transcript(['es', 'en'])
        except (NoTranscriptFound, TranscriptsDisabled):
            transcript = transcript_list.find_generated_transcript(['es', 'en'])
            
        return transcript.fetch()
    
    except TranscriptsDisabled:
        st.error("""
            🔇 Subtítulos deshabilitados para este video
            **Soluciones:**
            1. Prueba con otro video que tenga subtítulos habilitados
            2. Contacta al creador del video para activar subtítulos
            3. Usa videos de canales educativos/profesionales (generalmente tienen subtítulos)
        """)
        st.stop()
        
    except NoTranscriptFound:
        available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        languages = [f"{t.language} ({'manual' if t.is_generated else 'auto'})" 
                    for t in available_transcripts]
        
        st.error(f"""
            🌍 Subtítulos solo disponibles en otros idiomas:
            {', '.join(languages)}
            
            **Puedes:**\n
            - Usar la versión automática en inglés (traducción aproximada)
            - Copiar el enlace y usar traductores externos
        """)
        st.stop()
        
    except Exception as e:
        st.error(f"🚨 Error inesperado: {str(e)}")
        st.stop()

def chunk_text(text, max_words=MAX_WORDS_PER_CHUNK):
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
    prompt = f"""Genera un resumen parcial conciso que capture:
- Ideas principales y puntos clave
- Datos numéricos relevantes
- Conclusiones importantes

Texto: {text[:8000]}"""

    try:
        response = await async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Eres un experto en análisis y síntesis de contenido técnico."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error en resumen parcial: {str(e)}"

async def process_chunks_concurrently(chunks):
    tasks = [summarize_text_async(chunk) for chunk in chunks]
    return await asyncio.gather(*tasks)

async def generate_final_summary(partial_summaries):
    combined = "\n\n".join(partial_summaries)
    
    prompt = f"""Crea un resumen profesional con:
1. Título principal en **negritas**
2. Introducción contextual
3. Secciones temáticas con subtítulos
4. Puntos clave en viñetas
5. Conclusiones destacadas

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
        yield f"🚨 Error en resumen final: {str(e)}"

# UI Streamlit
st.set_page_config(
    page_title="YouTube Summarizer Pro",
    page_icon="▶️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1 style="font-size: 2.5rem; margin-bottom: 8px;">📹 YouTube Summarizer Pro</h1>
        <p style="color: #86868b; font-size: 1.1rem;">Solo funciona con videos que tengan subtítulos habilitados</p>
    </div>
""", unsafe_allow_html=True)

with st.form("input_form"):
    url = st.text_input("", placeholder="Pega aquí la URL de YouTube...")
    submitted = st.form_submit_button("Generar Resumen", use_container_width=True)
if submitted and url:
    with st.spinner("🔍 Analizando video..."):
        video_id = extraer_video_id(url)
        if not video_id:
            st.error("❌ URL de YouTube inválida")
            st.stop()
            
        try:
            transcript = get_transcript(video_id)
            full_text = " ".join([t['text'] for t in transcript])
            
            chunks = list(chunk_text(full_text))
            n_chunks = len(chunks)
            
            st.info(f"📦 Procesando {n_chunks} bloques de contenido...")
            progress_bar = st.progress(0)
            
            # Función asíncrona para procesamiento
            async def process_chunks():
                partial_summaries = await process_chunks_concurrently(chunks)
                
                # Actualizar progreso
                for i in range(n_chunks):
                    progress_bar.progress((i + 1) / n_chunks)
                    await asyncio.sleep(0.01)
                
                return partial_summaries
            
            partial_summaries = asyncio.run(process_chunks())
            
            st.success("✅ Resúmenes parciales completados")
            
            with st.container():
                st.markdown("## Resumen Final")
                summary_box = st.empty()
                full_summary = []
                
                async def display_final_summary():
                    async for chunk in generate_final_summary(partial_summaries):
                        full_summary.append(chunk)
                        summary_box.markdown("".join(full_summary))
                
                asyncio.run(display_final_summary())

            with st.expander("📄 Ver Transcripción Completa", expanded=False):
                st.text_area("", full_text, height=300, label_visibility="collapsed")

            st.download_button(
                label="⬇️ Descargar Resumen",
                data="".join(full_summary),
                file_name="resumen_video.md",
                mime="text/markdown",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"""
                ❌ Error crítico: {str(e)}
                **Posibles soluciones:**
                1. Intenta con otro video
                2. Verifica la conexión a Internet
                3. Reporta el error al soporte técnico
            """)