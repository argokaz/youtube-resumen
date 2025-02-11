import streamlit as st
import re
import asyncio
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

# Configuración inicial DEBE SER PRIMERO
st.set_page_config(
    page_title="YouTube Summarizer",
    page_icon="▶️",
    layout="centered"
)

# Configuración de la aplicación
MAX_WORDS_PER_CHUNK = 1500
MODEL_NAME = "gpt-4-turbo"
SUMMARY_TOKEN_LIMIT = 1500

# Cliente OpenAI
async_client = AsyncOpenAI(api_key=st.secrets["openai"]["api_key"])

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            transcript = transcript_list.find_manually_created_transcript(['es', 'en'])
        except (NoTranscriptFound, TranscriptsDisabled):
            transcript = transcript_list.find_generated_transcript(['es', 'en'])
            
        return transcript.fetch()
    
    except TranscriptsDisabled:
        st.error("Subtítulos deshabilitados para este video. Prueba con otro video.")
        st.stop()
        
    except NoTranscriptFound:
        try:
            available = YouTubeTranscriptApi.list_transcripts(video_id)
            langs = [t.language_code for t in available]
            st.error(f"Subtítulos disponibles solo en: {', '.join(langs)}")
        except:
            st.error("El video no tiene subtítulos disponibles")
        st.stop()
        
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
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
    prompt = f"Resume este texto destacando puntos clave y datos importantes:\n{text[:8000]}"
    
    try:
        response = await async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Eres un experto en resúmenes técnicos concisos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error en resumen parcial: {str(e)}"

async def process_chunks(chunks):
    tasks = [summarize_text_async(chunk) for chunk in chunks]
    return await asyncio.gather(*tasks)

async def generate_final_summary(partial_summaries):
    combined = "\n\n".join(partial_summaries)
    
    try:
        response = await async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Genera un resumen estructurado con puntos clave y conclusiones."},
                {"role": "user", "content": f"Texto a resumir:\n{combined[:12000]}"}
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
        yield f"Error en resumen final: {str(e)}"

# Interfaz de usuario simplificada
st.title("Resumidor de Videos de YouTube")

url = st.text_input("Ingresa la URL del video de YouTube:")
process_button = st.button("Generar Resumen")

if process_button and url:
    with st.spinner("Procesando..."):
        try:
            video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url).group(1)
            transcript = get_transcript(video_id)
            full_text = " ".join([t['text'] for t in transcript])
            
            chunks = list(chunk_text(full_text))
            progress = st.progress(0)
            
            # Procesamiento paralelo
            partial_summaries = await process_chunks(chunks)
            
            # Actualizar progreso
            for i in range(len(chunks)):
                progress.progress((i + 1) / len(chunks))
                await asyncio.sleep(0.01)
            
            # Generar resumen final
            final_summary = st.empty()
            full_content = []
            
            async for chunk in generate_final_summary(partial_summaries):
                full_content.append(chunk)
                final_summary.markdown("".join(full_content))
            
            # Mostrar transcripción
            with st.expander("Ver transcripción completa"):
                st.write(full_text)
                
            # Descarga
            st.download_button(
                "Descargar Resumen",
                "".join(full_content),
                file_name="resumen.md"
            )
            
        except Exception as e:
            st.error(f"Error crítico: {str(e)}")