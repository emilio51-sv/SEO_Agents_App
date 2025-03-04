import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from crewai import Crew, Agent, Task, Process
from langchain.chat_models import ChatOpenAI
from datetime import datetime
from textwrap import dedent
from langchain_community.utilities import GoogleSerperAPIWrapper

# -----------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------------
st.set_page_config(
    page_title="SEO Keyword Research & Content Suggestions",
    layout="wide"
)

# Recupera le API key dai secrets di Streamlit
openai_api_key = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key

SERPER_API_KEY = st.secrets["serper_api_key"]
os.environ["SERPER_API_KEY"] = SERPER_API_KEY

# Inizializza il wrapper per Google Serper
search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

def perform_search(query: str) -> str:
    """Esegue una ricerca con Google Serper e restituisce il risultato."""
    try:
        return search.run(query)
    except Exception as e:
        return f"Errore nella ricerca: {str(e)}"
    
    
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
# -----------------------------------------------------------------------------------
# Nuova funzione per l'audit del sito
# -----------------------------------------------------------------------------------
def perform_site_audit(url: str) -> dict:
    """Esegue un semplice audit del sito usando requests e BeautifulSoup."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return {"error": str(e)}
    
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.title.string.strip() if soup.title and soup.title.string else "Titolo non trovato"
    
    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_tag["content"].strip() if meta_tag and meta_tag.get("content") else "Meta description non trovata"
    
    headings = [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])]
    
    return {"title": title, "meta_description": meta_description, "headings": headings}

def format_site_audit(audit: dict) -> str:
    """Formatta i dati dell'audit del sito in una stringa leggibile."""
    if "error" in audit:
        return f"Errore durante l'analisi del sito: {audit['error']}"
    formatted = (
        f"Titolo: {audit.get('title')}\n"
        f"Meta Description: {audit.get('meta_description')}\n"
        f"Headings: {', '.join(audit.get('headings', []))}"
    )
    return formatted

# -----------------------------------------------------------------------------------
# 2. CUSTOM STYLES & MAIN TITLE
# -----------------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center;">
    <h1 style="color:#4A90E2; font-size: 3em; margin-bottom: 0.2em;">🔎 SEO Keyword Research & Content Suggestions</h1>
    <p style="font-size:1.2em; color:#444;">
       Scopri le keyword più rilevanti e ottimizza la tua strategia di contenuti per migliorare il posizionamento nei motori di ricerca.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------------
# 3. USER INPUTS
# -----------------------------------------------------------------------------------
with st.container():
    st.markdown("<h2 style='text-align:center;'>Focus Keyword o Argomento</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:1.2em'>Inserisci la keyword o il topic su cui vuoi fare analisi SEO.</p>", unsafe_allow_html=True)
    
    col_left, col_center, col_right = st.columns([1,2,1])
    with col_center:
        focus_keyword = st.text_input(
            "Keyword/Argomento per l'analisi SEO:",
            placeholder="Es. 'packaging sostenibile' o 'software CRM'..."
        )
        
with st.container():
    st.markdown("<h2 style='text-align:center;'>Inserisci l'URL dell'Azienda</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:1.2em'>Per eseguire un audit del sito aziendale, inserisci l'URL del sito.</p>", unsafe_allow_html=True)
    
    col_left2, col_center2, col_right2 = st.columns([1,2,1])
    with col_center2:
        site_url = st.text_input(
            "URL del sito:",
            placeholder="Es. 'https://www.grifal.it/'"
        )

current_date = datetime.now().strftime("%Y-%m-%d")

# -----------------------------------------------------------------------------------
# 4. AGENTS DEFINITION
# -----------------------------------------------------------------------------------

# Agente per l'analisi delle keyword e dei risultati SERP
seo_keyword_analyst = Agent(
    role="SEO Keyword Analyst",
    goal=f"Ricercare su Google le informazioni chiave riguardanti la keyword '{focus_keyword}' e analizzare snippet e risultati correlati.",
    backstory=dedent("""
        Il SEO Keyword Analyst è un esperto di Search Engine Optimization, specializzato nell'identificare 
        le keyword rilevanti e nell'analizzare i risultati di ricerca per capire come i contenuti sono posizionati.
        Utilizza Google Serper per estrarre snippet e pattern di query.
    """),
    personality="Analitico, orientato ai dati, preciso",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

# Agente per sviluppare strategie e suggerimenti di contenuto
seo_strategist = Agent(
    role="SEO Strategist",
    goal=f"Proporre suggerimenti di contenuto e ottimizzazioni SEO basati sui risultati di ricerca relativi a '{focus_keyword}'.",
    backstory=dedent("""
        Il SEO Strategist è un consulente specializzato in content marketing e ottimizzazione per i motori di ricerca. 
        Analizza le keyword principali, l'intento di ricerca, la struttura dei contenuti nei risultati di Google 
        e propone strategie di ottimizzazione on-page e di contenuto per migliorare il ranking.
    """),
    personality="Creativo, pragmatico, orientato ai risultati",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

# Agente per sintetizzare i risultati finali
summary_agent = Agent(
    role="Summary Agent",
    goal="Raccogliere i dati chiave dall'analisi SEO e dalle strategie proposte, producendo un riepilogo strutturato.",
    backstory=dedent("""
        Il Summary Agent esamina le uscite degli altri agenti, estrae le informazioni più rilevanti 
        e produce un riepilogo con punti chiave, raccomandazioni di contenuto e prossimi step di ottimizzazione.
    """),
    personality="Conciso, strutturato e diretto",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

# -----------------------------------------------------------------------------------
# 5. TASKS DEFINITION
# -----------------------------------------------------------------------------------

# Esegui una ricerca tramite Google Serper per ottenere info sulla keyword
search_query = f"{focus_keyword} - Principali risultati di ricerca e snippet"
search_results = perform_search(search_query)

# Se è stato inserito un URL, esegui anche l'audit del sito
if site_url:
    audit_data = perform_site_audit(site_url)
    audit_string = format_site_audit(audit_data)
else:
    audit_string = "Nessun URL fornito per l'audit del sito."

# Task 1: Analisi dettagliata della keyword e audit del sito
task_keyword_analysis = Task(
    description=f"""
    Il SEO Keyword Analyst deve raccogliere informazioni chiave sui risultati di ricerca per la keyword:
    '{focus_keyword}'

    Risultati di Google Serper (snippet):
    {search_results}

    Inoltre, deve analizzare il sito fornito:
    {audit_string}

    Richiedi di includere nel report:
    - Principali query correlate (keyword simili o "people also ask")
    - Analisi di snippet e meta tag (se disponibili)
    - Struttura dei contenuti (titoli, heading, formati)
    - Eventuali opportunità o gap di contenuto non coperto
    """,
    expected_output=f"""
    Report di Analisi SEO per '{focus_keyword}'

    Il report dovrebbe includere un'analisi strutturata con:
    - Elenco di keyword correlate
    - Principali osservazioni su snippet e meta tag
    - Potenziali argomenti ancora poco coperti
    - Informazioni raccolte dall'audit del sito
    """,
    agent=seo_keyword_analyst
)

# Task 2: Proposte di contenuti e strategie SEO
task_seo_strategy = Task(
    description=f"""
    Il SEO Strategist, basandosi sul report del SEO Keyword Analyst, deve proporre 3-4 strategie concrete per migliorare il posizionamento 
    della keyword '{focus_keyword}' e ottimizzare i contenuti.

    Dettagli richiesti:
    - Tipologie di contenuti da creare (blog post, landing page, FAQ)
    - Ottimizzazioni on-page (titoli, meta description, heading)
    - Potenziali link building o collaborazioni
    - Suggerimenti di struttura e formattazione
    """,
    expected_output=f"""
    Strategia SEO per '{focus_keyword}'

    Il report dovrebbe includere:
    - Lista di contenuti raccomandati con formati e argomenti
    - Suggerimenti di ottimizzazione (on-page e off-page)
    - Priorità di implementazione e metriche di successo
    """,
    agent=seo_strategist
)

# Task 3: Sintesi finale dei risultati
task_summary = Task(
    description=f"""
    Il Summary Agent deve raccogliere le uscite dei task di Analisi SEO e Strategie,
    e produrre un riepilogo finale che evidenzi:
    - Le principali keyword e query correlate
    - Le strategie proposte e i relativi potenziali impatti
    - Un elenco di raccomandazioni chiave e prossimi step
    """,
    expected_output=f"""
    Riepilogo Finale per '{focus_keyword}'

    Il riepilogo deve includere bullet points con i principali insight 
    e un breve paragrafo conclusivo su come procedere con l'ottimizzazione.
    """,
    agent=summary_agent
)

tasks = [task_keyword_analysis, task_seo_strategy, task_summary]

# -----------------------------------------------------------------------------------
# 6. CREW SETUP
# -----------------------------------------------------------------------------------
all_agents = [seo_keyword_analyst, seo_strategist, summary_agent]
crew = Crew(
    agents=all_agents,
    tasks=tasks,
    process=Process.sequential,
    full_output=True,
    share_crew=False,
)

def get_task_output(task):
    try:
        return task.output.raw
    except AttributeError:
        return "No data available."

# -----------------------------------------------------------------------------------
# 7. EXECUTION BUTTON
# -----------------------------------------------------------------------------------
with st.container():
    st.markdown("""
    <div style="text-align:center;">
        <h2 style="margin-bottom: 0.5em;">Analisi SEO & Suggerimenti</h2>
    </div>
    """, unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1,2,1])
    with btn_col:
        run_simulation = st.button("Esegui Analisi SEO", key="run_sim")

# -----------------------------------------------------------------------------------
# 8. RUN THE SIMULATION
# -----------------------------------------------------------------------------------
if run_simulation:
    if not focus_keyword:
        st.warning("Inserisci una keyword o un argomento per avviare l'analisi.")
    else:
        progress_bar = st.progress(0)

        # Step 1: Analisi Keyword (incluso l'audit del sito se fornito)
        progress_bar.progress(10)
        crew.tasks = [task_keyword_analysis]
        crew.kickoff()
        keyword_analysis_report = get_task_output(task_keyword_analysis)

        # Step 2: Strategia SEO
        progress_bar.progress(50)
        task_seo_strategy.description += f"\n\nReport di Analisi SEO:\n{keyword_analysis_report}"
        crew.tasks = [task_seo_strategy]
        crew.kickoff()
        seo_strategy_report = get_task_output(task_seo_strategy)

        # Step 3: Riepilogo Finale
        progress_bar.progress(80)
        final_text = "Report di Analisi SEO:\n" + keyword_analysis_report + "\n\n"
        final_text += "Strategia SEO:\n" + seo_strategy_report + "\n\n"
        task_summary.description += f"\n\nReport Completo:\n{final_text}"
        crew.tasks = [task_summary]
        crew.kickoff()
        progress_bar.progress(100)

        # -----------------------------------------------------------------------------------
        # DISPLAY DEI RISULTATI
        # -----------------------------------------------------------------------------------
        st.markdown("## Report Finale")
        st.markdown("---")

        final_summary_output = get_task_output(task_summary)
        with st.expander("Clicca per visualizzare il riepilogo finale"):
            st.markdown(final_summary_output)

        # TABS PER OGNI AGENTE (SEO Keyword Analyst e SEO Strategist)
        st.markdown("## Report Dettagliati per Agente")
        relevant_tasks = [task_keyword_analysis, task_seo_strategy]
        tab_labels = [t.agent.role for t in relevant_tasks]
        tabs = st.tabs(tab_labels)

        for i, task_obj in enumerate(relevant_tasks):
            with tabs[i]:
                st.markdown(f"### {task_obj.agent.role} Report")
                with st.expander("Clicca per visualizzare i dettagli"):
                    st.markdown(get_task_output(task_obj))

        st.markdown("---")
        st.markdown("## Fine Analisi")
        st.markdown("Grazie per aver utilizzato il tool di SEO Keyword Research e Content Suggestions!")
