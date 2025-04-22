import streamlit as st
import pandas as pd
import os
import datetime
from fpdf import FPDF
import textwrap
from uuid import uuid4
from langchain.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import pyttsx3
import tempfile

# Create temp directory if it doesn't exist
TEMP_DIR = "C:/temp/podcast_app"
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize Hugging Face (replace with your API token)
HUGGINGFACE_API_TOKEN = "API"

mock_jira_issues = [
    {"key": "PROJ-101", "summary": "Create login API", "type": "Task", "status": "In Progress", "assignee": "Alice"},
    {"key": "PROJ-102", "summary": "Fix session timeout bug", "type": "Bug", "status": "To Do", "assignee": "Bob"},
    {"key": "PROJ-103", "summary": "Design DB schema", "type": "Story", "status": "Done", "assignee": "Carol"},
]

mock_confluence_pages = [
    {"title": "Release Notes - v1.2", "author": "Dave", "last_updated": "2025-04-15", "content": "Summary of recent changes..."},
    {"title": "Onboarding Guide", "author": "Eve", "last_updated": "2025-04-10", "content": "Steps for new team members..."},
]


# Custom CSS for enhanced UI
def load_css():
    st.markdown("""
    <style>
        :root {
            --primary: #1a5276;
            --secondary: #2980b9;
            --accent: #e74c3c;
            --light: #ecf0f1;
            --dark: #2c3e50;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
        }
        
        .main {
            background-color: #f5f7fa;
        }
        
        .sidebar .sidebar-content {
            background-color: var(--dark);
            color: white;
        }
        
        .stButton>button {
            background-color: var(--primary);
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            border: none;
            font-weight: 500;
        }
        
        .stButton>button:hover {
            background-color: var(--secondary);
            color: white;
        }
        
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        .card-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--primary);
            border-bottom: 2px solid var(--light);
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border-left: 5px solid var(--warning);
        }
        
        .alert-danger {
            background-color: #f8d7da;
            border-left: 5px solid var(--danger);
        }
        
        .alert-success {
            background-color: #d4edda;
            border-left: 5px solid var(--success);
        }
        
        .document-item {
            padding: 15px;
            border-radius: 8px;
            background-color: white;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        
        .document-item:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        .tag {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 5px;
        }
        
        .tag-primary {
            background-color: var(--primary);
            color: white;
        }
        
        .tag-success {
            background-color: var(--success);
            color: white;
        }
        
        .tag-warning {
            background-color: var(--warning);
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'faqs' not in st.session_state:
        st.session_state.faqs = []
    if 'handovers' not in st.session_state:
        st.session_state.handovers = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'knowledge_base_initialized' not in st.session_state:
        st.session_state.knowledge_base_initialized = False

# Initialize knowledge base
def init_knowledge_base():
    if not st.session_state.knowledge_base_initialized and st.session_state.documents:
        try:
            # Initialize embeddings with explicit device and reduced model if needed
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                model_kwargs={'device': 'cpu'},  # Force CPU if GPU issues occur
                #encode_kwargs={'normalize_embeddings': False}
            )
            
            texts = [doc['content'] for doc in st.session_state.documents]
            metadatas = [{'source': doc['title'], 'type': doc['type']} for doc in st.session_state.documents]
            
            # Create FAISS index in smaller batches if needed
            st.session_state.vector_store = FAISS.from_texts(
                texts, 
                embeddings, 
                metadatas=metadatas
            )
            st.session_state.knowledge_base_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize knowledge base: {str(e)}")
            # Fallback to simple text storage if embedding fails
            st.session_state.knowledge_base_initialized = False

# Document repository functions
def save_document(file, title, description, tags, doc_type):

      # 👇 Add this debug line here
    print("Current Working Directory:", os.getcwd())

    file_id = str(uuid4())
    file_path = os.path.join("documents", f"{file_id}_{file.name}")

    os.makedirs("documents", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    document = {
        "id": file_id,
        "title": title,
        "description": description,
        "tags": tags,
        "type": doc_type,
        "file_path": file_path,
        "upload_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uploaded_by": "Current User"  # Replace with actual user
    }
    
    # Extract content based on file type
    if file.name.endswith('.txt'):
        with open(file_path, "r", encoding='utf-8') as f:
            document['content'] = f.read()
    elif file.name.endswith('.pdf'):
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            document['content'] = "\n".join([page.page_content for page in pages])
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            document['content'] = description
    else:
        document['content'] = description
    
    st.session_state.documents.append(document)
    st.session_state.knowledge_base_initialized = False
    return document

def get_documents_by_type(doc_type=None):
    if doc_type:
        return [doc for doc in st.session_state.documents if doc['type'] == doc_type]
    return st.session_state.documents

# Handover template functions
def create_handover_template(employee_name, last_working_day, projects):
    template = {
        "id": str(uuid4()),
        "employee_name": employee_name,
        "last_working_day": last_working_day,
        "created_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Draft",
        "projects": projects,
        "sections": {
            "current_projects": "",
            "key_contacts": "",
            "ongoing_issues": "",
            "critical_dates": "",
            "knowledge_transfer": ""
        }
    }
    st.session_state.handovers.append(template)
    return template

def generate_handover_pdf(handover):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Knowledge Handover Document", ln=1, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Employee: {handover['employee_name']}", ln=1)
    pdf.cell(200, 10, txt=f"Last Working Day: {handover['last_working_day']}", ln=1)
    pdf.cell(200, 10, txt=f"Generated On: {handover['created_date']}", ln=1)
    pdf.ln(10)
    
    # Sections
    sections = [
        ("1. Current Projects", handover['sections']['current_projects']),
        ("2. Key Contacts", handover['sections']['key_contacts']),
        ("3. Ongoing Issues", handover['sections']['ongoing_issues']),
        ("4. Critical Dates", handover['sections']['critical_dates']),
        ("5. Knowledge Transfer", handover['sections']['knowledge_transfer'])
    ]
    
    for title, content in sections:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=title, ln=1)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, txt=content)
    
    output_path = os.path.join("handovers", f"{handover['id']}.pdf")
    os.makedirs("handovers", exist_ok=True)
    pdf.output(output_path)
    return output_path

# FAQ functions
def add_faq(question, answer, tags):
    faq = {
        "id": str(uuid4()),
        "question": question,
        "answer": answer,
        "tags": tags,
        "created_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "created_by": "Current User",
        "upvotes": 0,
        "views": 0
    }
    st.session_state.faqs.append(faq)
    return faq

# AI functions
def generate_ai_recommendations(query):
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            temperature=0.3,
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
        )
        
        prompt = f"""
        [INST] As a knowledge management consultant, provide 3-5 actionable recommendations for:
        {query}
        
        Use bullet points and professional language. Focus on knowledge continuity and transfer. [/INST]
        """
        
        return llm.invoke(prompt)
    except Exception as e:
        return f"Could not generate recommendations: {str(e)}"

def search_knowledge_base(query):
    if st.session_state.vector_store:
        return st.session_state.vector_store.similarity_search(query, k=3)
    return []

# Text-to-speech functions
def init_tts_engine():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        return engine, voices
    except Exception as e:
        st.error(f"Failed to initialize TTS engine: {str(e)}")
        return None, None

engine, voices = init_tts_engine()

def generate_podcast(text):
    """Generate podcast audio using pyttsx3"""
    if engine is None:
        st.error("TTS engine not initialized")
        return None

    try:
        segments = textwrap.wrap(text, width=500)
        podcast_segments = []
        
        for i, segment in enumerate(segments[:6]):  # Limit to 6 segments
            try:
                # Configure voice and speed
                if voices and len(voices) > 1:
                    engine.setProperty('voice', voices[i % 2].id)
                engine.setProperty('rate', 180 if i % 2 == 0 else 160)
                
                # Save to temporary WAV file
                temp_wav = os.path.join(tempfile.gettempdir(), f"segment_{uuid4()}.wav")
                engine.save_to_file(segment, temp_wav)
                engine.runAndWait()
                
                # Read the generated WAV file
                with open(temp_wav, "rb") as f:
                    audio_data = f.read()
                os.remove(temp_wav)
                
                podcast_segments.append(audio_data)
                
            except Exception as seg_error:
                st.warning(f"Skipped segment {i}: {str(seg_error)}")
                continue
                
        return b"".join(podcast_segments) if podcast_segments else None
        
    except Exception as e:
        st.error(f"Podcast generation failed: {str(e)}")
        return None

def podcast_module():
    st.subheader("🎙️ PDF to Podcast")
    st.markdown("""
    <div style="background-color: #e7f5fe; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <b>Offline-capable podcast generator</b> - Uses system text-to-speech
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file and engine:
        with st.spinner("Creating podcast..."):
            try:
                # Extract text
                pdf = PdfReader(uploaded_file)
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                
                if not text:
                    st.error("No text found in PDF")
                    return
                
                # Generate audio (first 5000 chars)
                audio_data = generate_podcast(text[:5000])
                
                if audio_data:
                    # Display and download
                    st.audio(audio_data, format="audio/wav")
                    st.download_button(
                        "Download Podcast",
                        audio_data,
                        file_name="podcast.wav",
                        mime="audio/wav"
                    )
                    
                    # Transcript preview
                    with st.expander("Transcript Preview"):
                        st.write(text[:1000] + "...")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Main app
def main():
    st.set_page_config(
        page_title="Knowledge Continuity Portal",
        layout="wide",
        page_icon="🧠"
    )
    
    load_css()
    init_session_state()
    
    st.title("🧠 Knowledge Continuity Portal")
    st.markdown("""
    <div style="color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px;">
    Preserve critical knowledge, streamline handovers, and empower your team with collective intelligence
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    app_mode = st.sidebar.selectbox(
        "Choose a module",
        ["Dashboard", "Knowledge Repository", "Handover Manager", "FAQ System", "AI Recommendations", "Podcast Generator", "Project Workspace"]
    )

    # Dashboard
    if app_mode == "Dashboard":
        st.subheader("📊 Knowledge Continuity Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="card-header">Knowledge Repository</div>
                <h2>{len(st.session_state.documents)}</h2>
                <p>Documents stored</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <div class="card-header">Active Handovers</div>
                <h2>{len([h for h in st.session_state.handovers if h['status'] == 'Draft'])}</h2>
                <p>In progress</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="card">
                <div class="card-header">FAQ Knowledge</div>
                <h2>{len(st.session_state.faqs)}</h2>
                <p>Questions answered</p>
            </div>
            """, unsafe_allow_html=True)

        # 👉 Knowledge Score Section
        st.subheader("📈 Knowledge Score Insights")
        tabs = st.tabs(["👥 Team Score", "🧑‍💻 Individual Score", "📂 Project Score"])

        def calculate_knowledge_score(tasks, articles, interactions):
            return round(min((tasks * 0.4 + articles * 0.3 + interactions * 0.3), 100), 2)

        def get_mock_team_data():
            return [
                {"name": "Alice", "tasks": 45, "articles": 5, "interactions": 30},
                {"name": "Bob", "tasks": 32, "articles": 7, "interactions": 20},
                {"name": "Charlie", "tasks": 50, "articles": 2, "interactions": 15},
                {"name": "Dana", "tasks": 40, "articles": 4, "interactions": 25},
            ]

        def get_mock_project_data():
            return [
                {"project": "Payment Gateway", "tasks": 180, "articles": 15, "interactions": 85},
                {"project": "Loan Processing", "tasks": 145, "articles": 20, "interactions": 90},
                {"project": "Fraud Detection", "tasks": 100, "articles": 12, "interactions": 60},
            ]

        with tabs[0]:  # Team Score
            team_data = get_mock_team_data()
            for member in team_data:
                score = calculate_knowledge_score(member["tasks"], member["articles"], member["interactions"])
                with st.expander(f"👤 {member['name']} — Score: {score}/100"):
                    st.write(f"- Tasks: {member['tasks']}")
                    st.write(f"- Knowledge Articles: {member['articles']}")
                    st.write(f"- Jira Interactions: {member['interactions']}")

        with tabs[1]:  # Individual Score
            dev_name = st.selectbox("Select Developer", ["Alice", "Bob", "Charlie", "Dana"])
            selected = next(d for d in get_mock_team_data() if d["name"] == dev_name)
            score = calculate_knowledge_score(selected["tasks"], selected["articles"], selected["interactions"])
            st.metric(label="Knowledge Score", value=f"{score}/100", delta="AI Estimated")
            with st.expander("📊 Breakdown"):
                st.write(f"**Tasks Completed:** {selected['tasks']}")
                st.write(f"**Knowledge Articles Written:** {selected['articles']}")
                st.write(f"**Jira Interactions:** {selected['interactions']}")

        with tabs[2]:  # Project Score
            project_data = get_mock_project_data()
            for project in project_data:
                score = calculate_knowledge_score(project["tasks"], project["articles"], project["interactions"])
                with st.expander(f"📁 {project['project']} — Score: {score}/100"):
                    st.write(f"- Tasks: {project['tasks']}")
                    st.write(f"- Knowledge Articles: {project['articles']}")
                    st.write(f"- Jira Interactions: {project['interactions']}")

        # Upcoming handovers alert
        upcoming_handovers = [
            h for h in st.session_state.handovers 
            if datetime.datetime.strptime(h['last_working_day'], "%Y-%m-%d") - datetime.datetime.now() < datetime.timedelta(days=14)
        ]
        
        if upcoming_handovers:
            st.markdown("""
            <div class="alert alert-warning">
                <h4>⚠️ Upcoming Knowledge Transfers</h4>
                <ul>
            """ + "\n".join([
                f"<li>{h['employee_name']} (Last day: {h['last_working_day']}) - {h['status']}</li>" 
                for h in upcoming_handovers
            ]) + """
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent documents
        st.subheader("Recently Added Documents")
        recent_docs = sorted(st.session_state.documents, key=lambda x: x['upload_date'], reverse=True)[:5]
        
        if recent_docs:
            for doc in recent_docs:
                tags_html = " ".join([f'<span class="tag tag-primary">{tag}</span>' for tag in doc["tags"]])
                st.markdown(f"""
                <div class="document-item">
                    <h4>{doc['title']}</h4>
                    <p>{doc['description']}</p>
                    <div>{tags_html}</div>
                    <small>Uploaded on {doc['upload_date']} by {doc['uploaded_by']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No documents added yet. Add some in the Knowledge Repository.")
        
        # AI recommendations
        st.subheader("AI-Powered Knowledge Gaps")
        with st.expander("Get recommendations for improving knowledge continuity"):
            query = st.text_input("What knowledge continuity challenges are you facing?")
            if query and st.button("Get Recommendations"):
                with st.spinner("Analyzing with AI..."):
                    recommendations = generate_ai_recommendations(query)
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-header">Recommendations</div>
                        {recommendations.replace("\n", "<br>")}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Knowledge Repository
    elif app_mode == "Knowledge Repository":
        st.subheader("📚 Knowledge Repository")
        st.info(f"Documents are saved in: `{os.path.abspath('documents')}`")
        tab1, tab2 = st.tabs(["Browse Documents", "Upload New"])
        
        with tab1:
            st.markdown("""
            <div class="alert alert-success">
                <b>Centralized knowledge storage</b> for all project documentation, code snippets, and best practices
            </div>
            """, unsafe_allow_html=True)
            
            search_query = st.text_input("Search knowledge base")
            if search_query:
                st.markdown("### Search Results")
                docs = search_knowledge_base(search_query)
                if docs:
                    for doc in docs:
                        st.markdown(f"""
                        <div class="document-item">
                            <h4>{doc.metadata['source']}</h4>
                            <p>{doc.page_content[:200]}...</p>
                            <small>Type: {doc.metadata['type']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No matching documents found")
            
            doc_type_filter = st.selectbox(
                "Filter by document type",
                ["All", "Project Documentation", "Code Snippet", "Best Practice", "Meeting Notes", "Other"]
            )
            
            filtered_docs = get_documents_by_type() if doc_type_filter == "All" else get_documents_by_type(doc_type_filter)
            
            if filtered_docs:
                for doc in filtered_docs:
                    with st.expander(f"{doc['title']} - {doc['type']}"):
                        tags_html = " ".join([f'<span class="tag tag-primary">{tag}</span>' for tag in doc["tags"]])
                        st.markdown(f"""
                        <p><strong>Description:</strong> {doc['description']}</p>
                        <p><strong>Tags:</strong> {tags_html}</p>
                        <p><strong>Uploaded:</strong> {doc['upload_date']} by {doc['uploaded_by']}</p>
                        """, unsafe_allow_html=True)
                        
                        with open(doc['file_path'], "rb") as f:
                            st.download_button(
                                label="Download Document",
                                data=f,
                                file_name=os.path.basename(doc['file_path']),
                                mime="application/octet-stream"
                            )
            else:
                st.info("No documents found matching your criteria")
        
        with tab2:
            with st.form("upload_form"):
                st.markdown("### Upload New Document")
                file = st.file_uploader("Choose a file", type=["pdf", "txt", "md", "docx"])
                title = st.text_input("Document Title")
                description = st.text_area("Description")
                tags = st.multiselect(
                    "Tags",
                    ["Technical", "Process", "Client", "Internal", "Reference", "How-to"],
                    default=["Reference"]
                )
                doc_type = st.selectbox(
                    "Document Type",
                    ["Project Documentation", "Code Snippet", "Best Practice", "Meeting Notes", "Other"]
                )
                
                submitted = st.form_submit_button("Upload Document")
                if submitted and file and title:
                    document = save_document(file, title, description, tags, doc_type)
                    st.success(f"Document '{title}' uploaded successfully!")
                    st.balloons()
    
    # Handover Manager
    elif app_mode == "Handover Manager":
        st.subheader("🔄 Handover Manager")
        
        tab1, tab2, tab3 = st.tabs(["Active Handovers", "Create New", "Templates"])
        
        with tab1:
            st.markdown("""
            <div class="alert alert-warning">
                <b>Critical knowledge transfer</b> for employees transitioning out of roles
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.handovers:
                for handover in st.session_state.handovers:
                    status_color = "tag-success" if handover['status'] == "Completed" else "tag-warning"
                    
                    with st.expander(f"{handover['employee_name']} - {handover['last_working_day']}"):
                        st.markdown(f"""
                        <p><strong>Status:</strong> <span class="tag {status_color}">{handover['status']}</span></p>
                        <p><strong>Projects:</strong> {", ".join(handover['projects'])}</p>
                        """, unsafe_allow_html=True)
                        
                        # 🔁 Handover edit form
                        with st.form(f"edit_handover_{handover['id']}"):
                            st.markdown("### Current Projects")
                            handover['sections']['current_projects'] = st.text_area(
                                "Details",
                                value=handover['sections']['current_projects'],
                                height=150,
                                key=f"current_projects_{handover['id']}"
                            )

                            st.markdown("### Key Contacts")
                            handover['sections']['key_contacts'] = st.text_area(
                                "Details",
                                value=handover['sections']['key_contacts'],
                                height=150,
                                key=f"key_contacts_{handover['id']}"
                            )

                            st.markdown("### Ongoing Issues")
                            handover['sections']['ongoing_issues'] = st.text_area(
                                "Details",
                                value=handover['sections']['ongoing_issues'],
                                height=150,
                                key=f"ongoing_issues_{handover['id']}"
                            )

                            submitted = st.form_submit_button("Update Handover")
                            if submitted:
                                st.success("Handover updated successfully!")

                        # ✅ Generate and download PDF *outside* the form
                        if st.button(f"Generate PDF for {handover['employee_name']}", key=f"generate_pdf_{handover['id']}"):
                            pdf_path = generate_handover_pdf(handover)
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="Download Handover Document",
                                    data=f,
                                    file_name=f"handover_{handover['employee_name']}.pdf",
                                    mime="application/pdf",
                                    key=f"download_pdf_{handover['id']}"
                                )
            else:
                st.info("No active handovers. Create one to get started.")

        
        with tab2:
            with st.form("create_handover"):
                st.markdown("### Create New Handover Template")
                employee_name = st.text_input("Employee Name")
                last_working_day = st.date_input("Last Working Day")
                projects = st.multiselect(
                    "Projects Involved",
                    ["Project A", "Project B", "Project C", "Project D"],
                    default=["Project A"]
                )
                
                submitted = st.form_submit_button("Create Handover Template")
                if submitted and employee_name and last_working_day:
                    handover = create_handover_template(
                        employee_name,
                        last_working_day.strftime("%Y-%m-%d"),
                        projects
                    )
                    st.success(f"Handover template created for {employee_name}!")
        
        with tab3:
            st.markdown("""
            <div class="card">
                <div class="card-header">Standard Handover Template</div>
                <p>Use this structure for consistent knowledge transfer:</p>
                <ol>
                    <li><strong>Current Projects</strong> - Status, next steps, deadlines</li>
                    <li><strong>Key Contacts</strong> - Stakeholders, SMEs, relationships</li>
                    <li><strong>Ongoing Issues</strong> - Known problems, workarounds</li>
                    <li><strong>Critical Dates</strong> - Milestones, renewals, deadlines</li>
                    <li><strong>Knowledge Transfer</strong> - Specialized knowledge, tips</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # FAQ System
    elif app_mode == "FAQ System":
        st.subheader("❓ Knowledge Sharing & FAQ System")
        
        tab1, tab2 = st.tabs(["Browse FAQs", "Add New FAQ"])
        
        with tab1:
            st.markdown("""
            <div class="alert alert-success">
                <b>Collective intelligence</b> - Team-contributed answers to common questions
            </div>
            """, unsafe_allow_html=True)
            
            search_query = st.text_input("Search FAQs")
            filtered_faqs = [
                faq for faq in st.session_state.faqs 
                if not search_query or 
                   search_query.lower() in faq['question'].lower() or 
                   search_query.lower() in faq['answer'].lower()
            ]
            
            if filtered_faqs:
                for faq in filtered_faqs:
                    with st.expander(faq['question']):
                        tags_html = " ".join([f'<span class="tag tag-primary">{tag}</span>' for tag in faq["tags"]])
                        st.markdown(f"""
                        <p>{faq['answer']}</p>
                        <div>{tags_html}</div>
                        <small>Added by {faq['created_by']} on {faq['created_date']}</small>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"👍 Upvote ({faq['upvotes']})", key=f"upvote_{faq['id']}"):
                                faq['upvotes'] += 1
                                st.experimental_rerun()
            else:
                st.info("No FAQs found matching your criteria")
        
        with tab2:
            with st.form("add_faq"):
                st.markdown("### Add New FAQ")
                question = st.text_input("Question")
                answer = st.text_area("Answer")
                tags = st.multiselect(
                    "Tags",
                    ["Technical", "Process", "Client", "General", "How-to"],
                    default=["General"]
                )
                
                submitted = st.form_submit_button("Add FAQ")
                if submitted and question and answer:
                    faq = add_faq(question, answer, tags)
                    st.success("FAQ added successfully!")
                    st.balloons()
    
    # AI Recommendations
    elif app_mode == "AI Recommendations":
        st.subheader("🤖 AI-Powered Knowledge Recommendations")
        
        st.markdown("""
        <div class="alert alert-success">
            <b>Smart suggestions</b> for improving knowledge continuity based on your specific context
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("ai_recommendations"):
            context = st.text_area("Describe your situation or challenge")
            submit = st.form_submit_button("Get Recommendations")
            
            if submit and context:
                with st.spinner("Analyzing with AI..."):
                    recommendations = generate_ai_recommendations(context)
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-header">Recommendations</div>
                        {recommendations.replace("\n", "<br>")}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("### Common Scenarios")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("New Team Member Onboarding"):
                recommendations = generate_ai_recommendations(
                    "What are best practices for onboarding new team members to ensure knowledge transfer?"
                )
                st.markdown(f"""
                <div class="card">
                    <div class="card-header">Onboarding Recommendations</div>
                    {recommendations.replace("\n", "<br>")}
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("Critical Employee Leaving"):
                recommendations = generate_ai_recommendations(
                    "How to handle knowledge transfer when a critical employee is leaving the organization?"
                )
                st.markdown(f"""
                <div class="card">
                    <div class="card-header">Knowledge Transfer Recommendations</div>
                    {recommendations.replace("\n", "<br>")}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if st.button("Project Documentation Gaps"):
                recommendations = generate_ai_recommendations(
                    "Our project documentation is incomplete. What strategies can we use to improve documentation quality?"
                )
                st.markdown(f"""
                <div class="card">
                    <div class="card-header">Documentation Recommendations</div>
                    {recommendations.replace("\n", "<br>")}
                </div>  
                """, unsafe_allow_html=True)
            
            if st.button("Improving Team Knowledge Sharing"):
                recommendations = generate_ai_recommendations(
                    "Our team doesn't share knowledge effectively. What processes can we implement to improve?"
                )
                st.markdown(f"""
                <div class="card">
                    <div class="card-header">Knowledge Sharing Recommendations</div>
                    {recommendations.replace("\n", "<br>")}
                </div>
                """, unsafe_allow_html=True)
    
    #Podcast
    elif app_mode == "Podcast Generator":
        podcast_module()

    elif app_mode == "Project Workspace":
        st.subheader("🗂️ Project Workspace")

        tab1, tab2 = st.tabs(["Jira Issues", "Confluence Pages"])

        with tab1:
            st.markdown("### 🐞 Jira Issue Tracker (Mock)")
            for issue in mock_jira_issues:
                with st.expander(f"{issue['key']} - {issue['summary']}"):
                    st.markdown(f"""
                    **Type:** {issue['type']}  
                    **Status:** {issue['status']}  
                    **Assignee:** {issue['assignee']}
                    """)

        with tab2:
            st.markdown("### 📘 Confluence Knowledge Base (Mock)")
            for page in mock_confluence_pages:
                with st.expander(f"{page['title']}"):
                    st.markdown(f"""
                    **Author:** {page['author']}  
                    **Last Updated:** {page['last_updated']}  
                    **Content Preview:**  
                    {page['content']}
                    """)
    
    # Initialize knowledge base after any changes
    init_knowledge_base()

if __name__ == "__main__":
    main()
