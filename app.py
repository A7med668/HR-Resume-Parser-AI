import streamlit as st
import os
import tempfile
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import re
import base64
import time

# Set page configuration
st.set_page_config(
    page_title="HR Resume Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .json-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model with caching to avoid reloading on every interaction"""
    st.info("🔄 Loading AI model... This may take a few minutes.")
    model_name = "mistralai/Mistral-Nemo-Instruct-2407"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    st.success("✅ Model loaded successfully!")
    return model, tokenizer

def generate_text(prompt, max_new_tokens=800, temperature=0.7):
    """Text generation function with better token management"""
    model, tokenizer = st.session_state.model
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    # Calculate available tokens for generation
    input_length = inputs['input_ids'].shape[1]
    max_total_length = 4096  # Model's context window
    
    # Adjust max_new_tokens if needed
    if input_length + max_new_tokens > max_total_length:
        max_new_tokens = max_total_length - input_length - 10  # Leave some buffer
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode only the new tokens (skip the input/prompt)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def setup_parser():
    """Set up the output parser"""
    full_name_schema = ResponseSchema(
        name="full_name",
        description="The candidate's full name"
    )
    email_schema = ResponseSchema(
        name="email",
        description="The candidate's email address"
    )
    education_schema = ResponseSchema(
        name="education",
        description="List of education entries with degree, institution, and year"
    )
    skills_schema = ResponseSchema(
        name="skills",
        description="List of skills as strings"
    )
    experience_schema = ResponseSchema(
        name="experience", 
        description="List of work experience entries with role, company, and years"
    )

    response_schemas = [
        full_name_schema,
        email_schema,
        education_schema,
        skills_schema,
        experience_schema
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    return output_parser, format_instructions

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def clean_resume_text(text):
    """Clean and preprocess the extracted resume text"""
    # Remove extra whitespace and newlines
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'\x0c', '', text)  # Form feed characters
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces
    
    return text.strip()

def truncate_text(text, max_tokens=1500):
    """Truncate text to fit within token limits"""
    tokenizer = st.session_state.model[1]
    tokens = tokenizer.encode(text)
    
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    return text

def extract_json_from_response(text):
    """Extract JSON from model response"""
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    pattern = r'\{.*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    return text

def safe_get(data, key, default="N/A"):
    """Safely get value from dictionary with error handling"""
    if isinstance(data, dict):
        return data.get(key, default)
    return default

def safe_display_education(education):
    """Safely display education information with type checking"""
    if not education:
        st.write("No education information found")
        return
    
    if isinstance(education, str):
        st.write(f"**Education:** {education}")
        return
    
    if isinstance(education, list):
        for i, edu in enumerate(education, 1):
            st.write(f"**Education {i}:**")
            if isinstance(edu, dict):
                st.write(f"  - Degree: {safe_get(edu, 'degree')}")
                st.write(f"  - Institution: {safe_get(edu, 'institution')}")
                st.write(f"  - Year: {safe_get(edu, 'year')}")
            else:
                st.write(f"  - {str(edu)}")
    else:
        st.write(f"**Education:** {str(education)}")

def safe_display_experience(experience):
    """Safely display experience information with type checking"""
    if not experience:
        st.write("No experience information found")
        return
    
    if isinstance(experience, str):
        st.write(f"**Experience:** {experience}")
        return
    
    if isinstance(experience, list):
        for i, exp in enumerate(experience, 1):
            st.write(f"**Experience {i}:**")
            if isinstance(exp, dict):
                st.write(f"  - Role: {safe_get(exp, 'role')}")
                st.write(f"  - Company: {safe_get(exp, 'company')}")
                st.write(f"  - Years: {safe_get(exp, 'years')}")
            else:
                st.write(f"  - {str(exp)}")
    else:
        st.write(f"**Experience:** {str(experience)}")

def safe_display_skills(skills):
    """Safely display skills information with type checking"""
    if not skills:
        st.write("No skills information found")
        return
    
    if isinstance(skills, str):
        st.write(f"**Skills:** {skills}")
        return
    
    if isinstance(skills, list):
        st.write("**Skills:**")
        for skill in skills:
            st.write(f"  - {str(skill)}")
    else:
        st.write(f"**Skills:** {str(skills)}")

def parse_resume_from_pdf(pdf_file):
    """Parse PDF resume into structured JSON"""
    try:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name

        # Extract text from PDF
        resume_text = extract_text_from_pdf(tmp_path)
        if not resume_text:
            st.error("❌ Could not extract text from PDF")
            return None

        # Clean the text
        cleaned_text = clean_resume_text(resume_text)
        
        # Show text preview
        with st.expander("📝 Extracted Text Preview"):
            st.text_area("Raw extracted text", cleaned_text[:1000] + "..." if len(cleaned_text) > 1000 else cleaned_text, height=200)
        
        # Truncate text if too long
        if len(cleaned_text) > 3000:  # Rough character count estimate
            st.warning("⚠️ Resume text is long. Truncating to fit model limits...")
            cleaned_text = truncate_text(cleaned_text, max_tokens=1200)
        
        # Create prompt
        output_parser, format_instructions = st.session_state.parser
        
        resume_parsing_template = """
You are an HR assistant that extracts candidate information from resume text.

Extract the following information from the resume text:
- Full name
- Email address
- Education history (list of degrees with institution and year)
- Skills (list of strings)
- Work experience (list of roles with company and years)

Format the education as: [{{"degree": "degree name", "institution": "institution name", "year": "graduation year"}}]
Format the experience as: [{{"role": "job title", "company": "company name", "years": "employment years"}}]

Respond ONLY in JSON format as follows:

{format_instructions}

Now extract from the following resume text:
"{resume_text}"
"""

        prompt = PromptTemplate(
            template=resume_parsing_template,
            input_variables=["resume_text", "format_instructions"]
        ).format(resume_text=cleaned_text, format_instructions=format_instructions)

        # Show token count info
        tokenizer = st.session_state.model[1]
        prompt_tokens = len(tokenizer.encode(prompt))
        st.info(f"📊 Prompt length: {prompt_tokens} tokens (max: 4096)")

        # Generate response with progress
        with st.spinner("🤖 AI is parsing the resume... This may take 30-60 seconds."):
            progress_bar = st.progress(0)
            
            for i in range(100):
                time.sleep(0.3)  # Simulate progress
                progress_bar.progress(i + 1)
            
            response = generate_text(prompt, max_new_tokens=800)
        
        # Extract JSON from response
        json_text = extract_json_from_response(response)
        
        # Parse the JSON
        try:
            output_data = output_parser.parse(f"```json\n{json_text}\n```")
        except Exception as e:
            st.error(f"❌ Error parsing JSON response: {e}")
            st.code(response, language='text')
            return None
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return output_data
        
    except Exception as e:
        st.error(f"❌ Error parsing resume: {e}")
        # Clean up temporary file in case of error
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        return None

def get_download_link(data, filename):
    """Generate a download link for JSON data"""
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">Download JSON</a>'
    return href

# Main application
def main():
    st.markdown('<h1 class="main-header">🤖 HR Candidate Profile Parser</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = load_model()
    if 'parser' not in st.session_state:
        st.session_state.parser = setup_parser()
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This AI-powered tool extracts structured information from resume PDFs including:\n\n"
        "• Personal details (name, email)\n"
        "• Education history\n" 
        "• Skills\n"
        "• Work experience\n\n"
        "Upload a resume PDF to get started!"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("How to use:")
    st.sidebar.write("1. Upload a resume PDF file")
    st.sidebar.write("2. Wait for AI processing (30-60 seconds)")
    st.sidebar.write("3. View and download the parsed data")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tips:")
    st.sidebar.write("• Use clear, text-based PDFs for best results")
    st.sidebar.write("• Avoid scanned/image-only PDFs")
    st.sidebar.write("• Keep resumes under 3 pages for optimal parsing")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload a resume in PDF format (text-based PDFs work best)"
        )
        
        if uploaded_file is not None:
            st.info(f"📄 File uploaded: {uploaded_file.name}")
            st.write(f"File size: {uploaded_file.size / 1024:.2f} KB")
            
            if st.button("🚀 Parse Resume", type="primary", use_container_width=True):
                result = parse_resume_from_pdf(uploaded_file)
                
                if result:
                    st.session_state.result = result
                    st.session_state.filename = uploaded_file.name.replace('.pdf', '_parsed.json')
                    st.success("✅ Resume parsed successfully!")
                else:
                    st.error("❌ Failed to parse resume. Please try with a different file.")
    
    with col2:
        st.subheader("📊 Parsed Results")
        
        if 'result' in st.session_state:
            result = st.session_state.result
            
            # Display results in expandable sections
            with st.expander("👤 Personal Information", expanded=True):
                st.write(f"**Full Name:** {safe_get(result, 'full_name')}")
                st.write(f"**Email:** {safe_get(result, 'email')}")
            
            with st.expander("🎓 Education", expanded=True):
                education = safe_get(result, 'education', [])
                safe_display_education(education)
            
            with st.expander("💼 Skills", expanded=True):
                skills = safe_get(result, 'skills', [])
                safe_display_skills(skills)
            
            with st.expander("💼 Work Experience", expanded=True):
                experience = safe_get(result, 'experience', [])
                safe_display_experience(experience)
            
            # Raw JSON view
            with st.expander("📋 Raw JSON Output"):
                st.code(json.dumps(result, indent=2), language='json')
            
            # Download button
            st.markdown("---")
            st.markdown("### 📥 Download Results")
            st.markdown(get_download_link(result, st.session_state.filename), unsafe_allow_html=True)
        
        else:
            st.info("👆 Upload a PDF resume and click 'Parse Resume' to see results here")
            
            # Show sample output
            with st.expander("📋 Example Output Format"):
                sample_output = {
                    "full_name": "John Smith",
                    "email": "john.smith@email.com",
                    "education": [
                        {"degree": "B.Sc. Computer Science", "institution": "MIT", "year": "2020"}
                    ],
                    "skills": ["Python", "Machine Learning", "Data Analysis"],
                    "experience": [
                        {"role": "Software Engineer", "company": "Google", "years": "2020-2023"}
                    ]
                }
                st.json(sample_output)

if __name__ == "__main__":
    main()