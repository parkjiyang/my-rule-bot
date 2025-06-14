import streamlit as st
import json
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# --- 1. 설정 및 UI 최적화 ---
st.set_page_config(layout="centered", page_title="업무규정 AI", page_icon="🔗")
st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        .main .block-container {
            padding-top: 2rem; padding-left: 1rem; padding-right: 1rem;
        }
        .stButton>button {
            height: 3.5em; font-size: 1.1em; font-weight: bold; border-radius: 10px;
        }
        .stChatMessage {
            border-radius: 10px; padding: 1em 1.2em; margin-bottom: 1em;
            border: 1px solid #e0e0e0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# (카테고리 정의 및 파일 로딩 함수들은 이전과 동일)
CATEGORIES = {
    "1": {"name": "기본법규", "files": ["한국교통안전공단 정관"]},
    "2": {"name": "조직", "files": ["국외사무소 운영세칙", "위임전결규정", "이사회규정", "자동차안전연구원 운영규정", "자동차안전연구원 위임전결세칙", "자동차안전연구원 직제시행세칙", "직제규정", "직제시행세칙"]},
    "3": {"name": "인사", "files": ["인사관리세칙", "인사규정", "임금피크제 운영규정", "임원 추천위원회 운영규정", "자동차안전연구원 인사세칙", "채용업무처리세칙", "취업규정", "보직관리에 관한지침", "안전사고에 관한 임원 문책 규정"]},
    "4": {"name": "보수,복리", "files": ["상사료등제수당지급세칙", "경영성과지급지침", "보수규정", "복리후생규정", "실무직근로자 및 기간제근로자 관리규정"]},
    "5": {"name": "재무,계약", "files": ["계약사무처리세칙", "예산회계규정", "자금운용업무에관한세칙", "투자 및 자금운용업무 심사에 관한 세칙"]},
    "6": {"name": "소송,규정", "files": ["소송업무처리규정", "임직원소송지원규정", "제규정관리규정"]},
    "7": {"name": "윤리,인권", "files": ["공정거래자율준수프로그램 운영규정", "기업성장응답센터 운영규정", "임원직무청렴계약운영규정"]},
    "8": {"name": "감사", "files": ["감사규정", "내부통제규정", "부정청탁 및 금품 등 수수의 신고 사무 처리 지침", "이해충돌방지제도 운영지침"]},
    "9": {"name": "안전,보건", "files": ["안전보건관리규정"]},
    "10": {"name": "보안", "files": ["보안업무규칙"]},
    "11": {"name": "일반행정", "files": ["기업민원 보호서비스 헌장 운영규정", "민원처리규정", "여비규정", "청원심의회 운영규정", "행정업무의 효율적 운영에 관한 규정"]},
    "12": {"name": "자격,교육", "files": ["민간자격관리운영세칙", "운전적성정밀검사 운영세칙", "초경량비행장치 조종자 증명 운영세칙", "국가자격시험관리세칙", "무인비행장치 전문교육기관지정 및 관리업무 운영세칙", "무인비행장치 조종자 증명 운영세칙"]},
    "13": {"name": "철도", "files": ["정밀진단 성능평가 결과보고서 평가 규정"]},
    "14": {"name": "연구", "files": ["연구관리규정", "용역업무수탁에관한세칙", "자동차안전연구원 업무세칙"]},
    "15": {"name": "기타사업", "files": ["사내벤처 운영규정", "초경량비행장치 신고업무 운영세칙", "튜닝부품 안전확인 업무 등에 관한 규정"]},
}
ALL_FILES = list(set([file for category in CATEGORIES.values() for file in category['files']]))
JSON_FILE_PATH = '통합_수정완료.json'

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('jhgan/ko-sbert-nli')

@st.cache_data
def load_and_embed_data(path, _model):
    # This function is the same as the last version
    try:
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
    except: return None
    processed_docs = []
    for item in data:
        file_name = item.get('fileName', '')
        regulation_name = os.path.splitext(file_name)[0]
        regulation_name = re.sub(r'\(.*\)', '', regulation_name).strip()
        full_page_text = []
        for page in item.get('pages', []):
            for content_block in page.get('content', []):
                if content_block.get('type') == 'paragraph': full_page_text.append(content_block.get('data', ''))
                elif content_block.get('type') == 'table':
                    table_data = content_block.get('data', []); table_text = ' '.join([' '.join(map(str, row)) for row in table_data]); full_page_text.append(f"[표 내용] {table_text}")
        final_text = "\n".join(full_page_text)
        if final_text: processed_docs.append({"규정명": regulation_name, "내용": final_text, "id": len(processed_docs)})
    if not processed_docs: return None
    contents = [doc['내용'] for doc in processed_docs]
    embeddings = _model.encode(contents, convert_to_tensor=True, show_progress_bar=True)
    return {"docs": processed_docs, "embeddings": embeddings.cpu().numpy()}

# --- 2. 검색 및 답변 생성 엔진 ---
def hybrid_search(query, data, model, selected_files, top_k=3):
    # This function is the same as the last version
    query_keywords = set(query.split())
    docs_to_search = [doc for doc in data['docs'] if any(doc['규정명'].startswith(f) for f in selected_files)]
    keyword_results = []
    for doc in docs_to_search:
        doc_words = set(doc['내용'].split()); matched_keywords = query_keywords.intersection(doc_words)
        if matched_keywords: keyword_results.append({'doc': doc, 'score': len(matched_keywords)})
    keyword_results = sorted(keyword_results, key=lambda x: x['score'], reverse=True)
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()
    indices_to_search = [i for i, doc in enumerate(data['docs']) if any(doc['규정명'].startswith(f) for f in selected_files)]
    if not indices_to_search: return []
    filtered_docs_for_semantic = [data['docs'][i] for i in indices_to_search]
    filtered_embeddings = data['embeddings'][indices_to_search]
    semantic_similarities = cosine_similarity(query_embedding.reshape(1, -1), filtered_embeddings)[0]
    fused_scores = {}; k = 60
    semantic_top_indices = np.argsort(semantic_similarities)[::-1]
    for rank, idx in enumerate(semantic_top_indices):
        doc_id = filtered_docs_for_semantic[idx]['id']
        if doc_id not in fused_scores: fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + rank + 1)
    for rank, res in enumerate(keyword_results):
        doc_id = res['doc']['id']
        if doc_id not in fused_scores: fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + rank + 1)
    if not fused_scores: return []
    sorted_doc_ids = sorted(fused_scores.keys(), key=lambda id: fused_scores[id], reverse=True)
    id_to_doc = {doc['id']: doc for doc in data['docs']}
    return [id_to_doc[doc_id] for doc_id in sorted_doc_ids[:top_k]]

def generate_ultimate_answer(query, context_docs, chat_history):
    try:
        # ✨ 배포를 위해 Streamlit의 Secrets에서 API 키를 가져오도록 수정
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            return "오류: 관리자가 앱에 API 키를 설정하지 않았습니다."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        context_str = "\n\n---\n\n".join([f"문서 제목: {doc['규정명']}\n내용: {doc['내용']}" for doc in context_docs])
        history_str = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in chat_history])

        # ✨ 극도로 간결한 답변을 위한 최종 프롬프트
        prompt = f"""당신은 핵심만 말하는 과묵한 전문가입니다.

### 지시사항:
1.  '이전 대화'와 '관련 규정 내용'을 바탕으로 '현재 질문'의 핵심적인 답(Fact)을 찾으세요.
2.  당신의 답변은 **반드시 단 한 문장**으로만 구성되어야 합니다.
3.  절대 부연 설명을 하거나, 먼저 제안하거나, 인사말을 하지 마세요.
4.  오직 사실에 기반한 한 문장의 답변만 제공하고 사용자의 다음 질문을 기다리세요.

### 이전 대화:
{history_str}

### 관련 규정 내용:
{context_str}

### 현재 질문:
{query}

### 전문가의 한 문장 답변:
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

# --- 3. UI 구성 및 메인 로직 ---
st.title("🔗 업무규정 AI")

sbert_model = load_sbert_model()
data = load_and_embed_data(JSON_FILE_PATH, sbert_model)
if data is None: st.error("규정집 데이터 파일을 처리할 수 없습니다."); st.stop()

if "category" not in st.session_state: st.session_state.category = None
if "messages" not in st.session_state: st.session_state.messages = []

# --- 화면 그리기 ---
if st.session_state.category is None:
    st.info("통합 검색 또는 카테고리별 검색을 선택해주세요.")
    if st.button("🌐 전체 규정에서 검색", use_container_width=True, type="primary"):
        st.session_state.category = "global"; st.rerun()
    st.markdown("---")
    
    sorted_categories = sorted(CATEGORIES.items(), key=lambda item: int(item[0]))
    for key, value in sorted_categories:
        if st.button(value["name"], key=key, use_container_width=True):
            st.session_state.category = key; st.rerun()
else:
    if st.session_state.category == "global": category_name, files_to_search = "전체 규정", ALL_FILES
    else: info = CATEGORIES[st.session_state.category]; category_name, files_to_search = info['name'], info['files']
    
    st.header(f"💬 {category_name} 검색")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input(f"'{category_name}'에 대해 질문해보세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("가장 정확한 규정을 찾고 답변의 핵심을 요약하는 중입니다..."):
                chat_history = st.session_state.messages[-7:-1]
                context_docs = hybrid_search(prompt, data, sbert_model, files_to_search)
                if not context_docs:
                    response = "관련 규정을 찾지 못했습니다."
                else:
                    response = generate_ultimate_answer(prompt, context_docs, chat_history)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("↩️ 카테고리 선택으로"):
        st.session_state.category = None; st.session_state.messages = []; st.rerun()