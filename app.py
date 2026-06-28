import re
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Consensus Judge",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 12]


def get_similarity(text1, text2, model):
    embeddings = model.encode([text1, text2])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])


def get_question_type_rule(question_type):
    rules = {
        "사실 정보": {
            "rule": "사실 정보는 모델 간 합의도가 높더라도 공식 출처 확인이 필요합니다.",
            "risk": "중간"
        },
        "의견/예측": {
            "rule": "의견이나 미래 예측은 모델마다 관점이 다를 수 있으므로 낮은 합의도가 반드시 오류를 의미하지 않습니다.",
            "risk": "중간"
        },
        "의학/법률/금융": {
            "rule": "의학·법률·금융 질문은 AI 답변만으로 판단해서는 안 되며 전문가 또는 공식 자료 확인이 필수입니다.",
            "risk": "높음"
        },
        "코드/수학": {
            "rule": "코드와 수학 문제는 답변 비교뿐 아니라 실제 실행, 계산, 테스트를 통한 검증이 필요합니다.",
            "risk": "중간"
        },
        "창작": {
            "rule": "창작 요청은 정답 여부보다 사용자의 요구사항과 얼마나 잘 맞는지가 중요합니다.",
            "risk": "낮음"
        }
    }
    return rules.get(question_type, rules["사실 정보"])


def analyze_claims(gpt_answer, gemini_answer, model):
    gpt_sentences = split_sentences(gpt_answer)
    gemini_sentences = split_sentences(gemini_answer)

    common_claims = []
    gpt_unique = []
    gemini_unique = []

    # GPT 문장 기준으로 Gemini와 가장 비슷한 문장 찾기
    for gpt_s in gpt_sentences:
        best_score = 0
        best_match = ""

        for gemini_s in gemini_sentences:
            score = get_similarity(gpt_s, gemini_s, model)
            if score > best_score:
                best_score = score
                best_match = gemini_s

        similarity_percent = round(best_score * 100, 2)

        if best_score >= 0.60:
            common_claims.append((gpt_s, best_match, similarity_percent))
        else:
            gpt_unique.append(gpt_s)

    # Gemini만 강조한 내용 찾기
    for gemini_s in gemini_sentences:
        best_score = 0

        for gpt_s in gpt_sentences:
            score = get_similarity(gemini_s, gpt_s, model)
            if score > best_score:
                best_score = score

        if best_score < 0.60:
            gemini_unique.append(gemini_s)

    common_claims = sorted(common_claims, key=lambda x: x[2], reverse=True)

    return common_claims[:3], gpt_unique[:3], gemini_unique[:3]


def calculate_consensus_score(gpt_answer, gemini_answer, common_claims, gpt_unique, gemini_unique, model):
    # 1. 전체 답변 유사도
    full_similarity = get_similarity(gpt_answer, gemini_answer, model) * 100

    # 2. 공통 주장 평균 유사도
    if common_claims:
        claim_similarity = sum([claim[2] for claim in common_claims]) / len(common_claims)
    else:
        claim_similarity = 0

    # 3. 공통 주장 비율
    total_items = len(common_claims) + len(gpt_unique) + len(gemini_unique)
    if total_items > 0:
        common_ratio = (len(common_claims) / total_items) * 100
    else:
        common_ratio = 0

    # 4. 최종 점수
    final_score = (
        full_similarity * 0.30
        + claim_similarity * 0.50
        + common_ratio * 0.20
    )

    # 너무 낮게 왜곡되는 것 방지
    if claim_similarity >= 80 and len(common_claims) >= 1:
        final_score = max(final_score, 80)

    return round(final_score, 2), round(full_similarity, 2), round(claim_similarity, 2), round(common_ratio, 2)


def get_consensus_level(score):
    if score >= 80:
        return "HIGH", "🟢", "두 답변의 핵심 의미가 상당히 유사합니다."
    elif score >= 50:
        return "MEDIUM", "🟡", "두 답변이 일부 유사하지만 관점이나 강조점의 차이가 존재합니다."
    else:
        return "LOW", "🔴", "두 답변의 방향이나 핵심 내용에 큰 차이가 있습니다."


def make_report(
    question,
    question_type,
    final_score,
    full_similarity,
    claim_similarity,
    common_ratio,
    level,
    common_claims,
    gpt_unique,
    gemini_unique,
    warning
):
    lines = []
    lines.append("AI Consensus Judge - Analysis Report")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append(f"Question Type: {question_type}")
    lines.append(f"Final Consensus Score: {final_score}")
    lines.append(f"Full Response Similarity: {full_similarity}")
    lines.append(f"Claim Similarity: {claim_similarity}")
    lines.append(f"Common Claim Ratio: {common_ratio}")
    lines.append(f"Consensus Level: {level}")
    lines.append("")
    lines.append("Common Claims")
    if common_claims:
        for i, (gpt_s, gemini_s, s) in enumerate(common_claims, 1):
            lines.append(f"{i}. Similarity {s}%")
            lines.append(f"   GPT: {gpt_s}")
            lines.append(f"   Gemini: {gemini_s}")
    else:
        lines.append("- No strong common claims detected.")
    lines.append("")
    lines.append("GPT Emphasis / Unique Points")
    if gpt_unique:
        for i, item in enumerate(gpt_unique, 1):
            lines.append(f"{i}. {item}")
    else:
        lines.append("- No major GPT-only point detected.")
    lines.append("")
    lines.append("Gemini Emphasis / Unique Points")
    if gemini_unique:
        for i, item in enumerate(gemini_unique, 1):
            lines.append(f"{i}. {item}")
    else:
        lines.append("- No major Gemini-only point detected.")
    lines.append("")
    lines.append("User Warning")
    lines.append(warning)
    return "\n".join(lines)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<div class="main-title">AI Consensus Judge</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">생성형 AI 답변의 합의도 분석을 통한 신뢰성 평가 시스템</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    이 웹앱은 사용자의 질문에 대해 **Gemini 답변을 자동 생성**하고,  
    사용자가 붙여넣은 **GPT 답변**과 비교하여 공통 주장, 차이점, 사용자 주의 메시지를 제공합니다.
    """
)

with st.sidebar:
    st.header("Settings")

    api_key = st.secrets.get("GEMINI_API_KEY", "")

    question_type = st.selectbox(
        "Question Type",
        ["사실 정보", "의견/예측", "의학/법률/금융", "코드/수학", "창작"]
    )

    rule_info = get_question_type_rule(question_type)

    st.divider()
    st.markdown("### Question Type Rule")
    st.write(rule_info["rule"])
    st.write(f"Risk Level: {rule_info['risk']}")


question = st.text_area(
    "1. 질문을 입력하세요",
    height=110,
    placeholder="예: 전화기를 발명한 사람은 누구인가요?"
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Gemini Answer")

    if "gemini_answer" not in st.session_state:
        st.session_state.gemini_answer = ""

    if st.button("Gemini 답변 생성", use_container_width=True):
        if not api_key:
            st.error("Gemini API Key를 입력하세요.")
        elif not question:
            st.error("질문을 입력하세요.")
        else:
            with st.spinner("Gemini 답변 생성 중..."):
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel("gemini-2.5-flash")
                response = gemini_model.generate_content(question)
                st.session_state.gemini_answer = response.text

    gemini_answer = st.text_area(
        "Gemini가 생성한 답변",
        value=st.session_state.gemini_answer,
        height=330
    )

with col2:
    st.subheader("GPT Answer")
    gpt_answer = st.text_area(
        "ChatGPT에서 받은 답변을 붙여넣으세요",
        height=330,
        placeholder="GPT 답변을 여기에 붙여넣으세요."
    )

st.divider()

if st.button("합의도 분석 시작", type="primary", use_container_width=True):
    if not gpt_answer or not gemini_answer:
        st.error("GPT 답변과 Gemini 답변이 모두 필요합니다.")
    else:
        with st.spinner("답변 분석 중..."):
            embed_model = load_embedding_model()

            common_claims, gpt_unique, gemini_unique = analyze_claims(
                gpt_answer,
                gemini_answer,
                embed_model
            )

            final_score, full_similarity, claim_similarity, common_ratio = calculate_consensus_score(
                gpt_answer,
                gemini_answer,
                common_claims,
                gpt_unique,
                gemini_unique,
                embed_model
            )

            level, icon, level_message = get_consensus_level(final_score)
            rule_info = get_question_type_rule(question_type)

        st.subheader("Analysis Result")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Consensus Score", f"{final_score}")
        m2.metric("Consensus Level", f"{icon} {level}")
        m3.metric("Question Type", question_type)
        m4.metric("Risk Level", rule_info["risk"])

        st.progress(min(final_score / 100, 1.0))

        if level == "HIGH":
            st.success(level_message)
        elif level == "MEDIUM":
            st.warning(level_message)
        else:
            st.error(level_message)

        st.info(f"질문 유형별 평가 규칙: {rule_info['rule']}")

        st.subheader("Score Breakdown")
        b1, b2, b3 = st.columns(3)
        b1.metric("Full Similarity", f"{full_similarity}%")
        b2.metric("Claim Similarity", f"{claim_similarity}%")
        b3.metric("Common Claim Ratio", f"{common_ratio}%")

        st.subheader("🟢 공통 주장")
        if common_claims:
            for i, (gpt_s, gemini_s, score) in enumerate(common_claims, 1):
                with st.expander(f"공통 주장 {i} | 문장 유사도 {score:.2f}%"):
                    st.markdown("**GPT**")
                    st.write(gpt_s)
                    st.markdown("**Gemini**")
                    st.write(gemini_s)
        else:
            st.write("뚜렷하게 공통된 주장을 찾지 못했습니다.")

        col_gpt, col_gemini = st.columns(2)

        with col_gpt:
            st.subheader("🟡 GPT가 더 강조한 내용")
            if gpt_unique:
                for i, item in enumerate(gpt_unique, 1):
                    st.warning(f"{i}. {item}")
            else:
                st.write("GPT만 두드러지게 강조한 내용은 크게 감지되지 않았습니다.")

        with col_gemini:
            st.subheader("🟡 Gemini가 더 강조한 내용")
            if gemini_unique:
                for i, item in enumerate(gemini_unique, 1):
                    st.warning(f"{i}. {item}")
            else:
                st.write("Gemini만 두드러지게 강조한 내용은 크게 감지되지 않았습니다.")

        st.subheader("🔴 사용자 주의 메시지")
        user_warning = (
            "높은 합의도가 반드시 정답을 의미하지는 않습니다. "
            "특히 의학, 법률, 금융, 최신 정보, 미래 예측과 관련된 질문은 "
            "공식 자료나 전문가 의견을 통해 추가 검증해야 합니다."
        )
        st.error(user_warning)

        st.subheader("🔵 종합 해석")
        st.write(
            f"이 질문은 '{question_type}' 유형으로 평가되었으며, "
            f"최종 Consensus Score는 {final_score}점입니다. "
            f"이 점수는 전체 답변 유사도({full_similarity}%), "
            f"공통 주장 유사도({claim_similarity}%), "
            f"공통 주장 비율({common_ratio}%)을 함께 반영한 값입니다."
        )

        report_text = make_report(
            question,
            question_type,
            final_score,
            full_similarity,
            claim_similarity,
            common_ratio,
            level,
            common_claims,
            gpt_unique,
            gemini_unique,
            user_warning
        )

        st.download_button(
            label="분석 리포트 TXT 다운로드",
            data=report_text,
            file_name="ai_consensus_report.txt",
            mime="text/plain",
            use_container_width=True
        )
