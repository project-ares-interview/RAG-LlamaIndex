import os
import json
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI
from unidecode import unidecode
import re
import traceback

# RAG 시스템 클래스를 임포트합니다.
from new_azure_rag_llamaindex import AzureBlobRAGSystem
# 웹 검색 도구 임포트
from tool_code import google_search

def extract_json_from_response(text: str) -> str:
    """AI의 응답 텍스트에서 순수한 JSON 부분만 추출합니다."""
    # 코드 블록에서 JSON 추출
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    # 텍스트에서 가장 큰 JSON 객체 추출
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

class RAGInterviewBot:
    """[최종] 평가 결과를 면접 종료 후 일괄 제공하는 면접 시스템"""

    def __init__(self, company_name: str, job_title: str, container_name: str, index_name: str):
        print("🤖 RAG 전용 사업 분석 면접 시스템 초기화...")
        self.company_name = company_name
        self.job_title = job_title

        load_dotenv(dotenv_path=".env.keys", override=True)
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_KEY'),
            api_version=os.getenv('API_VERSION', '2024-02-15-preview')
        )
        self.model = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o')

        print("\n📊 Azure 사업 분석 RAG 시스템 연동...")
        self.rag_system = None
        self.rag_ready = False
        try:
            self.rag_system = AzureBlobRAGSystem(container_name=container_name, index_name=index_name)
            
            blobs = list(self.rag_system.container_client.list_blobs())
            if not blobs:
                print(f"⚠️ 경고: Azure Blob 컨테이너 '{container_name}'에 분석할 파일이 없습니다.")
                return

            print(f"✅ Azure RAG 시스템 준비 완료. {len(blobs)}개의 문서를 기반으로 합니다.")
            if input("Azure AI Search 인덱스를 동기화하시겠습니까? (y/n): ").lower() == 'y':
                self.rag_system.sync_index()
            self.rag_ready = True

        except Exception as e:
            print(f"❌ RAG 시스템 연동 실패: {e}")

    def generate_questions(self, num_questions: int = 3) -> list:
        """RAG 기반으로 사업 현황 심층 질문 생성"""
        if not self.rag_ready: return []
        print(f"\n🧠 {self.company_name} 맞춤 질문 생성 중...")
        try:
            business_info = self.rag_system.query(f"{self.company_name}의 핵심 사업, 최근 실적, 주요 리스크에 대해 요약해줘.")
            prompt = f"""
            당신은 {self.company_name}의 {self.job_title} 직무 면접관입니다.
            아래의 최신 사업 현황 데이터를 바탕으로, 지원자의 분석력과 전략적 사고를 검증할 수 있는 날카로운 질문 {num_questions}개를 생성해주세요.
            질문 목록만 JSON 형식으로 반환해주세요.
            ```json
            {{ "questions": ["생성된 질문 1", "생성된 질문 2"] }}
            ```
            """
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=1000, temperature=0.8, response_format={"type": "json_object"}
            )
            result = json.loads(extract_json_from_response(response.choices[0].message.content))
            questions = result.get("questions", [])
            print(f"✅ {len(questions)}개의 맞춤 질문 생성 완료.")
            return questions
        except Exception as e:
            print(f"❌ 질문 생성 실패: {e}")
            return [f"{self.company_name}의 주요 경쟁사와 비교했을 때, 우리 회사가 가진 핵심적인 기술적 우위는 무엇이라고 생각하십니까?"]

    def analyze_answer_with_rag(self, question: str, answer: str) -> dict:
        """개별 답변에 대한 상세 분석 (XAI 기반, 점수 없음)"""
        if not self.rag_ready: return {"error": "RAG 시스템 미준비"}

        print(f"    (답변 분석 중...)") # 사용자에게 시스템이 동작 중임을 알리는 최소한의 표시
        try:
            related_facts = self.rag_system.query(f"'{answer}'라는 주장에 대한 사실관계를 확인하고 관련 데이터를 찾아줘.")
            search_results = google_search.search(queries=[f"{self.company_name} {answer}"])
            web_context = f"웹 검색 결과:\n{search_results}"

            analysis_prompt = f"""
            당신은 시니어 사업 분석가입니다. 아래 자료를 종합하여 지원자의 답변을 상세히 평가해주세요.
            **'데이터 기반 사실 분석'과 '독창적인 전략적 통찰력'을 구분하여 평가하고, 점수 대신 서술형으로 평가 의견을 제시해주세요.**

            **면접 질문:** {question}
            **지원자 답변:** {answer}
            ---
            **[자료 1] 내부 사업 데이터:** {related_facts}
            **[자료 2] 외부 웹 검색 결과:** {web_context}
            ---
            **평가 지침:**
            1. **주장별 사실 확인:** 지원자의 핵심 주장을 1~2개 뽑아 자료 1, 2를 바탕으로 검증합니다.
            2. **내용 분석:** 데이터 활용 능력과 독창적인 비즈니스 논리를 평가합니다.
            3. **피드백:** 강점과 개선 제안을 서술합니다.
            
            **응답 형식 (JSON):**
            ```json
            {{
                "fact_checking": [{{ "claim": "추출한 주장", "verification": "확인 결과", "evidence": "판단 근거" }}],
                "content_analysis": {{
                    "analytical_depth": {{ "assessment": "평가 의견 (예: 데이터의 핵심을 정확히 파악함)", "comment": "상세 코멘트" }},
                    "strategic_insight": {{ "assessment": "평가 의견 (예: 실현 가능한 독창적 아이디어 제시)", "comment": "상세 코멘트" }}
                }},
                "actionable_feedback": {{ "strengths": [], "suggestions_for_improvement": [] }}
            }}
            ```
            """
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.2, max_tokens=2000, response_format={"type": "json_object"}
            )
            return json.loads(extract_json_from_response(response.choices[0].message.content))
        except Exception as e:
            print(f"❌ 답변 분석 중 오류: {e}")
            return {"error": str(e)}

    def print_individual_analysis(self, analysis: dict, question_num: int):
        """개별 답변에 대한 분석 결과 출력 형식"""
        if "error" in analysis:
            print(f"\n❌ 분석 오류: {analysis['error']}")
            return

        print("\n" + "="*70)
        print(f"📊 [질문 {question_num}] 답변 상세 분석 결과")
        print("="*70)

        print("\n" + "-"*30)
        print("✅ 주장별 사실 확인 (Fact-Checking)")
        fact_checks = analysis.get("fact_checking", [])
        if not fact_checks: print("  - 확인된 주장이 없습니다.")
        else:
            for check in fact_checks:
                print(f"  - 주장: \"{check.get('claim', 'N/A')}\"")
                print(f"    - 검증: {check.get('verification', 'N/A')}")
                print(f"    - 근거: {check.get('evidence', 'N/A')}")

        print("\n" + "-"*30)
        print("📝 내용 분석 (Content Analysis)")
        content = analysis.get("content_analysis", {})
        depth = content.get("analytical_depth", {})
        insight = content.get("strategic_insight", {})
        print(f"  - 데이터 분석 깊이: {depth.get('assessment', 'N/A')}")
        print(f"    - 코멘트: {depth.get('comment', 'N/A')}")
        print(f"  - 전략적 통찰력: {insight.get('assessment', 'N/A')}")
        print(f"    - 코멘트: {insight.get('comment', 'N/A')}")
        
        print("\n" + "-"*30)
        print("💡 실행 가능한 피드백 (Actionable Feedback)")
        feedback = analysis.get("actionable_feedback", {})
        strengths = feedback.get("strengths", [])
        suggestions = feedback.get("suggestions_for_improvement", [])
        if strengths:
            print("  - 강점:")
            for s in strengths: print(f"    ✓ {s}")
        if suggestions:
            print("  - 개선 제안:")
            for s in suggestions: print(f"    → {s}")
        print("="*70)

    def generate_follow_up_question(self, original_question: str, answer: str, analysis: dict) -> str:
        """분석 결과를 바탕으로 심층 꼬리 질문 생성"""
        try:
            suggestions = analysis.get("actionable_feedback", {}).get("suggestions_for_improvement", [])
            prompt = f"기존 질문: {original_question}\n지원자 답변: {answer}\n답변에 대한 AI 분석 내용(개선 제안): {', '.join(suggestions)}\n\n위 상황을 바탕으로, 지원자의 논리를 더 깊게 파고들기 위한 핵심 꼬리 질문 1개만 JSON 형식으로 생성해주세요. (예: {{\"follow_up_question\": \"생성된 꼬리 질문\"}})"
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=300, response_format={"type": "json_object"}
            )
            result = json.loads(extract_json_from_response(response.choices[0].message.content))
            return result.get("follow_up_question", "")
        except Exception as e:
            print(f"❌ 꼬리 질문 생성 실패: {e}")
            return ""

    def conduct_interview(self):
        """[수정] 평가 결과는 면접 종료 후 일괄 출력"""
        if not self.rag_ready:
            print("\n❌ RAG 시스템이 준비되지 않아 면접을 진행할 수 없습니다.")
            return

        questions = self.generate_questions()
        if not questions:
            print("\n❌ 면접 질문을 생성하지 못했습니다.")
            return

        print("\n" + "="*70)
        print(f"🏢 {self.company_name} {self.job_title} 직무 면접을 시작하겠습니다.")
        print("면접이 종료된 후 전체 답변에 대한 상세 분석이 제공됩니다.")
        print("="*70)

        interview_transcript = []

        for i, question in enumerate(questions, 1):
            print(f"\n--- [질문 {i}/{len(questions)}] ---")
            print(f"👨‍💼 면접관: {question}")
            answer = input("💬 답변: ")
            if answer.lower() in ['/quit', '/종료']: break

            # [핵심] 평가는 수행하되, 결과는 출력하지 않고 저장만 함
            analysis = self.analyze_answer_with_rag(question, answer)
            
            follow_up_question = ""
            follow_up_answer = ""
            if "error" not in analysis:
                follow_up_question = self.generate_follow_up_question(question, answer, analysis)
                if follow_up_question:
                    print(f"\n--- [꼬리 질문] ---")
                    print(f"👨‍💼 면접관: {follow_up_question}")
                    follow_up_answer = input("💬 답변: ")

            # 현재 질문, 답변, 분석 내용, 꼬리 질문/답변을 모두 기록
            interview_transcript.append({
                "question_num": i, "question": question, "answer": answer, "analysis": analysis,
                "follow_up_question": follow_up_question, "follow_up_answer": follow_up_answer
            })

        print("\n🎉 면접이 종료되었습니다. 수고하셨습니다.")
        
        # [핵심] 면접 종료 후, 저장된 모든 분석 결과를 일괄 출력
        if interview_transcript:
            print("\n\n" + "#"*70)
            print(" 면접 전체 답변에 대한 상세 분석 리포트")
            print("#"*70)

            # 1. 개별 답변 분석 결과부터 순서대로 출력
            for item in interview_transcript:
                self.print_individual_analysis(item['analysis'], item['question_num'])

            # 2. 최종 종합 리포트 생성 및 출력
            self.generate_final_report(interview_transcript)

    def generate_final_report(self, transcript: list):
        """면접 전체 기록을 바탕으로 최종 종합 리포트 생성"""
        print("\n\n" + "#"*70)
        print(" 최종 역량 분석 종합 리포트 생성 중...")
        print("#"*70)
        
        try:
            # 면접 전체 대화 내용과 개별 분석 결과를 요약하여 프롬프트에 전달
            conversation_summary = ""
            for item in transcript:
                q_num = item['question_num']
                analysis_assessment = item['analysis'].get('content_analysis', {}).get('strategic_insight', {}).get('assessment', '분석 미완료')
                conversation_summary += f"질문 {q_num}: {item['question']}\n답변 {q_num}: {item['answer']}\n(개별 분석 요약: {analysis_assessment})\n---\n"

            report_prompt = f"""
            당신은 시니어 채용 전문가입니다. 아래의 전체 면접 대화 및 개별 분석 요약을 종합하여, 지원자에 대한 '최종 역량 분석 종합 리포트'를 작성해주세요.
            
            **[자료] 면접 전체 요약:**
            {conversation_summary}
            ---
            **리포트 작성 지침:**
            1. **종합 총평:** 지원자의 일관성, 강점, 약점을 종합하여 최종 평가를 내립니다.
            2. **핵심 역량 분석:** {self.job_title} 직무에 필요한 핵심 역량(예: 문제 해결 능력, 비즈니스 이해도, 기술 전문성) 3가지를 식별하고, 면접 전체 내용을 근거로 [최상], [상], [중], [하]로 평가합니다. 각 평가에 대한 구체적인 근거를 제시해야 합니다.
            3. **성장 가능성:** 면접 과정에서 보인 태도나 답변의 깊이를 바탕으로 지원자의 잠재력을 평가합니다.

            **응답 형식 (JSON):**
            ```json
            {{
                "overall_summary": "종합적인 평가 요약...",
                "core_competency_analysis": [
                    {{"competency": "핵심 역량 1", "assessment": "[평가 등급]", "evidence": "판단 근거..."}}
                ],
                "growth_potential": "지원자의 성장 가능성에 대한 코멘트..."
            }}
            ```
            """
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": report_prompt}],
                temperature=0.3, max_tokens=3000, response_format={"type": "json_object"}
            )
            report_data = json.loads(extract_json_from_response(response.choices[0].message.content))
            self.print_final_report(report_data)

        except Exception as e:
            print(f"❌ 최종 리포트 생성 중 오류 발생: {e}")
            traceback.print_exc()

    def print_final_report(self, report: dict):
        """최종 종합 리포트 출력"""
        if not report: return
        
        print("\n\n" + "="*70)
        print(f"🏅 {self.company_name} {self.job_title} 지원자 최종 역량 분석 종합 리포트")
        print("="*70)

        print(f"\n■ 총평 (Overall Summary)\n" + "-"*50)
        print(report.get("overall_summary", "요약 정보 없음."))

        print(f"\n■ 핵심 역량 분석 (Core Competency Analysis)\n" + "-"*50)
        for comp in report.get("core_competency_analysis", []):
            print(f"  - {comp.get('competency', 'N/A')}: **{comp.get('assessment', 'N/A')}**")
            print(f"    - 근거: {comp.get('evidence', 'N/A')}")
        
        print(f"\n■ 성장 가능성 (Growth Potential)\n" + "-"*50)
        print(f"  {report.get('growth_potential', 'N/A')}")
        print("\n" + "="*70)

def main():
    try:
        target_container = 'interview-data'
        company_name = input("면접을 진행할 회사 이름 (예: SK하이닉스): ")
        safe_company_name_for_index = unidecode(company_name.lower()).replace(' ', '-')
        index_name = f"{safe_company_name_for_index}-report-index"
        job_title = input("지원 직무 (예: 사업분석가): ")

        print("\n" + "-"*40)
        print(f"대상 컨테이너: {target_container}")
        print(f"회사 이름: {company_name}")
        print(f"AI Search 인덱스: {index_name}")
        print("-"*40)

        bot = RAGInterviewBot(company_name=company_name, job_title=job_title, container_name=target_container, index_name=index_name)
        bot.conduct_interview()

    except Exception as e:
        print(f"\n❌ 시스템 실행 중 심각한 오류 발생: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()