# DecideRAG — BM25 + decide 판단 Agentic RAG

설계서: `plan\DecideRAG_설계서_20261126.md`. OpenSearch 설치·운영: `doc\OpenSearch_설치_운영가이드.md`.

흐름은 다음과 같다.
- 추출(struct_docparse.dll) → 청킹 → 검색 엔진(OpenSearch, Kiwi 토큰 BM25)
- decide 판단: 재정렬, 충분성, 재검색 계획
- 재검색 (최대 3라운드). 문서 라우팅은 후보 20개(근거 문서 + 문서 BM25 상위) 안에서 고른다
- Qwen 답변

## 실행

```powershell
python run.py build                                           # 증분 색인: 새·바뀐 문서만 추출 → 청킹 → OpenSearch
python run.py build --full-rebuild                            # 색인을 지우고 전부 다시
python run.py ask "나 다음달 결혼하는데 축의금이랑 휴가 얼마나 줘?"   # 답 + 근거 + 라운드 요약
python run.py ask "그럼 애 낳기 전에 미리 신청돼?" --history h.json  # 멀티턴 (h.json = [{"q","a"}, …])
python run.py ask "…" --full                                  # 라운드 trace 전체
python run.py eval                                            # 평가셋 100문항 → eval\runs\decide.jsonl (끊겨도 이어서)
python run.py eval --judge-only --out eval\runs\x.jsonl       # 답변 생성 없이 검색·판단만 (LLM 서버 불필요)
python run.py summary eval\runs\a.jsonl eval\runs\b.jsonl     # 나란히 비교
python run.py ask "…" --groups hr,legal                      # 이 그룹 권한으로 검색 (없으면 권한 필터 없음 — 운영자용)
python run.py serve                                           # HTTP API (기본 127.0.0.1:8090)
python tests\test_incremental.py                              # 증분 색인 시험 (임시 폴더·decide_test 색인 사용)
python tests\test_acl.py                                      # 권한·API 시험 (decide_acltest 색인, decide·Qwen 필요)
python tests\test_degraded.py                                 # 장애 처리·동시성 시험 (decide 꺼짐·지연·중간 실패, LLM 꺼짐, 검색 0건, 동시 4개, 대기열)
python tests\test_answer_fixes.py                             # 답 고치기 A1~A4 시험 (가짜 LLM)
python eval\answer_variants.py --name x --set retry_refusal=true  # 근거 고정, 답변 설정만 바꿔 실행 → eval\runs\ans_x.jsonl
python eval\ac.py packets --runs base x                       # 판정 안 된 답만 eval\ac\packets_*.jsonl 로 (판정은 Claude, eval\ac_judge_guide.md)
python eval\ac.py score --runs base x --base base             # 정답률(AC)·짝지은 95% 구간 → eval\runs\ac_summary.json
```

## 답 고치기 (`[answer]`, 설계서 10장·14.4)

Qwen 이 쓴 답을 보고, 문제가 있을 때만 한 번 더 묻는다. 측정 결과로 A3·A4 만 켰다.

| # | 조건 | 하는 일 | 설정 |
|---|---|---|---|
| A3 | 충분성 ≥ 0.8 인데 거절 | 가장 관련 높은 근거 번호를 짚어 다시 답하게 한다 | `retry_refusal = true` |
| A4 | 한자 비율 > 0.2 | 프롬프트 끝에 "반드시 한국어로" 를 붙여 새로 답하게 한다 | `force_korean = true` |
| A1 | decide 가 계산·조건 비교 질문으로 판단 | Qwen 생각 모드로 답한다 | `thinking_on_reasoning = false` (끔: 효과 확인 안 됨, 답변 평균 6초) |
| A2 | 답에 근거에 없는 숫자 | 근거 숫자로 계산식을 쓰고 다시 답하게 한다 | `verify_numbers = false` (끔: 효과 확인 안 됨) |

- 고친 기록은 trace 와 결과의 `fixes`(고치기 전 답 포함)에 남는다.
- 측정(근거 고정, 90문항): A3+A4 는 정답률 0.765 → 0.777, 답이 있는데 거절 5.6% → 2.2%, 문서에 없는 질문 거절 100% 유지.

## HTTP API (`src\decide_rag\service.py`)

| 메서드 | 경로 | 내용 |
|---|---|---|
| POST | `/ask` | `{"question", "history": [{"q","a"}], "debug": false}` → `{"answer", "sources": [{"chunk_id","title","path","page"}], "sufficient", "stop", "rounds", "timing", "degraded"}`. `debug` 면 `trace` 도 준다 |
| GET | `/health` | decide·LLM 응답, 엔진 문서·청크 수, 파싱 실패 수, 인증 방식. 인증 없이 부를 수 있다 |
| POST | `/index/rebuild` | `{"full": false}` 증분 색인을 백그라운드로 실행 (관리자 그룹만) |
| GET | `/index/status` | 마지막 색인 결과와 파싱 실패 문서 (관리자 그룹만) |

인증은 다음과 같다.
- 사용자·그룹은 인증 게이트웨이가 넣는 헤더에서만 읽는다: `X-Gateway-Token`, `X-User-Id`, `X-User-Groups`(쉼표 구분).
- 토큰은 환경변수 `DECIDE_GATEWAY_TOKEN` 과 같아야 한다. 요청 본문에 `groups` 를 넣어도 쓰지 않는다.
- 환경변수가 없으면 `/ask` 는 503 이다.
- 이 PC 시험용으로는 `[service] dev_mode = true` 를 켜면 `dev_user`·`dev_groups` 로 동작한다. 이때 헤더는 무시한다.
- 모든 사용자는 공개 그룹 `all` 을 자동으로 갖는다. 색인 명령은 `admin` 그룹만 쓸 수 있다.
- `/ask` 는 동시에 처리한다. decide 요청만 대기열(`QueuedDecide`)이 한 번에 하나씩 온 순서대로 보낸다. LLM 답변 단계는 요청끼리 겹쳐 진행된다.

장애 처리 (설계서 8.2, 3단계):

| 상황 | `/ask` 응답 |
|---|---|
| decide 꺼짐·오류·응답 지연 (재시도 2회 후) | 200, `degraded: true`. 판단해 둔 것이 있으면 그 순위, 없으면 BM25 상위 5개로 답한다. `degraded_reason` 에 이유 |
| decide 실패 직후 30초 (`cooldown_sec`) | decide 를 부르지 않고 바로 degraded (차단기) |
| decide 대기열 50개 초과 (`max_queue`) | 503 |
| LLM 꺼짐·오류 | 502, `sources` 에 찾아 둔 근거 |
| 검색 결과 0건 | LLM 을 부르지 않고 "찾을 수 없습니다" |
| 검색 엔진(OpenSearch) 오류 | 502 |

- `/health` 의 `decide.queue` 에 대기 수, 차단 남은 시간, 마지막 오류가 나온다.
- 평가(`run.py eval`)는 degraded 결과를 점수에 섞지 않는다. 오류로 남기고, 다음 실행 때 다시 한다.

```powershell
$env:DECIDE_GATEWAY_TOKEN = "<게이트웨이와 나눈 토큰>"
python run.py serve
```

## 권한 (`acl.toml`)

- 문서 폴더 기준 상대 경로 패턴 → 그룹 규칙이다. 예: `pattern = "37_*"`, `groups = ["legal"]`.
- 맞는 규칙이 없는 문서는 `[index] default_acl` (`["all"]` = 공개).
- 규칙을 바꾸고 `python run.py build` 를 돌리면 그 문서들의 `acl` 만 갱신한다. 파싱은 다시 하지 않는다.
- 모든 검색(1라운드·재검색·문서 후보)에 권한 필터가 먼저 걸린다. 볼 수 없는 문서는 제목도 decide 에 가지 않는다.

## trace 로그

- `logs\trace_YYYYMMDD.jsonl` 에 질문마다 한 줄을 쓴다.
- 담기는 항목: 시각, 사용자 ID·그룹, 질문, 답, 근거, 라운드(질의·청크별 관련성·충분성·라우팅 후보), 시간, decide 호출.
- 질문 원문이 들어가므로 보관 기간을 정해야 한다 (설계서 13장 4).

- 설정은 `config.toml` 에 있다.
- 검색 엔진은 `[search] engine` 으로 고른다.
  - `opensearch`: 운영용. 먼저 OpenSearch 를 띄워야 한다.
  - `memory`: rank_bm25, 문서 수천 개 이하 시험용.
- decide API 키는 환경변수 `DECIDE_API_KEY` 에서 먼저 읽는다. 없으면 `api_key_file` 의 첫 줄을 읽는다. 키는 어디에도 출력하지 않는다.
- OpenSearch 계정은 환경변수 `OPENSEARCH_USER`, `OPENSEARCH_PASSWORD` 에서만 읽는다.

## 색인 데이터

| 위치 | 내용 |
|---|---|
| `data\manifest.json` | 문서마다 상대 경로, 내용 SHA-256, 제목, 청크 수, 오류 |
| `data\parsed\<문서ID>.json` | 파싱 결과 |
| `data\chunks\<문서ID>.jsonl` | 청크. 엔진을 새로 깔면 여기서 다시 넣는다 (파싱 생략) |
| OpenSearch `decide_chunks` | 청크 색인 |
| OpenSearch `decide_docs` | 문서 색인 (라우팅 후보: 제목 + 헤딩 + 첫 청크 300자) |
| `data\legacy_stage1\` | 1단계 결과 (옛 ID `D01#000`). `tools\*_bench.py` 측정용 |

- 문서 ID 는 문서 폴더 기준 상대 경로(소문자)의 SHA-1 앞 12자리 앞에 `d` 를 붙인 것이다 (예: `d55cb0e974fe2`). 문서를 추가해도 다른 문서의 ID 가 바뀌지 않는다.
- 파일을 옮기거나 이름을 바꾸면 삭제 + 추가로 처리한다.
- 권한(`acl`)은 `acl.toml` 규칙으로 정한다 (아래 "권한").

## 구성

| 파일 | 내용 |
|---|---|
| `src\decide_rag\extract.py` | struct_docparse 로 블록 추출, 문서 ID(경로 해시), 내용 해시 |
| `src\decide_rag\chunker.py` | 헤딩 기준 청킹 (본문 700자, 150자 미만 합침, 표 머리글 반복), 문서별 청크 파일 |
| `src\decide_rag\indexer.py` | 증분 색인 (새·바뀜·그대로·사라짐 판단, 엔진 교체) |
| `src\decide_rag\search_engine.py` | `OpenSearchEngine`·`MemoryEngine`: 청크 검색(_msearch), 문서 후보 검색, PRF, ID 조회, 권한 필터 |
| `src\decide_rag\bm25_index.py` | Kiwi 형태소 토큰 규칙 + rank_bm25 (메모리 엔진) |
| `src\decide_rag\clients.py` | Qwen(OpenAI 호환) · decide(`/v1/systemone`) 클라이언트, 재시도, decide 대기열·차단기(`QueuedDecide`) |
| `src\decide_rag\judge.py` | `DecideJudge`: 재정렬 · 충분성 · 후보 문서 라우팅 · 재검색 계획 · 멀티턴 질의 |
| `src\decide_rag\agent.py` | 루프와 재검색 상한 |
| `src\decide_rag\answer.py` | 답변 프롬프트 · 생성 · 답 고치기 A1~A4 |
| `src\decide_rag\cli.py` | build / ask / eval / summary / serve |
| `src\decide_rag\service.py` | HTTP API (FastAPI), 게이트웨이 헤더 인증, 관리자 색인 명령 |
| `src\decide_rag\acl.py` | 권한 규칙(acl.toml) → 문서별 그룹, 사용자 그룹 |
| `src\decide_rag\tracelog.py` | 질문별 trace 로그 |
| `eval\run_eval.py` | 평가 실행 · 요약 (근거 포함률, 거절, 시간, decide 호출) |
| `tools\engine_bench_1m.py` | 합성 100만 청크에서 질문당 검색 시간 (서비스 엔진 코드 그대로) |

재검색 상한은 네 가지다. trace 의 `stop` 에 멈춘 이유가 남는다.
- `max_rounds`: 3
- `max_decide_calls`: 40
- `judge_budget_sec`: 6초
- 충분성 0.5 이상이면 멈춘다 (`sufficient`)

- Answer Correctness 채점: `eval\ac.py` (판정은 Claude, 프로토타입 판정과 같으면 다시 쓴다).
- RAGAS 4지표 채점: 프로토타입 `D:\Project\Q&A\AgenticRAG` 의 `ragas.py`.
