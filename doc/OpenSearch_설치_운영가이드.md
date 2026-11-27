# OpenSearch 설치·구동·중지·설정 가이드 (Windows / Linux)

작성일: 2026-11-27

이 문서는 DecideRAG 의 검색 엔진(설계서 §12.2)으로 쓰는 OpenSearch 의 설치와 운영 방법을 정리한 것이다.

**출처 표기**

| 표시 | 뜻 |
|---|---|
| **[이 PC 실측]** | 2026-11-26~27 이 PC(Windows 11)에 3.8.0 zip 을 설치해 실제로 실행한 내용 |
| **[공식 문서]** | docs.opensearch.org 의 최신 설치 문서(예시 버전 3.10.1). 이 PC 에서 실행해 보지는 않았다 |
| **미확인** | 공식 문서에 없고 직접 해 보지도 않은 부분 |

공식 문서의 예시 버전은 3.10.1 이다. 아래 예시는 이 PC 와 같은 **3.8.0** 으로 적었고, 버전은 `3.8.0` 부분만 바꾸면 된다.

---

## 0. 공통 사항

| 항목 | 값 |
|---|---|
| 라이선스 | Apache 2.0 |
| 포트 | 9200 REST API, 9300 노드 간 통신, 5601 Dashboards [공식 문서] |
| Java | 배포판 안에 JDK 가 들어 있다(`jdk` 폴더). 따로 설치할 필요가 없다 [이 PC 실측] |
| 관리자 비밀번호 | 2.12 부터 보안 플러그인을 켠 데모 설치에는 `OPENSEARCH_INITIAL_ADMIN_PASSWORD` 가 필요하다. 없으면 설치 스크립트가 OpenSearch 를 시작하지 않고 끝난다 [공식 문서] |
| 비밀번호 규칙 | 공식 페이지는 규칙을 링크로만 안내한다. 대문자·소문자·숫자·특수문자를 섞은 8자 이상이면 안전하다(오류 메시지 예시 기준, 공식 규칙 원문은 미확인) |
| 힙 메모리 | 시스템 RAM 의 절반 권장. 기본값은 1g. `-Xms` 와 `-Xmx` 를 같은 값으로 둔다 [공식 문서] |

다운로드 주소 형식은 다음과 같다.

```
https://artifacts.opensearch.org/releases/bundle/opensearch/<버전>/opensearch-<버전>-windows-x64.zip
https://artifacts.opensearch.org/releases/bundle/opensearch/<버전>/opensearch-<버전>-linux-x64.tar.gz
```

같은 주소 끝에 `.sha512` 를 붙이면 체크섬 파일을 받을 수 있다. 이 PC 에서는 zip 과 `.sha512` 를 함께 받아 해시가 일치하는 것을 확인했다 [이 PC 실측].

---

## 1. Windows (zip)

### 1.1 설치 [이 PC 실측]

**주의:** 압축을 풀 경로에 공백이 있으면 OpenSearch 가 시작되지 않는다 [공식 문서]. `D:\Tools` 처럼 공백 없는 경로를 쓴다.

```powershell
# 1) 내려받기 (약 수백 MB, 풀면 약 1.3GB)
New-Item -ItemType Directory -Force D:\Tools\opensearch_dl
cd D:\Tools\opensearch_dl
curl.exe -LO https://artifacts.opensearch.org/releases/bundle/opensearch/3.8.0/opensearch-3.8.0-windows-x64.zip
curl.exe -LO https://artifacts.opensearch.org/releases/bundle/opensearch/3.8.0/opensearch-3.8.0-windows-x64.zip.sha512

# 2) 체크섬 확인: 두 값이 같아야 한다
(Get-FileHash .\opensearch-3.8.0-windows-x64.zip -Algorithm SHA512).Hash.ToLower()
Get-Content .\opensearch-3.8.0-windows-x64.zip.sha512

# 3) 압축 풀기 → D:\Tools\opensearch-3.8.0
tar -xf .\opensearch-3.8.0-windows-x64.zip -C D:\Tools
```

### 1.2 설정

`D:\Tools\opensearch-3.8.0\config\opensearch.yml` 끝에 덧붙인다. 원본은 `.orig` 로 백업해 두었다.

**(가) 이 PC 측정용 — 보안 끔, 이 PC 에서만 접속** [이 PC 실측]

```yaml
cluster.name: decide-rag-bench
node.name: bench-1
discovery.type: single-node
network.host: 127.0.0.1
http.port: 9200
# 로컬 측정용이라 보안 플러그인을 끈다. 127.0.0.1 에만 열리므로 다른 PC 에서는 접속할 수 없다.
plugins.security.disabled: true
```

**(나) 보안을 켜고 쓰는 경우** [공식 문서]

- `plugins.security.disabled` 줄을 **지운다**. `false` 로 바꾸지 말고 줄 자체를 지워야 한다. 데모 설치 스크립트가 같은 설정을 다시 쓰면서 "Duplicate Object property" 오류가 날 수 있다.
- 다른 PC 에서 접속하게 하려면 `network.host: 0.0.0.0` 으로 둔다. 이 경우에는 반드시 보안을 켠다.

**힙 메모리** — `config\jvm.options` [이 PC 실측: 2g]

```
-Xms2g
-Xmx2g
```

### 1.3 구동

**보안 끔(이 PC 방식)** [이 PC 실측]

```powershell
cmd /c "cd /d D:\Tools\opensearch-3.8.0 && set PATH=%PATH%;D:\Tools\opensearch-3.8.0\plugins\opensearch-knn\lib && bin\opensearch.bat"
```

- 이 PC 에서는 k-NN 플러그인의 `lib` 폴더를 PATH 에 넣고 실행했다. 공식 Windows 문서에는 이 단계가 없다.
- 콘솔 로그를 파일로 남기려면 명령 끝에 `> D:\Tools\opensearch-3.8.0\logs_console.txt 2>&1` 을 붙인다.
- 창을 닫으면 OpenSearch 도 꺼진다.

**보안 켬(데모 인증서·admin 계정)** [공식 문서]

```powershell
$env:OPENSEARCH_INITIAL_ADMIN_PASSWORD = "<관리자 비밀번호>"
cd D:\Tools\opensearch-3.8.0
.\opensearch-windows-install.bat
```

CMD 에서는 `set OPENSEARCH_INITIAL_ADMIN_PASSWORD=<관리자 비밀번호>` 로 지정한다. 비밀번호를 스크립트나 저장소 파일에 적지 않는다.

**Windows 서비스로 등록** — 미확인

`bin` 폴더에 `opensearch-service.bat`, `opensearch-service-x64.exe`, `opensearch-service-mgr.exe` 가 있다 [이 PC 실측]. 공식 Windows 설치 페이지는 이 파일들을 설명하지 않는다. 이 PC 에서 서비스 등록은 해 보지 않았다.

### 1.4 동작 확인

```powershell
# 보안 끔
curl.exe http://127.0.0.1:9200
curl.exe "http://127.0.0.1:9200/_cluster/health?pretty"

# 보안 켬 (데모 인증서는 자체 서명이라 --insecure 가 필요하다)
curl.exe -X GET https://localhost:9200 -u "admin:<관리자 비밀번호>" --insecure
curl.exe -X GET "https://localhost:9200/_cat/plugins?v" -u "admin:<관리자 비밀번호>" --insecure
```

이 PC 의 응답은 `"number" : "3.8.0"`, `"build_type" : "zip"`, `"lucene_version" : "10.5.0"` 이었다 [이 PC 실측].

### 1.5 중지

| 방법 | 설명 |
|---|---|
| 실행 창에서 `Ctrl+C` 또는 창 닫기 | 공식 방법 [공식 문서] |
| 백그라운드로 띄운 경우 | 아래처럼 OpenSearch 의 java 프로세스를 찾아 종료한다 |

```powershell
# OpenSearch 가 쓰는 java 프로세스 찾기 (명령줄에 opensearch 가 들어 있는 것)
Get-CimInstance Win32_Process -Filter "Name='java.exe'" |
  Where-Object { $_.CommandLine -like "*opensearch*" } |
  Select-Object ProcessId, CommandLine
```

```powershell
# 찾은 PID 로 종료
Stop-Process -Id <PID>
```

### 1.6 제거

OpenSearch 를 끈 뒤 폴더를 지우면 된다. 레지스트리나 서비스는 쓰지 않는다(서비스로 등록하지 않은 경우).

```powershell
Remove-Item -Recurse -Force D:\Tools\opensearch-3.8.0
```

색인 데이터는 `D:\Tools\opensearch-3.8.0\data` 안에 있으므로 폴더와 함께 지워진다 [이 PC 실측].

---

## 2. Linux

Linux 내용은 모두 **[공식 문서]** 기준이며 이 PC 에서 실행하지 않았다.

### 2.1 운영체제 설정 (설치 전, 모든 방식 공통)

```bash
# 1) mmap 개수 한도: 운영 환경은 최소 262144
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
cat /proc/sys/vm/max_map_count     # 262144 확인

# 2) 스왑 끄기 (또는 opensearch.yml 에 bootstrap.memory_lock: true)
sudo swapoff -a
```

- 열 수 있는 파일 수는 최소 65,535 이다.
- tar 설치라면 `/etc/security/limits.conf` 에 다음을 넣는다.

```
opensearch - nofile 65535
opensearch - memlock unlimited     # bootstrap.memory_lock: true 를 쓸 때만
```

- 확인은 `ulimit -n`, `ulimit -l` 로 한다.

### 2.2 설치 방식 A — tar.gz (배포판과 무관)

```bash
cd /opt
sudo curl -fsSLO https://artifacts.opensearch.org/releases/bundle/opensearch/3.8.0/opensearch-3.8.0-linux-x64.tar.gz
sudo tar -xvf opensearch-3.8.0-linux-x64.tar.gz
sudo mv opensearch-3.8.0 opensearch          # 이후 /opt/opensearch 로 부른다
```

ARM 서버는 `linux-arm64` 파일을 받는다.

**설정 파일:** `/opt/opensearch/config/opensearch.yml`, `/opt/opensearch/config/jvm.options`

```yaml
# 단일 노드 예시 (공식 문서 권장값)
network.host: 0.0.0.0
discovery.type: single-node
plugins.security.disabled: false
```

```
# jvm.options: 8GB 서버 예시
-Xms4g
-Xmx4g
```

**직접 구동 / 중지**

```bash
cd /opt/opensearch

# 보안 켬: 데모 인증서와 admin 계정을 만든 뒤 시작
export OPENSEARCH_INITIAL_ADMIN_PASSWORD='<관리자 비밀번호>'
./opensearch-tar-install.sh

# 보안 끔: opensearch.yml 에 plugins.security.disabled: true 를 넣고
./bin/opensearch

# 중지: 실행 터미널에서 Ctrl+C
```

**systemd 서비스로 등록**

```bash
sudo adduser --system --shell /bin/bash -U --no-create-home opensearch
sudo usermod -aG opensearch $USER
sudo chown -R opensearch /opt/opensearch/
```

`/etc/systemd/system/opensearch.service` 를 만든다.

```ini
[Unit]
Description=OpenSearch
Wants=network-online.target
After=network-online.target

[Service]
Type=forking
User=opensearch
ExecStart=/opt/opensearch/bin/opensearch -d
LimitNOFILE=65535
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

공식 문서의 유닛 예시에서 확인한 항목은 `ExecStart`, `User`, `LimitNOFILE`, `LimitNPROC` 이다. `Type=forking` 과 `[Unit]`·`[Install]` 의 내용은 `-d`(데몬 실행)에 맞춘 일반적인 값이며, 공식 원문과 같은지는 미확인이다.

```bash
sudo systemctl daemon-reload
sudo systemctl enable opensearch.service   # 부팅 시 자동 시작
sudo systemctl start opensearch
sudo systemctl status opensearch
sudo systemctl stop opensearch             # 중지
```

### 2.3 설치 방식 B — Debian / Ubuntu (.deb, APT)

```bash
# 패키지 파일로 설치
sudo env OPENSEARCH_INITIAL_ADMIN_PASSWORD='<관리자 비밀번호>' dpkg -i opensearch-3.8.0-linux-x64.deb
```

```bash
# 또는 APT 저장소로 설치
# (키링 /etc/apt/keyrings/opensearch.gpg,
#  저장소 파일 /etc/apt/sources.list.d/opensearch-3.x.list,
#  주소 https://artifacts.opensearch.org/releases/bundle/opensearch/3.x/apt stable main)
sudo apt-get update
sudo env OPENSEARCH_INITIAL_ADMIN_PASSWORD='<관리자 비밀번호>' apt-get install opensearch
```

- 키 등록 명령은 공식 Debian 페이지의 원문을 따른다. 키 지문은 `A8B2 D9E0 4CD5 1FEF 6AA2 DB53 BA81 D999 8119 1457` 이다.
- 3.7 부터는 테스트·개발 용도로 `DISABLE_SECURITY_PLUGIN=true` 를 주면 보안 플러그인을 끈 채 설치된다.

```bash
sudo env DISABLE_SECURITY_PLUGIN=true dpkg -i opensearch-3.8.0-linux-x64.deb
```

### 2.4 설치 방식 C — RHEL / Rocky / Amazon Linux (.rpm, YUM)

```bash
sudo rpm --import https://artifacts.opensearch.org/publickeys/opensearch-release.pgp
```

```bash
# 패키지 파일로 설치
sudo env OPENSEARCH_INITIAL_ADMIN_PASSWORD='<관리자 비밀번호>' yum install opensearch-3.8.0-linux-x64.rpm
```

```bash
# 또는 YUM 저장소로 설치
sudo curl -SL https://artifacts.opensearch.org/releases/bundle/opensearch/3.x/opensearch-3.x.repo -o /etc/yum.repos.d/opensearch-3.x.repo
sudo yum clean all
sudo yum repolist
sudo env OPENSEARCH_INITIAL_ADMIN_PASSWORD='<관리자 비밀번호>' yum install opensearch
```

보안을 끈 개발용 설치(3.7 이상)는 `sudo env DISABLE_SECURITY_PLUGIN=true yum install ...` 로 한다.

### 2.5 패키지 설치(deb/rpm) 공통 — 경로와 서비스

| 항목 | 경로 |
|---|---|
| 설정 | `/etc/opensearch/opensearch.yml` |
| 힙 | `/etc/opensearch/jvm.options` |
| 데이터 | `/var/lib/opensearch` (Debian 문서 기준) |
| 로그 | `/var/log/opensearch` (Debian 문서 기준) |
| 사용자 계정 | `/etc/opensearch/opensearch-security/internal_users.yml` |
| 보안 도구 | `/usr/share/opensearch/plugins/opensearch-security/tools` (`hash.sh`, `securityadmin.sh`) |
| JDK | `/usr/share/opensearch/jdk` |

```bash
sudo systemctl enable opensearch    # 부팅 시 자동 시작
sudo systemctl start opensearch     # 구동
sudo systemctl status opensearch    # 상태
sudo systemctl restart opensearch   # 설정 변경 후 재시작
sudo systemctl stop opensearch      # 중지 (공식 페이지에 명시는 없음. systemd 표준 명령)
```

메모리 잠금을 쓰면 `sudo systemctl edit opensearch` 에 다음을 넣는다.

```ini
[Service]
LimitMEMLOCK=infinity
```

### 2.6 동작 확인

```bash
curl -X GET https://localhost:9200 -u 'admin:<관리자 비밀번호>' --insecure
curl -X GET "https://localhost:9200/_cat/plugins?v" -u 'admin:<관리자 비밀번호>' --insecure
# 보안 끔
curl -X GET http://localhost:9200
```

---

## 3. 자주 쓰는 API (Windows·Linux 공통)

아래 예시는 보안을 끈 이 PC(`http://127.0.0.1:9200`) 기준이다. 보안을 켰다면 `https://` 와 `-u "admin:..." --insecure` 를 붙인다. Windows PowerShell 에서는 `curl` 대신 `curl.exe` 를 쓴다.

```bash
# 클러스터 상태 (green/yellow/red). 단일 노드에 복제본 0 이면 green
curl "http://127.0.0.1:9200/_cluster/health?pretty"

# 색인 목록·문서 수·크기
curl "http://127.0.0.1:9200/_cat/indices?v"

# 노드 메모리·힙 사용량
curl "http://127.0.0.1:9200/_cat/nodes?v&h=name,heap.percent,ram.percent,cpu"

# 색인 매핑·설정 보기
curl "http://127.0.0.1:9200/chunks_real/_mapping?pretty"
curl "http://127.0.0.1:9200/chunks_real/_settings?pretty"

# 색인 삭제 (되돌릴 수 없다)
curl -X DELETE "http://127.0.0.1:9200/chunks_1m"
```

DecideRAG 측정에서 쓴 색인 설정은 다음과 같다(`tools\opensearch_bench.py` 의 `index_body()`) [이 PC 실측].

```json
PUT /chunks_real
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "refresh_interval": "-1",
    "similarity": { "bm25p": { "type": "BM25", "k1": 1.5, "b": 0.75 } }
  },
  "mappings": {
    "properties": {
      "chunk_id": { "type": "keyword" },
      "doc_id":   { "type": "keyword" },
      "acl":      { "type": "keyword" },
      "tokens":   { "type": "text", "analyzer": "whitespace", "similarity": "bm25p" },
      "text":     { "type": "text", "index": false },
      "title":    { "type": "keyword", "index": false },
      "path":     { "type": "keyword", "index": false },
      "page":     { "type": "integer", "index": false }
    }
  }
}
```

- `tokens` 필드에는 Kiwi 형태소 토큰을 공백으로 이어 넣는다. 그래서 OpenSearch 쪽에서는 공백 분석기만 쓴다.
- 권한은 `acl` 의 `terms` 필터로 건다.
- `refresh_interval: -1` 은 대량 적재 속도를 높이려고 자동 새로고침을 끈 것이다. 적재가 끝나면 `POST /<색인>/_refresh` 를 해야 검색에 보인다. 운영에서 증분 색인을 할 때는 이 값을 다시 정해야 한다.

---

## 4. 이 PC 현재 상태 (2026-11-27)

| 항목 | 값 |
|---|---|
| 설치 경로 | `D:\Tools\opensearch-3.8.0` (약 1.3GB) |
| 내려받은 파일 | `D:\Tools\opensearch_dl\opensearch-3.8.0-windows-x64.zip` (+ `.sha512`) |
| 주소 | `http://127.0.0.1:9200`. 보안 끔, 이 PC 에서만 접속 |
| 힙 | 2g |
| 색인 | `chunks_real`(실제 청크 606개), `chunks_1m`(합성 청크 100만 개, 약 1.08GB) |
| 측정 결과 | 설계서 §12.2.1 |

## 5. 운영 전환 시 체크리스트

- [ ] 운영 OS 확정 (설계서 열린 질문 2)
- [ ] 보안 플러그인 켜기. `plugins.security.disabled` 줄 삭제, 관리자 비밀번호는 환경변수로만 전달
- [ ] 데모 인증서 대신 정식 TLS 인증서
- [ ] 앱 전용 계정 분리. admin 계정을 앱에서 쓰지 않는다
- [ ] Linux 라면 `vm.max_map_count=262144`, 스왑 끄기, nofile 65535
- [ ] 힙 = RAM 의 절반 (`-Xms` = `-Xmx`)
- [ ] 데이터·로그 경로와 백업(스냅샷) 방법 결정 — 미정
- [ ] 실제 어휘의 100만 청크와 동시 사용자 부하로 다시 측정 (설계서 §12.2.1 의 미측정 항목)

## 참고

- 설치 개요·중요 설정: https://docs.opensearch.org/latest/install-and-configure/install-opensearch/index/
- Windows: https://docs.opensearch.org/latest/install-and-configure/install-opensearch/windows/
- tar: https://docs.opensearch.org/latest/install-and-configure/install-opensearch/tar/
- Debian: https://docs.opensearch.org/latest/install-and-configure/install-opensearch/debian/
- RPM: https://docs.opensearch.org/latest/install-and-configure/install-opensearch/rpm/
