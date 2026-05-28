# Vehicle Counter

YOLO + ByteTrack 기반 실시간 차량 감지, 추적 및 카운팅 시스템.
CCTV 영상에서 차량의 진입 방향(직진/좌회전/우회전)과 종류(승용차/버스/트럭)를 구분하여 카운트합니다.

## 요구사항

- Python 3.10 (Windows 64bit)
- YOLO 가중치 파일 (`yolov8s.pt`, 동봉)

## 설치 방법

### 인터넷 환경

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 폐쇄망 (오프라인) 환경

`wheels/` 폴더에 Python 3.10 Windows 64bit용 wheel 파일이 포함되어 있습니다.

```bash
# 1. 저장소를 USB 등으로 폐쇄망 PC에 복사

# 2. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate

# 3. wheel 파일로 오프라인 설치
pip install --no-index --find-links=wheels -r requirements.txt
```

> **참고**: `yolov8s.pt` 가중치 파일이 프로젝트 루트에 포함되어 있습니다.
> 폐쇄망에서는 Ultralytics가 자동으로 가중치를 다운로드할 수 없으므로, 반드시 `--weights` 옵션으로 로컬 파일 경로를 지정하세요.

## 사용 방법

### 1. 존(Zone) 설정

차량 카운팅에 사용할 감지 영역을 설정합니다.

```bash
# 저장된 이미지 사용
python zone_config.py --source road_image.jpg

# 화면 캡처 사용
python zone_config.py --source screen --screen-top 200 --screen-left 100 --screen-width 800 --screen-height 600
```

3개의 존을 순서대로 그립니다:
1. **3시 도로 (Entry)** — 차량 진입 감지 영역
2. **11시 도로 (Origin)** — 좌회전 출발지
3. **7시 도로 (Origin)** — 우회전 출발지

조작법:
- **마우스 좌클릭**: 다각형 꼭짓점 추가
- **R**: 현재 존 초기화
- **Enter**: 현재 존 확정
- **S**: 저장 (3개 존 완료 후)
- **Q**: 취소

결과는 `zones.json`에 저장됩니다.

### 2. 차량 카운팅 실행

#### 동영상 파일

```bash
python vehicle_counter.py --source video.mp4 --weights yolov8s.pt
```

#### RTSP 스트림

```bash
python vehicle_counter.py --source rtsp://username:password@192.168.1.10:554/stream1 --weights yolov8s.pt
```

#### 화면 캡처 (모니터에 띄운 CCTV 영상)

```bash
# 전체 화면
python vehicle_counter.py --source screen --weights yolov8s.pt

# 특정 영역 (왼쪽 상단 좌표 + 크기 지정)
python vehicle_counter.py --source screen --weights yolov8s.pt ^
    --screen-top 200 --screen-left 100 --screen-width 1200 --screen-height 800
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--source` | 영상 소스 (파일 경로, RTSP URL, 또는 `screen`) | 필수 |
| `--weights` | YOLO 가중치 파일 경로 | `yolov8s.pt` |
| `--y-line-ratio` | 카운팅 라인 수직 위치 (0.0~1.0) | `0.5` |
| `--lane-divider-ratio` | 차선 구분선 수평 위치 (0.0~1.0) | `0.5` |
| `--screen-top` | 화면 캡처 영역: 상단 좌표 (px) | 전체 화면 |
| `--screen-left` | 화면 캡처 영역: 좌측 좌표 (px) | 전체 화면 |
| `--screen-width` | 화면 캡처 영역: 너비 (px) | 전체 화면 |
| `--screen-height` | 화면 캡처 영역: 높이 (px) | 전체 화면 |
| `--db-url` | SQLAlchemy DB URL (선택) | 없음 |
| `--api-url` | 카운트 이벤트 전송 API URL (선택) | 없음 |

### 조작 키

- `q` : 프로그램 종료

## 프로젝트 구조

```
vehicle_counter/
├── vehicle_counter.py     # 메인 차량 카운팅 프로그램
├── zone_config.py         # 존 설정 도구 (개발 예정)
├── zones.json             # 존 설정 파일 (zone_config.py로 생성)
├── yolov8s.pt             # YOLO 가중치 (Git LFS)
├── road_image.jpg         # 참조용 도로 이미지
├── requirements.txt       # Python 의존성 목록
├── wheels/                # 오프라인 설치용 wheel 파일 (Python 3.10/Win64)
└── docs/                  # 설계 문서
```
