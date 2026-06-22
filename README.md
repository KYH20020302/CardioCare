# Quick Start (Reproducibility Guide)

프로젝트를 처음 받은 사용자가 아래 순서대로 실행하면 전체 과정을 재현할 수 있습니다.

## 1. Repository Clone

```bash
git clone https://github.com/KYH20020302/CardioCare.git
cd CardioCare
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Train Models

```bash
python src/train.py
```

실행 결과

- 데이터 분할
- 전처리 수행
- 특성 선택
- Logistic Regression 학습
- SVC 학습
- Random Forest 학습
- 5-Fold Cross Validation
- Hyperparameter Tuning
- MLflow 기록

이 수행됩니다.

## 4. Launch MLflow UI

```bash
mlflow ui
```

브라우저 접속

```text
http://127.0.0.1:5000
```

## 5. Run Inference

```bash
python src/inference.py
```

## 6. Run Monitoring & Drift Detection

```bash
python src/monitor.py
```

생성 파일

```text
inference.log
drift_report.csv
metric_timeseries.png
```

## 7. Run Unit Tests

```bash
python -m unittest tests/test_pipeline.py
```

## 8. Build Docker Image

```bash
docker build -t cardiocare:1.0 .
```

## 9. Run Docker Container

```bash
docker run --rm cardiocare:1.0
```

정상적으로 예측 결과가 출력되면 Docker 패키징이 성공적으로 재현된 것입니다.
