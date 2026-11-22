.PHONY: install data pipeline test notebook all clean stream-demo stream-live dashboard docker-up docker-down

install:
	pip install -r requirements.txt
	pip install -e .

data:            ## regenerate the synthetic raw dataset
	python -m pipeline_anomaly.data_generation

pipeline:        ## data -> validation -> features -> models -> evaluation -> figures
	python -m pipeline_anomaly.pipeline

test:
	pytest

notebook:        ## rebuild and execute the notebook
	python scripts/build_notebook.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/pipeline_anomaly_detection.ipynb

all: pipeline test notebook

stream-demo:     ## finite simulated stream with measured latency summary (~4 min)
	python -m pipeline_anomaly.streaming.run_demo

stream-live:     ## continuous simulated stream for the dashboard (Ctrl+C to stop)
	python -m pipeline_anomaly.streaming.run_demo --live

dashboard:       ## Streamlit dashboard (run stream-live in another terminal)
	streamlit run dashboard/app.py

docker-up:       ## producer + processor + dashboard in containers (http://localhost:8501)
	docker compose up --build

docker-down:
	docker compose down -v

clean:
	rm -f data/processed/*.csv models/*.joblib
	rm -rf runtime
