.PHONY: help install train predict smoke test clean

DATA ?= data/sample_comments_ok_toxic.csv
OUT ?= runs/comment_cls
NUM_CLASSES ?= 2
EPOCHS ?= 8
BATCH_SIZE ?= 64
TEXT ?= thanks for the help
LABELS ?= ok,toxic

help:
	@echo "Targets:"
	@echo "  install  - Install deps with Poetry (no-root)"
	@echo "  train    - Train on $(DATA) (outputs to $(OUT))"
	@echo "  predict  - Predict one text using $(OUT)/best.pt"
	@echo "  smoke    - Quick 1-epoch smoke run (outputs to runs/_smoke)"
	@echo "  clean    - Remove local run artifacts"

install:
	poetry install --no-root

train:
	poetry run python src/Trainer.py --data_csv "$(DATA)" --num_classes $(NUM_CLASSES) --epochs $(EPOCHS) --batch_size $(BATCH_SIZE) --out_dir "$(OUT)"

predict:
	poetry run python src/Predict.py --model "$(OUT)/best.pt" --tokenizer "$(OUT)/tokenizer.json" --text "$(TEXT)" --labels "$(LABELS)"

smoke:
	poetry run python src/Trainer.py --data_csv "$(DATA)" --num_classes $(NUM_CLASSES) --epochs 1 --batch_size 16 --out_dir runs/_smoke
	poetry run python src/Predict.py --model runs/_smoke/best.pt --tokenizer runs/_smoke/tokenizer.json --text "$(TEXT)" --labels "$(LABELS)"

test:
	poetry run pytest

clean:
	@powershell -NoProfile -Command "if (Test-Path runs) { Remove-Item -Recurse -Force runs }"
