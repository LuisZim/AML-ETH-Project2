SHELL := /bin/bash

venv:
	mv .venv .venv-old || true
	rm -rf .venv-old &

	uv venv .venv --system-site-packages
	uv pip compile requirements-ngc.txt -o requirements-ngc-subdeps.txt > /dev/null 2>&1

	sed -i '/^torch==/d' requirements-ngc-subdeps.txt
	sed -i '/^torchvision==/d' requirements-ngc-subdeps.txt


	source .venv/bin/activate && \
	uv pip install --no-deps -r requirements-ngc-subdeps.txt && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchvision; print(f'Torchvision Version: {torchvision.__version__}')"



# uv pip install --no-deps torchvision==0.21.0 && \