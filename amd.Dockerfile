FROM rocm/pytorch:rocm6.0.2_ubuntu22.04_py3.10_pytorch_2.1.2
RUN pip install --no-cache --no-build-isolation git+https://github.com/darkshapes/sdbx.git
EXPOSE 8188
WORKDIR /workspace
RUN sdbxui --quick-test-for-ci --cpu --cwd /workspace
CMD ["/usr/local/bin/sdbxui", "--listen"]
