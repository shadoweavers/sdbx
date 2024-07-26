FROM intel/intel-optimized-pytorch:2.3.0-pip-base
RUN pip install --no-cache --no-build-isolation git+https://github.com/darkshapes/sdbx.git
EXPOSE 8188
WORKDIR /workspace
RUN sdbxui --quick-test-for-ci --cpu --cwd /workspace
CMD ["/usr/local/bin/sdbxui", "--listen"]