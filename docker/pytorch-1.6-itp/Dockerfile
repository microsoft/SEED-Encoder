FROM docker/pytorch:pytorch-1.6

RUN conda install -c r -y conda pip=20.1.1

# ==================================================================
# apex
# ------------------------------------------------------------------
RUN cd $(mktemp -d) && \
git clone -q https://github.com/NVIDIA/apex.git && \
cd apex && \
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--fast_multihead_attn" ./ 


# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*