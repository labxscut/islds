# From Ubuntu Noble
FROM isl-conda
# All Necessary configs
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
WORKDIR /setup

# Install RStudio server, port 8787
# docker run -p 8787:8787 -v /mnt/e/tmp/:/home/rstudio/data --name rs -d isl-rpy rstudio-server start
RUN apt-get update --fix-missing && \
    apt-get -y install r-base r-base-dev && \
    apt-get -y install gdebi-core && \
    curl -sLo rstudio-server.deb https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2024.04.2-764-amd64.deb && \
    gdebi -n rstudio-server.deb && \
    rm rstudio-server.deb && \
    useradd -m -s /bin/bash rstudio && \
    echo "rstudio:rstudio" | chpasswd && \
    rm -rf /var/lib/apt/lists/*
    
# Install Jupyter Notebook server, port 8888
# docker run -p 8888:8888 -v /mnt/e/tmp:/mnt/tmp --name pnb -d isl-rpy jupyter notebook
ENV PATH=~/miniconda/bin:$PATH
RUN conda create -n islds -y python=3.8 && \
    echo "source activate islds" >> ~/.bashrc && \
    . ~/miniconda/etc/profile.d/conda.sh && \
    conda activate islds && \
    pip install numpy scipy matplotlib notebook

# docker run -p 8787:8787 -p 8888:8888 -v /mnt/e/tmp/:/home/rstudio/data -v /mnt/e/tmp:/mnt/tmp --name rpy -d isl-rpy sleep infinity
# rstudio-server start
# jupyter notebook --allow-root
