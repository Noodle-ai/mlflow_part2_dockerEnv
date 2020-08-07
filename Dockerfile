FROM continuumio/miniconda3

# Create the environment using conda
RUN conda install -c anaconda jupyter=1.0.0
RUN conda install -c conda-forge mlflow=1.8.0
RUN conda install -c anaconda scikit-learn=0.22.1
RUN conda install -c anaconda psycopg2=2.8.5
RUN conda install -c anaconda boto3=1.14.12

# Set up SSH
RUN apt-get update && apt-get install -y openssh-server
RUN useradd -m -s /bin/bash dockeruser
RUN mkdir /var/run/sshd
RUN echo 'dockeruser:123' | chpasswd
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]