# define base image
FROM python:3.6.12

# set working directory
WORKDIR /app 

# copy 'app' subfolder content to current directory, i.e. '/app'
COPY . .

# run command at build time
RUN pip install -r requirements.txt

# Volume to map during the run process
#VOLUME vol

# Declare Container Ports
EXPOSE 8888

# define Start-up Command, Entrypoint is null
ENTRYPOINT ["/bin/bash"]
#CMD ["/bin/bash"]
