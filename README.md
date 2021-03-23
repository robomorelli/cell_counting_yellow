# cell_counting_yellow
Starting point for cell_counting_yellow



DOCKER instruction

-Build the image from dockerfile
docker build -t .

-Run the image:
docker run --rm -it -p 8888:8888 -v ${PWD}/DATASET:app/DATASET cell

Run initialize a shell session inside the container where it is possible to run the scripts and a jupyter session with the command:
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root

connect following the instruction displayed on the terminal

run jupyter directly from docker:
docker run -it -p 8888:8888 -v ${PWD}/DATASET:/app/DATASET cell -c "jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root"

connect following the instruction displayed on the terminal
