cmake_minimum_required(VERSION 4.0)
project(cnn C)

set(CMAKE_C_STANDARD 99)

add_executable(cnn main.c
        headers/convlayer.h
        implementations/convlayer.c
        headers/pool.h
        implementations/pool.c
        headers/dense.h
        implementations/dense.c
        headers/activation.h
        implementations/activation.c
        headers/utils.h
        implementations/utils.c
        headers/backprop.h
        implementations/backprop.c
        headers/cnn.h
        implementations/cnn.c
)
