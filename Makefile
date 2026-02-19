# Compiler
CC = clang

# Compiler flags
CFLAGS = -Wall -Wextra -Icnn/headers -std=c99 -g

# Executable name
TARGET = main

# Source files
SRCS = $(wildcard cnn/*.c cnn/implementations/*.c)

# Object files
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile .c files into .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
