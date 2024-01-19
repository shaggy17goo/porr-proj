CC = gcc
CFLAGS = -Wall -Werror
#CFLAGS = -Wall -O3 -msse2 -mfpmath=sse -ffast-math
LDFLAGS = -lm -lpthread -fopenmp
OUT_DIR = out
BUILD_DIR = build
TARGET = $(OUT_DIR)/matrixfun

all: create_dir $(TARGET)

$(TARGET): $(BUILD_DIR)/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.c | create_dir
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

.PHONY: clean run create_dir

create_dir:
	@mkdir -p $(OUT_DIR) $(BUILD_DIR)

clean:
	rm -rf $(OUT_DIR) $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

profile: CFLAGS += -pg
profile: clean all
