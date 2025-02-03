#
# module.mk — Minimal Makefile to build all .cc files in the kernels directories
#

# Directories where sources are located
KERNELS_DIR  := tensorflow/lite/micro/kernels
KERNELS_AMBIQ := $(KERNELS_DIR)/ambiq

# Collect all .cc files
SRCS := $(wildcard $(KERNELS_DIR)/*.cc) \
        $(wildcard $(KERNELS_AMBIQ)/*.cc)

# Convert .cc to .o
OBJS := $(SRCS:.cc=.o)

# Compiler settings (adjust to your toolchain if needed)
CXX := arm-none-eabi-gcc
CXXFLAGS := -I. -O2 -Wall
AR       := ar

TARGET_LIB := libtensorflow-microlite-$(TFP)-gcc$(GCC_VERSION).a
# Default target
all: $(TARGET_LIB)

# Archive all object files into a static library
$(TARGET_LIB): $(OBJS)
	@echo "Creating static library $@"
	@$(AR) rcs $@ $(OBJS)

# Compile .cc -> .o
%.o: %.cc
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Cleanup
clean:
	rm -f $(OBJS) $(TARGET_LIB)
