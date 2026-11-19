# General configuration
ROOT_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

TARGET := hw
PLATFORM := $(ROOT_DIR)/../platform/kv260_custom.xpfm

DESIGN_FILE_NAME := fpga_kernels
KERNEL_FREQ := 200000000

# Directories
TEMP_DIR := ./tmp.$(TARGET)
BUILD_DIR := ./build.$(TARGET)
PACKAGE_OUT := ./package.$(TARGET)

XO_DIR := $(BUILD_DIR)/xo

# Output files
LINK_OUTPUT := $(BUILD_DIR)/$(DESIGN_FILE_NAME).link.xclbin

FPGA_BIN := $(PACKAGE_OUT)/$(DESIGN_FILE_NAME).bit.bin
XCL_BIN := $(PACKAGE_OUT)/$(DESIGN_FILE_NAME).xclbin
SHELL_JSON := $(PACKAGE_OUT)/shell.json
DTB_FILE := $(PACKAGE_OUT)/$(DESIGN_FILE_NAME).dtbo

# Compilation options
VPP := v++
JOBS := 4

VPP_FLAGS := --save-temps --jobs $(JOBS)
VPP_FLAGS += --profile.exec all:all

LINK_CONFIG := hw_link/fpga_kernels-link.cfg

# Kernel directory : HLS top function
KERNELS := \
	Conv2d:conv2d_forward \
	AvgPool2d:avg_pool2d_forward \
	Tanh:tanh_forward \
	Linear:linear_forward \
	PointWiseConv2d:pointwise_conv2d_forward \
	DepthWiseConv2d:depthwise_conv2d_forward \
	ReLU6:relu6_forward \
	LayerAdd:layer_add_forward \
	BatchNorm2d:batchnorm2d_forward

# Kernel information
kernel_dir = $(word 1,$(subst :, ,$(1)))
kernel_top = $(word 2,$(subst :, ,$(1)))

XO_FILES := $(foreach k,$(KERNELS),\
	$(XO_DIR)/$(call kernel_top,$(k)).xo)


# Main targets

.PHONY: all build kernels link clean cleanall

all: build

build: $(XCL_BIN) $(FPGA_BIN) $(SHELL_JSON) $(DTB_FILE)

kernels: $(XO_FILES)

link: $(LINK_OUTPUT)


# HLS kernel compilation

define KERNEL_RULE

$(XO_DIR)/$(2).xo: $(1)/hls.cfg $(1)/Makefile $(wildcard $(1)/HW/*.cpp) $(wildcard $(1)/HW/*.hpp)
	$$(MAKE) -C $(1) package
	mkdir -p $$(XO_DIR)
	cp $(1)/build/$(2)/$(2).xo $$@

endef

$(foreach k,$(KERNELS),$(eval $(call KERNEL_RULE,$(call kernel_dir,$(k)),$(call kernel_top,$(k)))))

# Hardware linking

$(LINK_OUTPUT): $(XO_FILES) $(LINK_CONFIG) $(PLATFORM)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l \
		-t $(TARGET) \
		--platform $(PLATFORM) \
		--clock.defaultFreqHz $(KERNEL_FREQ) \
		$(VPP_FLAGS) \
		--config $(LINK_CONFIG) \
		-o $@ \
		$(XO_FILES)


# Hardware packaging

$(XCL_BIN): $(LINK_OUTPUT)
	mkdir -p $(PACKAGE_OUT)
	cp $< $@

$(FPGA_BIN): $(LINK_OUTPUT)
	mkdir -p $(PACKAGE_OUT)
	cp $(TEMP_DIR)/link/int/system.bit $(PACKAGE_OUT)/system.bit
	printf 'all:{system.bit}\n' > $(PACKAGE_OUT)/bootgen.bif
	cd $(PACKAGE_OUT) && bootgen -w -arch zynqmp \
	    -process_bitstream bin -image bootgen.bif && \
	    mv system.bit.bin $(notdir $(FPGA_BIN))

$(SHELL_JSON): $(ROOT_DIR)/../platform/shell.json
	mkdir -p $(PACKAGE_OUT)
	cp $< $@

$(DTB_FILE): $(ROOT_DIR)/../platform/pl.dtbo
	mkdir -p $(PACKAGE_OUT)
	cp $< $@


# Cleaning

clean:
	rm -rf $(TEMP_DIR)
	rm -rf $(BUILD_DIR)
	rm -rf $(PACKAGE_OUT)
	rm -f *.log *.jou

cleanall: clean
	@for entry in $(KERNELS); do \
		dir=$${entry%%:*}; \
		$(MAKE) -C $$dir clean; \
	done
