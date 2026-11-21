VPP := v++
VITIS_RUN := vitis-run

CONFIG := hls.cfg
WORK_DIR := build

.PHONY: all synth package csim clean

all: package

synth:
	$(VPP) -c --mode hls --config $(CONFIG) --work_dir $(WORK_DIR)

package: synth
	$(VITIS_RUN) --mode hls --package --config $(CONFIG) --work_dir $(WORK_DIR)

csim:
	$(VITIS_RUN) --mode hls --csim --config $(CONFIG) --work_dir $(WORK_DIR)

clean:
	rm -rf $(WORK_DIR)
