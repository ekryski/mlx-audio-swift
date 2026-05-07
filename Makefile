# Makefile — Build orchestrator for mlx-audio-swift benchmarks
#
# Benchmarks must run release-mode so compile optimizations don't skew RTF /
# latency / memory metrics. This Makefile encapsulates the release build so the
# shell driver (`scripts/benchmark.sh`) doesn't have to repeat the flags on
# every iteration of the (model × quant × workload) sweep.
#
# Quick reference:
#   make              — full release build of test targets (default)
#   make build-tests  — same; explicit target name
#   make build-debug  — debug build (for local iteration on benchmark code)
#   make clean        — remove .build (keeps SPM checkouts)
#   make status       — show what's built

SHELL := /bin/bash
.DEFAULT_GOAL := build-tests

PROJECT_ROOT := $(CURDIR)
BUILD_DIR    := $(PROJECT_ROOT)/.build

CONFIG       ?= release
SWIFT_FLAGS  ?= -Xswiftc -enable-testing

.PHONY: build-tests build-debug clean status

# Release build of all test targets. Re-runs SPM, which is fast on a warm
# .build dir and deterministic — the benchmark driver runs `swift test
# --skip-build` afterward so the binaries are reused across permutations.
#
# `-enable-testing` is required so smoke-test files that use
# `@testable import` keep compiling in release mode. Without it, SPM's
# release mode strips internal-symbol visibility and the test target fails
# to link against the audio library targets.
build-tests:
	swift build --build-tests -c $(CONFIG) $(SWIFT_FLAGS)

build-debug:
	swift build --build-tests -c debug $(SWIFT_FLAGS)

clean:
	rm -rf $(BUILD_DIR)

status:
	@echo "BUILD_DIR: $(BUILD_DIR)"
	@if [ -d "$(BUILD_DIR)/arm64-apple-macosx/release" ]; then \
		echo "release: built"; \
	else \
		echo "release: not built — run 'make build-tests'"; \
	fi
	@if [ -d "$(BUILD_DIR)/arm64-apple-macosx/debug" ]; then \
		echo "debug:   built"; \
	else \
		echo "debug:   not built"; \
	fi
