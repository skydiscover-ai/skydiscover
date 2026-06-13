# Evaluator image for the kvstore family (portable tier).
#
# Builds the candidate against the shared C++ harness and runs the
# build -> 6-test gate -> benchmark pipeline. The harness files are copied in at
# build time from benchmarks/kvstore/_harness (populated from the runtime — see
# the family README). The hardware tier (NVMe/NUMA/cgroup/perf) runs on bare
# metal instead, not in this container.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /benchmark

# Evaluator scripts.
COPY evaluator.py wrapper.py evaluate.sh /benchmark/
RUN chmod +x /benchmark/evaluate.sh

# Shared harness (build context must include _harness/ — see how-to in the family
# README; the docker build is invoked with the kvstore family as context). The
# evaluator finds it via SKYKV_HARNESS_DIR.
COPY _harness/ /benchmark/_harness/
ENV SKYKV_HARNESS_DIR=/benchmark/_harness

ENTRYPOINT ["/benchmark/evaluate.sh"]
