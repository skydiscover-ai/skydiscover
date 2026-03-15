#!/bin/bash
set -euo pipefail

if [ "${DIND:-}" = "1" ]; then
    # Start Docker daemon in background (needed for Docker evaluators).
    # Try overlay2 first (fast), fall back to vfs (always works, needed
    # on Docker Desktop for Mac where overlay-on-overlay fails).
    dockerd --host=unix:///var/run/docker.sock \
            --storage-driver=overlay2 \
            > /var/log/dockerd.log 2>&1 &
    DOCKERD_PID=$!
    sleep 2
    if ! kill -0 $DOCKERD_PID 2>/dev/null; then
        echo "overlay2 failed, falling back to vfs storage driver..." >> /var/log/dockerd.log
        dockerd --host=unix:///var/run/docker.sock \
                --storage-driver=vfs \
                > /var/log/dockerd.log 2>&1 &
        DOCKERD_PID=$!
    fi

    # Wait for daemon to be ready.
    timeout=30
    while ! docker info > /dev/null 2>&1; do
        timeout=$((timeout - 1))
        if [ "$timeout" -le 0 ]; then
            echo "ERROR: dockerd failed to start" >&2
            cat /var/log/dockerd.log >&2
            exit 1
        fi
        sleep 1
    done

    # Load evaluator image if a tar was provided, then start a persistent
    # evaluator container. run_eval.sh uses docker exec against it.
    if [ -f /workspace/.evaluator-image.tar ]; then
        EVAL_IMAGE=$(docker load < /workspace/.evaluator-image.tar 2>/dev/null | grep -o '[^ ]*$')
        rm -f /workspace/.evaluator-image.tar
        EVAL_CID=$(docker run -d --rm --entrypoint sleep "$EVAL_IMAGE" infinity)
        echo "$EVAL_CID" > /workspace/.evaluator-container-id
    fi

    # Make workspace writable by the claude user. chown fails on Docker
    # Desktop for Mac volume mounts, so fall back to chmod.
    chown -R claude:claude /workspace 2>/dev/null || chmod -R 777 /workspace 2>/dev/null || true

    # Drop to non-root user. The controller writes the command to a
    # script file (.run.sh) to avoid quoting issues with su -c.
    exec su -s /bin/bash claude -c "export HOME=/workspace; $1"
else
    # Simple mode: just run the command (caller used --user).
    exec "$@"
fi
