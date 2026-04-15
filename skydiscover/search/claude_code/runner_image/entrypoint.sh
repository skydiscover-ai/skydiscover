#!/bin/bash
set -euo pipefail

if [ "${DIND:-}" = "1" ]; then
    # Start Docker daemon in background (needed for Docker evaluators).
    dockerd --host=unix:///var/run/docker.sock \
            --storage-driver=overlay2 \
            > /var/log/dockerd.log 2>&1 &
    DOCKERD_PID=$!
    sleep 2
    if ! kill -0 $DOCKERD_PID 2>/dev/null; then
        echo "overlay2 failed, falling back to vfs..." >> /var/log/dockerd.log
        dockerd --host=unix:///var/run/docker.sock \
                --storage-driver=vfs \
                > /var/log/dockerd.log 2>&1 &
        DOCKERD_PID=$!
    fi

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

    # Load evaluator image if provided, start persistent evaluator container.
    if [ -f /workspace/.evaluator-image.tar ]; then
        EVAL_IMAGE=$(docker load < /workspace/.evaluator-image.tar 2>/dev/null | grep -o '[^ ]*$')
        rm -f /workspace/.evaluator-image.tar
        EVAL_CID=$(docker run -d --rm --entrypoint sleep "$EVAL_IMAGE" infinity)
        echo "$EVAL_CID" > /workspace/.evaluator-container-id
    fi

    chown -R claude:claude /workspace 2>/dev/null || true
    chmod -R a+rX /workspace 2>/dev/null || true

    exec su -s /bin/bash claude -c "export HOME=/workspace; $1"
else
    exec "$@"
fi
