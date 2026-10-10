# The gr8 container

One image for GB10 boxes (linux/arm64, CUDA 13, sm_121a). The image holds the TP=4 engine (`bin/`, with the multi-session
path), the TP=3 engine (`bin3/`), the TP=6 engine (`bin6/`), the Rust front end (`bin/g6front-rs`), the shims and the
whole repository, on the CUDA 13 runtime image (which brings cuBLAS) with libibverbs and the mlx5 provider. Each box
runs one container: one rank (`g6m4`) per box, and the head (rank 0) also runs the shim and the front end. The model
cache stays on the host and is mounted in. `./gr8 glm53 up` starts bare binaries; running it inside the container is
not tested yet, so the steps below start the containers by hand.

CI builds this image but does not push it anywhere. Build the image yourself on one Spark (the compile takes a few minutes;
the CUDA base images are several GB to download) and copy it to the others:

    docker build -t gr8 .                                   # from the repo root, or from an unzipped tester zip
    for h in node2 node3 node4; do docker save gr8 | ssh $h docker load; done

## Container requirements

| need | flag |
|---|---|
| the GPU | `--gpus all` (the NVIDIA container toolkit, on DGX OS by default) |
| the RoCE fabric | `--network host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1` |
| the working directory (`s2/`, `out/`) and the model cache | `-v $G6_HOME:$G6_HOME -v $MODEL_CACHE:$MODEL_CACHE -w $G6_HOME` (the same paths on every box) |
| files the host user can read and write | `--user $(id -u):$(id -g)` |

## One rank

The engine's environment and arguments are the `engine:` line that `./gr8 glm53 up` writes to `out/supervisor.log` on
each box (fill the cache first with `./gr8 glm53 prepare`). Its `--s2-epoch` must be the same on every rank: pick one
(for example `$(date +%s)`). On box `r`, after `rm -rf $G6_HOME/s2 && mkdir -p $G6_HOME/s2 $G6_HOME/out`:

    docker run -d --name gr8-rank --gpus all --network host --device /dev/infiniband --cap-add IPC_LOCK \
      --ulimit memlock=-1 --user $(id -u):$(id -g) -v $G6_HOME:$G6_HOME -v $MODEL_CACHE:$MODEL_CACHE -w $G6_HOME gr8 \
      env $ENGINE_ENV g6m4 --rank $r --ips $IPS --ips2 $IPS2 --proxy-cpu $PROXY_CPU $ENGINE_ARGS

`docker logs gr8-rank` shows the engine's output. Rank 0 writes `$G6_HOME/s2/ready.s2` when all ranks have loaded.
TP=3 runs `/opt/gr8/bin3/g6m4`. TP=6 runs `/opt/gr8/bin6/g6m4`, and its head runs `frontend/g6shim.py` in place of
`g6s2shim.py`.

## The head: shim and front end

The shim writes requests into the head's own `s2/` directory, and the head engine passes them to the other ranks:

    docker run -d --name gr8-head --network host --user $(id -u):$(id -g) \
      -v $G6_HOME:$G6_HOME -v $MODEL_CACHE:$MODEL_CACHE:ro gr8 bash -c "
        mkdir -p $G6_HOME/run
        g6front-rs --model-dir $TOKENIZER_DIR --engine-sock $G6_HOME/run/engine.sock --port 8100 &
        exec python3 frontend/g6s2shim.py --sock $G6_HOME/run/engine.sock --dir $G6_HOME/s2"

`TOKENIZER_DIR` is the release's `tokenizer/` directory in the cache (the `--model-dir` of the front end's line in
`out/supervisor.log`). Start it after `ready.s2` exists. Then `curl http://<head>:8100/health`.

## Stopping

`docker stop gr8-head gr8-rank` on each box. If one rank stops, the whole engine stops. Restart every container.

## Not covered

`./gr8 glm53 up` does not start containers yet.
