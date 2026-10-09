# The gr8 container

One image for GB10 boxes (linux/arm64, CUDA 13, sm_121a). The image holds the TP=4 engine (`bin/`, with the multi-session
path), the TP=6 engine (`bin6/`), the Rust front end (`bin/g6front-rs`), the shims, the scripts and `tools/convert`, on
the CUDA 13 runtime image (which brings cuBLAS) with libibverbs and the mlx5 provider. Each box runs one container:
one rank (`g6m4`) per box, and the head (rank 0) also runs the shim and the front end. The packs stay on the host and
are mounted in.

CI builds this image but does not push it anywhere. Build the image yourself on one Spark (the compile takes a few minutes;
the CUDA base images are several GB to download) and copy it to the others:

    docker build -t gr8 .                                   # from the repo root, or from an unzipped tester zip
    for h in node2 node3 node4; do docker save gr8 | ssh $h docker load; done

## Container requirements

| need | flag |
|---|---|
| the GPU | `--gpus all` (the NVIDIA container toolkit, on DGX OS by default) |
| the RoCE fabric | `--network host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1` |
| the working directory with `packs/`, `s2/`, `out/` | `-v $G6_HOME:$G6_HOME -w $G6_HOME` (the same path on every box, as `config/cluster.env` says) |
| files the host user can read and write | `--user $(id -u):$(id -g)` |

## One rank

The engine settings are the `ENGINE_ENV`, `ARGS`, `SRCARGS` and `DFARGS` lines of `scripts/launch_s2.sh` (TP=4); use
the same values here. `ARGS` carries `--s2-epoch $EPOCH`: pick one `EPOCH` (for example `$(date +%s)`) and give every
rank the same one. On box `r`, after `rm -rf $G6_HOME/s2 && mkdir -p $G6_HOME/s2 $G6_HOME/out`:

    docker run -d --name gr8-rank --gpus all --network host --device /dev/infiniband --cap-add IPC_LOCK \
      --ulimit memlock=-1 --user $(id -u):$(id -g) -v $G6_HOME:$G6_HOME -w $G6_HOME gr8 \
      env $ENGINE_ENV g6m4 --rank $r --ips $IPS --ips2 $IPS2 --proxy-cpu $PROXY_CPU \
        --manifest packs/rank$r.manifest.json --pack packs/rank$r.pack --image --layers all $SRCARGS $ARGS $DFARGS

`docker logs gr8-rank` shows the engine's output. Rank 0 writes `$G6_HOME/s2/ready.s2` when all ranks have loaded.
TP=6 runs `/opt/gr8/bin6/g6m4` with the settings of `scripts/launch_tp6.sh`, and its head runs `frontend/g6shim.py`
in place of `g6s2shim.py`.

## The head: shim and front end

The shim writes requests into the other ranks' `s2/` directories over ssh, so the head container needs the host
user's ssh keys and passwd entry:

    docker run -d --name gr8-head --network host --user $(id -u):$(id -g) -e HOME=$HOME \
      -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v $HOME/.ssh:$HOME/.ssh:ro \
      -v $G6_HOME:$G6_HOME -v $MODEL_DIR:$MODEL_DIR:ro gr8 bash -c "
        mkdir -p $G6_HOME/run
        g6front-rs --model-dir $MODEL_DIR --engine-sock $G6_HOME/run/engine.sock --port 8100 &
        exec python3 frontend/g6s2shim.py --sock $G6_HOME/run/engine.sock \
          --rank-dirs $G6_HOME/s2 node2:$G6_HOME/s2 node3:$G6_HOME/s2 node4:$G6_HOME/s2"

Start it after `ready.s2` exists. Then `curl http://<head>:8100/health`.

## Stopping

`docker stop gr8-head gr8-rank` on each box. If one rank stops, the whole engine stops. Restart every container.

## Not covered

`scripts/launch_tp4.sh`, `launch_s2.sh` and `stop.sh` start bare binaries over ssh. Container support in them is still to do.
The image also runs the pack build (`python3 -m tools.convert.convert ...` with the data directories mounted).
