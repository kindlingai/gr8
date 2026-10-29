"""Launch options: the command line, one config file and the environment, in that order of precedence.

Every option has a KEY (the name in a config file and the environment) and a --flag. A config file is `@FILE` on the
command line, else $G6_CLUSTER, else ./cluster.env, else <repo>/config/cluster.env, else none (--no-config skips
it). The file is parsed, not sourced: `KEY=value` lines, `#` comments, optional `export`, quotes stripped, no `$`.

Scopes decide what the handshake compares. `cluster` options must be equal on every box (they reach every rank's
engine, or the cluster's shape). `box` options are this box's own (paths, the proxy CPU), and `run` options only change
how this launcher waits or checks.
"""
import hashlib
import ipaddress
import os
import re
import socket
import struct
import subprocess

import models

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# key: (scope, default, meaning). A default of None means unset, or the engine's default for the TP
# (scripts/engine_args.sh). The flag is --<key in lower case, _ as ->, except where FLAGS says otherwise.
KEYS = {
    'GR8_MODEL': ('cluster', None, 'the model: glm53 (GLM-5.3) or glm53f (GLM-5.3-Flash). ./gr8 <model> sets it; the '
                                   'image reads it from docker/.env. up, check, prepare and show-config refuse to run '
                                   "without it, and MODEL must be a release of it. ./gr8 models lists each model's "
                                   'release, TP sizes and variants'),
    'TP': ('cluster', None, '4, 6 or 3 (glm53f: 4 or 2); needed with mentat. Default: the number of peers.'),
    'SHAPE': ('cluster', 'mesh', "mesh (a switch), or ring (TP=3 or 4, no switch). With --peers, each box's f0 port "
                                 "is cabled to the next box's f1 port, in --peers order. With mentat (mentatd 0.24.0 "
                                 'or later on every box), the boxes claim a ring from mentat, which solves it once '
                                 'over the cables its probes find, on any cabling. They take the rank order and their '
                                 '--ring-devs from that one view, and refuse the boot, naming the pair, when the '
                                 "links form no ring. --head and the model set's offers turn the ring. Takes no --ips"),
    'PEERS': ('cluster', None, "the boxes' LAN addresses in rank order, the same on every box. The first is the head. "
                               'HOSTS in an old config file is read as PEERS'),
    'MENTAT': ('box', None, 'find the boxes through the local mentat daemon (the default when PEERS is unset). '
                            'Default: localhost.'),
    'GROUP': ('cluster', None, 'the mentat group the boxes join. Default: gr8-<model>.'),
    'HEAD': ('cluster', None, "mentat mode: the head's LAN address, or a name that resolves to it (a "
                              "LeaderWorkerSet's LWS_LEADER_ADDRESS, deploy/k8s), which the boxes wait for until "
                              'JOIN_TIMEOUT. Default: the lowest address.'),
    'IPS': ('cluster', None, "each rank's rail-0 and rail-1 fabric address. Without them each box finds its own two "
                             'RoCE v2 addresses. Default: discovered.'),
    'IPS2': ('cluster', None, "each rank's rail-0 and rail-1 fabric address. Without them each box finds its own two "
                              'RoCE v2 addresses. Default: discovered.'),
    'CONTROL_PORT': ('cluster', '7340', "the supervisors' TCP port on the LAN (handshake, status, logs, down)"),
    'PORT': ('cluster', '8100', 'the OpenAI API on the head'),
    'S2PORT': ('cluster', None, 'the multi-session control port on the head'),
    'MODEL': ('box', None, 'the exlr8 release: https://huggingface.co/<org>/<repo>[@rev] (each box fetches its part '
                           'into MODEL_CACHE), or a local copy of it (dense/, experts/, index-slices/, manifests/, '
                           'the tokenizer files) used in place, on a local disk or a shared mount such as NFS. The '
                           "engine plans every rank's weights from the release's files (V2_DENSE)"),
    'MODEL_CACHE': ('box', None, "with an https:// MODEL: this box's local directory for its rank's part of the "
                                 'release, and for the converted drafter. With FETCH=model-loader, the model-loader '
                                 'store: gr8 reads models/<name>/<revision>/ in place and writes nothing there, so it '
                                 'may be a read-only mount'),
    'FETCH': ('box', 'legacy', 'with an https:// MODEL: who fetches the weights. legacy: gr8 itself, from Hugging '
                               "Face, into MODEL_CACHE. model-loader: the box's model-loader "
                               '(kindlingai/model-loader). gr8 puts the pieces its rank needs (g6load fetch-plan) as '
                               'the request gr8-<model> on MODEL_LOADER, waits until it is held, and checks every '
                               "expert half's xxhash64 before boot. The request pins those pieces while it stands, "
                               "and is put again at every boot. With DRAFTER, the drafter's own files come through a "
                               'second request (gr8-<model>-drafter) and each box converts its rank under GR8_HOME: '
                               'model-loader never holds a converted drafter. With STEERING_PROFILE=hf:NAME, the '
                               "steering index and that one artifact come through the release's request, and the "
                               'profile is unpacked under GR8_HOME. Setting MODEL_LOADER alone selects model-loader, '
                               "as model-loader's own deploy trees (deploy/compose/gr8.yml, "
                               'deploy/spark-os/gr8.override.yml, deploy/k8s/examples/gr8) do; FETCH=legacy with '
                               'MODEL_LOADER is an error'),
    'MODEL_LOADER': ('box', None, "FETCH=model-loader: model-loader's local request API, unix:PATH or host:port. On "
                                  'the socket, its file mode is the boundary. host:port needs the token model-loader '
                                  "serves TCP with, as MODEL_LOADER_LOCAL_TOKEN in gr8's environment (kept out of "
                                  'argv); gr8 sends it to the socket as well when set. The token authenticates gr8 to '
                                  'model-loader over plain HTTP, so keep TCP on the box or a link you trust. No '
                                  'weights cross it: gr8 reads them from the store, and the revision model-loader '
                                  'answers must name a directory under it. Default: '
                                  'unix:<MODEL_CACHE>/model-loader.sock.'),
    'MODEL_LOADER_DRAFTER_MODEL': ('box', None, "FETCH=model-loader with DRAFTER: the drafter's name in "
                                                "model-loader's catalog. Unset, gr8 asks model-loader for the entry "
                                                "whose Hugging Face source is DRAFTER's repo, as for "
                                                "MODEL_LOADER_MODEL. The revision is DRAFTER's @rev, else the "
                                                "catalog's only revision of it. Default: looked up."),
    'MODEL_LOADER_MODEL': ('box', None, "FETCH=model-loader: the release's name in model-loader's catalog. Unset, gr8 "
                                        'asks model-loader (GET /catalog) for the entry whose Hugging Face source is '
                                        "MODEL's repo, so the catalog may name it anything and no box is told the "
                                        'name. Set it only for a catalog entry with another kind of source, or a '
                                        "model-loader too old to list its catalog (then the default is the repo's "
                                        "name). The revision is MODEL's @rev, else the catalog's only revision of it. "
                                        'A commit in MODEL must be the commit model-loader resolved the revision to. '
                                        'Default: looked up.'),
    'MODEL_DIR': ('box', None, 'the tokenizer files, when the release directory has none'),
    'G6_HOME': ('box', REPO, 'working directory: bin/, out/, s2/, run/'),
    'PROXY_CPU': ('box', '19', "CPU core of this rank's fabric proxy thread. The other proxies take the 7 cores below "
                               "it, and the ring's relay thread the one below those"),
    'RING_DEVS': ('box', None, "--shape ring: this box's RDMA devices as relay-out,next,relay-in,prev. With mentat, "
                               'the devices behind its addresses toward the next and previous ranks '
                               '(/sys/class/infiniband); with --peers, '
                               'rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1. An explicit list wins. Default: from '
                               "mentat's probes, or the GB10 CX-7 names."),
    'NODE_IP': ('box', None, "this box's LAN address, when several peers are addresses of this box. Default: found."),
    'PY': ('box', None, 'the python that runs the shim'),
    'FRONT_BIN': ('box', None, 'the Rust front end binary'),
    'DEFAULT_MAX_TOKENS': ('box', '0', "the head's front end: max_tokens for a chat request that names none. 0 = what "
                                       'keeps the request within SHARE_LIMIT, but at least 8,192, and never more than '
                                       'the room the context leaves; another value is used as it is'),
    'DEFAULT_REASONING_EFFORT': ('box', None, "the head's front end: reasoning_effort for a chat request that names "
                                              'none: minimal, low, medium, high, xhigh or max, which the template '
                                              'hears as the nearest level it names (README, "Thinking"). Unset = the '
                                              "template's default (max on GLM 5.3). none is refused: GLM 5.3 cannot "
                                              'turn thinking off'),
    'DEFAULT_THINKING_BUDGET': ('box', '0', "the head's front end: thinking_budget for a chat request that names "
                                            "none, or names 0 or null (TensorFold's rule). The front end closes the "
                                            "think block at the reply's Nth token. 0 = no limit. A request with "
                                            'logprobs gets no default budget'),
    'ALLOW_THINKING_OFF': ('box', '0', "experimental. 1 = the head's front end takes enable_thinking: false and "
                                       'reasoning_effort: "none" instead of refusing them: the prompt drops its '
                                       'Reasoning Effort turn and closes the think block, so the reply starts on '
                                       "content. GLM 5.3's official template always opens <think>; a closed block "
                                       'gave degenerate output on an earlier serving stack, so it is off by default '
                                       '(frontend/README.md)'),
    'RANK': ('run', None, 'prepare: the rank to fetch, when this box cannot tell (mentat mode)'),
    'JOIN_TIMEOUT': ('run', '600', 'seconds for the mentat group to fill, for a worker to reach the head, and on the '
                                   'head from the first hello to the last (0 = forever)'),
    'READY_TMO': ('run', None, 'seconds to wait for the ranks to load'),
    'CHECK': ('run', '1', "0 = skip this box's checks (the model files are checked anyway)"),
    'CAP': ('cluster', None, 'KV pool in tokens, and so the longest context. Lower it first when a box runs short of '
                             'memory. Unset, it follows SHARE_LIMIT: each rank keeps the same KV memory (2,280 pages '
                             'of 64 tokens at TP=4, 2,026 at TP=6, 1,941 at TP=3)'),
    'KV_FORMAT': ('cluster', None, 'experimental. The KV latent row: fp8 (656 B) or int4 (416 B, INT4 after a '
                                   'Hadamard rotation). int4 runs bin/g6m4-int4 (in the image; KV_FORMAT=int4 ./gr8 '
                                   "glm53 build builds it), and its default CAP keeps fp8's memory a rank: 415,232 at "
                                   'TP=4 (2,390 pages a rank, measured against fp8 at 1,920; the fp8 default is 2,280 '
                                   'now), 345,088 at TP=3 and 857,600 at TP=6 (estimates). At TP=4 it decodes about '
                                   '3% slower and prefills about 6% faster, and finds the same needles. docs/KV4.md'),
    'SHARE_LIMIT': ('cluster', None, 'requests up to this many tokens (prompt + max_tokens) share decode steps; a '
                                     'longer one waits for the others and runs alone. Every rank holds the first '
                                     'SHARE_LIMIT tokens of a session and a share of the rest, so a higher limit '
                                     'leaves a shorter longest context: at TP=4, 65536 gives CAP 387072, 98304 gives '
                                     '288768. A multiple of 64 from 4096; a limit off the 2048 doubling is a decode '
                                     'bucket boundary of its own'),
    'SESS': ('cluster', None, 'requests decoding at once (1-8); every reachable step graph is captured at boot (about '
                              '2.5 GiB at 8)'),
    'MTP': ('cluster', None, 'MTP draft count (TP=4: for sessions that share steps, or with DFLASH=0; TP=6 and TP=3: '
                             'every round)'),
    'DFLASH': ('cluster', None, '1 = the DFlash2 drafter (--drafter, or dflash.r<r>.* in the model set); 0 = MTP '
                                "only. TP=6: the drafter's body runs whole on every rank (its 8 KV heads do not split "
                                'six ways; packs converted with --nr 6, about 2.6 GB a rank, and 1.3 GB more with '
                                'DFL_BODY_I4=1), drafting what TP=4 drafts. TP=3 refuses 1. Default: TP=4 1 with a '
                                'drafter, else 0, TP=6 0, TP=3 0 (fixed)'),
    'DFLASH_SEL': ('cluster', None, "1 = the DFlash2 selector walk, 0 = the drafter's argmax"),
    'DFLASH_K': ('cluster', None, 'verify at most the first k of the 7 drafts'),
    'SPEC_ADAPT': ('cluster', None, "1 = choose each round's draft count from step cost and acceptance"),
    'SPEC_LEVELS': ('cluster', None, 'the draft counts SPEC_ADAPT may choose, e.g. 7,5,3,2'),
    'SPEC_ACCEPT': ('cluster', None, 'rejection = rejection sampling of drafts at temperature > 0'),
    'REUSE': ('cluster', None, '1 = --reuse-prefix: a request that continues an earlier one re-prefills only the new '
                               'part. 0 = off'),
    'FLASH_CACHE': ('cluster', None, 'glm53f, experimental: terminal = aligned plus a checkpoint at the end of every '
                                     'request, NOT cold-exact (docs/FLASH.md). aligned = session checkpoints, so a '
                                     "prompt that starts with an earlier prompt's tokens is prefilled from that "
                                     "prompt's last 8,192-row grid point, bit-exact (GR8_FLASH_CACHE, docs/FLASH.md). "
                                     'The store is the one KV_STORE (no persist=on), else memory. On glm53f, KV_STORE '
                                     'needs it. The prefill checkpoint is taken at the boundary after the quantum '
                                     "that reaches the grid point, once that boundary's events have committed, so a "
                                     'store Resize that lands there comes first. Default: off.'),
    'FLASH_STATE_BUDGET_MIB': ('cluster', None, 'FLASH_CACHE with no KV_STORE: the in-memory checkpoint store a rank, '
                                                'in MiB (GR8_FLASH_STATE_BUDGET_MIB). Default: 8192.'),
    'KV_STORE': ('cluster', None, 'experimental. Block stores below the KV pool (--kv-store, given once per store, or '
                                  'stores separated by ; in a file or the environment), tried in order: '
                                  'STORE,size=N(M|G|T)[,persist=on|off][,keep=system|sessions|all][,name=NAME][,cache=NAME][,group=NAME[,cap=N(M|G|T)][,weight=N]], '
                                  'and caches as entries of their own, cache:NAME,kind=spare_ram[,floor=N(M|G)]. '
                                  'group= (on remote:) makes the stores of a cluster one group lease that grows to '
                                  'cap= and gives to other clusters on the same node (KV_SHARE_WHEN). keep= is that '
                                  "store's KV_KEEP: "
                                  'ssd:PATH,size=4G,persist=on,keep=system;remote:auto,size=4G,keep=all keeps system '
                                  'prompts on the SSD and every other evicted run (with its tail past SHARE_LIMIT) on '
                                  "the KV node. STORE is ssd:PATH, local SSD offload (a folder on each box's own SSD: "
                                  'rank r uses PATH/r<r>), or remote:HOST:PORT / remote:auto, a KV node on another '
                                  'machine (auto: the one the mentat group announces, ./gr8 kv up). An SSD store is '
                                  'cleared at every boot. With persist=on its system prefixes and the counts survive '
                                  'restarts while the model, layout, block format and engine build stay the same; a '
                                  'boot that served wrong output would hand them on, so persist only a store you '
                                  'trust. cache=NAME puts a cache in spare RAM in front of a store (not with group=): '
                                  'each block it writes is also copied into memory the kernel may take back when it '
                                  "needs it (MADV_FREE), while MemAvailable stays above the cache's floor= (default "
                                  '2G). A read that misses caches what it read, so on '
                                  'cache:spare,kind=spare_ram;ssd:PATH,size=4G,persist=on,keep=system,cache=spare the '
                                  'system prompts kept from an earlier boot reach RAM on first use. name= names a '
                                  'store, by default its kind, or kind-N when the kind repeats. A read whose copy the '
                                  'kernel took, wholly or in part, reads the store, so the cache never changes what '
                                  'is read, only how fast (engine/kvman-rs/src/ram.rs). Sizes are per rank, G = 2^30. '
                                  '@FILE reads the same settings from a YAML file (engine/kvman-rs/REMOTE-TIER.md, '
                                  'Configuration). The two forms accept the same stores, and the boot log prints '
                                  'both. Needs REUSE. engine/kvman-rs/REMOTE-TIER.md'),
    'KV_KEEP': ('cluster', None, 'what the stores keep. system: each system prompt with its tool definitions (the '
                                 "span the front end sends; without one, the prompt's grid cells), once, after "
                                 'KV_SYSTEM_AFTER conversations started with it. sessions: those, plus the runs the '
                                 'pool evicts that continued their conversation at least once, stored on the system '
                                 "prefix they start with, a run past SHARE_LIMIT with each rank's shard of its tail. "
                                 'all: every run the pool evicts, for benchmarks. system suits most; sessions suits '
                                 "agent fleets larger than the KV pool. On the cluster's 8-agent replay (TP=4) "
                                 'sessions cut TTFT p50 from 234 s to 126 s writing 98 GB/h a rank, all was about the '
                                 'same, and system wrote 1.2 GB/h (TTFT p50 213 s); engine/kvman-rs/REMOTE-TIER.md '
                                 'has the table. In kvman-sim on the opencode captures system writes under 4 GB/h per '
                                 'rank and all 90-135'),
    'KV_SYSTEM_AFTER': ('cluster', None, 'the conversations a system prefix must start before the stores keep it. A '
                                         'request that continues a live, retained or stored run counts toward nothing'),
    'KV_SHARE_WHEN': ('cluster', None, 'a KV_STORE group= lease (a KV node shared by clusters): idle = the engine '
                                       "offers its blocks to other groups only while it has no session, so a give's "
                                       'compaction never pauses a turn; always = whenever the node asks. A group '
                                       'starting below its floor is served either way (GR8_KV_SHARE_WHEN, '
                                       'engine/kvman-rs/REMOTE-TIER.md Sharing a node)'),
    'KV_WRITE_LIMIT': ('cluster', None, 'a cap on store writes: at most this many page blocks per page of KV the '
                                        'engine computes (prefill and decode), so 0.25 writes at most a quarter of '
                                        'what the engine fills. Counted the same on every rank, never by the clock; a '
                                        'store refuses what would go past it'),
    'KV_ROWS': ('cluster', None, "experimental. Remote attention rows: each rank's attention rows past SHARE_LIMIT go "
                                 'to a rows store as their pages fill, and decode reads back the rows its selection '
                                 'picks (--kv-rows): '
                                 'STORE,size=N(M|G|T)[,local=PAGES][,pool=SLOTS][,place=PAGES][,fmt=fp8|int4]. STORE '
                                 'is remote:auto (a KV node the mentat group announces, odd ranks on its second rail, '
                                 "as KV_STORE), remote:HOST:PORT or mem:. size is each rank's lease; with local a "
                                 "session waits until its tail's records fit it. local=PAGES: release, a session's "
                                 'pages past SHARE_LIMIT cycle through that many local pages a rank, so its tail no '
                                 "longer needs local memory; place=PAGES then caps the main model's pages a rank, so "
                                 'CAP may pass what they hold. pool is the resident slots per indexer group (with '
                                 'local it defaults to what the sessions past SHARE_LIMIT that place holds can select '
                                 'in a step, and a smaller value admits fewer of them at once), fmt=int4 stores 416 B '
                                 "rows for 656. Sets the engine's G6_RT_* (docs/SETTINGS.md)"),
    'KV_TIER': ('cluster', None, 'deprecated, for one release: KV_TIER is KV_STORE with keep=evicted|shared|all per '
                                 'tier, and maps to KV_KEEP=all when any tier kept evicted runs (the old default), '
                                 'else system; KV_SHARED_AFTER is KV_SYSTEM_AFTER. A warning says so'),
    'KV_SHARED_AFTER': ('cluster', None, 'deprecated, for one release: KV_TIER is KV_STORE with '
                                         'keep=evicted|shared|all per tier, and maps to KV_KEEP=all when any tier '
                                         'kept evicted runs (the old default), else system; KV_SHARED_AFTER is '
                                         'KV_SYSTEM_AFTER. A warning says so'),
    'FLASH_KV_DONATE_GIB': ('box', None, 'once the serve has booted, each Flash box starts ./gr8 kv up --gb N '
                                         '--lazy-lock --ledger out/kv/ledger with N = this, or less when MemAvailable '
                                         'then - FLASH_KV_GROWTH_GIB - KV_MIN_FREE_GB is less (flash_donor_budget, '
                                         'whole GiB). No budget, or a node that fails, leaves the serve as it is; the '
                                         "serve's stop stops the node. 0 = no node. Measured effect: not measured. "
                                         'Default: 0.'),
    'FLASH_KV_GROWTH_GIB': ('box', None, 'with FLASH_KV_DONATE_GIB: the GiB the serve may still allocate after boot. '
                                         "Required: the Flash serve's growth after boot is not measured yet"),
    'FLASH_KV_GROUP': ('box', None, 'with FLASH_KV_DONATE_GIB: the mentat group the KV node joins, the cluster that '
                                    "borrows from it (the GLM-5.3 serve's), not the Flash serve's. Required: a "
                                    'donation without it is refused at boot'),
    'FLASH_CAP_POLICY': ('cluster', None, "protected: the configured CAP's memory is never lent; the host ledger "
                                          'counts only what is lent. borrow-idle (needs FLASH_KV_DONATE_GIB): the '
                                          'Flash KV pool in chunks (GR8_FLASH_KV_CHUNKS=1); once the serve has been '
                                          'idle FLASH_KV_IDLE_S, every chunk but one (16,384 tokens) is lent to the '
                                          'KV node, and a request that needs more waits for their recall before its '
                                          'admission (its TTFT includes the recall). The ledger books the lendable '
                                          "chunks as the GPU's. Measured effect: not measured. Default: protected."),
    'FLASH_KV_IDLE_S': ('cluster', None, 'with FLASH_CAP_POLICY=borrow-idle: idle seconds before a lend '
                                         '(GR8_FLASH_KV_LEND_MS). Not measured. Default: 60.'),
    'PF_OWN': ('cluster', None, 'prefill attention ownership (GR8_PF_OWN); 3 also loads packs/qo.pack. TP=6 with '
                                'PF_OWN=1 needs no QO pack. TP=5: 3'),
    'QP_ROLL': ('cluster', None, 'PF_OWN=4: rolling pair-weight slots (GR8_PF_QP_ROLL)'),
    'QP_CTX': ('cluster', None, 'PF_OWN=4: QO key-table size; a longer request runs OWN=2'),
    'QP_EARLY': ('cluster', None, "PF_OWN=4: send the counterpart's partial before the own rows' o_proj"),
    'DFL_REUSE': ('cluster', None, "1 = a request whose prefix-reuse hit skipped into the DFlash2 drafter's window "
                                   'keeps the drafter; 2 = also a request alone on MTP takes it (GR8_DFL_REUSE). '
                                   'Defaults to 2 when serving DFlash2 with MTP. Needs DFLASH=1'),
    'DFL_REUSE_MIN': ('cluster', None, 'DFL_REUSE: context rows the drafter must hold before it drafts; fewer = MTP '
                                       'warm-up rounds (GR8_DFL_REUSE_MIN)'),
    'MTP_KSTOP': ('cluster', None, "a request's solo MTP rounds chain and verify the drafts the MTP head is confident "
                                   "in: 1 = thresholds (MTP_KSTOP_TAU), 2 = the drafter choice's MTP estimates "
                                   '(GR8_MTP_KSTOP). Needs MTP 2+; without DFLASH (TP=6, TP=3): 1 only, for a request '
                                   'decoding alone, MTP_KSTOP_KMAX at most MTP. Default: on (2) when serving with '
                                   'DFlash2 + MTP 2+ at TP=4'),
    'MTP_KSTOP_KMAX': ('cluster', None, 'MTP_KSTOP: the most drafts a solo MTP round chains, 1..3 '
                                        '(GR8_MTP_KSTOP_KMAX)'),
    'MTP_KSTOP_TAU': ('cluster', None, "MTP_KSTOP=1: thresholds t1,t2,t3 on the drafts' cumulative probability "
                                       '(GR8_MTP_KSTOP_TAU)'),
    'DRAFT_PAR': ('cluster', None, "a solo DFlash2 round's drafter step on its own stream under MTP's first form, 1, "
                                   'or 2 = drafting steps at high priority (GR8_DRAFT_PAR; bit-exact). Needs '
                                   'DFLASH=1. Default: on (2) when serving with DFlash2 + MTP at TP=4'),
    'QA_SPLIT': ('cluster', None, 'decode: the layer-input GEMVs split by output tiles across the ranks and '
                                  'all-gathered (GR8_QA_SPLIT, 1 or 2; bit-exact). Measured at TP=4 only. Default: '
                                  'TP=4 on (2) when TP>1, TP=6 unset (0), TP=3 unset (0)'),
    'IDXQ_SPLIT': ('cluster', None, "decode: the indexer's wq_b split the same way (GR8_IDXQ_SPLIT; bit-exact). "
                                    'Default: TP=4 on (1) when TP>1, TP=6 unset (0), TP=3 unset (0)'),
    'MTP_SPLIT': ('cluster', None, "MTP draft steps: eh_proj and the draft layer's input GEMVs split the same way "
                                   '(GR8_MTP_SPLIT; bit-exact). Default: TP=4 on (1) when TP>1 + MTP, TP=6 unset (0), '
                                   'TP=3 unset (0)'),
    'DRAFT_HEAD_I4': ('cluster', None, "1 = the MTP draft's lm_head on INT4 g32 copies made at boot "
                                       '(GR8_DRAFT_HEAD_I4; drafts only, +134 MB a rank at TP=4). Default: TP=4 1 '
                                       '(with MTP 1+), TP=6 unset (0), TP=3 unset (0)'),
    'DFL_HEAD_I4': ('cluster', None, "1 = the DFlash2 drafter's lm_head on the same INT4 records (GR8_DFL_HEAD_I4; "
                                     'drafts only). Needs DFLASH=1'),
    'DFL_BODY_I4': ('cluster', None, "1 = the DFlash2 drafter's body GEMVs on INT4 g32 copies (GR8_DFL_BODY_I4; "
                                     'drafts only, +382 MB a rank). Needs DFLASH=1'),
    'ROUTE_OVL': ('cluster', None, 'decode MoE layers of 2+ rows: the trellis shared expert on a forked stream during '
                                   'the router, 1 = fc1 + fc2, 2 = fc1 (GR8_ROUTE_OVL; bit-exact). Measured at TP=4 '
                                   'only'),
    'ATTN_SC2': ('cluster', None, 'decode attention scores with S = 1, 2, 4 or 8 key slots per warp (GR8_ATTN_SC2; '
                                  'bit-exact)'),
    'ATTN_SC2_TR': ('cluster', None, 'with ATTN_SC2: the transpose reduction, 1 or 2 = eager (GR8_ATTN_SC2_TR; '
                                     'bit-exact)'),
    'ATTN_CTX2': ('cluster', None, 'decode attention context kernel variant 1..4 (GR8_ATTN_CTX2; bit-exact)'),
    'ATTN_CTX3': ('cluster', None, 'decode attention context with one test per batch of slots, 1..4 (GR8_ATTN_CTX3; '
                                   'bit-exact)'),
    'ATTN_CTX3_SP': ('cluster', None, 'with ATTN_CTX3: 64 or 128 key slots per context partition instead of 256 '
                                      '(GR8_ATTN_CTX3_SP; deterministic, not bit-exact)'),
    'ATTN_POSIDX': ('cluster', None, '1 = decode steps in the buckets up to 2048 list the keys by position instead of '
                                     'running the indexer, which selects all of them there (GR8_ATTN_POSIDX; '
                                     'deterministic, not bit-exact)'),
    'ATTN_SEL2': ('cluster', None, "decode index selection's final sort as a rank by count over 8, 16, 32 or 64 CTAs "
                                   'per row (GR8_ATTN_SEL2; bit-exact)'),
    'IDX_GRP': ('cluster', None, 'S2 batched decode: the indexer reads each key tile once for up to 1..64 rows of a '
                                 'session (GR8_IDX_GRP; bit-exact)'),
    'ATTN_QT': ('cluster', None, '1 = q and rope stored lane-transposed for the scores kernel (GR8_ATTN_QT; needs '
                                 'ATTN_SC2 2 or 4; bit-exact)'),
    'ATTN_FRONT': ('cluster', None, "1 = a decode step's first attention kernels in one launch (GR8_ATTN_FRONT; "
                                    'bit-exact)'),
    'IDX_VEC': ('cluster', None, "1 = the decode indexer's vector loads (GR8_IDX_VEC)"),
    'DEC_BK': ('cluster', None, 'decode context bucket in tokens: the step graphs are captured per bucket '
                                '(GR8_DEC_BK; 0 = the engine default 2048)'),
    'S2_ADAPT': ('cluster', None, "S2: each round's MTP draft count from the sessions' accept rates and step costs "
                                  '(GR8_S2_ADAPT; drafts only; about 1 GiB more step graphs). Default: TP=4 1 (with '
                                  'MTP 1+), TP=6 unset (0), TP=3 unset (0)'),
    'S2_DECODE_PF': ('cluster', None, 'S2: while a session decodes, a prefill quantum takes at most this many rows on '
                                      'one lane (GR8_S2_DECODE_PF; a multiple of 64, at most --prefill)'),
    'S2_ADMIT_HOLD': ('cluster', None, 'S2: a burst of queued requests is admitted at one boundary instead of one per '
                                       'notice (GR8_S2_ADMIT_HOLD)'),
    'PF_SELX_CTX': ('cluster', None, 'prefill index tiles at context >= this select with selx instead of SEL32, same '
                                     'ids in the same order (GR8_PF_SELX_CTX; 0 = SEL32)'),
    'DSEL_MODEL': ('cluster', None, "1 = the drafter choice prices both drafters from DSEL_TABLE and a request's MTP "
                                    'rounds as the confidence stop runs them (GR8_S2_DSEL_MODEL; drafts only: the '
                                    'target decides every token). Needs DFLASH=1'),
    'DSEL_TABLE': ('cluster', None, 'DSEL_MODEL: the cost table, a key:value list (v1..v8, d, m, e, p, optional g, k, '
                                    'n, w, r) or a file holding one, measured on the cluster with '
                                    'tools/s2_dsel_table.py (GR8_S2_DSEL_TABLE)'),
    'NGRAM': ('cluster', None, 'a request decoding alone with the DFlash2 drafter also drafts from this n-gram table '
                               "file: a round verifies the table's chain instead of the drafter's drafts when its "
                               'calibrated probabilities promise more tokens per verify ms (GR8_NGRAM; drafts only: '
                               'the target decides every token; #56). Write the file with tools/ngram/ngram.py '
                               'export. Needs DFLASH=1'),
    'NGRAM_KMAX': ('cluster', None, 'NGRAM: the longest chain a round verifies, 1..--dflash-k (GR8_NGRAM_KMAX)'),
    'NGRAM_KMIN': ('cluster', None, 'NGRAM: the shortest chain worth a round, 1..NGRAM_KMAX (GR8_NGRAM_KMIN)'),
    'NGRAM_MARGIN': ('cluster', None, 'NGRAM: a chain must promise this much more tokens per verify ms than the '
                                      "drafter's drafts (GR8_NGRAM_MARGIN)"),
    'MTP_KSTOPB': ('cluster', None, '1 = in rounds that several sessions share, each session verifies its own count '
                                    "of the MTP drafts, from the MTP head's confidence (GR8_MTP_KSTOPB; drafts only: "
                                    'the target decides every token). Captures a main graph per width the rule '
                                    'reaches'),
    'MTP_KSTOPB_KMIN': ('cluster', None, 'MTP_KSTOPB: the least drafts a session verifies, 0 or 1 '
                                         '(GR8_MTP_KSTOPB_KMIN)'),
    'MTP_KSTOPB_TAU': ('cluster', None, "MTP_KSTOPB: thresholds t1,t2,... (up to 7) on each session's cumulative "
                                        'draft probability, at every session count (GR8_MTP_KSTOPB_TAU)'),
    'MTP_KSTOPB_TAU_S': ('cluster', None, 'MTP_KSTOPB: thresholds per session count, n:t1,t2,... entries separated by '
                                          '/, e.g. 1:0.7,0.7/2:0.7,0.8 (GR8_MTP_KSTOPB_TAU_S<n>)'),
    'L2PF': ('cluster', None, 'decode: a paced L2 prefetch of the weights the step reads next, on a side stream in '
                              'the windows where DRAM is idle; window mask 1 = MLP all-reduce wait, 2 = gather waits, '
                              '4 = attention, 8 = attention all-reduce wait (GR8_L2PF; bit-exact). Measured at TP=4 '
                              'only, as 15'),
    'CP_STEP_ROWS': ('cluster', None, 'S2: a decode step past SHARE_LIMIT carries at most this many rows, and the CP '
                                      'fabric is sized for it instead of for the widest S2 step (SESS x (MTP + 1) = '
                                      '24 rows, 1.358 GB pinned per rank at TP=4; 8 rows: 0.453 GB). While a session '
                                      "past the limit decodes and the sessions' rows do not fit, they take turns, "
                                      'each once per cycle. At least the rows of a solo step (the boot names the '
                                      'minimum, 8 with DFlash2) (GR8_CP_STEP_ROWS)'),
    'CP_STEP_SEG': ('cluster', None, 'with CP_STEP_ROWS: 1 = a round past SHARE_LIMIT takes only sessions past it, so '
                                     'a shorter session never runs the CP attention because a long one is resident '
                                     '(GR8_CP_STEP_SEG)'),
    'CP_ONECHAIN': ('cluster', None, "1 = past SHARE_LIMIT, each rank runs the other ranks' head groups over its keys "
                                     'as one attention call instead of one per group (GR8_CP_ONECHAIN; bit-exact). '
                                     'TP=4 only (the engine refuses it elsewhere), not with GR8_ATTN_FLIP'),
    'KVB': ('cluster', None, "fp8 = kv_b (UK / UV) as the release's block FP8 bytes and scales instead of their BF16 "
                             '(GR8_KVB; half the bytes, bit-exact). Needs V2_DENSE (a release boot)'),
    'PF_GEMMF32': ('cluster', None, "1 = with PF_OWN=3, prefill's row-owner QO o_proj on the FP32-epilogue GEMM "
                                    '(GR8_PF_GEMMF32; byte-exact). TP=6 K3: +1.0 % cold prefill (1693.6 -> 1710.4, '
                                    'one point, inside the 2 % band). Set for the boot'),
    'MTP_LASTROW': ('cluster', None, "1 = an MTP first-form draft step routes only each session's last row through "
                                     "layer 78's routed experts (GR8_MTP_LASTROW; same caches, drafts and "
                                     'log-probabilities). Skips the routed experts of every row but the last; no '
                                     'end-to-end number. In the TP=6 K3 recipe'),
    'DRAFT_HEAD2': ('cluster', None, 'experimental: 1 = the MTP draft head in two stages, a 2-bit screen then the '
                                     'exact MXFP8 logits of its best 64 (GR8_DRAFT_HEAD2; drafts only, +74.5 MB a '
                                     'rank at TP=4); 2 = also the MXFP8 check (measurement). No end-to-end number. In '
                                     'the TP=6 K3 recipe'),
    'MTP_L2PF': ('cluster', None, 'with GR8_L2PF: the paced L2 prefetch in MTP draft steps, 1 = the draft step, 2 = '
                                  'the head waits, 3 = both (GR8_MTP_L2PF; same bytes). TP=6: about 0.1 ms per ~39 ms '
                                  'round, below speedbench resolution. In the TP=6 K3 recipe (3)'),
    'MTP_KSTOPB_WIDTHS': ('cluster', None, 'with GR8_MTP_KSTOPB: the main widths captured besides the multiples of '
                                           'MTP + 1, w1,w2,... (GR8_MTP_KSTOPB_WIDTHS); fewer widths = less graph '
                                           'memory (about 115 MiB a width a rank at TP=4) and a shorter boot.'),
    'TR3_W2ALL': ('cluster', None, '1 = the window-2 fc1 split (KS1x2, WG1) on every MoE layer at M > 1, not only the '
                                   'layers the k2 gate picks (GR8_TR3_W2ALL; bit-exact). TP=4: prose +1.3 %, c4 +2.7 '
                                   'to +3.4 %, exact; not measured at TP=5 or TP=6'),
    'QO_CACHE': ('cluster', None, "PF_OWN=3: a directory; the first boot writes each layer's full-head q_b / o_proj / "
                                  'UK / UV tensors to it and later boots read them (GR8_QO_CACHE; same bits). TP=5: '
                                  'second boot /health at 46 s against 199 s without; 13 GB of disk a box. In the '
                                  'TP=5 T5X recipe'),
    'PF_ISGRT_RG': ('cluster', None, 'prefill index scores: rows per CTA, a multiple of 8 (GR8_PF_ISGRT_RG; same '
                                     'bits). TP=5: 16 = 157K TTFT about -2 %; 32 with PF_ATTN_ROWS=512 = 32K prefill '
                                     '+2.2 %, 157K TTFT -2.8 %. In the TP=5 T5X recipe (32)'),
    'PF_ATTN_ROWS': ('cluster', None, 'prefill attention chunk rows inside a sub-batch (GR8_PF_ATTN_ROWS). TP=5: 512 '
                                      '= about 0.5 GB more free memory. In the TP=5 T5X recipe (512)'),
    'PF_F12': ('cluster', None, '0 = the stock prefill fc1 instead of the fat-CTA k_pf_fc1f (GR8_PF_F12; byte-exact). '
                                'Effect not measured on its own (TP=5 T5X added it with PF_OWN=3 and PF_GEMMF32=1 in '
                                'one step). Default: TP=4 unset (1 when built with PF_F12), TP=6 unset, TP=3 unset'),
    'ATTN_P11': ('cluster', None, '1 = the P11 decode attention tile retune; takes precedence over ATTN_SC2, '
                                  'ATTN_CTX2, ATTN_CTX3 (GR8_ATTN_P11; read before graph capture). Effect not '
                                  'measured'),
    'DTR': ('cluster', None, "0 = the int4 / MXFP8 records instead of the exlr8 release's dense trellis matrices, "
                             'where the layer has both (GR8_DTR; A/B). Effect not measured; inert on pack boots'),
    'PF_AOP': ('cluster', None, '0 = no fc1 A-operand share in prefill (GR8_PF_AOP; byte-exact). Effect not measured. '
                                'Default: TP=4 unset (1 when built with PF_AOP), TP=6 unset, TP=3 unset'),
    'NOHASH': ('cluster', None, '1 = skip the sha256 check of every pack tensor at load (run once without it first)'),
    'G6ENV_X': ('cluster', None, 'more engine environment for every rank ("A=1 B=2")'),
    'G6ARGS_X': ('cluster', None, 'more engine arguments for every rank'),
    'V2': ('box', None, 'exlr8 v2 experts from their raw width files: the width files and the index slices'),
    'V2_SLICES': ('box', None, 'exlr8 v2 experts from their raw width files: the width files and the index slices'),
    'V2_WIDTHS': ('box', None, 'the width manifest as a path: the older spelling of --variant <path>. Giving both is '
                               'refused'),
    'VARIANT': ('cluster', None, 'the exlr8 v2 width manifest: bit225, bit243, bit260, bit325, bit338 (routed experts '
                                 'at about 2.25-3.38 bpw, spent where they lower KL against bf16 the most), home '
                                 '(3.25 bpw), down1 (about 2.25), up1 (about 4.0), or a manifest path. A name is '
                                 'exlr8-manifest-<name>.json in the model set or in V2 (directly or under '
                                 'manifests/). At TP=3, a release without bit225 boots down1. Without V2, only home'),
    'V2_DENSE': ('box', None, 'the exlr8 release directory whose dense/ files give every non-expert weight (g6m4 '
                              '--v2-dense). Default: the release.'),
    'V2_DENSE_FORM': ('cluster', None, "with V2_DENSE: trellis keeps q_b, o_proj and the dense MLP as the release's "
                                       'trellis entries for the dense trellis kernels, so the weights are the '
                                       "release's own. rounded is the fallback: it rounds them to the MXFP8 and int4 "
                                       'forms of a converted pack. Default: trellis.'),
    'LAYOUT': ('box', None, 'experimental. With V2_DENSE: a layout file that says which pieces each rank holds '
                            '(bin/g6layout, engine/loader-rs/src/layout.rs). Rank r holds what the layout gives its '
                            "r-th member, so list PEERS in the layout's member order. bin/g6layout plan makes one "
                            "from the current layout, keeping each box's pieces where it can. A box reads the slice "
                            "rows of halves the formulas give other ranks, so it needs every rank's index slices at "
                            'the target TP: a local release (--model DIR) must hold them '
                            "(index-slices/tp<N>/k<K>/rank*.safetensors), and with --model https://... the box's "
                            'cache fetches them and the halves the layout gives its rank. Halves the cache already '
                            'holds, from another TP or layout, are not fetched again. GLM 5.3 only: GLM-5.3-Flash '
                            'plans the formulas. Default: the formulas.'),
    'GRAMMAR': ('cluster', '1', '1 = the engine masks response_format json_object / json_schema replies with their '
                                "XGrammar grammar from the think close on, so the reply's content is valid JSON of "
                                'the schema (each rank masks its own vocabulary slice; GLM-5.3 only: a Flash engine '
                                'gets no --grammar-dir, and an engine built without XGrammar serves without grammars. '
                                "scripts/build.sh finds XGrammar through the front end's xgrammar crate). The front "
                                'end checks the reply either way. PROTOCOL.md grammar'),
    'GRAMMAR_DIR': ('box', None, 'with GRAMMAR=1: the directory whose tokenizer.json and generation_config.json the '
                                 "grammars compile against. Default: the model set's tokenizer."),
    'DRAFTER': ('box', None, "TP=4: the DFlash2 drafter's repo, https://huggingface.co/incoai/GLM-5.3-DFlash2. Each "
                             'box fetches it and converts its rank into MODEL_CACHE. Its repo and commit go into the '
                             'id the boxes compare'),
    'DFLASH_DIR': ('box', None, "a directory holding dflash.r<r>.*, when it is not the model set's"),
    'STEERING_PROFILE': ('box', None, 'verified schema-v1 profile directory, or hf:default / hf:NAME from the model '
                                      'release'),
    'STEERING_CHAT_TEMPLATE': ('box', None, 'actual served template file, resolved by the supervisor. Default: '
                                            'resolved model.'),
    'STEERING_FORCE': ('cluster', None, 'force qualification requests to this positive FP32 scale, at most 4'),
    'STEERING_MODEL_REVISION': ('cluster', None, "profile's exact model revision assertion; checked against config "
                                                 'metadata where present'),
    'STEERING_TOKENIZER_REVISION': ('cluster', None, "profile's exact tokenizer revision assertion; checked where "
                                                     'present'),
    'STEERING_CHAT_TEMPLATE_SHA256': ('cluster', None, 'must match the actual served template bytes'),
    'STEERING_WEIGHT_SET': ('cluster', None, 'content identity; explicit mismatches fail. Default: resolved model.'),
}
FLAGS = {'S2PORT': '--s2-port', 'G6_HOME': '--g6-home'}
# Read from old config files and ignored here, so they do not draw a warning.
IGNORED = {'HOSTS', 'PACKS', 'PACK_DIR', 'BIN_DIR', 'G6_CLUSTER', 'G6_NR', 'BUNDLE_LIBS', 'ARCH', 'EXTRA', 'SHARED',
           'RANK_HEALTH', 'PROBE_ARGS', 'QP_PACK', 'PLAIN', 'FRONT'}
SWITCHES = {'--no-config', '--no-check', '--foreground'}
ENGINE_KNOBS = [k for k in list(KEYS)[list(KEYS).index('CAP'):]]
FABRIC_PORT = 18600   # engine/comms/fabric.h: the fabric bootstrap and per-collective ports start here


class OptionError(Exception):
    pass


def flag_of(key):
    return FLAGS.get(key, '--' + key.lower().replace('_', '-'))


KEY_OF_FLAG = {flag_of(k): k for k in KEYS}


def parse_file(path):
    """{KEY: value} from a cluster.env-style file. Raises OptionError naming the line."""
    out = {}
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].lstrip()
            m = re.match(r'([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
            if not m:
                raise OptionError(f'{path}:{n}: not KEY=value: {raw.rstrip()}')
            key, v = m.group(1), m.group(2)
            if v[:1] in ('"', "'"):
                end = v.find(v[0], 1)
                if end < 0:
                    raise OptionError(f'{path}:{n}: unterminated quote')
                rest = v[end + 1:].strip()
                if rest and not rest.startswith('#'):
                    raise OptionError(f'{path}:{n}: text after the closing quote')
                v = v[1:end]
            else:
                v = re.split(r'\s+#', v, maxsplit=1)[0].strip()
            if '$' in v or '`' in v:
                raise OptionError(f'{path}:{n}: {key}: the file is parsed, not sourced, so $ and ` are not expanded')
            out[key] = v
    return out


def default_file(environ):
    for p in (environ.get('G6_CLUSTER'), 'cluster.env', os.path.join(REPO, 'config', 'cluster.env')):
        if p and os.path.isfile(p):
            return p
    return None


class Options:
    """values: {KEY: value}; origin: {KEY: 'command line' | '@file' | 'environment' | 'default'}."""

    def __init__(self, values, origin, file, args, switches, warnings):
        self.values, self.origin, self.file, self.args, self.switches, self.warnings = \
            values, origin, file, args, switches, warnings

    def get(self, key, default=None):
        v = self.values.get(key)
        return default if v is None or v == '' else v

    def int(self, key):
        v = self.get(key)
        try:
            return int(v)
        except (TypeError, ValueError):
            raise OptionError(f'{key}={v!r}: want a whole number')


def parse(argv, environ=None):
    """argv after the command. Returns Options; positional words that are not options stay in .args."""
    environ = os.environ if environ is None else environ
    cmd, at, args, switches = {}, None, [], set()
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith('@'):
            if at:
                raise OptionError('one @FILE only')
            at = a[1:]
        elif a in SWITCHES:
            switches.add(a)
        elif re.match(r'[A-Z][A-Z0-9_]*=', a):
            key, _, val = a.partition('=')
            if key not in KEYS:
                raise OptionError(f'unknown option {key} (in {a}): ./gr8 knobs lists the options')
            cmd[key] = val
        elif a.startswith('--'):
            name, eq, val = a.partition('=')
            key = KEY_OF_FLAG.get(name)
            if key is None:
                raise OptionError(f'unknown option {name}')
            if not eq:
                i += 1
                if i >= len(argv):
                    raise OptionError(f'{name} needs a value')
                val = argv[i]
            if key in ('KV_STORE', 'KV_TIER') and cmd.get(key):   # one store per --kv-store
                val = cmd[key] + ';' + val
            cmd[key] = val
        else:
            args.append(a)
        i += 1
    warnings = []
    path = at or (None if '--no-config' in switches else default_file(environ))
    filev = {}
    if path:
        if not os.path.isfile(path):
            raise OptionError(f'no config file {path}')
        filev = parse_file(path)
        # An old file names the boxes in HOSTS, by ssh name. HOSTS stands for PEERS when PEERS is absent.
        if 'HOSTS' in filev and 'PEERS' not in filev:
            filev['PEERS'] = ','.join(filev['HOSTS'].split())
        for k in sorted(filev):
            if k not in KEYS and k not in IGNORED:
                warnings.append(f'{path}: unknown key {k}, ignored')
    values, origin = {}, {}
    for k, (_, dflt, _) in KEYS.items():
        for src, d in (('command line', cmd), ('@file' if at else 'config file', filev), ('environment', environ)):
            if k in d:
                values[k], origin[k] = d[k], src
                break
        else:
            values[k], origin[k] = dflt, 'default'
    # The mode comes from the strongest source that names one.
    for src, d in (('command line', cmd), ('file', filev), ('environment', environ)):
        has_p, has_m = bool(d.get('PEERS')), bool(d.get('MENTAT'))
        if has_p and has_m:
            raise OptionError(f'the {src} sets both PEERS (--peers) and MENTAT (--mentat): choose one')
        if has_p:
            values['MENTAT'] = None
            break
        if has_m:
            values['PEERS'] = None
            break
    else:
        values['MENTAT'], origin['MENTAT'] = 'localhost', 'default'
    if '--no-check' in switches:
        values['CHECK'], origin['CHECK'] = '0', 'command line'
    # A box that names its model-loader fetches through it: model-loader's own deploy trees (deploy/compose/gr8.yml,
    # deploy/spark-os/gr8.override.yml, deploy/k8s/examples/gr8) set MODEL_LOADER and MODEL_CACHE, not FETCH.
    if origin['FETCH'] == 'default' and values['MODEL_LOADER']:
        values['FETCH'], origin['FETCH'] = 'model-loader', f'MODEL_LOADER, from the {origin["MODEL_LOADER"]}'
    deprecated_kv(values, origin, warnings)
    kv_store_files(values, origin, path)
    model = values['GR8_MODEL']
    if model and model not in models.MODELS:
        raise OptionError(f'GR8_MODEL={model}, from the {origin["GR8_MODEL"]}: not a model gr8 serves. Want '
                          f'{models.CHOICES} (./gr8 models lists them)')
    if model and not values['GROUP']:
        values['GROUP'] = models.MODELS[model].group
    opts = Options(values, origin, path, args, switches, warnings)
    opts.plain = filev.get('PLAIN', environ.get('PLAIN', ''))   # PLAIN=1 is retired. validate() refuses it
    opts.g6env = sorted(k for k in environ if k.startswith('G6_') and k not in KEYS and k not in IGNORED)
    for k in ('MODEL', 'MODEL_CACHE', 'MODEL_DIR', 'G6_HOME', 'V2', 'V2_SLICES', 'V2_WIDTHS', 'VARIANT',
              'DFLASH_DIR', 'V2_DENSE', 'STEERING_PROFILE', 'STEERING_CHAT_TEMPLATE', 'LAYOUT', 'GRAMMAR_DIR'):
        if not values.get(k) or (k == 'MODEL' and looks_remote(values[k])) or \
                (k == 'STEERING_PROFILE' and values[k].startswith('hf:')) or \
                (k == 'VARIANT' and '/' not in values[k] and not values[k].endswith('.json')):
            continue
        # the engine runs in G6_HOME. STEERING_PROFILE may list several directories
        values[k] = ','.join(d if not d or d.startswith('hf:') else os.path.abspath(d) for d in values[k].split(',')) \
            if k == 'STEERING_PROFILE' else os.path.abspath(values[k])
    return opts


def looks_remote(model):
    """A Hugging Face source is an https:// URL. Anything else is a local path."""
    return model.startswith('https://')


def mode(o):
    return 'peers' if o.get('PEERS') else 'mentat'


def validate(o):
    """Refuse what this version cannot boot, with the reason."""
    if not o.get('GR8_MODEL'):
        raise OptionError(f'no model: set GR8_MODEL to {models.CHOICES}, in the config file or the environment, or '
                          'run ./gr8 <model> (./gr8 models lists them)')
    model = models.of(o)
    tp = o.get('TP')
    if tp is None:
        if mode(o) == 'peers':
            tp = str(len(split_list(o.get('PEERS'))))
            o.values['TP'], o.origin['TP'] = tp, 'the number of peers'
        else:
            raise OptionError(f'--tp is needed with --mentat ({model.tp_sizes})')
    if o.g6env:
        raise OptionError(f'{", ".join(o.g6env)} set in the environment: the engine does not inherit G6_* variables. Pass '
                          'them with --g6env-x "NAME=value ..." so every box compares them')
    if o.get('VARIANT') and o.get('V2_WIDTHS'):
        raise OptionError('--variant and --v2-widths both name the width manifest: give one')
    if o.get('DRAFTER') and not o.get('DRAFTER').startswith('https://'):
        raise OptionError(f'--drafter {o.get("DRAFTER")}: want https://huggingface.co/<org>/<repo>[@rev]')
    if not model.drafter and (o.get('DRAFTER') or o.get('DFLASH') == '1'):
        raise OptionError(f'--drafter, --dflash 1: {model.name} has no DFlash2 drafter')
    if o.get('DRAFTER') and tp not in ('4', '6'):
        raise OptionError(f'--drafter: the DFlash2 drafter runs at TP=4 (split) and TP=6 (replicated) only, not TP={tp}')
    if not str(o.get('DEFAULT_MAX_TOKENS')).isdigit():
        raise OptionError(f'--default-max-tokens {o.get("DEFAULT_MAX_TOKENS")}: want a whole number (0 = fit SHARE_LIMIT)')
    effort = o.get('DEFAULT_REASONING_EFFORT')
    if effort == 'none' and not model.thinking_off:
        raise OptionError('--default-reasoning-effort none: GLM 5.3 cannot turn thinking off (use low, or '
                          '--default-thinking-budget)')
    if effort not in (None, 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
        raise OptionError(f'--default-reasoning-effort {effort}: want minimal, low, medium, high, xhigh or max')
    if o.get('GRAMMAR') not in ('0', '1'):
        raise OptionError(f'--grammar {o.get("GRAMMAR")}: want 0 or 1')
    if o.get('ALLOW_THINKING_OFF') not in ('0', '1'):
        raise OptionError(f'--allow-thinking-off {o.get("ALLOW_THINKING_OFF")}: want 0 or 1')
    if not str(o.get('DEFAULT_THINKING_BUDGET')).isdigit():
        raise OptionError(f'--default-thinking-budget {o.get("DEFAULT_THINKING_BUDGET")}: want a whole number '
                          '(0 = no limit)')
    share = o.get('SHARE_LIMIT')
    if share is not None and not (str(share).isdigit() and int(share) >= 4096 and int(share) % 64 == 0):
        # the engine takes any multiple of a KV page (64 tokens) and makes it a decode bucket boundary (context_buckets.h)
        raise OptionError(f'--share-limit {share}: want a multiple of 64 from 4096')
    if o.get('V2') and not o.get('V2_SLICES'):
        raise OptionError('--v2 needs --v2-slices')
    if o.get('V2_DENSE_FORM') not in (None, 'trellis', 'rounded'):
        raise OptionError(f'--v2-dense-form {o.get("V2_DENSE_FORM")}: want trellis or rounded')
    if o.get('KVB') not in (None, 'bf16', 'fp8'):
        raise OptionError(f'--kvb {o.get("KVB")}: want bf16 or fp8')
    # a release boot sets V2_DENSE, or --model names a release (resolve_model sets V2_DENSE after this); the engine refuses
    # KVB=fp8 on a pack boot at start
    if o.get('KVB') == 'fp8' and not (o.get('V2_DENSE') or o.get('MODEL')):
        raise OptionError('--kvb fp8 needs a release boot (--model <release> or --v2-dense): packs hold kv_b in BF16')
    if o.get('LAYOUT') and not os.path.isfile(o.get('LAYOUT')):
        raise OptionError(f'--layout {o.get("LAYOUT")}: no such file')
    if o.get('FETCH') not in ('legacy', 'model-loader'):
        raise OptionError(f'--fetch {o.get("FETCH")}: want legacy or model-loader')
    if o.get('FETCH') == 'legacy' and o.get('MODEL_LOADER'):
        raise OptionError(f'--fetch legacy with --model-loader {o.get("MODEL_LOADER")}: gr8 fetches its own weights then, '
                          'and does not ask model-loader. Unset one')
    if o.plain == '1':
        raise OptionError('PLAIN=1 (the one-request loop) is retired at every TP: unset it')
    steering = o.get('STEERING_PROFILE')
    if o.get('STEERING_FORCE') and not steering:
        raise OptionError('--steering-force needs --steering-profile')
    if steering:
        if o.get('NOHASH') == '1':
            raise OptionError('--steering-profile requires weight hash checks (unset NOHASH)')
        bundled = steering.startswith('hf:')
        if bundled and not re.fullmatch(r'hf:[A-Za-z0-9][A-Za-z0-9._-]{0,127}', steering):
            raise OptionError('--steering-profile: want profile directories, hf:default or hf:NAME')
        dirs = steering.split(',')
        if len(dirs) > 1 and (any(d.startswith('hf:') or not d for d in dirs) or len(dirs) > 8):
            raise OptionError('--steering-profile: a list holds up to 8 profile directories (hf: selects one profile)')
        if len(dirs) > 1 and o.get('STEERING_FORCE'):
            raise OptionError('--steering-force needs exactly one --steering-profile')
        missing = [flag_of(k) for k in ('STEERING_MODEL_REVISION', 'STEERING_TOKENIZER_REVISION',
                                        'STEERING_CHAT_TEMPLATE_SHA256') if not o.get(k)]
        if missing and not bundled:
            raise OptionError('--steering-profile needs ' + ', '.join(missing))
        for key in ('STEERING_CHAT_TEMPLATE_SHA256', 'STEERING_WEIGHT_SET'):
            if o.get(key) and not re.fullmatch(r'[0-9a-f]{64}', o.get(key)):
                raise OptionError(flag_of(key) + ': want 64 lowercase hex digits')
        if o.get('STEERING_FORCE'):
            try:
                force = float(o.get('STEERING_FORCE'))
            except ValueError:
                force = 0.0
            if not (0.0 < force <= 4.0) or struct.unpack('<f', struct.pack('<f', force))[0] == 0:
                raise OptionError('--steering-force: want a finite number in (0, 4]')
    if tp not in model.tp:
        raise OptionError(f'--tp {tp}: want {model.tp_sizes} ({model.name})')
    if o.get('SHAPE') not in ('mesh', 'ring'):
        raise OptionError(f'--shape {o.get("SHAPE")}: want mesh or ring')
    if o.get('SHAPE') == 'ring':
        if tp not in ('3', '4'):
            raise OptionError(f'--shape ring: TP=3 or 4, not {tp}')
        if o.get('IPS'):
            raise OptionError('--shape ring takes no --ips/--ips2: the engine finds its devices by name (--ring-devs) '
                              'and bootstraps over the LAN')
    if o.get('RING_DEVS') and len(split_list(o.get('RING_DEVS'))) != 4:
        raise OptionError(f'--ring-devs {o.get("RING_DEVS")}: want four devices')
    if mode(o) == 'peers' and len(split_list(o.get('PEERS'))) != int(tp):
        raise OptionError(f'--peers names {len(split_list(o.get("PEERS")))} boxes for --tp {tp}')
    for k in ('IPS', 'IPS2'):
        if o.get(k) and len(split_list(o.get(k))) != int(tp):
            raise OptionError(f'{k} has {len(split_list(o.get(k)))} addresses for --tp {tp}')
    if bool(o.get('IPS')) != bool(o.get('IPS2')):
        raise OptionError('IPS and IPS2 go together')
    for k in ('CONTROL_PORT', 'PORT', 'PROXY_CPU', 'JOIN_TIMEOUT'):
        o.int(k)
    stores = kv_stores(o)
    cache = o.get('FLASH_CACHE')
    if cache not in (None, 'off', 'aligned', 'terminal'):
        raise OptionError(f'--flash-cache {cache}: want off, aligned or terminal')
    if cache not in (None, 'off') and model.name != 'glm53f':
        raise OptionError(f'--flash-cache: Flash session checkpoints are glm53f only, not {model.name}')
    budget = o.get('FLASH_STATE_BUDGET_MIB')
    if budget is not None and not (str(budget).isdigit() and int(budget) >= 1):
        raise OptionError(f'--flash-state-budget-mib {budget}: want a whole number of MiB from 1')
    if model.name == 'glm53f' and stores:
        if cache in (None, 'off'):
            raise OptionError('--kv-store on glm53f holds Flash session checkpoints: it needs --flash-cache aligned or terminal')
        if len(stores) > 1:
            raise OptionError('--flash-cache keeps its checkpoints in one store: give one --kv-store')
        if any(t['persist'] for t in stores):
            raise OptionError('--flash-cache: Flash checkpoints do not persist across boots (persist=on)')
    elif stores and o.get('REUSE') == '0':
        raise OptionError('--kv-store keeps the runs prefix reuse retains: it needs REUSE (on by default)')
    if stores:
        if any(t['host'] == 'auto' for t in stores) and mode(o) == 'peers':
            raise OptionError('--kv-store remote:auto finds a KV node through mentat: with --peers, name it (remote:HOST:PORT)')
    rows = kv_rows(o)
    if rows and rows['host'] == 'auto' and mode(o) == 'peers':
        raise OptionError('--kv-rows remote:auto finds a KV node through mentat: with --peers, name it (remote:HOST:PORT)')
    if rows and rows['fmt'] == 'int4' and o.get('KV_FORMAT') == 'int4':
        raise OptionError('--kv-rows fmt=int4: the int4 KV build\'s rows are INT4 already')
    if o.get('KV_FORMAT') == 'int4' and not model.int4_kv:
        raise OptionError(f'--kv-format int4: {model.name} has an fp8 KV build only')
    if o.get('KV_FORMAT') not in (None, 'fp8', 'int4'):
        raise OptionError(f'--kv-format {o.get("KV_FORMAT")}: want fp8 or int4')
    if o.get('KV_SHARE_WHEN') not in (None, 'idle', 'always'):
        raise OptionError(f'--kv-share-when {o.get("KV_SHARE_WHEN")}: want idle or always')
    if o.get('KV_KEEP') not in (None, 'system', 'sessions', 'all'):
        raise OptionError(f'--kv-keep {o.get("KV_KEEP")}: want system, sessions or all')
    sa = o.get('KV_SYSTEM_AFTER')
    if sa is not None and not (str(sa).isdigit() and int(sa) >= 1):
        raise OptionError(f'--kv-system-after {sa}: want a whole number from 1')
    wl = o.get('KV_WRITE_LIMIT')
    if wl is not None and not re.fullmatch(r'\d+(\.\d+)?', str(wl)):
        raise OptionError(f'--kv-write-limit {wl}: want a number of page blocks per page computed, such as 0.25')


def deprecated_kv(values, origin, warnings):
    """KV_TIER and KV_SHARED_AFTER (one release, deprecated) become KV_STORE, KV_KEEP and KV_SYSTEM_AFTER."""
    old = values.get('KV_TIER')
    if old not in (None, 'off') and values.get('KV_STORE') is None:
        texts = [t.strip() for t in old.split(';') if t.strip()]
        keeps = [m.group(1) if (m := re.search(r',keep=(\w+)', t)) else 'evicted' for t in texts]   # the old default
        values['KV_STORE'] = ';'.join(re.sub(r',keep=\w+', '', t) for t in texts)
        origin['KV_STORE'] = origin['KV_TIER']
        if values.get('KV_KEEP') is None:
            values['KV_KEEP'] = 'all' if any(k in ('evicted', 'all') for k in keeps) else 'system'
            origin['KV_KEEP'] = origin['KV_TIER']
        warnings.append(f'KV_TIER (--kv-tier) is deprecated: KV_STORE={values["KV_STORE"]} KV_KEEP={values["KV_KEEP"]}')
    if values.get('KV_SHARED_AFTER') is not None:
        if values.get('KV_SYSTEM_AFTER') is None:
            values['KV_SYSTEM_AFTER'], origin['KV_SYSTEM_AFTER'] = values['KV_SHARED_AFTER'], origin['KV_SHARED_AFTER']
        warnings.append(f'KV_SHARED_AFTER (--kv-shared-after) is deprecated: KV_SYSTEM_AFTER={values["KV_SYSTEM_AFTER"]}')
    values['KV_TIER'] = values['KV_SHARED_AFTER'] = None


# KV_STORE's one-line form: entries separated by ;, each a head then ,key=value settings:
#   cache:NAME,kind=spare_ram[,floor=N(M|G)]
#   ssd:PATH,size=N(M|G|T)[,persist=on|off][,keep=RULE][,name=NAME][,cache=NAME]
#   remote:HOST:PORT|remote:auto,size=N(M|G|T)[,keep=RULE][,name=NAME][,cache=NAME][,group=NAME[,cap=N(M|G|T)][,weight=N]]
# A YAML store file says the same things (kv_store_file), with these key names: {kind: {one-line key: YAML key}}
KV_KEYS = {'cache': {'kind': 'kind', 'floor': 'floor'},
           'ssd': {'size': 'size_per_rank', 'persist': 'persist', 'keep': 'keep', 'name': 'name', 'cache': 'cache'},
           'remote': {'size': 'size_per_rank', 'keep': 'keep', 'name': 'name', 'cache': 'cache', 'group': 'group',
                      'cap': 'cap_per_rank', 'weight': 'weight'}}
CACHE_KINDS = ('spare_ram',)
SIZE = r'\d+(\.\d+)?[MGT]'
WORD = r'[A-Za-z0-9_.-]+'
# in the design but not built: refused by name so a file written for them fails loudly
NOT_BUILT = {'caches': ('takes', 'holds', 'size', 'overflow'), 'tiers': ('transforms', 'tag_restored'),
             'top': ('runs',)}
TYPE_WORDS = {str: 'a string', bool: 'true or false', int: 'a whole number', dict: 'a mapping', list: 'a list'}


class KvError(OptionError):
    """A bad KV_STORE. `where` is ('cache', NAME), ('tier', INDEX) or None, so a store file can name its key."""

    def __init__(self, msg, where=None):
        super().__init__(msg)
        self.where = where


def kv_parse(v):
    """KV_STORE's one-line form parsed and checked: (caches {NAME: {'kind', 'floor'}}, stores in cascade order
    [{'text', 'store', 'kind': 'ssd' | 'remote', 'path', 'host', 'port', 'size', 'persist', 'keep', 'name', 'cache',
    'group', 'cap', 'weight', 'settings': [(key, value) as written]}]). keep None = KV_KEEP; name defaults to the kind.
    Raises KvError naming the entry."""
    caches, stores = {}, []
    for text in [t.strip() for t in (v or '').split(';') if t.strip()]:
        head, *kvs = text.split(',')
        if re.search(r'\s', text):
            raise KvError(f'--kv-store {text}: no spaces')
        settings = [kv.partition('=')[::2] for kv in kvs]
        keys = [k for k, _ in settings]
        if len(set(keys)) != len(keys):
            raise KvError(f'--kv-store {text}: a setting given twice')
        if head.startswith('cache:'):
            name, where = head[6:], ('cache', head[6:])
            if not re.fullmatch(WORD, name):
                raise KvError(f'--kv-store {text}: cache:NAME, NAME of letters, digits and _.-', where)
            if name in caches:
                raise KvError(f'--kv-store {text}: a second cache {name}', where)
            c = dict(settings)
            for k in c:
                if k not in KV_KEYS['cache']:
                    raise KvError(f'--kv-store {text}: {k}: a cache takes kind= and floor=', where)
            if c.get('kind') not in CACHE_KINDS:
                raise KvError(f'--kv-store {text}: kind={" or ".join(CACHE_KINDS)}', where)
            if 'floor' in c and not (re.fullmatch(r'\d+(\.\d+)?[MG]', c['floor']) and float(c['floor'][:-1]) > 0):
                raise KvError(f'--kv-store {text}: floor=N(M|G)', where)
            caches[name] = {'kind': c['kind'], 'floor': c.get('floor')}
            continue
        where = ('tier', len(stores))
        m = re.fullmatch(r'ssd:(.+)', head) or re.fullmatch(r'remote:([^:,]+)(?::(\d+))?', head)
        if not m or (head.startswith('remote:auto') and m.group(2)):
            raise KvError(f'--kv-store {text}: the store is ssd:PATH, remote:HOST:PORT or remote:auto', where)
        t = {'text': text, 'store': head, 'kind': 'ssd' if head.startswith('ssd:') else 'remote', 'path': None,
             'host': None, 'port': None, 'size': None, 'persist': False, 'keep': None, 'name': None, 'cache': None,
             'group': None, 'cap': None, 'weight': None, 'settings': settings}
        if t['kind'] == 'ssd':
            t['path'] = m.group(1)
        else:
            t['host'], t['port'] = m.group(1), m.group(2)
        for k, x in settings:
            if k == 'persist' and t['kind'] != 'ssd':
                raise KvError(f'--kv-store {text}: persist= is for ssd: stores', where)
            if k in ('group', 'cap', 'weight') and t['kind'] != 'remote':
                raise KvError(f'--kv-store {text}: {k}= is for remote: stores', where)
            if k in ('spare_ram_cache', 'spare_ram_floor'):
                raise KvError(f'--kv-store {text}: {k}= is the engine\'s; name a cache: cache:NAME,kind=spare_ram '
                              'and ,cache=NAME on the store', where)
            if k in ('size', 'cap') and re.fullmatch(SIZE, x) and float(x[:-1]) > 0:
                t[k] = x
            elif k == 'persist' and x in ('on', 'off'):
                t['persist'] = x == 'on'
            elif k in ('group', 'name', 'cache') and re.fullmatch(WORD, x):
                t[k] = x
            elif k == 'weight' and re.fullmatch(r'[1-9]\d*', x) and int(x) <= 1000000:
                t['weight'] = int(x)
            elif k == 'keep':
                if x not in ('system', 'sessions', 'all'):
                    raise KvError(f'--kv-store {text}: keep=system, sessions or all (this store\'s KV_KEEP)', where)
                t['keep'] = x
            else:
                raise KvError(f'--kv-store {text}: {k}={x}: want size=N(M|G|T), persist=on|off (ssd:), keep=RULE, '
                              'name=NAME, cache=NAME, or group=NAME[,cap=N(M|G|T)][,weight=N] (remote:)', where)
        if not t['size']:
            raise KvError(f'--kv-store {text}: size=N(G|T) is needed (per rank)', where)
        if (t['cap'] or t['weight']) and not t['group']:
            raise KvError(f'--kv-store {text}: cap= and weight= need group=', where)
        stores.append(t)
    # an unnamed store is named for its kind, or kind-N when its kind repeats
    unnamed = {k: [t for t in stores if t['kind'] == k and not t['name']] for k in ('ssd', 'remote')}
    for k, ts in unnamed.items():
        for n, t in enumerate(ts, 1):
            t['auto'] = k if len(ts) == 1 else f'{k}-{n}'
    names = {}
    for i, t in enumerate(stores):
        name = t['name'] or t['auto']
        if name in names:
            raise KvError(f'--kv-store {t["text"]}: the name {name} is also store {names[name] + 1}\'s', ('tier', i))
        names[name] = i
        t['name'] = name
        if t['cache']:
            if t['cache'] not in caches:
                raise KvError(f'--kv-store {t["text"]}: no cache {t["cache"]} (caches: {", ".join(caches) or "none"})',
                              ('tier', i))
            if t['group']:
                raise KvError(f'--kv-store {t["text"]}: a cache needs a fixed lease (no group=)', ('tier', i))
    for name in caches:
        if not any(t['cache'] == name for t in stores):
            raise KvError(f'--kv-store cache:{name}: no store uses it', ('cache', name))
    if caches and not stores:
        raise KvError('--kv-store: caches but no store')
    return caches, stores


def kv_stores(o):
    """KV_STORE's stores in cascade order, as kv_parse gives them ([] when unset or off)."""
    v = o.get('KV_STORE')
    return [] if v in (None, 'off') else kv_parse(v)[1]


def kv_line(caches, stores):
    """kv_parse's result as KV_STORE's one-line form: caches first by name, then the stores, settings in KV_KEYS
    order."""
    def store(t):
        given = dict(t['settings'])
        return ','.join([t['store']] + [f'{k}={given[k]}' for k in KV_KEYS[t['kind']] if k in given])

    return ';'.join([f'cache:{n},kind={c["kind"]}' + (f',floor={c["floor"]}' if c['floor'] else '') for n, c in sorted(caches.items())] +
                    [store(t) for t in stores])


def kv_yaml(caches, stores):
    """kv_parse's result as a YAML store file (kv_store_text reads it back to the same result)."""
    import yaml

    def q(x):   # a YAML scalar, quoted where YAML would read it as something else
        return yaml.safe_dump(x, default_flow_style=True, width=1 << 20).strip().removesuffix('\n...').removesuffix('...').strip()

    def flow(d):
        return '{' + ', '.join(f'{k}: {q(x)}' for k, x in d.items()) + '}'

    def value(k, x):
        return x == 'on' if k == 'persist' else int(x) if k == 'weight' else x

    lines = []
    if caches:
        lines.append('caches:')
        lines += [f'  {q(n)}: ' + flow({k: c[k] for k in KV_KEYS['cache'] if c[k]}) for n, c in sorted(caches.items())]
    lines.append('tiers:')
    for t in stores:
        given = dict(t['settings'])
        tier = {KV_KEYS[t['kind']][k]: given[k] for k in ('keep', 'name', 'cache') if k in given}
        store = {('path' if t['kind'] == 'ssd' else 'address'): t['store'].split(':', 1)[1]}
        store.update({KV_KEYS[t['kind']][k]: value(k, x) for k, x in t['settings'] if k not in ('keep', 'name', 'cache')})
        head = '- '
        for k, x in tier.items():
            lines.append(f'{head}{k}: {q(x)}')
            head = '  '
        lines += [f'{head}store:', f'    {t["kind"]}: {flow(store)}']
    return '\n'.join(lines) + '\n'


def kv_describe(o):
    """Log lines for KV_STORE: the one-line form, then the same stores as a YAML store file."""
    v = o.get('KV_STORE')
    if v in (None, 'off'):
        return []
    caches, stores = kv_parse(v)
    out = [f'KV_STORE={kv_line(caches, stores)}']
    try:
        out += ['KV_STORE as YAML:'] + ['  ' + line for line in kv_yaml(caches, stores).splitlines()]
    except ImportError:
        out.append('KV_STORE as YAML: needs PyYAML (python3-yaml)')
    return out


def kv_store_with(o, target):
    """KV_STORE with each remote:auto naming the KV node at `target`."""
    caches, stores = kv_parse(o.get('KV_STORE'))
    for t in stores:
        if t['host'] == 'auto':
            t['store'] = f'remote:{target}'
    return kv_line(caches, stores)


def kv_engine(v):
    """KV_STORE as the engine takes it (--kv-store each, g6m4.cu): a store's cache becomes spare_ram_cache=on and
    spare_ram_floor=, and name= and the cache entries go. Settings come in KV_KEYS order."""
    if v in (None, 'off'):
        return v
    caches, stores = kv_parse(v)
    out = []
    for t in stores:
        given = dict(t['settings'])
        parts = [t['store']] + [f'{k}={given[k]}' for k in KV_KEYS[t['kind']] if k in given and k not in ('name', 'cache')]
        if t['cache']:
            floor = caches[t['cache']]['floor']
            parts += ['spare_ram_cache=on'] + ([f'spare_ram_floor={floor}'] if floor else [])
        out.append(','.join(parts))
    return ';'.join(out)


def kv_store_files(values, origin, config):
    """Each @FILE in KV_STORE replaced by the stores it holds, in the one-line form. A relative FILE is relative to the
    config file that names it, else to the working directory."""
    v = values.get('KV_STORE')
    if not v or '@' not in v:
        return
    base = os.path.dirname(config) if config and origin['KV_STORE'] in ('config file', '@file') else os.getcwd()
    texts, files = [], []
    for t in [t.strip() for t in v.split(';') if t.strip()]:
        if t.startswith('@'):
            files.append(os.path.join(base, t[1:]))
            t = kv_store_file(files[-1])
        texts.append(t)
    values['KV_STORE'] = ';'.join(texts)
    origin['KV_STORE'] += ' via ' + ', '.join(files)


def _yaml_loader():
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader, node):   # PyYAML keeps the last of two equal keys: refuse them
        seen = set()
        for k, _ in node.value:
            key = loader.construct_object(k)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f'{key} given twice', k.start_mark)
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node)

    def integer(loader, node):   # YAML 1.1 reads 0777 as octal and 1:20 as base 60: plain decimal only
        if not re.fullmatch(r'-?(0|[1-9]\d*)', node.value):
            raise yaml.constructor.ConstructorError(None, None, f'{node.value}: write a whole number in plain decimal',
                                                    node.start_mark)
        return int(node.value)

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    Loader.add_constructor('tag:yaml.org,2002:int', integer)
    return yaml, Loader


def kv_store_text(text, path='-'):
    """A YAML store file's text as KV_STORE's one-line form (kv_line). Shapes and types are checked here; everything
    else by kv_parse, so the two forms refuse the same things. Raises OptionError naming the file and the key.

        caches:
          spare: {kind: spare_ram, floor: 2G}      # memory the kernel may take back
        tiers:                                     # tried in order: a full tier passes entries to the next
          - keep: system
            cache: spare
            store:
              ssd: {path: /nvme/gr8kv, size_per_rank: 4G, persist: true}
          - keep: all
            cache: spare
            store:
              remote: {address: auto, size_per_rank: 4G}
    """
    def fail(where, msg):
        raise OptionError(f'KV_STORE=@{path}: {where}: {msg}')

    def want(where, x, t):
        if type(x) is not t:   # a YAML 1.1 on/yes is a bool, a bare 4 an int, 1:20 a base-60 int
            fail(where, f'want {TYPE_WORDS[t]}, got {x!r}')
        return x

    def keys(where, d, allowed, not_built=()):
        for k in d:
            if k in not_built:
                fail(f'{where}.{k}', 'not built yet')
            if k not in allowed:
                fail(f'{where}.{k}', f'want {", ".join(allowed)}')

    def one(x):   # a YAML value in the one-line form
        return ('on' if x else 'off') if isinstance(x, bool) else str(x)

    try:
        yaml, loader = _yaml_loader()
    except ImportError:
        raise OptionError(f'KV_STORE=@{path}: needs PyYAML (python3-yaml)') from None
    try:
        doc = yaml.load(text, Loader=loader)
    except yaml.YAMLError as e:
        raise OptionError(f'KV_STORE=@{path}: {e}') from None
    want('the file', doc, dict)
    keys('the file', doc, ('caches', 'tiers'), NOT_BUILT['top'])
    caches = want('caches', doc.get('caches', {}), dict)
    entries, tier_keys = [], {}
    for name, c in caches.items():
        where = f'caches.{name}'
        want(where, name, str)
        keys(where, want(where, c, dict), tuple(KV_KEYS['cache'].values()), NOT_BUILT['caches'])
        for k, x in c.items():
            want(f'{where}.{k}', x, str)
            if re.search(r'[,;=\s]', x):
                fail(f'{where}.{k}', f'{x!r}: no commas, semicolons, = or spaces')
        if re.search(r'[,;=\s]', name):
            fail(where, f'{name!r}: no commas, semicolons, = or spaces in a cache name')
        entries.append(','.join([f'cache:{name}'] + [f'{k}={c[k]}' for k in KV_KEYS['cache'] if k in c]))
    if 'tiers' not in doc:
        fail('tiers', 'no tiers')
    tiers = want('tiers', doc['tiers'], list)
    if not tiers:
        fail('tiers', 'no tiers')
    for i, t in enumerate(tiers):
        where = f'tiers[{i}]'
        keys(where, want(where, t, dict), ('name', 'keep', 'cache', 'store'), NOT_BUILT['tiers'])
        if 'store' not in t:
            fail(where, 'no store')
        store = want(f'{where}.store', t['store'], dict)
        if len(store) != 1 or next(iter(store)) not in ('ssd', 'remote'):
            fail(f'{where}.store', 'want one of ssd: or remote:')
        kind, s = next(iter(store.items()))
        where_s = f'{where}.store.{kind}'
        address = 'path' if kind == 'ssd' else 'address'
        on_store = {y: k for k, y in KV_KEYS[kind].items() if y not in ('keep', 'name', 'cache')}
        keys(where_s, want(where_s, s, dict), (address, *on_store))
        if address not in s:
            fail(where_s, f'no {address}')
        settings = [(f'{where}.{y}', k, t[y]) if y in ('keep', 'name', 'cache') else (f'{where_s}.{y}', k, s[y])
                    for k, y in KV_KEYS[kind].items() if y in (t if y in ('keep', 'name', 'cache') else s)]
        for w, k, x in [(f'{where_s}.{address}', None, s[address])] + settings:
            want(w, x, {'persist': bool, 'weight': int}.get(k, str))
            if isinstance(x, str) and re.search(r'[,;=\s]', x):
                fail(w, f'{x!r}: no commas, semicolons, = or spaces')
        entries.append(','.join([f'{kind}:{s[address]}'] + [f'{k}={one(x)}' for _, k, x in settings]))
        tier_keys[i] = where
    try:
        return kv_line(*kv_parse(';'.join(entries)))
    except KvError as e:
        where = {'cache': lambda n: f'caches.{n}', 'tier': lambda i: tier_keys.get(i, 'tiers')}[e.where[0]](e.where[1]) \
            if e.where else 'the file'
        fail(where, str(e))


def kv_store_file(path):
    """kv_store_text of the file at `path`."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError as e:
        raise OptionError(f'KV_STORE=@{path}: {e.strerror}') from None
    return kv_store_text(text, path)


def kv_rows(o):
    """KV_ROWS parsed: {'text', 'store', 'kind': 'remote' | 'mem', 'host', 'port', 'size', 'local', 'pool', 'place',
    'fmt'}, or None when unset or off. Raises OptionError naming the bad part."""
    v = o.get('KV_ROWS')
    if v in (None, 'off'):
        return None
    text = v.strip()
    store, *kvs = text.split(',')
    m = re.fullmatch(r'remote:([^:,]+)(?::(\d+))?', store)
    if store != 'mem:' and (not m or (m.group(1) == 'auto' and m.group(2))):
        raise OptionError(f'--kv-rows {text}: the store is remote:auto, remote:HOST:PORT or mem:')
    t = {'text': text, 'store': store, 'kind': 'mem' if store == 'mem:' else 'remote', 'host': m.group(1) if m else None,
         'port': m.group(2) if m else None, 'size': None, 'local': None, 'pool': None, 'place': None, 'fmt': None}
    for kv in kvs:
        k, _, x = kv.partition('=')
        if k == 'size' and re.fullmatch(r'\d+(\.\d+)?[MGT]', x) and float(x[:-1]) > 0:
            t['size'] = x
        elif k in ('local', 'pool', 'place') and x.isdigit() and int(x) > 0:
            t[k] = int(x)
        elif k == 'fmt' and x in ('fp8', 'int4'):
            t['fmt'] = x
        else:
            raise OptionError(f'--kv-rows {text}: {kv}: want size=N(M|G|T), local=PAGES, pool=SLOTS, place=PAGES or fmt=fp8|int4')
    if not t['size']:
        raise OptionError(f'--kv-rows {text}: size=N(M|G|T) is needed (each rank\'s lease)')
    if t['place'] and not t['local']:
        raise OptionError(f'--kv-rows {text}: place= needs local= (the pages past it are reached through the rows store only)')
    if t['kind'] == 'remote' and t['host'] != 'auto' and not t['port']:
        raise OptionError(f'--kv-rows {text}: remote:HOST:PORT names the port')
    return t


def kv_rows_with(o, target):
    """KV_ROWS with remote:auto replaced by the KV node host:port this box uses (None: as written)."""
    t = kv_rows(o)
    if not t or t['host'] != 'auto' or not target:
        return t and t['text']
    return t['text'].replace('remote:auto', f'remote:{target}', 1)


def split_list(v):
    return [x for x in re.split(r'[,\s]+', v or '') if x]


def resolve_peers(o):
    """The peer list as IPv4 addresses, resolved on this box."""
    out = []
    for name in split_list(o.get('PEERS')):
        try:
            addr = socket.getaddrinfo(name, None, socket.AF_INET)[0][4][0]
        except OSError:
            raise OptionError(f'cannot resolve peer {name!r}. PEERS (or HOSTS) must be LAN addresses or names every box '
                              'resolves, not ssh aliases; literal addresses are best')
        if ipaddress.ip_address(addr).is_loopback and name != addr and len(split_list(o.get('PEERS'))) > 1:
            raise OptionError(f'peer {name!r} resolves to {addr} here; name the LAN address instead')
        out.append(addr)
    if len(set(out)) != len(out):
        raise OptionError(f'the peer list names one box twice: {",".join(out)}')
    return out


# ---- the engine plan: scripts/engine_args.sh --------------------------------------------------------------------------
def engine_plan(o, rank='@R', rank_dir='@RANKDIR', aux_dir='@AUXDIR', home='@HOME', epoch='@EPOCH', set_id='',
                kv_store=None, kv_rows=None):
    """{'knobs': {K: resolved value}, 'env': [K=V..], 'args': [..]} from scripts/engine_args.sh.
    The defaults give the canonical template, with every per-box value as a placeholder. kv_store, kv_rows: this box's
    KV_STORE and KV_ROWS with the KV node resolved (remote:GB names none)."""
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'GR8_MODEL': models.of(o).name, 'TP': o.get('TP'),
           'SET_ID': set_id}
    for k in ENGINE_KNOBS + ['S2PORT', 'READY_TMO']:
        if o.get(k) is not None:
            env[k] = o.get(k)
    if o.get('STEERING_PROFILE'):
        for key in ('STEERING_PROFILE', 'STEERING_CHAT_TEMPLATE', 'STEERING_MODEL_REVISION', 'STEERING_TOKENIZER_REVISION'):
            if any(c.isspace() for c in o.get(key) or ''):
                raise OptionError(flag_of(key) + ': whitespace is not supported by the engine launcher')
    if o.get('STEERING_PROFILE') and not env.get('STEERING_CHAT_TEMPLATE'):
        env['STEERING_CHAT_TEMPLATE'] = '@STEERING_CHAT_TEMPLATE'
    if kv_store:
        env['KV_STORE'] = kv_store
    if env.get('KV_STORE'):
        env['KV_STORE'] = kv_engine(env['KV_STORE'])
    if kv_rows:
        env['KV_ROWS'] = kv_rows
    if rank == '@R':   # the template: v2 paths are per box
        for k in ('V2', 'V2_SLICES', 'V2_WIDTHS', 'DFLASH_DIR', 'V2_DENSE', 'STEERING_PROFILE', 'STEERING_CHAT_TEMPLATE', 'LAYOUT',
                  'GRAMMAR_DIR'):
            if env.get(k):
                env[k] = '@' + k
    names = ' '.join(ENGINE_KNOBS + ['S2PORT', 'READY_TMO'])
    script = ('set -e; . "$0"; engine_defaults; engine_env; for k in ' + names + '; do echo "knob $k=${!k-}"; done; '
              'echo "env $ENGINE_ENV"; echo "args $(engine_args "$@")"')
    r = subprocess.run(['bash', '-c', script, os.path.join(REPO, 'scripts', 'engine_args.sh'),
                        str(rank), rank_dir, aux_dir, home, str(epoch)], env=env, capture_output=True, text=True)
    if r.returncode:
        raise OptionError(f'scripts/engine_args.sh: {r.stderr.strip()}')
    plan = {'knobs': {}, 'env': [], 'args': []}
    for line in r.stdout.splitlines():
        kind, _, rest = line.partition(' ')
        if kind == 'knob':
            k, _, v = rest.partition('=')
            plan['knobs'][k] = v
        elif kind == 'env':
            plan['env'] = rest.split()
        elif kind == 'args':
            plan['args'] = rest.split()
    plan['drafter_dir'] = o.get('DFLASH_DIR')
    return plan


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(o, plan, members, variant=None, ring=''):
    """The cluster options as sorted key=value lines, and their sha256. members: the peer list (peers mode) or None.
    variant: the resolved exlr8 v2 variant (modelset.Variant), whose name and manifest sha256 count. ring: --shape ring
    with mentat, the ring this box planned from mentat's claim (mentat.ring_text), so every box must have the same."""
    c = {'tp': o.get('TP'), 'shape': o.get('SHAPE'), 'mode': mode(o), 'fabric_port': str(FABRIC_PORT)}
    if models.of(o) is not models.GLM53:   # GLM-5.3's lines stay as they were, so a mixed upgrade still agrees
        c['model'] = models.of(o).name
    if members:
        c['peers'] = ','.join(members)
    else:
        c['group'] = o.get('GROUP')
        c['head'] = o.get('HEAD', '')
    for k in ('IPS', 'IPS2', 'CONTROL_PORT', 'PORT'):
        c[k] = o.get(k) or ''
    # Every knob that reaches the engine is in engine.env and engine.args below. CAP is also here by name, so a refusal
    # names it. DFLASH and PF_OWN decide which pack files a rank needs.
    for k in ('CAP', 'DFLASH', 'PF_OWN'):
        c[k] = plan['knobs'].get(k, '')
    if variant:
        c['variant'] = variant.name
        c['variant.sha256'] = variant.sha256 or ''
    if ring:
        c['ring'] = ring
    if o.get('LAYOUT'):   # a path per box: its bytes are what every rank must agree on
        c['layout.sha256'] = file_sha256(o.get('LAYOUT'))
    c['engine.env'] = ' '.join(sorted(plan['env']))
    c['engine.args'] = ' '.join(plan['args'])
    text = ''.join(f'{k}={c[k]}\n' for k in sorted(c))
    return c, text, hashlib.sha256(text.encode()).hexdigest()


def summary(o):
    """The effective options, one line each, with where each value came from."""
    lines = []
    for k, (scope, _, _) in KEYS.items():
        v = o.values.get(k)
        if v is None or v == '':
            continue
        lines.append(f'  {k:<13} {v:<28} {scope:<8} ({o.origin.get(k, "default")})')
    return '\n'.join(lines)
