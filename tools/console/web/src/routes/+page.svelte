<script>
  import { onMount, tick } from 'svelte';
  let snapshot = $state(null);
  let error = $state('');
  let stale = $state(false);
  let presets = $state(null);
  let dialog;
  let review = $state(null);
  let sending = $state(false);
  let job = $state(null);
  let completedJob = '';
  const busy = $derived(job?.status === 'running');
  const current = $derived(snapshot?.models?.find(m => m.id === snapshot.active));
  const enabled = $derived(snapshot?.can_control && !busy && !stale);
  const presetLabel = $derived(presets?.current === 'custom' ? 'custom' : presets?.presets?.find(p => p.id === presets.current)?.label);
  const presetCaption = $derived(presets?.presets?.find(p => p.id === presets.current)?.caption);
  const rows = $derived([
    ['Decode', snapshot?.stats.decode, 'tok/s', 'generation_tokens_total / wall s'],
    ['Prefill', snapshot?.stats.prefill, 'tok/s', 'prompt_tokens_total / wall s'],
    ['First token', snapshot?.stats.ttft, 's avg', 'time_to_first_token sum ÷ count'],
    ['Chats', snapshot?.stats.running, `/ ${fmt(snapshot?.stats.slots)} · ${fmt(snapshot?.stats.waiting)} waiting`, 'num_requests_running / waiting'],
    ['KV cache', snapshot?.stats.kv_used, `/ ${fmt(snapshot?.stats.kv_size)}`, 'kv_active_tokens / cache_size'],
    ['Prefix hits', snapshot?.stats.prefix_hits, '%', 'prefix_hits ÷ queried tokens'],
    ['Draft accept', snapshot?.stats.draft_accept, '', 'accepted tokens ÷ draft tokens'],
    ['Requests finished', snapshot?.stats.finished, '', 'all finish reasons, including errors']
  ]);
  function fmt(n) { return n == null || !Number.isFinite(n) ? '—' : n.toLocaleString(undefined, {maximumFractionDigits: n < 1 ? 3 : 1}); }
  async function api(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch('/api/' + path, {...options, signal: controller.signal});
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      return payload.value;
    } finally { clearTimeout(timer); }
  }
  async function loadPresets(model) {
    if (!model) { presets = null; return; }
    const next = await api('presets/' + model);
    if (snapshot?.active === model) presets = next;
  }
  async function refresh() {
    try {
      const next = await api('snapshot');
      snapshot = next; if (stale) error = ''; stale = false;
      job = next.operation;
      const finished = job && job.status !== 'running' && job.id !== completedJob;
      if (finished) completedJob = job.id;
      if (!next.active) presets = null;
      else if (!review) {
        try { await loadPresets(next.active); }
        catch(e) { presets = null; error = e.message; }
      }
    } catch(e) { stale = true; error = `Connection lost. Values below are stale. ${e.message}`; }
  }
  async function openReview(action, model) {
    review = {action, model, expect_active: snapshot?.active ?? null, backup_id: job?.backup_id};
    await tick(); dialog.showModal();
  }
  async function openPreset(id) {
    if (!presets || !enabled || !presets.observed || presets.current === id) return;
    const choice = presets.presets.find(item => item.id === id);
    review = {action: 'restart', model: snapshot.active, preset: id, label: choice.label,
      expect_active: snapshot.active ?? null, revision: presets.revision, before: presets.current_setting, after: choice.setting,
      diff: choice.diff || [], download: choice.download || null};
    await tick(); dialog.showModal();
  }
  function closeReview() { if (!sending) { dialog.close(); review = null; } }
  async function submit() {
    if (!review || sending) return;
    sending = true; error = '';
    try {
      job = await api('operations', { method: 'POST', headers: {'Content-Type':'application/json', 'X-Console-Token':snapshot.csrf},
        body: JSON.stringify({action: review.action, model: review.model, preset: review.preset,
          revision: review.revision, expect_active: review.expect_active, backup_id: review.backup_id, confirm: true,
          accept_download: review.download ? true : undefined}) });
      dialog.close(); review = null;
    } catch(e) { error = e.message; }
    finally { sending = false; }
  }
  onMount(() => {
    let alive = true, timer;
    const poll = async () => { await refresh(); if (alive) timer = setTimeout(poll, 2000); };
    poll();
    return () => { alive = false; clearTimeout(timer); };
  });
</script>

{#if snapshot?.fixture}<div class="fixture">SYNTHETIC FIXTURE · not live model measurements · controls affect only the fake deployment</div>{/if}
<header>
  <div class="brand">KINDLING <b>DASH</b></div>
  <div class="live"><span class="dot" class:off={!current || stale}></span><span class="hide-s" id="hstate">{stale ? 'disconnected' : current ? current.name + ' · ready' : 'no verified ready model'}</span></div>
</header>
<main>
  {#if error}<div class="notice error" role="alert">{error} {#if !stale}<button class="btn" onclick={() => error = ''}>Dismiss</button>{/if}</div>{/if}
  {#if snapshot?.error}<div class="notice warn" role="status">{snapshot.error}</div>{/if}
  {#if snapshot && !snapshot.can_control}<div class="notice off">Read-only · {snapshot.control_error || 'control is disabled in the Kindling dash configuration'}</div>{/if}
  {#if !snapshot}<p class="sub">Connecting to gr8…</p>{/if}

  {#if snapshot}
    <div class="bar">
      <span class="name">{current?.name || 'No verified ready model'}</span>
      <span class="pill" class:ok={current && !stale}>{stale ? 'stale' : current ? '● ready' : 'unknown / unavailable'}</span>
      {#if presetLabel && !stale}<span class="pill preset">{presetLabel}</span>{/if}
      <span class="meta">{current ? `TP${current.width} · ${current.width} boxes · ${presetCaption || `ctx ${fmt(current.context)}`} · ` : ''}{current?.started ? `up ${Math.floor((snapshot.observed_at - current.started) / 60)}m · ` : ''}2 s refresh</span>
    </div>
    <div class="cols">
      <div class="panel">
        <div class="ph">Loaded model<span class="r">source</span></div>
        <table><tbody>
          {#each rows as [label, value, unit, source]}
            <tr><td>{label}</td><td class="v">{fmt(value)} <small>{unit}</small></td><td class="source hide-s">{source}</td></tr>
          {/each}
        </tbody></table>
      </div>
      <div class="stack">
        <div class="panel">
          <div class="ph">Preset<span class="r">{presets?.source || 'no model running'}</span></div>
          {#if presets}
            <div class="presets" role="group" aria-label="Presets">
              {#each presets.presets as item}
                <button class="pb" class:on={presets.current === item.id} aria-pressed={presets.current === item.id} disabled={!enabled || !presets.observed} onclick={() => openPreset(item.id)}><b>{item.label}</b><span>{item.caption}</span></button>
              {/each}
            </div>
            <div class="psets">
              {#each presets.presets as item}<div><b>{item.label}</b> · {item.setting} · {item.summary}</div>{/each}
            </div>
            {#if presets.current === 'custom'}<p class="sub custom">Current settings are custom.</p>{/if}
            {#if !presets.observed}<p class="sub">Dotenv not observed; presets require the switch integration.</p>{/if}
          {:else}<p class="sub">No model running.</p>{/if}
        </div>
        <div class="panel">
          <div class="ph">Boxes<span class="r">supervisor status</span></div>
          <table><thead><tr><th>box</th><th>state</th><th class="n">free mem</th><th class="hide-s">last line</th></tr></thead><tbody>
            {#each snapshot.boxes || [] as box, i}
              <tr><td class="m">{i + 1}{i === 0 ? ' head' : ''}</td><td class:ok={box.state === 'serving'}>{box.state || 'unknown'}</td><td class="n">{fmt(box.mem_gib)} GiB</td><td class="source hide-s">{box.last_line || '—'}</td></tr>
            {:else}<tr><td colspan="4" class="off">No supervisor data</td></tr>{/each}
          </tbody></table>
        </div>
        <div class="panel"><div class="ph">Log · head<span class="r">supervisor logs</span></div><pre class="logs">{snapshot.logs || 'No log available'}</pre></div>
      </div>
    </div>

    <div class="panel models">
      <div class="ph">Models<span class="r">one model at a time · gr8 switch</span></div>
      <table><thead><tr><th></th><th>model</th><th class="hide-s">layout</th><th>state</th><th class="n hide-s">last switch</th><th></th></tr></thead><tbody>
        {#each snapshot.models || [] as model}
          <tr class:cur={model.state === 'ready'}><td><span class="dot" class:off={model.state !== 'ready' || stale}></span></td><td class="m model-name">{model.name}</td><td class="source hide-s">{model.width ? `TP${model.width} · ${model.width} boxes` : '—'}</td><td class:ok={model.state === 'ready'}>{stale ? 'stale' : model.state}</td><td class="n source hide-s">{model.last_switch == null ? '—' : fmt(model.last_switch) + ' s'}</td><td>
            <div class="acts">
              {#if model.state === 'ready'}
                <button class="btn" disabled={!enabled} onclick={() => openReview('restart', model.id)}>Restart</button><button class="btn danger" disabled={!enabled} onclick={() => openReview('stop', model.id)}>Stop</button>
              {:else}<button class="btn pri" disabled={!enabled || model.state === 'not configured'} onclick={() => openReview('switch', model.id)}>Switch</button>{/if}
            </div>
          </td></tr>
        {/each}
      </tbody></table>
    </div>
  {/if}
  {#if job}
    <section class="panel operation" aria-live="polite" aria-label="Operation status">
      <div class="ph">{job.action} · {job.model}<span class="r" class:error={job.status === 'failed' || job.status === 'interrupted'} class:ok={job.status === 'succeeded'}>{job.status} · {fmt(job.elapsed)} s</span></div>
      {#if job.error}<div class="notice error" role="alert">{job.error}</div>{/if}
      <ol class="steps">{#each job.steps || [] as step}<li>{step}</li>{/each}</ol>
      {#if job.can_restore && !busy}<div class="notice"><button class="btn" disabled={!enabled} onclick={() => openReview('restore', job.model)}>Restore previous settings</button> <span class="meta">Restores this operation’s saved dotenv files; does not restart.</span></div>{/if}
    </section>
  {/if}
</main>
<dialog bind:this={dialog} oncancel={e => { if (sending) e.preventDefault(); else review = null; }} aria-labelledby="review-title">
  {#if review}
    <h3 id="review-title">{review.label ? `Restart ${snapshot?.models.find(m => m.id === review.model)?.name} with ${review.label}?` : `${review.action === 'restore' ? 'Restore previous settings for' : review.action === 'switch' ? 'Switch to' : review.action === 'stop' ? 'Stop' : review.expect_active && review.expect_active !== review.model ? 'Switch and restart' : 'Restart'} ${snapshot?.models.find(m => m.id === review.model)?.name}?`}</h3>
    {#if review.action !== 'restore' && review.expect_active && review.expect_active !== review.model}<p class="warn">{snapshot?.models.find(m => m.id === review.expect_active)?.name} stops on its configured boxes.</p>{/if}
    <p>{review.action === 'restore' ? 'Restores the saved previous dotenv files only if nobody has edited them since. Does not restart the current engine. Then use Restart or Switch to apply.' : review.action === 'stop' ? 'Stops the target on every configured box. No replacement starts.' : 'Stops and starts the configured boxes. Chats in progress end. Completion waits for a ready model.'}</p>
    {#if review.after}<div class="diff"><div class="error">− {review.before}</div><div class="ok">+ {review.after}</div>{#each (review.diff || []).filter(row => row.after == null && row.before != null) as row}<div class="error">− {row.key}={row.before}</div><div class="ok">+ {row.key} removed</div>{/each}{#each (review.diff || []).filter(row => row.key === 'CAP' && row.after != null) as row}<div class="error">− CAP={row.before}</div><div class="ok">+ CAP={row.after}</div>{/each}</div>{/if}
    {#if review.download}<p class="warn">{review.download}</p>{/if}
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    <div class="row"><button class="btn" disabled={sending} onclick={closeReview}>Cancel</button><button class="btn pri" disabled={sending || busy || stale} onclick={submit}>{sending ? 'Submitting…' : review.action === 'restore' ? 'Restore settings now' : review.action === 'restart' ? 'Restart now' : review.action === 'stop' ? 'Stop now' : 'Switch now'}</button></div>
  {/if}
</dialog>
