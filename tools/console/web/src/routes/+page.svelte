<script>
  import { onMount, tick } from 'svelte';
  let tab = $state('overview');
  let snapshot = $state(null);
  let error = $state('');
  let stale = $state(false);
  let selected = $state('glm53');
  let settings = $state(null);
  let values = $state({});
  let touched = $state({});
  let search = $state('');
  let loading = $state(false);
  let dialog;
  let review = $state(null);
  let sending = $state(false);
  let job = $state(null);
  let completedJob = '';
  const busy = $derived(job?.status === 'running');
  const current = $derived(snapshot?.models?.find(m => m.id === snapshot.active));
  const enabled = $derived(snapshot?.can_control && !busy && !stale);
  const changes = $derived(Object.fromEntries(Object.entries(values).filter(([k,v]) => v !== settings?.values[k] || (settings?.mixed.includes(k) && touched[k]))));
  const count = $derived(Object.keys(changes).length);
  const groups = ['fast switching', 'speed', 'capacity'];
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
  function show(v) { return v == null ? 'auto (unset)' : v; }
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
  async function refresh() {
    try {
      const next = await api('snapshot');
      snapshot = next; if (stale) error = ''; stale = false;
      job = next.operation;
      if (job && job.status !== 'running' && job.id !== completedJob) {
        completedJob = job.id;
        if (job.status === 'succeeded' && ['restart','restore'].includes(job.action) && settings && selected === job.model) await loadKnobs();
      }
    } catch(e) { stale = true; error = `Connection lost. Values below are stale. ${e.message}`; }
  }
  async function loadKnobs() {
    loading = true; error = '';
    const target = selected;
    try {
      const next = await api('knobs/' + target);
      if (selected === target) { settings = next; values = {...next.values}; touched = {}; }
    } catch(e) { settings = null; error = e.message; }
    finally { loading = false; }
  }
  async function navigate(next) {
    tab = next;
    if (next === 'knobs' && !settings) {
      selected = snapshot?.active || selected;
      await loadKnobs();
    }
  }
  function setValue(key, value) { values = {...values, [key]: value === '' ? null : value}; touched = {...touched, [key]:true}; }
  function discard() { values = {...settings.values}; touched = {}; }
  async function openReview(action, model, edits = {}) {
    review = {action, model, edits, expect_active: snapshot?.active ?? null, backup_id: job?.backup_id, before: {...settings?.values}, mixed: settings?.mixed || [], revision: settings?.revision};
    await tick(); dialog.showModal();
  }
  function closeReview() { if (!sending) { dialog.close(); review = null; } }
  async function submit() {
    if (!review || sending) return;
    sending = true; error = '';
    try {
      job = await api('operations', { method: 'POST', headers: {'Content-Type':'application/json', 'X-Console-Token':snapshot.csrf},
        body: JSON.stringify({action: review.action, model: review.model, changes: review.edits,
          revision: review.revision, expect_active: review.expect_active, backup_id: review.backup_id, confirm: true}) });
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
  <div class="brand">GR<b>8</b></div>
  <nav aria-label="Console sections">
    {#each ['overview','knobs','models'] as name}
      <button class:on={tab === name} aria-current={tab === name ? 'page' : undefined} onclick={() => navigate(name)}>{name[0].toUpperCase() + name.slice(1)}</button>
    {/each}
  </nav>
  <div class="live"><span class="dot" class:off={!current || stale}></span><span class="hide-s">{stale ? 'disconnected' : current ? current.name + ' · ready' : 'no verified ready model'}</span></div>
</header>
<main>
  {#if error}<div class="notice error" role="alert">{error} {#if !stale}<button class="btn" onclick={() => error = ''}>Dismiss</button>{/if}</div>{/if}
  {#if snapshot?.error}<div class="notice warn" role="status">{snapshot.error}</div>{/if}
  {#if snapshot && !snapshot.can_control}<div class="notice off">Read-only · {snapshot.control_error || 'control is disabled in console configuration'}</div>{/if}
  {#if !snapshot}<p class="sub">Connecting to gr8…</p>{/if}

  {#if tab === 'overview'}
    <div class="bar">
      <span class="name">{current?.name || 'No verified ready model'}</span>
      <span class="pill" class:ok={current && !stale}>{stale ? 'stale' : current ? '● ready' : 'unknown / unavailable'}</span>
      <span class="meta">{current ? `TP${current.width} · ${current.width} boxes · ctx ${fmt(current.context)} · ` : ''}{current?.started ? `up ${Math.floor((snapshot.observed_at - current.started) / 60)}m · ` : ''}2 s refresh</span>
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
          <div class="ph">Boxes<span class="r">supervisor status</span></div>
          <table><thead><tr><th>box</th><th>state</th><th class="n">free mem</th><th class="hide-s">last line</th></tr></thead><tbody>
            {#each snapshot?.boxes || [] as box, i}
              <tr><td class="m">{i + 1}{i === 0 ? ' head' : ''}</td><td class:ok={box.state === 'serving'}>{box.state || 'unknown'}</td><td class="n">{fmt(box.mem_gib)} GiB</td><td class="source hide-s">{box.last_line || '—'}</td></tr>
            {:else}<tr><td colspan="4" class="off">No supervisor data</td></tr>{/each}
          </tbody></table>
        </div>
        <div class="panel"><div class="ph">Log · head<span class="r">supervisor logs</span></div><pre class="logs">{snapshot?.logs || 'No log available'}</pre></div>
      </div>
    </div>
    <p class="sub" style="margin-top:6px">Rates are aggregate counter deltas, not single-request speed. TTFT and ratios cover the current boot. — means unavailable or no samples yet.</p>
  {:else if tab === 'knobs'}
    <div class="bar">
      <label for="knob-model" class="meta">Model</label>
      <select id="knob-model" bind:value={selected} disabled={count > 0 || busy || loading} onchange={() => { settings = null; loadKnobs(); }}>
        {#each snapshot?.models || [] as model}<option value={model.id} disabled={model.state === 'not configured'}>{model.name}</option>{/each}
      </select>
      <button class="btn" disabled={count > 0 || busy || loading} onclick={loadKnobs}>Refresh overrides</button>
    </div>
    <p class="sub">Every knob is read at boot. Changes stay in this browser until a confirmed restart. Current chats end.</p>
    {#if loading}<p class="sub">Reading configured overrides…</p>{/if}
    {#if settings}
      <p class="sub">{settings.note}</p>
      {#if settings.mixed.length}<div class="notice warn">Different overrides across boxes: {settings.mixed.join(', ')}. Change explicitly to reconcile; otherwise existing values stay.</div>{/if}
      {#if !settings.observed}<div class="notice warn">Dotenv overrides not observed; controls require the switch integration.</div>{/if}
      <div class="panel">
        <div class="ph">Knobs<span class="r"><input class="search" aria-label="Search knobs" placeholder="search knobs…" bind:value={search}></span></div>
        <table class="knobs"><colgroup><col class="input-col"><col class="label-col"><col class="tag-col"><col class="hide-s"></colgroup><tbody>
          <tr class="fastrow"><td><button class="btn pri" disabled={!enabled || !settings.observed || loading || !Object.keys(settings.fast_values).length} onclick={() => { values = {...values, ...settings.fast_values}; touched = {...touched, ...Object.fromEntries(Object.keys(settings.fast_values).map(k => [k,true]))}; }}>Stage fast mode</button></td><td class="fast-label">Fast switching · SCREEN</td><td><span class="tag">restart</span></td><td class="doc">{Object.keys(settings.fast_values).length ? Object.keys(settings.fast_values).join(', ') : 'No prepared fast profile configured. Individual knobs remain available.'}</td></tr>
          {#each groups as group}
            {#if settings.catalog.some(k => k.group === group && (k.key + k.label + k.description).toLowerCase().includes(search.toLowerCase()))}
              <tr class="grp"><td colspan="4">{group}</td></tr>
              {#each settings.catalog.filter(k => k.group === group && (k.key + k.label + k.description).toLowerCase().includes(search.toLowerCase())) as knob}
                <tr class:chg={Object.hasOwn(changes, knob.key)}>
                  <td>
                    {#if knob.kind === 'bool' || knob.kind === 'merge'}
                      <select class="knob-input" aria-label={knob.key} value={settings.mixed.includes(knob.key) && !touched[knob.key] ? '__mixed__' : values[knob.key] ?? ''} disabled={!enabled || !settings.observed || loading} onchange={e => setValue(knob.key, e.currentTarget.value)}>
                        <!-- Mixed is distinct from removing every override. -->
                        {#if settings.mixed.includes(knob.key)}<option value="__mixed__" disabled>mixed</option>{/if}<option value="">auto</option><option value="0">off · 0</option><option value="1">on · 1</option>
                      </select>
                    {:else}
                      <input class="knob-input" aria-label={knob.key} type={knob.kind === 'int' ? 'number' : 'text'} min={knob.bounds?.[0]} max={knob.bounds?.[1]}
                        placeholder={settings.mixed.includes(knob.key) ? 'mixed' : 'auto / unset'} value={values[knob.key] ?? ''} disabled={!enabled || !settings.observed || loading} oninput={e => setValue(knob.key, e.currentTarget.value)}>
                    {/if}
                  </td>
                  <td class="m" title={knob.description}>{knob.label} · {knob.key}</td><td><span class="tag">restart</span></td><td class="doc">{knob.description}</td>
                </tr>
              {/each}
            {/if}
          {/each}
        </tbody></table>
      </div>
      {#if count}<div class="stagebar"><span class="msg">{count} change{count === 1 ? '' : 's'} staged · needs restart</span><button class="btn" disabled={busy} onclick={discard}>Discard</button><button class="btn pri" disabled={!enabled || loading} onclick={() => openReview('restart', selected, changes)}>Review + restart…</button></div>{/if}
    {/if}
  {:else}
    <p class="sub">Switch uses the gr8 switch command: stop other configured deployments, start the target, wait for ready. No automatic rollback after a boot failure.</p>
    {#if count}<div class="notice warn">Knob changes are still staged. Apply or discard them before a model action.</div>{/if}
    <div class="panel"><div class="ph">Models</div>
      <table><thead><tr><th></th><th>model</th><th class="hide-s">layout</th><th>state</th><th class="n hide-s">last switch</th><th></th></tr></thead><tbody>
        {#each snapshot?.models || [] as model}
          <tr class:cur={model.state === 'ready'}><td><span class="dot" class:off={model.state !== 'ready' || stale}></span></td><td class="m model-name">{model.name}</td><td class="source hide-s">{model.width ? `TP${model.width} · ${model.width} boxes` : '—'}</td><td class:ok={model.state === 'ready'}>{stale ? 'stale' : model.state}</td><td class="n source hide-s">{model.last_switch == null ? '—' : fmt(model.last_switch) + ' s'}</td><td>
            <div class="acts">
              {#if model.state === 'ready'}
                <button class="btn" disabled={!enabled || count > 0} onclick={() => openReview('restart', model.id)}>Restart</button><button class="btn danger" disabled={!enabled || count > 0} onclick={() => openReview('stop', model.id)}>Stop</button>
              {:else}<button class="btn pri" disabled={!enabled || count > 0 || model.state === 'not configured'} onclick={() => openReview('switch', model.id)}>Switch</button>{/if}
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
    <h3 id="review-title">{review.action === 'restore' ? 'Restore previous settings for' : review.action === 'switch' ? 'Switch to' : review.action === 'stop' ? 'Stop' : review.expect_active && review.expect_active !== review.model ? 'Switch and restart' : 'Restart'} {snapshot?.models.find(m => m.id === review.model)?.name}?</h3>
    {#if review.action !== 'restore' && review.expect_active && review.expect_active !== review.model}<p class="warn">{snapshot?.models.find(m => m.id === review.expect_active)?.name} stops on its configured boxes.</p>{/if}
    <p>{review.action === 'restore' ? 'Restores the saved previous dotenv files only if nobody has edited them since. Does not restart the current engine. Then use Restart or Switch to apply.' : review.action === 'stop' ? 'Stops the target on every configured box. No replacement starts.' : 'Stops and starts the configured boxes. Chats in progress end. Completion waits for a ready model.'}</p>
    {#if Object.keys(review.edits).length}<div class="diff">{#each Object.entries(review.edits) as [key,value]}<div class="error">− {key} = {review.mixed.includes(key) ? 'mixed across boxes' : show(review.before[key])}</div><div class="ok">+ {key} = {show(value)}</div>{/each}</div>{/if}
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    <div class="row"><button class="btn" disabled={sending} onclick={closeReview}>Cancel</button><button class="btn pri" disabled={sending || busy || stale} onclick={submit}>{sending ? 'Submitting…' : review.action === 'restore' ? 'Restore settings now' : review.action === 'restart' ? 'Restart now' : review.action === 'stop' ? 'Stop now' : 'Switch now'}</button></div>
  {/if}
</dialog>
