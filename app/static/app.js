/*
 * Frontend Client Logic for Web Testing AI Agent Dashboard (Local Vision Engine).
 * Supports Single & Parallel Multi-App Batch Auditing, Live Tabbed Stream Switching, & Master Batch Dashboard.
 */

document.addEventListener('DOMContentLoaded', () => {
  let activeRunId = null;
  let activeBatchId = null;
  let isBatchMode = false;
  let socket = null;

  // Active run state map: { run_id: { url, status, logs: [], step: 0, state: 'Idle', screenshot: null, summary: null } }
  const runStates = {};

  // DOM Elements
  const testForm = document.getElementById('test-form');
  const startBtn = document.getElementById('start-btn');
  const btnSingleApp = document.getElementById('btn-single-app');
  const btnMultiApp = document.getElementById('btn-multi-app');
  const singleAppInputs = document.getElementById('single-app-inputs');
  const batchAppInputs = document.getElementById('batch-app-inputs');

  const targetUrlInput = document.getElementById('target-url');
  const usernameInput = document.getElementById('username');
  const passwordInput = document.getElementById('password');
  const multiAppUrlsInput = document.getElementById('multi-app-urls');
  const modelSelect = document.getElementById('model-select');
  const headlessCheck = document.getElementById('headless-check');

  const geminiStatusText = document.getElementById('gemini-status-text');
  const liveRunTabs = document.getElementById('live-run-tabs');
  const liveScreenshot = document.getElementById('live-screenshot');
  const screenPlaceholder = document.getElementById('screen-placeholder');
  const stepIndicator = document.getElementById('step-indicator');
  const agentStateIndicator = document.getElementById('agent-state-indicator');
  const terminalLogs = document.getElementById('terminal-logs');

  const historyContainer = document.getElementById('history-container');
  const refreshHistoryBtn = document.getElementById('refresh-history-btn');

  const statSteps = document.getElementById('stat-steps');
  const statDuration = document.getElementById('stat-duration');
  const statIssues = document.getElementById('stat-issues');
  const statUx = document.getElementById('stat-ux');
  const reportDetailsBox = document.getElementById('report-details-box');
  const reportDownloadActions = document.getElementById('report-download-actions');
  const btnDownloadDocx = document.getElementById('btn-download-docx');
  const btnDownloadJson = document.getElementById('btn-download-json');
  const btnDownloadMd = document.getElementById('btn-download-md');

  // Dynamic App Rows System
  const btnAddAppRow = document.getElementById('btn-add-app-row');
  const appRowsContainer = document.getElementById('app-rows-container');

  if (btnAddAppRow && appRowsContainer) {
    btnAddAppRow.addEventListener('click', () => {
      const row = document.createElement('div');
      row.className = 'app-config-row';
      row.style.cssText = 'background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px; margin-top: 6px;';
      row.innerHTML = `
        <div class="form-group" style="margin-bottom: 6px;">
          <input type="text" class="form-input app-target-url" placeholder="Target URL e.g. http://localhost:3000/login">
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
          <input type="text" class="form-input app-username" placeholder="Username / ID" style="flex: 1; font-size: 11px;">
          <input type="password" class="form-input app-password" placeholder="Password" style="flex: 1; font-size: 11px;">
          <button type="button" class="btn-secondary btn-remove-app-row" style="padding: 6px 10px; font-size: 12px; color: #ef4444; border-color: rgba(239,68,68,0.3);" title="Remove App Target" aria-label="Remove App Target">❌</button>
        </div>
      `;
      appRowsContainer.appendChild(row);
    });

    appRowsContainer.addEventListener('click', (e) => {
      if (e.target.classList.contains('btn-remove-app-row')) {
        const rows = appRowsContainer.querySelectorAll('.app-config-row');
        if (rows.length > 1) {
          e.target.closest('.app-config-row').remove();
        } else {
          alert('At least one application row must remain.');
        }
      }
    });
  }

  // Mode Switcher Handlers
  if (btnSingleApp) {
    btnSingleApp.addEventListener('click', () => {
      isBatchMode = false;
      btnSingleApp.style.background = 'var(--primary)';
      btnSingleApp.style.color = 'white';
      btnSingleApp.style.border = 'none';
      if (btnMultiApp) {
        btnMultiApp.style.background = 'transparent';
        btnMultiApp.style.color = 'var(--text-muted)';
      }
      singleAppInputs.style.display = 'block';
      batchAppInputs.style.display = 'none';
    });
  }

  if (btnMultiApp) {
    btnMultiApp.addEventListener('click', () => {
      isBatchMode = true;
      btnMultiApp.style.background = 'var(--primary)';
      btnMultiApp.style.color = 'white';
      btnMultiApp.style.border = 'none';
      if (btnSingleApp) {
        btnSingleApp.style.background = 'transparent';
        btnSingleApp.style.color = 'var(--text-muted)';
      }
      singleAppInputs.style.display = 'none';
      batchAppInputs.style.display = 'block';
    });
  }

  // Password Visibility Toggle
  const togglePasswordBtn = document.getElementById('toggle-password-btn');
  const eyeIcon = document.getElementById('eye-icon');
  const eyeOffIcon = document.getElementById('eye-off-icon');

  if (togglePasswordBtn && passwordInput && eyeIcon && eyeOffIcon) {
    togglePasswordBtn.addEventListener('click', () => {
      const isPassword = passwordInput.type === 'password';
      passwordInput.type = isPassword ? 'text' : 'password';
      eyeIcon.style.display = isPassword ? 'none' : 'block';
      eyeOffIcon.style.display = isPassword ? 'block' : 'none';
      const labelText = isPassword ? 'Hide password' : 'Show password';
      togglePasswordBtn.setAttribute('title', labelText);
      togglePasswordBtn.setAttribute('aria-label', labelText);
    });
  }

  // Initialize App
  initWebSocket();
  fetchModels();
  loadHistory();

  // 1. Initialize WebSocket Connection with session switching support
  function initWebSocket(sessionId = null) {
    if (socket) {
      try { socket.close(); } catch (e) {}
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = sessionId 
      ? `${protocol}//${window.location.host}/ws/${sessionId}`
      : `${protocol}//${window.location.host}/ws`;

    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      appendLog(`[WebSocket] Live stream connected ${sessionId ? 'for session ' + sessionId : ''}.`);
    };

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    socket.onclose = () => {
      setTimeout(() => initWebSocket(activeRunId), 5000);
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  // 2. Handle incoming WebSocket events
  function handleWebSocketMessage(msg) {
    const runId = msg.run_id;

    if (msg.type === 'BATCH_STARTED') {
      activeBatchId = msg.batch_id;
      appendLog(`[Batch Started] Launching parallel audit across ${msg.total_apps} apps. Batch ID: ${msg.batch_id}`);
      startBtn.disabled = true;
      startBtn.innerHTML = '<span>⏳ Parallel Batch Audit Running...</span>';
    }
    else if (msg.type === 'RUN_STARTED') {
      if (!runStates[runId]) {
        runStates[runId] = {
          run_id: runId,
          target_url: msg.target_url,
          status: 'running',
          logs: [],
          step: 0,
          state: 'Started',
          screenshot: null,
          summary: null
        };
      }
      activeRunId = runId;
      renderRunTabs();
      switchRunTab(runId);

      appendLog(`[Run Started] Target: ${msg.target_url}`, '', runId);
      startBtn.disabled = true;
      startBtn.innerHTML = '<span>⏳ Agent Exploration Running...</span>';
    } 
    else if (msg.type === 'STEP_UPDATE') {
      const data = msg.data;
      if (!runStates[runId]) {
        runStates[runId] = { run_id: runId, target_url: data.target_url || 'Active App', status: 'running', logs: [], step: 0, state: 'Running', screenshot: null, summary: null };
      }

      runStates[runId].step = data.step;
      if (data.screenshot_url) {
        runStates[runId].screenshot = `${data.screenshot_url}?t=${Date.now()}`;
      }

      appendLog(`[Step ${data.step}] Action: ${data.action.toUpperCase()} ${data.target_selector ? '(' + data.target_selector + ')' : ''}`, '', runId);
      if (data.reasoning) {
        appendLog(`  └─ Reasoning: ${data.reasoning}`, '', runId);
      }
      if (data.issues && data.issues.length > 0) {
        data.issues.forEach(issue => appendLog(`  🚨 Issue: ${issue}`, 'log-issue', runId));
      }

      if (activeRunId === runId) {
        updateViewportUI(runId);
      }
    } 
    else if (msg.type === 'AGENT_STATE') {
      if (runStates[runId]) {
        runStates[runId].state = msg.message;
      }
      appendLog(`[Agent State] ${msg.message}`, '', runId);
      
      const multiAgentStream = document.getElementById('multi-agent-stream');
      if (multiAgentStream) {
        const timeStr = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.style.marginBottom = '4px';
        if (msg.message.includes('👁️')) entry.style.color = '#38bdf8';
        else if (msg.message.includes('🧠')) entry.style.color = '#a78bfa';
        else if (msg.message.includes('🔒')) entry.style.color = '#f87171';
        else if (msg.message.includes('🤝')) entry.style.color = '#34d399';
        else entry.style.color = '#94a3b8';
        
        entry.innerHTML = `<span style="color: #64748b;">[${timeStr}]</span> ${msg.message}`;
        multiAgentStream.appendChild(entry);
        multiAgentStream.scrollTop = multiAgentStream.scrollHeight;
      }

      if (activeRunId === runId) {
        agentStateIndicator.textContent = `Agent: ${msg.message}`;
      }
    }
    else if (msg.type === 'RUN_COMPLETED') {
      if (runStates[runId]) {
        runStates[runId].status = 'completed';
        runStates[runId].summary = msg.summary;
      }
      appendLog(`[Run Completed] Audit session finished successfully!`, '', runId);
      renderRunTabs();
      loadHistory();

      if (!activeBatchId) {
        startBtn.disabled = false;
        startBtn.innerHTML = '<span>▶ Start Autonomous Audit</span>';
        loadRunDetails(runId);
      }
    } 
    else if (msg.type === 'RUN_FAILED') {
      if (runStates[runId]) {
        runStates[runId].status = 'failed';
      }
      appendLog(`[Run Failed] Error: ${msg.error || 'Execution interrupted'}`, 'log-issue', runId);
      renderRunTabs();
      loadHistory();

      if (!activeBatchId) {
        startBtn.disabled = false;
        startBtn.innerHTML = '<span>▶ Start Autonomous Audit</span>';
        loadRunDetails(runId);
      }
    }
    else if (msg.type === 'BATCH_COMPLETED') {
      appendLog(`[Batch Completed] Parallel multi-app audit finished successfully!`);
      startBtn.disabled = false;
      startBtn.innerHTML = '<span>▶ Start Autonomous Audit</span>';
      renderMasterBatchReport(msg.master_report);
      loadHistory();
    }
  }

  // 3. Tabbed UI Switching Logic
  function renderRunTabs() {
    liveRunTabs.innerHTML = '';
    const keys = Object.keys(runStates);
    if (keys.length === 0) return;

    keys.forEach(rId => {
      const state = runStates[rId];
      const pill = document.createElement('button');
      pill.className = 'btn-secondary';
      pill.style.fontSize = '12px';
      pill.style.padding = '6px 12px';
      pill.style.borderRadius = '20px';
      pill.style.whiteSpace = 'nowrap';

      if (rId === activeRunId) {
        pill.style.background = 'var(--primary)';
        pill.style.color = 'white';
      } else {
        pill.style.background = 'rgba(255,255,255,0.05)';
      }

      const shortUrl = state.target_url.replace("http://", "").replace("https://", "");
      pill.textContent = `📍 ${shortUrl.substring(0, 18)}... (Step ${state.step})`;
      pill.addEventListener('click', () => switchRunTab(rId));
      liveRunTabs.appendChild(pill);
    });

    if (activeBatchId) {
      const masterPill = document.createElement('button');
      masterPill.className = 'btn-secondary';
      masterPill.style.fontSize = '12px';
      masterPill.style.padding = '6px 12px';
      masterPill.style.borderRadius = '20px';
      masterPill.style.background = activeRunId === 'MASTER_BATCH' ? 'var(--accent)' : 'rgba(6, 182, 212, 0.2)';
      masterPill.style.color = 'white';
      masterPill.textContent = `🏆 Master Batch Summary`;
      masterPill.addEventListener('click', () => switchRunTab('MASTER_BATCH'));
      liveRunTabs.appendChild(masterPill);
    }
  }

  function switchRunTab(runId) {
    activeRunId = runId;
    renderRunTabs();

    if (runId === 'MASTER_BATCH') {
      if (activeBatchId) fetchBatchDetails(activeBatchId);
      return;
    }

    const state = runStates[runId];
    if (!state) return;

    updateViewportUI(runId);

    // Render Tab Logs
    terminalLogs.innerHTML = '';
    state.logs.forEach(l => {
      const entry = document.createElement('div');
      entry.className = `log-entry ${l.className}`;
      entry.innerHTML = `<span class="log-time">[${l.time}]</span> ${l.text}`;
      terminalLogs.appendChild(entry);
    });
    terminalLogs.scrollTop = terminalLogs.scrollHeight;

    if (state.summary) {
      loadRunDetails(runId);
    }
  }

  function updateViewportUI(runId) {
    const state = runStates[runId];
    if (!state) return;

    stepIndicator.textContent = `Step ${state.step}`;
    agentStateIndicator.textContent = `Agent: ${state.state}`;

    if (state.screenshot) {
      screenPlaceholder.style.display = 'none';
      liveScreenshot.style.display = 'block';
      liveScreenshot.className = 'browser-screen w-full h-full object-contain max-h-[500px] block mx-auto';
      liveScreenshot.src = state.screenshot;
    }
  }

  // 4. Form Submissions (Single vs Batch)
  testForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    startBtn.disabled = true;
    startBtn.innerHTML = '<span>⏳ Launching Audit...</span>';

    if (isBatchMode) {
      const rawText = (multiAppUrlsInput ? multiAppUrlsInput.value : "").trim();
      const lines = rawText.split('\n').filter(l => l.trim().length > 0);
      if (lines.length === 0) {
        alert('Please enter at least one target application URL.');
        startBtn.disabled = false;
        startBtn.innerHTML = '<span>▶ Start Autonomous Audit</span>';
        return;
      }

      const apps = lines.map(line => {
        const parts = line.split(',').map(p => p.trim());
        let url = parts[0];
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
          url = 'http://' + url;
        }
        return {
          target_url: url,
          username: parts[1] || null,
          password: parts[2] || null
        };
      });

      const payload = {
        apps: apps,
        model_name: modelSelect.value,
        max_steps: 200,
        headless: headlessCheck.checked
      };

      try {
        const res = await fetch('/api/runs/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok) {
          activeBatchId = data.batch_id;
          appendLog(`[System] Multi-App Batch Audit initiated: ${data.batch_id} (${data.total_apps} apps)`);
        } else {
          alert(`Failed to launch batch audit: ${data.detail || 'Error'}`);
          startBtn.disabled = false;
          startBtn.innerHTML = '<span>▶ Start Autonomous Audit</span>';
        }
      } catch (err) {
        alert(`Error launching batch audit: ${err.message}`);
        startBtn.disabled = false;
        startBtn.innerHTML = '<span>▶ Start Autonomous Audit</span>';
      }
    } else {
      let rawUrl = targetUrlInput.value.trim();
      if (!rawUrl) {
        alert('Please enter a target application URL.');
        startBtn.disabled = false;
        startBtn.innerHTML = '<span>▶ Start Autonomous Audit</span>';
        return;
      }
      if (!rawUrl.startsWith('http://') && !rawUrl.startsWith('https://')) {
        rawUrl = 'http://' + rawUrl;
        targetUrlInput.value = rawUrl;
      }

      const payload = {
        target_url: rawUrl,
        username: usernameInput.value.trim() || null,
        password: passwordInput.value.trim() || null,
        model_name: modelSelect.value,
        max_steps: 200,
        headless: headlessCheck.checked
      };

      try {
        const res = await fetch('/api/runs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok) {
          appendLog(`[System] Local AI Audit initiated: ${data.run_id}`);
        } else {
          alert(`Failed to start run: ${data.detail || 'Error'}`);
          startBtn.disabled = false;
          startBtn.innerHTML = '<span>▶ Start Autonomous Audit</span>';
        }
      } catch (err) {
        alert(`Error launching run: ${err.message}`);
        startBtn.disabled = false;
        startBtn.innerHTML = '<span>▶ Start Autonomous Audit</span>';
      }
    }
  });

  // 5. Fetch Models & Status
  async function fetchModels() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      
      modelSelect.innerHTML = '';
      data.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        if (model === data.default_model) {
          option.selected = true;
          option.textContent += ' (Default)';
        }
        modelSelect.appendChild(option);
      });

      geminiStatusText.textContent = 'Local Vision Engine Ready';
      geminiStatusText.parentElement.style.borderColor = 'rgba(52, 211, 153, 0.4)';
    } catch (err) {
      geminiStatusText.textContent = 'Server Offline';
    }
  }

  // 6. Load Audit History
  async function loadHistory() {
    try {
      const res = await fetch('/api/runs');
      const data = await res.json();
      historyContainer.innerHTML = '';

      if (!data.runs || data.runs.length === 0) {
        historyContainer.innerHTML = '<div style="font-size: 12px; color: var(--text-dim); text-align: center; padding: 20px;">No past audits found.</div>';
        return;
      }

      data.runs.forEach(run => {
        const item = document.createElement('div');
        item.className = 'history-item';
        const isCompleted = run.status === 'completed';
        item.innerHTML = `
          <div>
            <div class="history-url">${run.target_url}</div>
            <div class="history-meta">${run.run_id} • ${run.status}</div>
          </div>
          <span style="font-size: 11px; font-weight: bold; color: ${isCompleted ? '#34d399' : '#f87171'}">${run.duration_seconds ? run.duration_seconds + 's' : ''}</span>
        `;
        item.addEventListener('click', () => loadRunDetails(run.run_id));
        historyContainer.appendChild(item);
      });
    } catch (err) {
      console.error('History load error:', err);
    }
  }

  refreshHistoryBtn.addEventListener('click', loadHistory);

  async function downloadDocxFile(runId) {
    if (!runId) return;
    try {
      const url = `/api/reports/${runId}/docx`;
      const res = await fetch(url);
      if (res.status === 202) {
        alert('⏳ Audit report DOCX is currently generating. Please try again in a few seconds.');
        return;
      }
      if (!res.ok) {
        window.open(url, '_blank');
        return;
      }
      const blob = await res.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = `${runId}_audit_report.docx`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(blobUrl);
    } catch (err) {
      console.warn('Direct fetch failed, opening fallback window:', err);
      window.open(`/api/reports/${runId}/docx`, '_blank');
    }
  }

  // 7. Render Run Details
  async function loadRunDetails(runId) {
    activeRunId = runId;
    initWebSocket(runId);
    try {
      const res = await fetch(`/api/runs/${runId}`);
      if (!res.ok) return;
      const data = await res.json();

      const summary = data.summary || {};
      statSteps.textContent = data.steps ? data.steps.length : summary.total_steps || 0;
      statDuration.textContent = summary.duration_seconds ? `${summary.duration_seconds}s` : '0s';
      statIssues.textContent = summary.critical_bugs ? summary.critical_bugs.length : 0;
      statUx.textContent = summary.overall_ux_rating || 'N/A';

      reportDownloadActions.style.display = 'flex';
      if (btnDownloadDocx) btnDownloadDocx.onclick = () => downloadDocxFile(runId);
      btnDownloadJson.onclick = () => window.open(`/api/runs/${runId}/download/json`, '_blank');
      btnDownloadMd.onclick = () => window.open(`/api/runs/${runId}/download/markdown`, '_blank');

      let html = '<div style="display: flex; flex-direction: column; gap: 20px;">';

      html += '<div><h3 style="color: #f87171; font-size: 14px; margin-bottom: 8px;">🚨 Critical Bugs & Visual Anomalies</h3>';
      if (summary.critical_bugs && summary.critical_bugs.length > 0) {
        html += '<ul style="padding-left: 20px; color: var(--text-main); font-size: 13px;">';
        summary.critical_bugs.forEach(b => html += `<li style="margin-bottom: 6px;">${b}</li>`);
        html += '</ul>';
      } else {
        html += '<p style="font-size: 13px; color: #34d399;">✅ Zero critical visual bugs or DOM errors observed.</p>';
      }
      html += '</div>';

      html += '<div><h3 style="color: #38bdf8; font-size: 14px; margin-bottom: 8px;">💡 UX & Usability Recommendations</h3>';
      if (summary.ux_recommendations && summary.ux_recommendations.length > 0) {
        html += '<ul style="padding-left: 20px; color: var(--text-main); font-size: 13px;">';
        summary.ux_recommendations.forEach(r => html += `<li style="margin-bottom: 6px;">${r}</li>`);
        html += '</ul>';
      } else {
        html += '<p style="font-size: 13px; color: var(--text-muted);">No UX issues flagged.</p>';
      }
      html += '</div>';

      if (data.steps && data.steps.length > 0) {
        html += '<div><h3 style="font-size: 14px; margin-bottom: 12px;">🐾 Exploration Step Trajectory</h3>';
        html += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px;">';
        data.steps.forEach(s => {
          html += `
            <div style="background: rgba(10,15,29,0.6); border: 1px solid var(--border-color); border-radius: 8px; padding: 10px;">
              <div style="font-size: 11px; font-weight: bold; color: var(--primary);">Step ${s.step_number}: ${s.action.toUpperCase()}</div>
              <div style="font-size: 11px; color: var(--text-dim); margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${s.target_selector || 'N/A'}</div>
              ${s.screenshot_path ? `<img src="/storage/runs/${runId}/${s.screenshot_path}" style="width:100%; height:120px; object-fit:cover; border-radius:4px;">` : ''}
            </div>
          `;
        });
        html += '</div></div>';
      }

      html += '</div>';
      reportDetailsBox.innerHTML = html;

    } catch (err) {
      console.error('Error loading run details:', err);
    }
  }

  // 8. Render Master Batch Report Dashboard
  function renderMasterBatchReport(masterData) {
    if (!masterData) return;

    statSteps.textContent = masterData.total_steps || 0;
    statDuration.textContent = masterData.parallel_duration_seconds ? `${masterData.parallel_duration_seconds.toFixed(1)}s` : '0s';
    statIssues.textContent = masterData.total_critical_bugs || 0;
    statUx.textContent = 'Batch Audit Complete';

    reportDownloadActions.style.display = 'flex';
    btnDownloadJson.onclick = () => {};
    btnDownloadMd.onclick = () => window.open(`/api/batch/${masterData.batch_id}/download/markdown`, '_blank');

    let html = '<div style="display: flex; flex-direction: column; gap: 20px;">';
    html += '<h3 style="color: var(--accent); font-size: 16px;">🚀 Master Batch Multi-App Comparative Analysis</h3>';
    
    html += `
      <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;">
        <thead>
          <tr style="border-bottom: 1px solid var(--border-color); color: var(--text-muted);">
            <th style="padding: 8px;">Target Application</th>
            <th style="padding: 8px;">Steps</th>
            <th style="padding: 8px;">Duration</th>
            <th style="padding: 8px;">Critical Bugs</th>
            <th style="padding: 8px;">UX Rating</th>
            <th style="padding: 8px;">Action</th>
          </tr>
        </thead>
        <tbody>
    `;

    (masterData.app_summaries || []).forEach(s => {
      html += `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
          <td style="padding: 10px; font-weight: 500;">${s.target_url}</td>
          <td style="padding: 10px;">${s.total_steps || 0}</td>
          <td style="padding: 10px;">${s.duration_seconds ? s.duration_seconds.toFixed(1) + 's' : '0s'}</td>
          <td style="padding: 10px; color: ${s.critical_bugs && s.critical_bugs.length > 0 ? '#f87171' : '#34d399'};">${s.critical_bugs ? s.critical_bugs.length : 0}</td>
          <td style="padding: 10px; font-weight: bold; color: #34d399;">${s.overall_ux_rating || 'N/A'}</td>
          <td style="padding: 10px;"><button class="btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="loadRunDetails('${s.run_id}')">View Details</button></td>
        </tr>
      `;
    });

    html += '</tbody></table></div>';
    reportDetailsBox.innerHTML = html;
  }

  async function fetchBatchDetails(batchId) {
    try {
      const res = await fetch(`/api/batch/${batchId}`);
      if (res.ok) {
        const data = await res.json();
        renderMasterBatchReport(data);
      }
    } catch (e) {
      console.error("Batch details load error:", e);
    }
  }

  // Helper log appender with run_id routing
  function appendLog(text, className = '', runId = null) {
    const timeStr = new Date().toLocaleTimeString();
    const logObj = { time: timeStr, text: text, className: className };

    if (runId && runStates[runId]) {
      runStates[runId].logs.push(logObj);
    }

    if (!runId || runId === activeRunId) {
      const entry = document.createElement('div');
      entry.className = `log-entry ${className}`;
      entry.innerHTML = `<span class="log-time">[${timeStr}]</span> ${text}`;
      terminalLogs.appendChild(entry);
      terminalLogs.scrollTop = terminalLogs.scrollHeight;
    }
  }
});

