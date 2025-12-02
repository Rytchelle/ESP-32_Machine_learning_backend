/**
 * Monitor de Vibração em Tempo Real
 */
(() => {
  const $ = id => document.getElementById(id);
  
  // Elementos
  const statusBadge = $('status-badge');
  const statusText = $('status-text');
  const situationEl = $('situation');
  const mainIcon = $('main-icon');
  const mainStatusCard = $('main-status-card');
  const confidenceEl = $('confidence');
  const confidenceBar = $('confidence-bar');
  const distanceEl = $('distance');
  const thresholdEl = $('threshold');
  const updatedAtEl = $('updatedAt');
  const samplesCountEl = $('samples-count');
  const alertBanner = $('alert-banner');
  const alertText = $('alert-text');
  const sensorIndicator = $('sensor-indicator');
  const sensorStatusText = $('sensor-status-text');
  const fpsCounter = $('fps-counter');

  let lastStatus = 'green';
  let lastDataTime = 0;
  let updateCount = 0;
  let lastFpsTime = Date.now();

  // Chart.js
  const ctx = $('vibrationChart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'X', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.4 },
        { label: 'Y', data: [], borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.1)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.4 },
        { label: 'Z', data: [], borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.4 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(51,65,85,0.5)' } }
      }
    }
  });

  const icons = {
    check: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    warning: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    alert: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    offline: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="1" y1="1" x2="23" y2="23"/><path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>'
  };

  function saveToHistory(color, state) {
    if (color === 'green') return;
    try {
      const events = JSON.parse(localStorage.getItem('vibration_events') || '[]');
      const counts = JSON.parse(localStorage.getItem('vibration_counts') || '{"normal":0,"alerts":0,"anomalies":0}');
      
      events.push({
        type: color === 'red' ? 'anomaly' : 'alert',
        timestamp: Date.now(),
        confidence: state?.confidence || 0,
        distance: state?.distance || 0,
        threshold: state?.threshold || 0
      });
      
      if (events.length > 500) events.splice(0, events.length - 500);
      if (color === 'red') counts.anomalies++;
      else counts.alerts++;
      
      localStorage.setItem('vibration_events', JSON.stringify(events));
      localStorage.setItem('vibration_counts', JSON.stringify(counts));
    } catch(e) {}
  }

  function updateUI(data) {
    const color = data.status_color || 'green';
    const conf = data.confidence || 0;
    const dist = data.distance || 0;
    const thresh = data.threshold || 0;
    
    if (color !== lastStatus) {
      saveToHistory(color, data);
      lastStatus = color;
    }
    
    statusBadge.className = 'status-badge' + (color !== 'green' ? ' ' + color : '');
    mainStatusCard.className = 'metric-card main-status' + (color !== 'green' ? ' ' + color : '');
    
    const labels = {
      green: ['Normal', icons.check, 'Sistema operando normalmente'],
      yellow: ['Alerta', icons.warning, 'Comportamento suspeito'],
      red: ['ANOMALIA', icons.alert, 'Anomalia detectada!']
    };
    const [text, icon, footer] = labels[color] || labels.green;
    
    statusText.textContent = text;
    situationEl.textContent = text;
    mainIcon.innerHTML = icon;
    mainStatusCard.querySelector('.metric-footer').textContent = footer;
    
    const confPct = (conf * 100).toFixed(1);
    confidenceEl.textContent = confPct + '%';
    confidenceBar.style.width = confPct + '%';
    distanceEl.textContent = dist.toFixed(3);
    thresholdEl.textContent = thresh.toFixed(3);
    updatedAtEl.textContent = new Date().toLocaleTimeString('pt-BR');
    
    if (color === 'red') {
      alertText.textContent = 'Anomalia detectada! Confiança: ' + confPct + '%';
      alertBanner.className = 'alert-banner error';
    } else if (color === 'yellow') {
      alertText.textContent = 'Comportamento suspeito - monitorando...';
      alertBanner.className = 'alert-banner warning';
    } else {
      alertBanner.className = 'alert-banner hidden';
    }
    
    updateCount++;
  }

  function updateChart(samples) {
    if (!samples || !samples.length) return;
    const data = samples.slice(-200);
    chart.data.labels = data.map((_, i) => i);
    chart.data.datasets[0].data = data.map(s => s.x);
    chart.data.datasets[1].data = data.map(s => s.y);
    chart.data.datasets[2].data = data.map(s => s.z);
    chart.update('none');
    samplesCountEl.textContent = data.length;
  }

  function showOffline() {
    statusBadge.className = 'status-badge disconnected';
    mainStatusCard.className = 'metric-card main-status disconnected';
    statusText.textContent = 'Sem Dados';
    situationEl.textContent = 'Aguardando';
    mainIcon.innerHTML = icons.offline;
    mainStatusCard.querySelector('.metric-footer').textContent = 'ESP32 desconectado';
    alertBanner.className = 'alert-banner hidden';
    sensorIndicator.className = 'sensor-indicator error';
    sensorStatusText.textContent = 'Sem dados';
  }

  // WebSocket
  let ws = null;
  
  function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(protocol + '//' + location.host + '/ws');
    
    ws.onopen = () => {
      sensorIndicator.className = 'sensor-indicator connected';
      sensorStatusText.textContent = 'Conectado';
    };
    
    ws.onmessage = (e) => {
      lastDataTime = Date.now();
      try {
        const d = JSON.parse(e.data);
        if (d.type === 'update' || d.type === 'init') {
          updateUI(d);
          if (d.samples) updateChart(d.samples);
          sensorIndicator.className = 'sensor-indicator connected';
          sensorStatusText.textContent = 'Recebendo';
        }
      } catch(err) {}
    };
    
    ws.onclose = () => {
      sensorIndicator.className = 'sensor-indicator error';
      sensorStatusText.textContent = 'Reconectando...';
      setTimeout(connect, 1500);
    };
    
    ws.onerror = () => ws.close();
  }

  // Ping
  setInterval(() => {
    if (ws && ws.readyState === 1) ws.send('{"type":"ping"}');
  }, 15000);

  // Timeout sensor
  setInterval(() => {
    if (lastDataTime && Date.now() - lastDataTime > 8000) showOffline();
  }, 2000);

  // FPS counter
  setInterval(() => {
    const now = Date.now();
    const fps = Math.round(updateCount / ((now - lastFpsTime) / 1000));
    if (fpsCounter) fpsCounter.textContent = fps + ' upd/s';
    updateCount = 0;
    lastFpsTime = now;
  }, 1000);

  // Init
  connect();
  
  // Fallback polling caso WebSocket falhe
  setTimeout(() => {
    if (!lastDataTime) {
      setInterval(async () => {
        try {
          const res = await fetch('/realtime/state');
          if (res.ok) {
            const data = await res.json();
            updateUI(data);
            lastDataTime = Date.now();
          }
        } catch(e) {}
      }, 1000);
    }
  }, 5000);
})();
