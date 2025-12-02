/**
 * Monitor de Vibração - Versão Otimizada
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

  // Estado mínimo
  let lastStatus = 'green';
  let lastDataTime = 0;
  let threshold = 0;

  // Chart.js config otimizada
  const ctx = $('vibrationChart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'X', data: [], borderColor: '#3b82f6', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
        { label: 'Y', data: [], borderColor: '#8b5cf6', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
        { label: 'Z', data: [], borderColor: '#22c55e', borderWidth: 1.5, pointRadius: 0, tension: 0.3 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(51,65,85,0.3)' } }
      }
    }
  });

  // Ícones
  const icons = {
    check: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    warning: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    alert: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    offline: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="1" y1="1" x2="23" y2="23"/><path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"/><path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>'
  };

  function updateUI(color, conf, dist, isAnomaly) {
    // Atualiza classes
    statusBadge.className = 'status-badge ' + (color !== 'green' ? color : '');
    mainStatusCard.className = 'metric-card main-status ' + (color !== 'green' ? color : '');
    
    // Labels
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
    
    // Métricas
    const confPct = (conf * 100).toFixed(1);
    confidenceEl.textContent = confPct + '%';
    confidenceBar.style.width = confPct + '%';
    distanceEl.textContent = dist.toFixed(3);
    thresholdEl.textContent = threshold.toFixed(3);
    updatedAtEl.textContent = new Date().toLocaleTimeString('pt-BR');
    
    // Alerta
    if (color === 'red') {
      alertText.textContent = 'Anomalia detectada! Confiança: ' + confPct + '%';
      alertBanner.className = 'alert-banner error';
    } else if (color === 'yellow') {
      alertText.textContent = 'Comportamento suspeito - monitorando...';
      alertBanner.className = 'alert-banner warning';
    } else {
      alertBanner.className = 'alert-banner hidden';
    }
    
    lastStatus = color;
  }

  function updateChart(samples) {
    if (!samples || !samples.length) return;
    chart.data.labels = samples.map((_, i) => i);
    chart.data.datasets[0].data = samples.map(s => s.x);
    chart.data.datasets[1].data = samples.map(s => s.y);
    chart.data.datasets[2].data = samples.map(s => s.z);
    chart.update('none');
    samplesCountEl.textContent = samples.length;
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
      const d = JSON.parse(e.data);
      
      if (d.type === 'init') {
        threshold = d.t || 0;
        updateUI(d.s || 'green', d.c || 0, d.d || 0, false);
      } else if (d.type === 'update') {
        updateUI(d.s, d.c, d.d, d.a);
        if (d.samples) updateChart(d.samples);
        sensorIndicator.className = 'sensor-indicator connected';
        sensorStatusText.textContent = 'Recebendo';
      } else if (d.type === 'pong') {
        // keepalive ok
      }
    };
    
    ws.onclose = () => {
      sensorIndicator.className = 'sensor-indicator error';
      sensorStatusText.textContent = 'Reconectando...';
      setTimeout(connect, 2000);
    };
    
    ws.onerror = () => ws.close();
  }

  // Ping keepalive
  setInterval(() => {
    if (ws && ws.readyState === 1) {
      ws.send('{"type":"ping"}');
    }
  }, 20000);

  // Detecta timeout do sensor
  setInterval(() => {
    if (lastDataTime && Date.now() - lastDataTime > 8000) {
      showOffline();
    }
  }, 3000);

  // Init
  connect();
})();
