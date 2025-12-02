/**
 * Sistema de Monitoramento de Vibração
 * Dashboard em tempo real com WebSocket
 */

(() => {
  // Elementos do DOM
  const statusBadge = document.getElementById('status-badge');
  const statusText = document.getElementById('status-text');
  const situationEl = document.getElementById('situation');
  const mainIcon = document.getElementById('main-icon');
  const mainStatusCard = document.getElementById('main-status-card');
  const confidenceEl = document.getElementById('confidence');
  const confidenceBar = document.getElementById('confidence-bar');
  const distanceEl = document.getElementById('distance');
  const thresholdEl = document.getElementById('threshold');
  const updatedAtEl = document.getElementById('updatedAt');
  const samplesCountEl = document.getElementById('samples-count');
  const alertBanner = document.getElementById('alert-banner');
  const alertText = document.getElementById('alert-text');
  const sensorIndicator = document.getElementById('sensor-indicator');
  const sensorStatusText = document.getElementById('sensor-status-text');
  const fpsCounter = document.getElementById('fps-counter');

  // Estado
  let updateCount = 0;
  let lastFpsUpdate = Date.now();
  let samplesBuffer = [];
  const MAX_CHART_POINTS = 200;
  let sensorConnected = false;
  let lastSensorDataTime = null;
  
  // Estado para histórico (salva no localStorage)
  let lastStatus = 'green';

  // Cores do gráfico
  const colorX = '#3b82f6';
  const colorY = '#8b5cf6';
  const colorZ = '#22c55e';

  // Configuração do Chart.js
  const ctx = document.getElementById('vibrationChart').getContext('2d');
  
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { 
          label: 'X', 
          data: [], 
          borderColor: colorX, 
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4
        },
        { 
          label: 'Y', 
          data: [], 
          borderColor: colorY, 
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4
        },
        { 
          label: 'Z', 
          data: [], 
          borderColor: colorZ, 
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: false },
        tooltip: { 
          enabled: true,
          backgroundColor: '#1e293b',
          titleColor: '#f8fafc',
          bodyColor: '#94a3b8',
          borderColor: '#334155',
          borderWidth: 1,
          padding: 12,
          cornerRadius: 8
        }
      },
      scales: {
        x: {
          display: false
        },
        y: {
          ticks: { 
            color: '#64748b',
            font: { size: 11 }
          },
          grid: { 
            color: 'rgba(51, 65, 85, 0.5)',
            drawBorder: false
          }
        }
      }
    }
  });

  // Ícones SVG
  const icons = {
    check: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
    </svg>`,
    warning: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>`,
    alert: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>`
  };

  // Atualiza status do sensor
  function updateSensorStatus(status, message) {
    sensorIndicator.classList.remove('connected', 'error');
    
    if (status === 'connected') {
      sensorIndicator.classList.add('connected');
      sensorStatusText.textContent = message || 'Sensor conectado';
    } else if (status === 'error') {
      sensorIndicator.classList.add('error');
      sensorStatusText.textContent = message || 'Sensor desconectado';
    } else {
      sensorStatusText.textContent = message || 'Conectando...';
    }
  }

  // Atualiza UI com novo status
  function applyStatus(color, state) {
    // Salva eventos no localStorage para a página de histórico
    if (color !== lastStatus) {
      saveEventToHistory(color, state);
      lastStatus = color;
    }
    
    // Status badge
    statusBadge.classList.remove('yellow', 'red');
    if (color !== 'green') statusBadge.classList.add(color);
    
    // Main status card
    mainStatusCard.classList.remove('yellow', 'red');
    if (color !== 'green') mainStatusCard.classList.add(color);
    
    // Labels e ícones
    const labels = {
      green: { text: 'Normal', icon: icons.check, footer: 'Sistema operando normalmente' },
      yellow: { text: 'Alerta', icon: icons.warning, footer: 'Comportamento suspeito detectado' },
      red: { text: 'ANOMALIA', icon: icons.alert, footer: 'Anomalia detectada!' }
    };
    
    const { text, icon, footer } = labels[color] || labels.green;
    statusText.textContent = text;
    situationEl.textContent = text;
    mainIcon.innerHTML = icon;
    mainStatusCard.querySelector('.metric-footer').textContent = footer;
    
    // Métricas
    if (state) {
      const conf = (state.confidence * 100).toFixed(1);
      confidenceEl.textContent = `${conf}%`;
      confidenceBar.style.width = `${conf}%`;
      distanceEl.textContent = state.distance?.toFixed(3) ?? '—';
      thresholdEl.textContent = state.threshold?.toFixed(3) ?? '—';
      updatedAtEl.textContent = state.timestamp 
        ? new Date(state.timestamp).toLocaleTimeString('pt-BR')
        : '—';
    }
    
    // Alerta
    if (color === 'red') {
      alertText.textContent = `Anomalia detectada! Confiança: ${(state?.confidence * 100 || 0).toFixed(1)}%`;
      alertBanner.classList.remove('hidden', 'warning');
      alertBanner.classList.add('error');
    } else if (color === 'yellow') {
      alertText.textContent = 'Comportamento suspeito detectado - monitorando...';
      alertBanner.classList.remove('hidden', 'error');
      alertBanner.classList.add('warning');
    } else {
      alertBanner.classList.add('hidden');
    }
    
    updateCount++;
  }
  
  // Salva evento no localStorage para a página de histórico
  function saveEventToHistory(color, state) {
    if (color === 'green') return;
    
    const events = JSON.parse(localStorage.getItem('vibration_events') || '[]');
    const counts = JSON.parse(localStorage.getItem('vibration_counts') || '{"normal":0,"alerts":0,"anomalies":0}');
    
    const event = {
      type: color === 'red' ? 'anomaly' : 'alert',
      timestamp: Date.now(),
      confidence: state?.confidence || 0,
      distance: state?.distance || 0,
      threshold: state?.threshold || 0
    };
    
    events.push(event);
    
    // Mantém últimos 500 eventos
    if (events.length > 500) {
      events.splice(0, events.length - 500);
    }
    
    // Atualiza contadores
    if (color === 'red') {
      counts.anomalies++;
    } else if (color === 'yellow') {
      counts.alerts++;
    }
    
    localStorage.setItem('vibration_events', JSON.stringify(events));
    localStorage.setItem('vibration_counts', JSON.stringify(counts));
  }

  // Atualiza gráfico
  function updateChart(samples) {
    if (!samples || samples.length === 0) return;
    
    samplesBuffer = samples.slice(-MAX_CHART_POINTS);
    
    chart.data.labels = samplesBuffer.map((_, i) => i);
    chart.data.datasets[0].data = samplesBuffer.map(s => s.x);
    chart.data.datasets[1].data = samplesBuffer.map(s => s.y);
    chart.data.datasets[2].data = samplesBuffer.map(s => s.z);
    chart.update('none');
    
    samplesCountEl.textContent = samplesBuffer.length;
  }

  // WebSocket
  let ws = null;
  let reconnectTimer = null;

  function connectWebSocket() {
    updateSensorStatus('connecting', 'Conectando...');
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    try {
      ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('[WS] Conectado');
        updateSensorStatus('connected', 'Servidor conectado');
        if (reconnectTimer) {
          clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          switch (data.type) {
            case 'prediction':
            case 'state':
            case 'connected':
              if (data.status) {
                applyStatus(data.status.status_color || 'green', data.status);
              }
              sensorConnected = true;
              lastSensorDataTime = Date.now();
              updateSensorStatus('connected', 'Recebendo dados');
              break;
              
            case 'sensor_disconnected':
              sensorConnected = false;
              updateSensorStatus('error', 'Sensor desconectado');
              break;
              
            case 'sensor_reconnected':
              sensorConnected = true;
              lastSensorDataTime = Date.now();
              updateSensorStatus('connected', 'Sensor reconectado');
              break;
          }
          
          if (data.samples) {
            updateChart(data.samples);
          }
          
          if (data.type === 'prediction') {
            fetchSamples();
          }
        } catch (e) {
          console.error('[WS] Erro:', e);
        }
      };
      
      ws.onclose = () => {
        ws = null;
        updateSensorStatus('error', 'Reconectando...');
        reconnectTimer = setTimeout(connectWebSocket, 2000);
      };
      
      ws.onerror = () => {
        updateSensorStatus('error', 'Erro de conexão');
      };
      
    } catch (e) {
      console.error('[WS] Falha:', e);
      startPolling();
    }
  }

  // Ping keepalive
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping' }));
    }
  }, 15000);

  // Fallback: Polling HTTP
  let pollingActive = false;

  function startPolling() {
    if (pollingActive) return;
    pollingActive = true;
    
    setInterval(async () => {
      await fetchState();
      await fetchSamples();
    }, 200);
  }

  async function fetchState() {
    try {
      const res = await fetch('/realtime/state');
      if (res.ok) {
        const state = await res.json();
        applyStatus(state.status_color || 'green', state);
      }
    } catch (e) { /* ignore */ }
  }

  async function fetchSamples() {
    try {
      const res = await fetch('/realtime/samples?limit=200');
      if (res.ok) {
        const { samples } = await res.json();
        if (samples && samples.length > 0) {
          updateChart(samples);
        }
      }
    } catch (e) { /* ignore */ }
  }

  // FPS Counter
  setInterval(() => {
    const now = Date.now();
    const elapsed = (now - lastFpsUpdate) / 1000;
    const fps = Math.round(updateCount / elapsed);
    fpsCounter.textContent = `${fps} upd/s`;
    updateCount = 0;
    lastFpsUpdate = now;
  }, 1000);

  // Monitor sensor timeout - mostra "Sem dados" quando ESP desconecta
  setInterval(() => {
    if (lastSensorDataTime) {
      const timeSinceLastData = Date.now() - lastSensorDataTime;
      if (timeSinceLastData > 10000 && sensorConnected) {
        sensorConnected = false;
        showNoDataState();
      }
    }
  }, 3000);
  
  // Mostra estado "Sem dados" quando ESP desconecta
  function showNoDataState() {
    updateSensorStatus('error', 'Sem dados do sensor');
    
    // Atualiza status para cinza/desconectado
    statusBadge.classList.remove('yellow', 'red');
    statusBadge.classList.add('disconnected');
    statusText.textContent = 'Sem Dados';
    situationEl.textContent = 'Aguardando Sensor';
    mainIcon.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <line x1="1" y1="1" x2="23" y2="23"/><path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"/>
      <path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39"/><path d="M10.71 5.05A16 16 0 0 1 22.58 9"/>
      <path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/>
      <line x1="12" y1="20" x2="12.01" y2="20"/>
    </svg>`;
    mainStatusCard.classList.remove('yellow', 'red');
    mainStatusCard.classList.add('disconnected');
    mainStatusCard.querySelector('.metric-footer').textContent = 'ESP32 desconectado';
    
    // Esconde alerta
    alertBanner.classList.add('hidden');
    
    // Limpa métricas
    confidenceEl.textContent = '—';
    confidenceBar.style.width = '0%';
    distanceEl.textContent = '—';
    updatedAtEl.textContent = '—';
  }

  // Init
  console.log('[App] Iniciando monitor...');
  fetchSamples();
  fetchState();
  connectWebSocket();
  setInterval(fetchSamples, 500);
})();
