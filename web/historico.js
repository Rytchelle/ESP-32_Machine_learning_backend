/**
 * Histórico de Eventos - VibrationML
 */

(() => {
  // Elementos
  const historyList = document.getElementById('history-list');
  const totalNormal = document.getElementById('total-normal');
  const totalAlerts = document.getElementById('total-alerts');
  const totalAnomalies = document.getElementById('total-anomalies');
  const sessionTime = document.getElementById('session-time');
  const urgentBanner = document.getElementById('urgent-banner');
  const urgentDesc = document.getElementById('urgent-desc');
  const sensorIndicator = document.getElementById('sensor-indicator');
  const sensorStatusText = document.getElementById('sensor-status-text');

  // Estado
  let events = JSON.parse(localStorage.getItem('vibration_events') || '[]');
  let counts = JSON.parse(localStorage.getItem('vibration_counts') || '{"normal":0,"alerts":0,"anomalies":0}');
  let sessionStart = parseInt(localStorage.getItem('session_start') || Date.now());
  let lastStatus = 'green';

  // Salva início da sessão se não existir
  if (!localStorage.getItem('session_start')) {
    localStorage.setItem('session_start', sessionStart.toString());
  }

  // Atualiza contadores na tela
  function updateDisplay() {
    totalNormal.textContent = counts.normal;
    totalAlerts.textContent = counts.alerts;
    totalAnomalies.textContent = counts.anomalies;
    
    // Verifica aviso urgente (5 ou mais anomalias)
    if (counts.anomalies >= 5) {
      urgentBanner.classList.remove('hidden');
      urgentDesc.textContent = `${counts.anomalies} anomalias detectadas - VERIFICAR MÁQUINA IMEDIATAMENTE!`;
    } else {
      urgentBanner.classList.add('hidden');
    }
    
    // Renderiza lista de eventos
    renderEvents();
  }

  // Renderiza lista de eventos
  function renderEvents() {
    if (events.length === 0) {
      historyList.innerHTML = `
        <div class="history-empty">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>
          </svg>
          <p>Nenhum evento registrado ainda</p>
        </div>
      `;
      return;
    }

    // Mostra últimos 50 eventos (mais recentes primeiro)
    const recentEvents = events.slice(-50).reverse();
    
    historyList.innerHTML = recentEvents.map(event => {
      const time = new Date(event.timestamp).toLocaleTimeString('pt-BR');
      const date = new Date(event.timestamp).toLocaleDateString('pt-BR');
      
      let icon, className, label;
      if (event.type === 'anomaly') {
        icon = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>`;
        className = 'event-anomaly';
        label = 'ANOMALIA';
      } else if (event.type === 'alert') {
        icon = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>`;
        className = 'event-alert';
        label = 'ALERTA';
      } else {
        return '';
      }
      
      return `
        <div class="history-item ${className}">
          <div class="event-icon">${icon}</div>
          <div class="event-info">
            <span class="event-label">${label}</span>
            <span class="event-details">Confiança: ${(event.confidence * 100).toFixed(1)}% | Distância: ${event.distance?.toFixed(2) || '—'}</span>
          </div>
          <div class="event-time">
            <span class="event-date">${date}</span>
            <span class="event-hour">${time}</span>
          </div>
        </div>
      `;
    }).join('');
  }

  // Atualiza tempo de sessão
  function updateSessionTime() {
    const elapsed = Date.now() - sessionStart;
    const hours = Math.floor(elapsed / 3600000);
    const minutes = Math.floor((elapsed % 3600000) / 60000);
    const seconds = Math.floor((elapsed % 60000) / 1000);
    sessionTime.textContent = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }

  // Adiciona evento
  function addEvent(type, data) {
    const event = {
      type,
      timestamp: Date.now(),
      confidence: data.confidence || 0,
      distance: data.distance || 0,
      threshold: data.threshold || 0
    };
    
    events.push(event);
    
    // Mantém apenas últimos 500 eventos
    if (events.length > 500) {
      events = events.slice(-500);
    }
    
    // Atualiza contadores
    if (type === 'anomaly') {
      counts.anomalies++;
    } else if (type === 'alert') {
      counts.alerts++;
    } else {
      counts.normal++;
    }
    
    // Salva no localStorage
    localStorage.setItem('vibration_events', JSON.stringify(events));
    localStorage.setItem('vibration_counts', JSON.stringify(counts));
    
    updateDisplay();
  }

  // Limpa histórico
  window.clearHistory = function() {
    if (confirm('Tem certeza que deseja limpar todo o histórico?')) {
      events = [];
      counts = { normal: 0, alerts: 0, anomalies: 0 };
      sessionStart = Date.now();
      localStorage.setItem('vibration_events', '[]');
      localStorage.setItem('vibration_counts', JSON.stringify(counts));
      localStorage.setItem('session_start', sessionStart.toString());
      updateDisplay();
    }
  };

  // WebSocket para receber eventos em tempo real
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    try {
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        sensorIndicator.classList.add('connected');
        sensorStatusText.textContent = 'Conectado';
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'prediction' && data.status) {
            const color = data.status.status_color || 'green';
            
            // Registra evento quando muda de estado
            if (color !== lastStatus) {
              if (color === 'red') {
                addEvent('anomaly', data.status);
              } else if (color === 'yellow') {
                addEvent('alert', data.status);
              } else if (lastStatus !== 'green') {
                // Voltou ao normal
                counts.normal++;
                localStorage.setItem('vibration_counts', JSON.stringify(counts));
                updateDisplay();
              }
              lastStatus = color;
            }
          }
        } catch (e) {
          console.error('[WS] Erro:', e);
        }
      };
      
      ws.onclose = () => {
        sensorIndicator.classList.remove('connected');
        sensorStatusText.textContent = 'Desconectado';
        setTimeout(connectWebSocket, 2000);
      };
      
    } catch (e) {
      console.error('[WS] Falha:', e);
    }
  }

  // Inicialização
  updateDisplay();
  setInterval(updateSessionTime, 1000);
  connectWebSocket();
})();
