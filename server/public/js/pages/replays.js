const ReplaysPage = {
  _matches: [],

  render() {
    return `<div class="replays-page" id="replaysPage">
      <div class="replays-section-title">AI Highlights</div>

      <!-- Billiards AI highlight -->
      <div class="replay-feature-card">
        <div class="replay-feature-header">
          <div class="replay-feature-meta">
            <span class="replay-sport-badge">🎱 Billiards</span>
            <span class="replay-ai-badge">AI</span>
          </div>
          <div class="replay-feature-title">Real Game · Goals Reel</div>
          <div class="replay-feature-sub">Auto-detected goals · Wide + Zoom + Slo-Mo</div>
        </div>
        <div class="replay-video-wrap">
          <video
            id="billiardsVideo"
            src="/highlights/billiards"
            controls
            playsinline
            preload="metadata"
            poster=""
          ></video>
        </div>
      </div>

      <!-- Full-game 16-min replay -->
      <div class="replay-feature-card" style="margin-top:16px" onclick="navigate('watch', { code: 'GAME02' })" style="cursor:pointer">
        <div class="replay-feature-header">
          <div class="replay-feature-meta">
            <span class="replay-sport-badge">🎱 Billiards</span>
            <span class="replay-ai-badge">FULL GAME</span>
          </div>
          <div class="replay-feature-title">Real Game 2 · Full 16-min</div>
          <div class="replay-feature-sub">3 synced angles · Ball tracking · Draggable timeline</div>
        </div>
        <div class="replay-video-hint">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          <span>Tap to watch</span>
        </div>
      </div>

      <div class="replays-section-title" style="margin-top:24px">Match Replays</div>
      <div id="replaysList"></div>
    </div>`;
  },

  async mount() {
    document.getElementById('topbarTitle').textContent = 'Replays';
    try {
      const res = await fetch('/api/matches');
      this._matches = await res.json();
      this._renderList();
    } catch {}
  },

  unmount() {},

  _renderList() {
    const el = document.getElementById('replaysList');
    if (!el) return;
    const finished = this._matches.filter(m => !m.isLive && m.cameras.length > 0);
    if (!finished.length) {
      el.innerHTML = `<div class="replays-empty">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        <p>Finished matches will appear here</p>
      </div>`;
      return;
    }
    el.innerHTML = finished.map(m => {
      const colA = WebRTCHelper.avatarColor(m.teamA);
      const colB = WebRTCHelper.avatarColor(m.teamB);
      const date = new Date(m.createdAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      const cams = m.cameras.length;
      return `
        <div class="replay-match-card" onclick="navigate('watch', { code: '${m.code}' })">
          <div class="replay-match-teams">
            <div class="replay-match-team">
              <div class="avatar ${colA}" style="width:36px;height:36px;font-size:12px">${WebRTCHelper.initials(m.teamA)}</div>
              <span>${m.teamA}</span>
            </div>
            <div class="replay-match-score">
              <span>–</span>
              <span class="replay-match-sport">${m.sport || 'Sport'}</span>
            </div>
            <div class="replay-match-team replay-match-team-right">
              <div class="avatar ${colB}" style="width:36px;height:36px;font-size:12px">${WebRTCHelper.initials(m.teamB)}</div>
              <span>${m.teamB}</span>
            </div>
          </div>
          <div class="replay-match-meta">
            <span>${date}</span>
            <span>·</span>
            <span>${cams} cam${cams !== 1 ? 's' : ''}</span>
          </div>
        </div>
      `;
    }).join('');
  },
};
