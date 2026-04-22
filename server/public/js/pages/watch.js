function _fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

// Full-game sync constants (lateral is reference)
const FG_OFFSETS  = { 1: 0, 2: 11, 3: 6 };   // seconds each camera is ahead of lateral
const FG_DURATION = 982;                        // total lateral duration in seconds
const FG_EVENTS   = [
  { t:  32, type: 'goal'     },
  { t:  90, type: 'goal'     },
  { t: 195, type: 'goal'     },
  { t: 347, type: 'goal'     },
  { t: 384, type: 'scratch'  },
  { t: 414, type: 'goal'     },
  { t: 502, type: 'scratch'  },
  { t: 503, type: 'goal'     },
  { t: 529, type: 'goal'     },
  { t: 581, type: 'goal'     },
  { t: 656, type: 'scratch'  },
  { t: 680, type: 'goal'     },
  { t: 693, type: 'goal'     },
  { t: 720, type: 'goal'     },
  { t: 957, type: 'game_over'},
];

const WatchPage = {
  _match: null,
  _pcs: new Map(),
  _camSocketIds: new Map(),
  _streams: new Map(),
  _featured: null,
  _unsubMatch: null,
  _unsubOffer: null,
  _unsubIce: null,
  _unsubCut: null,
  _directorOverride: false,
  _demoReady: false,
  _demoVids: new Map(),
  // full-game state
  _fgVids: new Map(),        // camNum → <video>
  _fgRaf: null,              // requestAnimationFrame handle
  _fgDragging: false,

  render(params) {
    this._code = params.code;
    return `
      <div class="watch-page" id="watchPage">
        <!-- Main featured video -->
        <div class="watch-video-wrap" id="watchMainWrap">
          <video id="watchVideo" autoplay playsinline></video>
          <div class="watch-placeholder" id="watchPlaceholder">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            <p>Waiting for cameras...</p>
          </div>
          <div class="watch-overlay">
            <button class="icon-btn" onclick="WatchPage.leave()">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M19 12H5M12 5l-7 7 7 7"/></svg>
            </button>
            <div id="watchLiveBadge" class="hidden"></div>
            <div id="watchCamLabel" class="watch-cam-label hidden"></div>
          </div>
        </div>

        <!-- Thumbnail rail for other angles -->
        <div class="watch-thumb-rail" id="watchThumbRail"></div>

        <!-- Match info strip (live / demo) -->
        <div class="watch-bottom" id="watchBottom">
          <div class="watch-match-info">
            <div class="watch-title" id="watchTitle">Loading...</div>
            <div class="watch-teams" id="watchTeams"></div>
          </div>
          <div class="watch-stats">
            <div class="watch-stat">
              <div class="watch-stat-val" id="statCams">–</div>
              <div class="watch-stat-label">Cameras</div>
            </div>
            <div class="watch-stat-divider"></div>
            <div class="watch-stat">
              <div class="watch-stat-val" id="statStreaming">–</div>
              <div class="watch-stat-label">Live</div>
            </div>
            <div class="watch-stat-divider"></div>
            <div class="watch-stat">
              <div class="watch-stat-val" id="statQuality">–</div>
              <div class="watch-stat-label">Quality</div>
            </div>
          </div>
        </div>

        <!-- Full-game timeline (shown for full game replays) -->
        <div class="fg-timeline hidden" id="fgTimeline">
          <button class="fg-playbtn" id="fgPlayBtn" onclick="WatchPage._fgTogglePlay()">
            <svg id="fgPlayIcon" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            <svg id="fgPauseIcon" width="18" height="18" viewBox="0 0 24 24" fill="currentColor" class="hidden"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
          </button>
          <span class="fg-time" id="fgCurrentTime">0:00</span>
          <div class="fg-bar" id="fgBar">
            <div class="fg-bar-track">
              <div class="fg-fill" id="fgFill"></div>
              <div class="fg-markers" id="fgMarkers"></div>
              <div class="fg-thumb" id="fgThumb"></div>
            </div>
          </div>
          <span class="fg-time" id="fgTotalTime">16:22</span>
          <button class="fg-playbtn" id="fgSoundBtn" onclick="WatchPage._fgToggleMute()">
            <svg id="fgSoundIcon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
            <svg id="fgMuteIcon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="hidden"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>
          </button>
        </div>
      </div>
    `;
  },

  mount(params) {
    const socket = SocketClient.get();

    this._unsubOffer = async (data) => {
      const pc = this._pcs.get(data.cameraNumber);
      if (!pc) return;
      this._camSocketIds.set(data.cameraNumber, data.cameraSocketId);
      try {
        await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        socket.emit('webrtc-answer', { cameraSocketId: data.cameraSocketId, sdp: pc.localDescription });
      } catch (e) { console.warn('offer handling failed', e); }
    };

    this._unsubIce = (data) => {
      if (!data.candidate) return;
      this._pcs.forEach(pc => {
        pc.addIceCandidate(new RTCIceCandidate(data.candidate)).catch(() => {});
      });
    };

    // Director-cut: broadcaster forces a camera on all viewers
    this._unsubCut = (data) => {
      if (data.cameraNumber !== null) {
        this._directorOverride = true;
        this._showDirectorBanner(`Director cut → CAM ${data.cameraNumber}`);
        if (this._match?.isDemo) {
          this._featured = data.cameraNumber;
          this._layoutDemoPlayers();
        } else if (this._pcs.has(data.cameraNumber)) {
          this._setFeatured(data.cameraNumber);
        } else {
          const cam = this._match?.cameras.find(c => c.number === data.cameraNumber);
          if (cam) { this._connectCam(cam.number, cam.role); this._featured = cam.number; }
        }
      } else {
        this._directorOverride = false;
        this._hideDirectorBanner();
      }
    };
    socket.on('director-cut', this._unsubCut);

    socket.on('webrtc-offer', this._unsubOffer);
    socket.on('webrtc-ice-candidate', this._unsubIce);

    socket.emit('watch-match', { code: this._code }, (resp) => {
      if (resp.error) { navigate('matches'); return; }
      this._match = resp.match;
      if (resp.match.isFullGame) { this._initFullGame(); this._updateUI(); return; }
      this._syncCameras();
      this._updateUI();
    });

    this._unsubMatch = (match) => {
      if (match.code !== this._code) return;
      this._match = match;
      this._syncCameras();
      this._updateUI();
    };
    socket.on('match-updated', this._unsubMatch);
  },

  unmount() {
    const socket = SocketClient.get();
    if (this._unsubOffer) socket.off('webrtc-offer', this._unsubOffer);
    if (this._unsubIce)   socket.off('webrtc-ice-candidate', this._unsubIce);
    if (this._unsubMatch) socket.off('match-updated', this._unsubMatch);
    if (this._unsubCut)   socket.off('director-cut', this._unsubCut);
    this._pcs.forEach(pc => pc.close());
    this._pcs.clear();
    this._camSocketIds.clear();
    this._streams.clear();
    this._featured = null;
    this._directorOverride = false;
    this._demoReady = false;
    this._demoVids.forEach(v => { v.pause(); v.src = ''; });
    this._demoVids.clear();
    if (this._fgRaf) { cancelAnimationFrame(this._fgRaf); this._fgRaf = null; }
    this._fgVids.forEach(v => { v.pause(); v.src = ''; });
    this._fgVids.clear();
    this._fgDragging = false;
    this._match = null;
  },

  // ── Camera management ────────────────────────────────────────
  _syncCameras() {
    const streaming = this._match?.cameras.filter(c => c.isStreaming) || [];

    if (this._match?.isDemo) {
      if (this._demoReady) return;
      this._demoReady = true;
      if (this._featured === null && streaming.length > 0) {
        this._featured = streaming[0].number;
      }
      this._initDemoPlayers();
      this._layoutDemoPlayers();
      return;
    }

    const streamNums = new Set(streaming.map(c => c.number));

    // Disconnect cameras that stopped streaming
    this._pcs.forEach((pc, num) => {
      if (!streamNums.has(num)) {
        pc.close();
        this._pcs.delete(num);
        this._camSocketIds.delete(num);
        if (this._featured === num) {
          this._featured = null;
          const vid = document.getElementById('watchVideo');
          if (vid) vid.srcObject = null;
          const ph = document.getElementById('watchPlaceholder');
          if (ph) ph.style.removeProperty('display');
          const sq = document.getElementById('statQuality');
          if (sq) sq.textContent = '–';
        }
        this._streams.delete(num);
      }
    });

    // Connect to new cameras
    streaming.forEach(cam => {
      if (!this._pcs.has(cam.number)) this._connectCam(cam.number, cam.role);
    });

    // Auto-select first available if nothing featured
    if (this._featured === null && streaming.length > 0) {
      this._setFeatured(streaming[0].number);
    }

    this._renderThumbs();
  },


  _connectCam(num, role) {
    const pc = WebRTCHelper.createPC();
    this._pcs.set(num, pc);

    pc.ontrack = (e) => {
      const stream = e.streams[0];
      this._streams.set(num, stream);

      // Attach to thumb video
      const thumb = document.getElementById(`watchThumb${num}`);
      if (thumb) { thumb.srcObject = stream; thumb.play().catch(() => {}); }

      // Attach to main video if this is the featured cam
      if (this._featured === num) {
        const vid = document.getElementById('watchVideo');
        if (vid) { vid.srcObject = stream; vid.play().catch(() => {}); }
        const ph = document.getElementById('watchPlaceholder');
        if (ph) ph.style.display = 'none';
        document.getElementById('statQuality').textContent = 'HD';
      }
    };

    pc.onicecandidate = (e) => {
      const socketId = this._camSocketIds.get(num);
      if (e.candidate && socketId) {
        SocketClient.get().emit('webrtc-ice-candidate', {
          targetSocketId: socketId,
          candidate: e.candidate.toJSON(),
        });
      }
    };

    pc.oniceconnectionstatechange = () => {
      if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
        pc.close();
        this._pcs.delete(num);
        this._camSocketIds.delete(num);
        this._streams.delete(num);
        if (this._featured === num) {
          this._featured = null;
          const vid = document.getElementById('watchVideo');
          if (vid) vid.srcObject = null;
          const ph = document.getElementById('watchPlaceholder');
          if (ph) ph.style.removeProperty('display');
          const sq = document.getElementById('statQuality');
          if (sq) sq.textContent = '–';
          // Fall back to another connected camera
          const fallback = [...this._pcs.keys()][0];
          if (fallback != null) this._setFeatured(fallback);
        }
        this._renderThumbs();
      }
    };

    SocketClient.get().emit('webrtc-request-stream', { matchCode: this._code, cameraNumber: num });
  },

  // ── Feature (main) a camera ──────────────────────────────────
  _setFeatured(num) {
    this._featured = num;
    const stream = this._streams.get(num);
    const vid = document.getElementById('watchVideo');

    if (stream && vid) {
      vid.srcObject = stream;
      vid.play().catch(() => {});
      const ph = document.getElementById('watchPlaceholder');
      if (ph) ph.style.display = 'none';
      const sq = document.getElementById('statQuality');
      if (sq) sq.textContent = 'HD';
    }

    const cam = this._match?.cameras.find(c => c.number === num);
    const label = document.getElementById('watchCamLabel');
    if (label) {
      label.textContent = `CAM ${num}${cam?.role ? ' · ' + cam.role : ''}`;
      label.classList.remove('hidden');
    }

    this._renderThumbs();
  },

  switchCam(num) {
    if (num === this._featured) return;
    if (this._directorOverride) return;
    if (this._match?.isDemo) {
      this._featured = num;
      this._layoutDemoPlayers();
      return;
    }
    this._setFeatured(num);
  },

  // ── Demo player: create once, move DOM nodes on switch ───────
  _initDemoPlayers() {
    if (this._demoVids.size > 0) return;
    const cameras = this._match.cameras.filter(c => c.isStreaming && c.videoSrc);
    cameras.forEach(cam => {
      const v = document.createElement('video');
      v.src = cam.videoSrc;
      v.autoplay = true;
      v.playsInline = true;
      v.muted = true;
      v.dataset.camNum = String(cam.number);
      v.dataset.role = cam.role;
      this._demoVids.set(cam.number, v);
      v.load();
    });
  },

  _layoutDemoPlayers() {
    const mainWrap = document.getElementById('watchMainWrap');
    const rail     = document.getElementById('watchThumbRail');
    if (!mainWrap || !rail) return;

    // Hide the static placeholder video element
    const staticVid = document.getElementById('watchVideo');
    if (staticVid) staticVid.style.display = 'none';
    const ph = document.getElementById('watchPlaceholder');
    if (ph) ph.style.display = 'none';

    const featured = this._featured;
    const cameras  = this._match.cameras.filter(c => c.isStreaming && c.videoSrc);
    const others   = cameras.filter(c => c.number !== featured);

    // Move featured video into main wrap (replace any existing demo video there)
    const featuredVid = this._demoVids.get(featured);
    if (featuredVid) {
      featuredVid.className = 'demo-main-video';
      featuredVid.muted = false;
      featuredVid.play().catch(() => {});
      // Remove any other demo video from main wrap first
      mainWrap.querySelectorAll('.demo-main-video').forEach(el => {
        if (el !== featuredVid) el.remove();
      });
      if (!mainWrap.contains(featuredVid)) mainWrap.insertBefore(featuredVid, mainWrap.firstChild);
    }

    // Update cam label
    const cam = cameras.find(c => c.number === featured);
    const label = document.getElementById('watchCamLabel');
    if (label && cam) {
      label.textContent = `CAM ${featured} · ${cam.role}`;
      label.classList.remove('hidden');
    }
    const sq = document.getElementById('statQuality');
    if (sq) sq.textContent = 'HD';

    // Build thumbnail rail from the other video elements (move, don't recreate)
    rail.innerHTML = '';
    others.forEach(c => {
      const v = this._demoVids.get(c.number);
      if (!v) return;
      v.className = '';
      v.muted = true;
      v.play().catch(() => {});

      const thumb = document.createElement('div');
      thumb.className = 'watch-thumb';
      thumb.onclick = () => WatchPage.switchCam(c.number);
      thumb.appendChild(v);

      const lbl = document.createElement('div');
      lbl.className = 'watch-thumb-label';
      lbl.textContent = `CAM ${c.number} · ${c.role}`;
      thumb.appendChild(lbl);

      rail.appendChild(thumb);
    });
    rail.style.display = others.length ? 'flex' : 'none';
  },

  _showDirectorBanner(msg) {
    let banner = document.getElementById('watchDirectorBanner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'watchDirectorBanner';
      banner.className = 'watch-director-banner';
      const wrap = document.getElementById('watchMainWrap');
      if (wrap) wrap.appendChild(banner);
    }
    banner.textContent = msg;
    banner.style.display = 'flex';
  },

  _hideDirectorBanner() {
    const banner = document.getElementById('watchDirectorBanner');
    if (banner) banner.style.display = 'none';
  },

  // ── Thumbnail rail (WebRTC only — demo uses _layoutDemoPlayers) ─
  _renderThumbs() {
    const rail = document.getElementById('watchThumbRail');
    if (!rail) return;
    const streaming = this._match?.cameras.filter(c => c.isStreaming) || [];
    const others = streaming.filter(c => c.number !== this._featured);

    if (others.length === 0) {
      rail.innerHTML = '';
      rail.style.display = 'none';
      return;
    }
    rail.style.display = 'flex';
    rail.innerHTML = others.map(c => `
      <div class="watch-thumb" onclick="WatchPage.switchCam(${c.number})">
        <video id="watchThumb${c.number}" autoplay playsinline muted
          ${c.videoSrc ? `src="${c.videoSrc}"` : ''}></video>
        <div class="watch-thumb-label">CAM ${c.number}${c.role ? ' · ' + c.role : ''}</div>
      </div>
    `).join('');

    // Attach streams (WebRTC) or sync demo videos
    others.forEach(c => {
      const thumb = document.getElementById(`watchThumb${c.number}`);
      if (!thumb) return;
      if (c.videoSrc) {
        // Sync demo thumb to main video time
        const mainVid = document.getElementById('watchVideo');
        thumb.addEventListener('canplay', () => {
          if (mainVid) thumb.currentTime = mainVid.currentTime;
          thumb.play().catch(() => {});
        }, { once: true });
        thumb.load();
      } else {
        const stream = this._streams.get(c.number);
        if (stream) { thumb.srcObject = stream; thumb.play().catch(() => {}); }
      }
    });
  },

  // ── UI ────────────────────────────────────────────────────────
  _updateUI() {
    const m = this._match;
    if (!m) return;
    const titleEl = document.getElementById('watchTitle');
    const teamsEl = document.getElementById('watchTeams');
    const badge   = document.getElementById('watchLiveBadge');
    if (titleEl) titleEl.textContent = m.title;
    if (teamsEl) teamsEl.textContent = `${m.teamA} vs ${m.teamB}`;
    if (badge) {
      badge.className = m.isLive ? 'badge-live' : 'hidden';
      badge.textContent = m.isLive ? 'LIVE' : '';
    }
    const streaming = m.cameras.filter(c => c.isStreaming);
    const statCams = document.getElementById('statCams');
    const statStreaming = document.getElementById('statStreaming');
    if (statCams) statCams.textContent = m.cameras.length;
    if (statStreaming) statStreaming.textContent = streaming.length;
  },

  leave() {
    this.unmount();
    navigate('matches');
  },

  // ── Full-game player ──────────────────────────────────────────
  _initFullGame() {
    const cameras = this._match.cameras.filter(c => c.isStreaming && c.videoSrc);

    // Create persistent video elements
    cameras.forEach(cam => {
      const v = document.createElement('video');
      v.src         = cam.videoSrc;
      v.preload     = 'auto';
      v.playsInline = true;
      v.muted       = true;   // start muted so autoplay works; unmute cam1 after layout
      v.dataset.camNum = String(cam.number);
      this._fgVids.set(cam.number, v);
    });

    // Layout: featured = cam 1 (lateral/annotated)
    this._featured = 1;
    this._fgLayout();

    // Show timeline, hide stats strip
    document.getElementById('watchBottom')?.classList.add('hidden');
    const tl = document.getElementById('fgTimeline');
    if (tl) tl.classList.remove('hidden');

    // Seed goal markers
    const markers = document.getElementById('fgMarkers');
    if (markers) {
      markers.innerHTML = FG_EVENTS.map(ev => {
        const pct = (ev.t / FG_DURATION * 100).toFixed(3);
        const cls = ev.type === 'game_over' ? 'fg-marker fg-marker-end'
                  : ev.type === 'scratch'   ? 'fg-marker fg-marker-scratch'
                  : 'fg-marker fg-marker-goal';
        return `<div class="${cls}" style="left:${pct}%" title="${ev.type} @ ${_fmtTime(ev.t)}"></div>`;
      }).join('');
    }

    // Wire up the drag timeline
    const bar = document.getElementById('fgBar');
    if (bar) {
      const seek = (e) => {
        const rect = bar.getBoundingClientRect();
        const x    = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
        const pct  = Math.max(0, Math.min(1, x / rect.width));
        const t    = pct * FG_DURATION;
        this._fgSeekTo(t);
      };
      bar.addEventListener('pointerdown', (e) => {
        this._fgDragging = true;
        bar.setPointerCapture(e.pointerId);
        seek(e);
      });
      bar.addEventListener('pointermove', (e) => { if (this._fgDragging) seek(e); });
      bar.addEventListener('pointerup',   () => { this._fgDragging = false; });
    }

    // Start RAF loop
    this._fgRafLoop();
  },

  _fgLayout() {
    const mainWrap = document.getElementById('watchMainWrap');
    const rail     = document.getElementById('watchThumbRail');
    if (!mainWrap || !rail) return;

    const staticVid = document.getElementById('watchVideo');
    if (staticVid) staticVid.style.display = 'none';
    document.getElementById('watchPlaceholder')?.style.setProperty('display', 'none');

    // Main video — keep muted for autoplay policy; user can unmute via sound btn
    const main = this._fgVids.get(this._featured);
    if (main) {
      main.className = 'demo-main-video';
      main.muted = true;
      mainWrap.querySelectorAll('.demo-main-video').forEach(el => { if (el !== main) el.remove(); });
      if (!mainWrap.contains(main)) mainWrap.insertBefore(main, mainWrap.firstChild);
      main.load();
      main.play().catch(() => {
        document.getElementById('fgPlayIcon')?.classList.remove('hidden');
        document.getElementById('fgPauseIcon')?.classList.add('hidden');
      });
    }

    // Thumbnail rail
    rail.innerHTML = '';
    this._fgVids.forEach((v, num) => {
      if (num === this._featured) return;
      v.className = '';
      v.muted = true;
      v.play().catch(() => {});
      const cam = this._match.cameras.find(c => c.number === num);
      const thumb = document.createElement('div');
      thumb.className = 'watch-thumb';
      thumb.onclick = () => { this._featured = num; this._fgLayout(); };
      thumb.appendChild(v);
      const lbl = document.createElement('div');
      lbl.className = 'watch-thumb-label';
      lbl.textContent = `CAM ${num} · ${cam?.role || ''}`;
      thumb.appendChild(lbl);
      rail.appendChild(thumb);
    });
    rail.style.display = this._fgVids.size > 1 ? 'flex' : 'none';

    const label = document.getElementById('watchCamLabel');
    const cam = this._match.cameras.find(c => c.number === this._featured);
    if (label && cam) {
      label.textContent = `CAM ${this._featured} · ${cam.role}`;
      label.classList.remove('hidden');
    }
    document.getElementById('statQuality').textContent = 'HD';
  },

  _fgSeekTo(lateralT) {
    this._fgVids.forEach((v, num) => {
      const t = Math.max(0, lateralT + (FG_OFFSETS[num] || 0));
      v.currentTime = t;
    });
    this._fgUpdateTimeline(lateralT);
  },

  _fgToggleMute() {
    const main = this._fgVids.get(1);
    if (!main) return;
    main.muted = !main.muted;
    document.getElementById('fgSoundIcon')?.classList.toggle('hidden', main.muted);
    document.getElementById('fgMuteIcon')?.classList.toggle('hidden', !main.muted);
  },

  _fgTogglePlay() {
    const main = this._fgVids.get(1);
    if (!main) return;
    if (main.paused) {
      this._fgVids.forEach(v => v.play().catch(() => {}));
      document.getElementById('fgPlayIcon')?.classList.add('hidden');
      document.getElementById('fgPauseIcon')?.classList.remove('hidden');
    } else {
      this._fgVids.forEach(v => v.pause());
      document.getElementById('fgPlayIcon')?.classList.remove('hidden');
      document.getElementById('fgPauseIcon')?.classList.add('hidden');
    }
  },

  _fgRafLoop() {
    const tick = () => {
      if (!this._fgVids.size) return;
      const lat = this._fgVids.get(1);
      if (lat && !this._fgDragging) this._fgUpdateTimeline(lat.currentTime);
      this._fgRaf = requestAnimationFrame(tick);
    };
    this._fgRaf = requestAnimationFrame(tick);
  },

  _fgUpdateTimeline(t) {
    const pct = Math.min(1, t / FG_DURATION) * 100;
    const fill  = document.getElementById('fgFill');
    const thumb = document.getElementById('fgThumb');
    const cur   = document.getElementById('fgCurrentTime');
    if (fill)  fill.style.width = `${pct}%`;
    if (thumb) thumb.style.left = `${pct}%`;
    if (cur)   cur.textContent  = _fmtTime(t);
  },
};
