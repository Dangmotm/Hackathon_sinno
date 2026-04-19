const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const resetBtn = document.getElementById('resetBtn');
const overlayResetBtn = document.getElementById('overlayResetBtn');
const endOverlay = document.getElementById('endOverlay');
const serverStatus = document.getElementById('serverStatus');

const playerHpText = document.getElementById('playerHpText');
const enemyHpText = document.getElementById('enemyHpText');
const playerShield = document.getElementById('playerShield');
const enemyShield = document.getElementById('enemyShield');
const playerHpBar = document.getElementById('playerHpBar');
const enemyHpBar = document.getElementById('enemyHpBar');
const timeText = document.getElementById('timeText');
const resultBadge = document.getElementById('resultBadge');
const projectileCount = document.getElementById('projectileCount');
const streakCount = document.getElementById('streakCount');
const stepCount = document.getElementById('stepCount');
const rewardText = document.getElementById('rewardText');
const botType = document.getElementById('botType');
const overlayTitle = document.getElementById('overlayTitle');
const overlaySubtitle = document.getElementById('overlaySubtitle');

const keys = {
  left: false,
  right: false,
  up: false,
  down: false,
  shoot: false,
  blink: false,
  shield: false,
};

const mouse = {
  canvasX: canvas.width * 0.7,
  canvasY: canvas.height * 0.3,
  worldX: 8.4,
  worldY: 8.4,
  inside: false,
};

let state = null;
let inFlight = false;
let lastFrameAt = 0;
const STEP_INTERVAL_MS = 1000 / 30;

const VISUAL = {
  pad: 72,
  arenaGlow: 'rgba(91, 196, 255, 0.18)',
  gridColor: 'rgba(173, 216, 255, 0.08)',
  borderColor: 'rgba(214, 234, 255, 0.28)',
  playerColor: '#61c5ff',
  enemyColor: '#ff7686',
  shieldColor: '#74ffd2',
  projectilePlayer: '#8fd7ff',
  projectileEnemy: '#ff9ba8',
};

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function hpPercent(hp, maxHp) {
  if (!maxHp) return 0;
  return clamp((hp / maxHp) * 100, 0, 100);
}

function resizeCanvas() {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const size = Math.max(300, Math.floor(Math.min(rect.width, rect.height || rect.width)));
  canvas.width = Math.floor(size * dpr);
  canvas.height = Math.floor(size * dpr);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
}

function arenaRect() {
  const width = canvas.getBoundingClientRect().width;
  const height = canvas.getBoundingClientRect().height;
  const pad = Math.max(42, width * 0.07);
  return {
    x: pad,
    y: pad,
    w: width - pad * 2,
    h: height - pad * 2,
    pad,
    width,
    height,
  };
}

function worldToScreen(x, y, cfg) {
  const rect = arenaRect();
  return {
    x: rect.x + (x / cfg.arena_size) * rect.w,
    y: rect.y + rect.h - (y / cfg.arena_size) * rect.h,
  };
}

function screenToWorld(clientX, clientY, cfg) {
  const canvasRect = canvas.getBoundingClientRect();
  const localX = clientX - canvasRect.left;
  const localY = clientY - canvasRect.top;
  const rect = arenaRect();
  const wx = ((localX - rect.x) / rect.w) * cfg.arena_size;
  const wy = ((rect.y + rect.h - localY) / rect.h) * cfg.arena_size;
  return {
    canvasX: localX,
    canvasY: localY,
    worldX: clamp(wx, 0, cfg.arena_size),
    worldY: clamp(wy, 0, cfg.arena_size),
  };
}

function setBadgeAppearance(result) {
  const normalized = (result || 'ongoing').toUpperCase();
  resultBadge.textContent = normalized;
  resultBadge.style.color = '#ebf4ff';
  resultBadge.style.boxShadow = 'none';

  if (normalized === 'WIN') {
    resultBadge.style.color = '#75ffd1';
    resultBadge.style.boxShadow = '0 0 28px rgba(117,255,209,0.18)';
  } else if (normalized === 'LOSS') {
    resultBadge.style.color = '#ff9eab';
    resultBadge.style.boxShadow = '0 0 28px rgba(255,120,136,0.18)';
  } else if (normalized === 'TIMEOUT') {
    resultBadge.style.color = '#ffd36b';
    resultBadge.style.boxShadow = '0 0 28px rgba(255,211,107,0.14)';
  }
}

function updateHud(snapshot) {
  const cfg = snapshot.config;
  const hud = snapshot.hud;
  const player = snapshot.player;
  const enemy = snapshot.enemy;

  playerHpText.textContent = `${player.hp.toFixed(0)} / ${cfg.max_hp.toFixed(0)}`;
  enemyHpText.textContent = `${enemy.hp.toFixed(0)} / ${cfg.max_hp.toFixed(0)}`;
  playerShield.textContent = `Shield x${player.shield_charges}`;
  enemyShield.textContent = `Shield x${enemy.shield_charges}`;
  playerHpBar.style.width = `${hpPercent(player.hp, cfg.max_hp)}%`;
  enemyHpBar.style.width = `${hpPercent(enemy.hp, cfg.max_hp)}%`;

  timeText.textContent = `${snapshot.hud.time_left_seconds.toFixed(1)}s`;
  projectileCount.textContent = String(hud.active_projectiles);
  streakCount.textContent = String(hud.steps_since_last_damage_dealt);
  stepCount.textContent = String(hud.step);
  rewardText.textContent = hud.reward.toFixed(2);
  botType.textContent = snapshot.meta.using_model ? 'Model bot' : 'Scripted bot';
  serverStatus.textContent = snapshot.meta.status_message;
  setBadgeAppearance(hud.result);

  const finished = hud.terminated || hud.truncated;
  if (finished) {
    endOverlay.classList.remove('hidden');
    overlayTitle.textContent = hud.result.toUpperCase();
    overlaySubtitle.textContent = `Player ${player.hp.toFixed(1)} HP • Enemy ${enemy.hp.toFixed(1)} HP`;
  } else {
    endOverlay.classList.add('hidden');
  }
}

function drawBackground(snapshot) {
  const rect = arenaRect();
  const bgGradient = ctx.createLinearGradient(0, 0, 0, rect.height);
  bgGradient.addColorStop(0, '#122a4f');
  bgGradient.addColorStop(0.5, '#0d1a34');
  bgGradient.addColorStop(1, '#08111f');

  ctx.clearRect(0, 0, rect.width + rect.x * 2, rect.height + rect.y * 2);
  ctx.fillStyle = bgGradient;
  ctx.fillRect(0, 0, rect.width + rect.x * 2, rect.height + rect.y * 2);

  const arenaGradient = ctx.createRadialGradient(
    rect.x + rect.w * 0.35,
    rect.y + rect.h * 0.35,
    0,
    rect.x + rect.w * 0.5,
    rect.y + rect.h * 0.55,
    rect.w * 0.85,
  );
  arenaGradient.addColorStop(0, 'rgba(50, 89, 138, 0.35)');
  arenaGradient.addColorStop(1, 'rgba(9, 18, 35, 0.92)');
  ctx.fillStyle = arenaGradient;
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

  ctx.strokeStyle = VISUAL.borderColor;
  ctx.lineWidth = 3;
  ctx.shadowColor = VISUAL.arenaGlow;
  ctx.shadowBlur = 18;
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
  ctx.shadowBlur = 0;

  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();

  const cell = rect.w / 12;
  ctx.strokeStyle = VISUAL.gridColor;
  ctx.lineWidth = 1;
  for (let i = 1; i < 12; i += 1) {
    const gx = rect.x + cell * i;
    const gy = rect.y + cell * i;
    ctx.beginPath();
    ctx.moveTo(gx, rect.y);
    ctx.lineTo(gx, rect.y + rect.h);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(rect.x, gy);
    ctx.lineTo(rect.x + rect.w, gy);
    ctx.stroke();
  }

  const t = performance.now() / 1000;
  for (let i = 0; i < 14; i += 1) {
    const px = rect.x + ((i * 131.7 + t * 18) % rect.w);
    const py = rect.y + ((i * 97.2 + t * 12) % rect.h);
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.beginPath();
    ctx.arc(px, py, 1.6 + ((i + t) % 3), 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function drawObstacles(snapshot) {
  const rect = arenaRect();
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();

  for (const obs of snapshot.obstacles) {
    const p1 = worldToScreen(obs.x, obs.y + obs.h, snapshot.config);
    const p2 = worldToScreen(obs.x + obs.w, obs.y, snapshot.config);
    const w = p2.x - p1.x;
    const h = p2.y - p1.y;

    const g = ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
    g.addColorStop(0, 'rgba(111, 143, 180, 0.95)');
    g.addColorStop(1, 'rgba(44, 58, 84, 0.96)');
    ctx.fillStyle = g;
    ctx.shadowColor = 'rgba(0,0,0,0.35)';
    ctx.shadowBlur = 18;
    roundRect(ctx, p1.x, p1.y, w, h, 14, true, false);
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(207, 227, 255, 0.18)';
    ctx.lineWidth = 2;
    roundRect(ctx, p1.x, p1.y, w, h, 14, false, true);
  }

  ctx.restore();
}

function drawAimLine(agent, cfg, color) {
  const start = worldToScreen(agent.x, agent.y, cfg);
  const end = worldToScreen(agent.x + agent.aim_dx * 1.2, agent.y + agent.aim_dy * 1.2, cfg);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.85;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();
  ctx.restore();
}

function drawAgent(agent, cfg, bodyColor, accentColor) {
  const rect = arenaRect();
  const pos = worldToScreen(agent.x, agent.y, cfg);
  const pxRadius = (cfg.agent_radius / cfg.arena_size) * rect.w;

  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();

  if (agent.shield_active) {
    const shieldRadius = (cfg.shield_radius / cfg.arena_size) * rect.w;
    const pulse = 1 + 0.03 * Math.sin(performance.now() / 110);
    ctx.strokeStyle = 'rgba(116,255,210,0.95)';
    ctx.lineWidth = 4;
    ctx.shadowColor = 'rgba(116,255,210,0.35)';
    ctx.shadowBlur = 22;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, shieldRadius * pulse, 0, Math.PI * 2);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  ctx.fillStyle = 'rgba(0,0,0,0.24)';
  ctx.beginPath();
  ctx.ellipse(pos.x, pos.y + pxRadius * 1.1, pxRadius * 1.2, pxRadius * 0.55, 0, 0, Math.PI * 2);
  ctx.fill();

  const glow = ctx.createRadialGradient(pos.x, pos.y, pxRadius * 0.2, pos.x, pos.y, pxRadius * 2.1);
  glow.addColorStop(0, `${accentColor}cc`);
  glow.addColorStop(1, `${accentColor}00`);
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, pxRadius * 2.15, 0, Math.PI * 2);
  ctx.fill();

  const body = ctx.createRadialGradient(pos.x - pxRadius * 0.35, pos.y - pxRadius * 0.35, pxRadius * 0.2, pos.x, pos.y, pxRadius * 1.2);
  body.addColorStop(0, '#f5fbff');
  body.addColorStop(0.28, bodyColor);
  body.addColorStop(1, shadeColor(bodyColor, -28));
  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, pxRadius, 0, Math.PI * 2);
  ctx.fill();

  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(10, 18, 34, 0.9)';
  ctx.stroke();

  drawAimLine(agent, cfg, accentColor);
  ctx.restore();
}

function drawProjectiles(snapshot) {
  const rect = arenaRect();
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();

  for (const proj of snapshot.projectiles) {
    const pos = worldToScreen(proj.x, proj.y, snapshot.config);
    const radius = Math.max(4, (snapshot.config.projectile_radius / snapshot.config.arena_size) * rect.w);
    const color = proj.owner === 'player' ? VISUAL.projectilePlayer : VISUAL.projectileEnemy;

    ctx.strokeStyle = `${color}77`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    ctx.lineTo(pos.x - proj.vx * 14, pos.y + proj.vy * 14);
    ctx.stroke();

    const glow = ctx.createRadialGradient(pos.x, pos.y, radius * 0.2, pos.x, pos.y, radius * 3);
    glow.addColorStop(0, `${color}`);
    glow.addColorStop(1, `${color}00`);
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius * 3, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function drawCrosshair(snapshot) {
  if (!snapshot || !mouse.inside) return;
  const pos = worldToScreen(mouse.worldX, mouse.worldY, snapshot.config);
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.8)';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(pos.x - 18, pos.y);
  ctx.lineTo(pos.x - 6, pos.y);
  ctx.moveTo(pos.x + 6, pos.y);
  ctx.lineTo(pos.x + 18, pos.y);
  ctx.moveTo(pos.x, pos.y - 18);
  ctx.lineTo(pos.x, pos.y - 6);
  ctx.moveTo(pos.x, pos.y + 6);
  ctx.lineTo(pos.x, pos.y + 18);
  ctx.stroke();
  ctx.restore();
}

function draw(snapshot) {
  if (!snapshot) return;
  drawBackground(snapshot);
  drawObstacles(snapshot);
  drawProjectiles(snapshot);
  drawAgent(snapshot.enemy, snapshot.config, VISUAL.enemyColor, '#ff6b7f');
  drawAgent(snapshot.player, snapshot.config, VISUAL.playerColor, '#4aa8ff');
  drawCrosshair(snapshot);
}

function roundRect(context, x, y, width, height, radius, fill, stroke) {
  let r = radius;
  if (width < 2 * r) r = width / 2;
  if (height < 2 * r) r = height / 2;
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
  if (fill) context.fill();
  if (stroke) context.stroke();
}

function shadeColor(hex, percent) {
  const stripped = hex.replace('#', '');
  const num = parseInt(stripped, 16);
  const amt = Math.round(2.55 * percent);
  const r = clamp(((num >> 16) & 0xff) + amt, 0, 255);
  const g = clamp(((num >> 8) & 0xff) + amt, 0, 255);
  const b = clamp((num & 0xff) + amt, 0, 255);
  return `rgb(${r}, ${g}, ${b})`;
}

async function api(path, method = 'GET', body = null) {
  const options = { method, headers: {} };
  if (body) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function resetRound() {
  try {
    state = await api('/api/reset', 'POST');
    updateHud(state);
    draw(state);
  } catch (error) {
    serverStatus.textContent = `Lỗi reset: ${error.message}`;
  }
}

async function stepGame(now) {
  if (!state) return;
  if (inFlight) return;
  if (now - lastFrameAt < STEP_INTERVAL_MS) return;
  if (state.hud.terminated || state.hud.truncated) return;

  inFlight = true;
  lastFrameAt = now;
  try {
    state = await api('/api/step', 'POST', {
      keys,
      aim: {
        x: mouse.worldX,
        y: mouse.worldY,
      },
    });
    updateHud(state);
  } catch (error) {
    serverStatus.textContent = `Mất kết nối: ${error.message}`;
  } finally {
    inFlight = false;
  }
}

function animationLoop(now) {
  if (state) {
    draw(state);
    stepGame(now);
  }
  window.requestAnimationFrame(animationLoop);
}

function onPointerMove(event) {
  if (!state) return;
  mouse.inside = true;
  Object.assign(mouse, screenToWorld(event.clientX, event.clientY, state.config));
}

canvas.addEventListener('mousemove', onPointerMove);
canvas.addEventListener('mouseenter', (event) => {
  mouse.inside = true;
  onPointerMove(event);
});
canvas.addEventListener('mouseleave', () => {
  mouse.inside = false;
});

canvas.addEventListener('mousedown', (event) => {
  if (event.button === 0) keys.shoot = true;
  if (event.button === 2) keys.shield = true;
});

canvas.addEventListener('mouseup', (event) => {
  if (event.button === 0) keys.shoot = false;
  if (event.button === 2) keys.shield = false;
});

canvas.addEventListener('contextmenu', (event) => event.preventDefault());

document.addEventListener('keydown', (event) => {
  if (event.repeat) return;
  const key = event.key.toLowerCase();
  if (key === 'a') keys.left = true;
  if (key === 'd') keys.right = true;
  if (key === 'w') keys.up = true;
  if (key === 's') keys.down = true;
  if (key === ' ') {
    keys.blink = true;
    event.preventDefault();
  }
  if (key === 'q') keys.shield = true;
  if (key === 'r') resetRound();
});

document.addEventListener('keyup', (event) => {
  const key = event.key.toLowerCase();
  if (key === 'a') keys.left = false;
  if (key === 'd') keys.right = false;
  if (key === 'w') keys.up = false;
  if (key === 's') keys.down = false;
  if (key === ' ') keys.blink = false;
  if (key === 'q') keys.shield = false;
});

window.addEventListener('mouseup', () => {
  keys.shoot = false;
  keys.shield = false;
});

window.addEventListener('blur', () => {
  Object.keys(keys).forEach((k) => {
    keys[k] = false;
  });
});

window.addEventListener('resize', () => {
  resizeCanvas();
  if (state) draw(state);
});

resetBtn.addEventListener('click', resetRound);
overlayResetBtn.addEventListener('click', resetRound);

(async function init() {
  resizeCanvas();
  try {
    state = await api('/api/state');
    updateHud(state);
    draw(state);
    serverStatus.textContent = state.meta.status_message;
  } catch (error) {
    serverStatus.textContent = `Không kết nối được server: ${error.message}`;
  }
  window.requestAnimationFrame(animationLoop);
})();
