// Cloudflare Pages Function — AI 프록시 (앱 → 여기 → Gemini / OpenRouter 백업)
// 앱에서 { system, prompt } 를 받아 무료 최상위 Gemini 모델 체인
// (Flash → Flash-Lite 자동 폴백 + 선택적 OpenRouter :free 2차 백업)으로 호출하고
// 순수 JSON 결과만 돌려준다. API 키는 서버 환경변수(GEMINI_API_KEY)에만 존재.
//
// 경로: POST /api/analyze

const PROXY_VERSION = 'v2-fallback-5models';

const DEFAULT_GEMINI_MODELS = [
  'gemini-2.5-flash',
  'gemini-2.5-flash-lite',
  'gemini-3.1-flash-lite',
  'gemini-3.8-flash',
  'gemini-2.0-flash',
];

const DEFAULT_OPENROUTER_MODEL = 'qwen/qwen3-235b-a22b:free';
const MODEL_TIMEOUT_MS = 22000;

const STEMS = ['갑', '을', '병', '정', '무', '기', '경', '신', '임', '계'];
const STEMS_HANJA = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'];
const BRANCHES = ['자', '축', '인', '묘', '진', '사', '오', '미', '신', '유', '술', '해'];
const BRANCHES_HANJA = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'];

function yearGanzhi(y) {
  const idx = ((y - 4) % 60 + 60) % 60;
  const s = idx % 10;
  const b = idx % 12;
  return `${STEMS[s]}${BRANCHES[b]}(${STEMS_HANJA[s]}${BRANCHES_HANJA[b]})`;
}

function buildKstTimeAnchor() {
  const kst = new Date(Date.now() + 9 * 60 * 60 * 1000);
  const rawY = kst.getUTCFullYear();
  const y = rawY >= 2026 ? rawY : 2026;
  const m = kst.getUTCMonth() + 1;
  const d = kst.getUTCDate();
  const sajuYear = (m === 1 || (m === 2 && d < 4)) ? y - 1 : y;
  return [
    `[시스템 절대 시간 락 (KST 실시간 기준 — 하네스 1 최우선 강제)]`,
    `- 오늘 실제 날짜: ${y}년 ${m}월 ${d}일`,
    `- '올해(현재 연도)' = ${y}년 ${yearGanzhi(sajuYear)}년 세운`,
    `- '내년' = ${y + 1}년 ${yearGanzhi(sajuYear + 1)}년 / '작년' = ${y - 1}년 ${yearGanzhi(sajuYear - 1)}년`,
    `- [절대 금지 경고]: 모델의 과거 사전학습 시점(2024년 갑진년이나 2025년 을사년)을 '올해/현재'라고 부르거나 2024년 기준으로 나이와 대운을 역산하는 환각을 엄격히 금지합니다. 모든 '올해/지금' 표현은 반드시 ${y}년(${yearGanzhi(sajuYear)}년)으로만 해석하세요.`,
  ].join('\n');
}

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Expose-Headers': 'X-Fate-Model, X-Fate-Proxy-Version',
};

function json(obj, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'X-Fate-Proxy-Version': PROXY_VERSION,
      ...CORS,
      ...extraHeaders,
    },
  });
}

function parseJsonSafely(text) {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch (_) {
    let cleaned = text.replace(/```json/gi, '').replace(/```/g, '').trim();
    const firstBrace = cleaned.indexOf('{');
    const lastBrace = cleaned.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace > firstBrace) {
      cleaned = cleaned.slice(firstBrace, lastBrace + 1);
    }
    try {
      return JSON.parse(cleaned);
    } catch (_) {
      return null;
    }
  }
}

async function callGeminiModel({ model, apiKey, finalSystem, finalPrompt }) {
  const endpoint =
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;

  const generationConfig = {
    responseMimeType: 'application/json',
    temperature: 0.65,
    maxOutputTokens: 8192,
  };
  if (model.startsWith('gemini-2.5-')) {
    generationConfig.thinkingConfig = { thinkingBudget: 0 };
  }

  const geminiBody = {
    system_instruction: { parts: [{ text: finalSystem }] },
    contents: [{ role: 'user', parts: [{ text: finalPrompt }] }],
    generationConfig,
    safetySettings: [
      { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH' },
      { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_ONLY_HIGH' },
      { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_ONLY_HIGH' },
      { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_ONLY_HIGH' },
    ],
  };

  let resp;
  try {
    resp = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': apiKey,
      },
      body: JSON.stringify(geminiBody),
      signal: AbortSignal.timeout(MODEL_TIMEOUT_MS),
    });
  } catch (_) {
    return { ok: false, status: 502, reason: 'network_or_timeout' };
  }

  if (!resp.ok) {
    const detail = (await resp.text()).slice(0, 250);
    return { ok: false, status: resp.status, reason: `http_${resp.status}`, detail };
  }

  let data;
  try {
    data = await resp.json();
  } catch (_) {
    return { ok: false, status: 502, reason: 'response_json_error' };
  }

  const parts = data?.candidates?.[0]?.content?.parts ?? [];
  const textPart =
    parts.find((p) => !p.thought && typeof p.text === 'string') ||
    parts[parts.length - 1];
  const text = textPart?.text ?? '';
  const parsed = parseJsonSafely(text);
  if (!parsed || typeof parsed !== 'object') {
    return { ok: false, status: 502, reason: 'parse_error', detail: text.slice(0, 200) };
  }

  return { ok: true, status: 200, model, parsed };
}

async function callOpenRouterFallback({ apiKey, model, finalSystem, finalPrompt }) {
  let resp;
  try {
    resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
        'HTTP-Referer': 'https://fate-report.github.io/',
        'X-Title': 'Fate Report',
      },
      body: JSON.stringify({
        model,
        temperature: 0.65,
        max_tokens: 4096,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: finalSystem },
          { role: 'user', content: finalPrompt },
        ],
      }),
      signal: AbortSignal.timeout(MODEL_TIMEOUT_MS),
    });
  } catch (_) {
    return { ok: false, status: 502, reason: 'openrouter_network_or_timeout' };
  }

  if (!resp.ok) {
    return { ok: false, status: resp.status, reason: `openrouter_${resp.status}` };
  }

  let data;
  try {
    data = await resp.json();
  } catch (_) {
    return { ok: false, status: 502, reason: 'openrouter_json_error' };
  }

  const text = data?.choices?.[0]?.message?.content ?? '';
  const parsed = parseJsonSafely(text);
  if (!parsed || typeof parsed !== 'object') {
    return { ok: false, status: 502, reason: 'openrouter_parse_error' };
  }
  return { ok: true, status: 200, model: `openrouter:${model}`, parsed };
}

export async function onRequestOptions() {
  return new Response(null, { status: 204, headers: CORS });
}

export async function onRequestGet(context) {
  const { env } = context;
  const modelList = env.GEMINI_MODELS
    ? env.GEMINI_MODELS.split(',').map((m) => m.trim()).filter(Boolean)
    : DEFAULT_GEMINI_MODELS;
  return json({
    ok: true,
    version: PROXY_VERSION,
    models: modelList,
    hasGeminiKey: Boolean(env.GEMINI_API_KEY),
    hasOpenRouterKey: Boolean(env.OPENROUTER_API_KEY),
  });
}

export async function onRequestPost(context) {
  const { request, env } = context;

  const geminiKey = env.GEMINI_API_KEY;
  const openRouterKey = env.OPENROUTER_API_KEY;
  if (!geminiKey && !openRouterKey) {
    return json({ error: 'server_misconfigured', message: 'API 키(GEMINI_API_KEY) 없음' }, 500);
  }

  let body;
  try {
    body = await request.json();
  } catch (_) {
    return json({ error: 'bad_request', message: '잘못된 요청' }, 400);
  }
  const system = (body.system || '').toString();
  const prompt = (body.prompt || '').toString();
  if (!prompt) {
    return json({ error: 'bad_request', message: 'prompt 없음' }, 400);
  }

  const timeAnchor = buildKstTimeAnchor();
  const finalSystem = system.includes('Temporal Hard Lock')
    ? system
    : (system ? `${timeAnchor}\n\n${system}` : timeAnchor);
  const finalPrompt = prompt.includes('Temporal Hard Lock')
    ? prompt
    : `${timeAnchor}\n\n${prompt}`;

  const modelList = env.GEMINI_MODELS
    ? env.GEMINI_MODELS.split(',').map((m) => m.trim()).filter(Boolean)
    : DEFAULT_GEMINI_MODELS;

  let sawRateLimit = false;
  let sawCreditError = false;
  let lastError = null;

  if (geminiKey) {
    for (const model of modelList) {
      const result = await callGeminiModel({
        model,
        apiKey: geminiKey,
        finalSystem,
        finalPrompt,
      });
      if (result.ok) {
        return json(result.parsed, 200, { 'X-Fate-Model': result.model });
      }
      if (result.status === 429) sawRateLimit = true;
      if (result.status === 402 || result.status === 403) sawCreditError = true;
      lastError = result;
    }
  }

  if (openRouterKey) {
    const orModel = (env.OPENROUTER_MODEL || DEFAULT_OPENROUTER_MODEL).trim();
    const fallback = await callOpenRouterFallback({
      apiKey: openRouterKey,
      model: orModel,
      finalSystem,
      finalPrompt,
    });
    if (fallback.ok) {
      return json(fallback.parsed, 200, { 'X-Fate-Model': fallback.model });
    }
    if (fallback.status === 429) sawRateLimit = true;
    lastError = fallback;
  }

  if (sawRateLimit) {
    return json({ error: 'rate_limit', message: '요청 과다 (모든 무료 모델 한도 도달)' }, 429);
  }
  if (sawCreditError) {
    return json({ error: 'credit', message: '크레딧/권한 문제' }, 402);
  }
  return json(
    {
      error: 'upstream',
      message: lastError?.reason || 'AI 응답 실패',
      detail: lastError?.detail || '',
    },
    502,
  );
}
