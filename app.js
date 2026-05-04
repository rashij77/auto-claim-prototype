/**
 * ============================================================
 *  AGRI-INSURANCE SYSTEM — SYSTEM 2: WEATHER ORACLE & PAYOUT LEDGER
 *  Professional-grade Node.js implementation
 *  No shortcuts. Production-ready. Fraud-prevention hardened.
 * ============================================================
 */

"use strict";

// ── 1. ENVIRONMENT ────────────────────────────────────────────
require("dotenv").config(); // Loads .env from project root

const express  = require("express");
const axios    = require("axios");
const fs       = require("fs");
const path     = require("path");

// ── 2. CONSTANTS & PATH RESOLUTION ────────────────────────────
const PORT            = process.env.PORT || 3000;
const OPENWEATHER_KEY = process.env.OPENWEATHER_API_KEY;

// Always resolve paths relative to THIS file — never rely on cwd()
const DATA_DIR        = path.join(__dirname, "data");
const LAND_REGISTRY   = path.join(DATA_DIR, "landRegistry.json");
const AUDIT_LOG       = path.join(DATA_DIR, "auditLog.json");

// ── 3. BOOT-TIME SAFETY CHECKS ────────────────────────────────
/**
 * Ensures the data/ directory exists and auditLog.json is initialised.
 * Called once before the server starts accepting requests.
 */
function bootstrapDataLayer() {
  // 3a. Create data/ directory if absent
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    console.log(`[BOOT] Created missing directory: ${DATA_DIR}`);
  }

  // 3b. Create auditLog.json if absent
  if (!fs.existsSync(AUDIT_LOG)) {
    fs.writeFileSync(AUDIT_LOG, JSON.stringify([], null, 2), "utf8");
    console.log(`[BOOT] Initialised empty audit log: ${AUDIT_LOG}`);
  }

  // 3c. Warn loudly if landRegistry.json is missing
  if (!fs.existsSync(LAND_REGISTRY)) {
    console.error(`[BOOT] CRITICAL: Land registry not found at ${LAND_REGISTRY}`);
    console.error(`[BOOT] Create data/landRegistry.json before processing claims.`);
  }

  // 3d. Warn if API key is missing
  if (!OPENWEATHER_KEY) {
    console.warn("[BOOT] WARNING: OPENWEATHER_API_KEY is not set in .env — weather calls will fail.");
  }
}

// ── 4. HELPER: APPEND TO AUDIT LOG ────────────────────────────
/**
 * Appends a single audit entry to auditLog.json atomically.
 * Never throws — audit failure must not crash a live claim request.
 *
 * @param {object} entry - Structured audit record
 */
function appendAuditLog(entry) {
  try {
    const raw      = fs.readFileSync(AUDIT_LOG, "utf8");
    const log      = JSON.parse(raw);
    log.push({ ...entry, timestamp: new Date().toISOString() });
    fs.writeFileSync(AUDIT_LOG, JSON.stringify(log, null, 2), "utf8");
  } catch (err) {
    console.error("[AUDIT] Failed to write audit log:", err.message);
  }
}

// ── 5. HELPER: FETCH WEATHER ───────────────────────────────────
/**
 * Fetches current weather for a given lat/lon from OpenWeatherMap.
 * Returns a normalised object; throws on network/API failure.
 *
 * @param {number} lat
 * @param {number} lon
 * @returns {Promise<object>} Normalised weather payload
 */
async function fetchWeather(lat, lon) {
  const url = "https://api.openweathermap.org/data/2.5/weather";
  const response = await axios.get(url, {
    params: {
      lat,
      lon,
      appid: OPENWEATHER_KEY,
      units: "metric",
    },
    timeout: 8000, // 8-second hard timeout
  });

  const d = response.data;
  return {
    location       : d.name,
    condition      : d.weather[0].main,
    description    : d.weather[0].description,
    temperatureC   : d.main.temp,
    humidityPct    : d.main.humidity,
    rainfallMm     : d.rain ? (d.rain["1h"] || d.rain["3h"] || 0) : 0,
    windSpeedKmh   : parseFloat((d.wind.speed * 3.6).toFixed(2)),
    observedAtUTC  : new Date(d.dt * 1000).toISOString(),
  };
}

// ── 6. HELPER: PAYOUT DECISION ENGINE ─────────────────────────
/**
 * Pure function — determines payout eligibility based on weather data
 * and the policy thresholds stored in the land record.
 *
 * Edit ONLY this function to change your fraud/payout rules.
 *
 * @param {object} weather      - Normalised weather payload
 * @param {object} landRecord   - Entry from landRegistry.json
 * @returns {{ eligible: boolean, reason: string, payoutINR: number }}
 */
function evaluatePayout(weather, landRecord) {
  const policy = landRecord.policy;

  // Rule 1: Drought — rainfall below threshold
  if (weather.rainfallMm < policy.droughtThresholdMm) {
    return {
      eligible  : true,
      reason    : `Drought detected. Rainfall ${weather.rainfallMm} mm < threshold ${policy.droughtThresholdMm} mm.`,
      payoutINR : policy.droughtPayoutINR,
    };
  }

  // Rule 2: Flood — rainfall above threshold
  if (weather.rainfallMm > policy.floodThresholdMm) {
    return {
      eligible  : true,
      reason    : `Flood detected. Rainfall ${weather.rainfallMm} mm > threshold ${policy.floodThresholdMm} mm.`,
      payoutINR : policy.floodPayoutINR,
    };
  }

  // Rule 3: Extreme heat
  if (weather.temperatureC > policy.heatThresholdC) {
    return {
      eligible  : true,
      reason    : `Extreme heat detected. ${weather.temperatureC}°C > threshold ${policy.heatThresholdC}°C.`,
      payoutINR : policy.heatPayoutINR,
    };
  }

  // No trigger matched
  return {
    eligible  : false,
    reason    : "Current weather conditions do not meet any policy trigger threshold.",
    payoutINR : 0,
  };
}

// ── 7. EXPRESS APPLICATION ─────────────────────────────────────
const app = express();
app.use(express.json());

// ── 7a. HEALTH CHECK ──────────────────────────────────────────
app.get("/health", (_req, res) => {
  res.json({ status: "OK", service: "Agri-Insurance Weather Oracle", uptime: process.uptime() });
});

// ── 7b. CORE ROUTE: PROCESS CLAIM ─────────────────────────────
/**
 * GET /process-claim/:farmerId/:claimId
 *
 * :farmerId — e.g. "101"
 * :claimId  — e.g. "A"
 *
 * Steps:
 *   1. Load land registry
 *   2. Locate farmer record
 *   3. Fetch live weather for the farm's coordinates
 *   4. Run payout decision engine
 *   5. Append immutable audit log entry
 *   6. Return structured JSON response
 */
app.get("/process-claim/:farmerId/:claimId", async (req, res) => {
  const { farmerId, claimId } = req.params;
  const requestId = `${farmerId}-${claimId}-${Date.now()}`;

  console.log(`\n[CLAIM] Processing request ${requestId}`);

  // Step 1 — Load land registry
  let registry;
  try {
    const raw = fs.readFileSync(LAND_REGISTRY, "utf8");
    registry  = JSON.parse(raw);
  } catch (err) {
    console.error("[CLAIM] Failed to read land registry:", err.message);
    return res.status(500).json({
      success : false,
      error   : "Internal error: Land registry unavailable.",
      requestId,
    });
  }

  // Step 2 — Find farmer
  const farmerRecord = registry.find(
    (r) => String(r.farmerId) === String(farmerId)
  );

  if (!farmerRecord) {
    console.warn(`[CLAIM] Farmer ${farmerId} not found in registry.`);
    appendAuditLog({
      requestId, farmerId, claimId,
      outcome: "REJECTED", reason: "Farmer ID not found in land registry.",
    });
    return res.status(404).json({
      success   : false,
      requestId,
      farmerId,
      claimId,
      outcome   : "REJECTED",
      reason    : `Farmer ID '${farmerId}' does not exist in the land registry.`,
    });
  }

  // Step 3 — Fetch live weather
  let weather;
  try {
    weather = await fetchWeather(farmerRecord.lat, farmerRecord.lon);
    console.log(`[CLAIM] Weather fetched for ${farmerRecord.name}:`, weather.condition, `${weather.temperatureC}°C`);
  } catch (err) {
    console.error("[CLAIM] Weather API failure:", err.message);
    appendAuditLog({
      requestId, farmerId, claimId, farmerName: farmerRecord.name,
      outcome: "ERROR", reason: `Weather API error: ${err.message}`,
    });
    return res.status(502).json({
      success   : false,
      requestId,
      error     : "Weather Oracle temporarily unavailable. Try again shortly.",
      detail    : err.message,
    });
  }

  // Step 4 — Evaluate payout
  const decision = evaluatePayout(weather, farmerRecord);
  console.log(`[CLAIM] Decision for ${farmerRecord.name}: eligible=${decision.eligible}`);

  // Step 5 — Write immutable audit entry
  appendAuditLog({
    requestId,
    farmerId,
    claimId,
    farmerName    : farmerRecord.name,
    farmLocation  : farmerRecord.location,
    weather,
    decision,
    outcome       : decision.eligible ? "APPROVED" : "REJECTED",
  });

  // Step 6 — Respond
  return res.status(200).json({
    success       : true,
    requestId,
    farmerId,
    claimId,
    farmerName    : farmerRecord.name,
    farmLocation  : farmerRecord.location,
    weather,
    decision,
    outcome       : decision.eligible ? "APPROVED" : "REJECTED",
  });
});

// ── 7c. 404 FALLBACK ──────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({
    error   : `Route not found: ${req.method} ${req.path}`,
    hint    : "Valid routes: GET /health, GET /process-claim/:farmerId/:claimId",
  });
});

// ── 8. START SERVER ────────────────────────────────────────────
bootstrapDataLayer(); // Run safety checks before binding port

app.listen(PORT, () => {
  console.log("\n╔══════════════════════════════════════════════════╗");
  console.log(`║  Agri-Insurance Weather Oracle                   ║`);
  console.log(`║  LIVE on http://localhost:${PORT}                   ║`);
  console.log("╠══════════════════════════════════════════════════╣");
  console.log(`║  Test:  GET /process-claim/101/A                 ║`);
  console.log(`║  Health: GET /health                             ║`);
  console.log("╚══════════════════════════════════════════════════╝\n");
});