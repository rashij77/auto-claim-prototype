/**
 * Automated Insurance Claim Backend
 * app.js — Express server with biometric integration, claim processing, and JWT security
 */

"use strict";

const express = require("express");
const { execFile } = require("child_process");
const fs = require("fs");
const path = require("path");
const https = require("https");
const jwt = require("jsonwebtoken");
const multer = require("multer");
const cors = require("cors");

// ─── Config ──────────────────────────────────────────────────────────────────

const PORT = process.env.PORT || 3000;
const JWT_SECRET = process.env.JWT_SECRET || "CHANGE_ME_IN_PRODUCTION";
const OPENWEATHER_API_KEY = process.env.OPENWEATHER_API_KEY || "";
const WEATHER_CITY = process.env.WEATHER_CITY || "Mumbai";

const DATA_DIR = path.join(__dirname, "data");
const LAND_REGISTRY_PATH = path.join(DATA_DIR, "landRegistry.json");
const AUDIT_LOG_PATH = path.join(DATA_DIR, "auditLog.json");
const PYTHON_SCRIPT = path.join(__dirname, "biometric-system", "face_recognition_system.py");

// ─── App Setup ────────────────────────────────────────────────────────────────

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// Multer: store uploaded biometric images in memory (no disk persistence)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 }, // 5 MB cap
  fileFilter: (_req, file, cb) => {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("Only image files are accepted for biometric verification."));
    }
    cb(null, true);
  },
});

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Safely read and parse a JSON file.
 * Returns `defaultValue` if the file is missing or malformed — server never crashes.
 */
function readJSON(filePath, defaultValue = null) {
  try {
    const raw = fs.readFileSync(filePath, "utf8");
    return JSON.parse(raw);
  } catch (err) {
    if (err.code === "ENOENT") {
      console.warn(`[WARN] File not found: ${filePath}. Using default.`);
    } else {
      console.error(`[ERROR] Failed to parse JSON at ${filePath}:`, err.message);
    }
    return defaultValue;
  }
}

/**
 * Safely write a JSON file atomically (write to tmp, then rename).
 * Prevents corruption if the process dies mid-write.
 */
function writeJSON(filePath, data) {
  const tmp = filePath + ".tmp";
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2), "utf8");
  fs.renameSync(tmp, filePath);
}

/**
 * Fetch current weather from OpenWeatherMap.
 * Resolves with weather object; rejects with a descriptive Error.
 */
function fetchWeather(city) {
  return new Promise((resolve, reject) => {
    if (!OPENWEATHER_API_KEY) {
      // Graceful fallback when key is absent (dev/testing)
      return resolve({ condition: "unknown", temperature: null, note: "API key not configured" });
    }
    const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&appid=${OPENWEATHER_API_KEY}&units=metric`;
    https.get(url, (res) => {
      let body = "";
      res.on("data", (chunk) => (body += chunk));
      res.on("end", () => {
        try {
          const json = JSON.parse(body);
          if (json.cod !== 200) return reject(new Error(`Weather API error: ${json.message}`));
          resolve({
            condition: json.weather?.[0]?.main ?? "N/A",
            description: json.weather?.[0]?.description ?? "N/A",
            temperature: json.main?.temp ?? null,
            humidity: json.main?.humidity ?? null,
            city: json.name,
            fetchedAt: new Date().toISOString(),
          });
        } catch {
          reject(new Error("Failed to parse weather response."));
        }
      });
    }).on("error", (err) => reject(new Error(`Weather HTTP error: ${err.message}`)));
  });
}

/**
 * JWT middleware — verifies the bearer token and injects `req.user`.
 */
function requireAuth(req, res, next) {
  const authHeader = req.headers.authorization || "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : null;
  if (!token) return res.status(401).json({ error: "Authorization token required." });

  try {
    req.user = jwt.verify(token, JWT_SECRET);
    next();
  } catch (err) {
    const message = err.name === "TokenExpiredError" ? "Token has expired." : "Invalid token.";
    return res.status(401).json({ error: message });
  }
}

/**
 * Middleware — ensures the JWT payload confirms biometric verification was done.
 * A token issued without `biometricVerified: true` is rejected here.
 */
function requireBiometric(req, res, next) {
  if (!req.user?.biometricVerified) {
    return res.status(403).json({
      error: "Biometric verification is required before submitting a claim.",
    });
  }
  next();
}

// ─── Route: POST /verify-identity ─────────────────────────────────────────────
/**
 * Accepts a multipart image upload, pipes it to the Python face-recognition
 * script via stdin, and returns a JWT that encodes the biometric result.
 *
 * Data flow:
 *   Browser  →  (multipart/form-data image)  →  /verify-identity
 *   Node     →  spawn python face_recognition_system.py
 *   Node     →  write image bytes to python stdin
 *   Python   →  reads stdin, runs TensorFlow model, prints JSON to stdout
 *   Node     →  parses stdout, issues JWT { userId, biometricVerified }
 *   Browser  ←  JWT (stored client-side, attached to /submit-claim)
 */
app.post("/verify-identity", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image file received. Send it as 'image' field." });
  }

  const userId = req.body.userId;
  if (!userId) {
    return res.status(400).json({ error: "'userId' field is required." });
  }

  // Spawn Python script — we communicate via stdin/stdout (no temp files)
  const python = execFile(
    "python3",
    [PYTHON_SCRIPT],
    { timeout: 30_000, maxBuffer: 1024 * 512 },
    (err, stdout, stderr) => {
      if (err) {
        // Python crashed or timed out — log the stderr for ops, return generic error to client
        console.error("[biometric] Python error:", err.message);
        console.error("[biometric] stderr:", stderr);
        return res.status(503).json({
          error: "Biometric service is temporarily unavailable. Please try again.",
        });
      }

      // Parse the JSON line printed by the Python script
      let result;
      try {
        result = JSON.parse(stdout.trim());
      } catch {
        console.error("[biometric] Unexpected Python output:", stdout);
        return res.status(500).json({ error: "Biometric system returned an unreadable response." });
      }

      const verified = result.status === "Success";

      // Issue a short-lived JWT encoding the biometric outcome
      const token = jwt.sign(
        { userId, biometricVerified: verified, verifiedAt: new Date().toISOString() },
        JWT_SECRET,
        { expiresIn: "15m" } // claim must be submitted within 15 minutes
      );

      return res.json({
        verified,
        message: verified ? "Identity verified successfully." : "Biometric verification failed.",
        token, // sent back only so the client can use it on /submit-claim
        confidence: result.confidence ?? null,
      });
    }
  );

  // Stream image buffer into Python stdin
  python.stdin.write(req.file.buffer);
  python.stdin.end();
});

// ─── Route: POST /submit-claim ────────────────────────────────────────────────
/**
 * Protected by:
 *   1. requireAuth  — valid JWT must be present
 *   2. requireBiometric — JWT must carry biometricVerified: true
 *
 * Data flow:
 *   Browser  →  POST /submit-claim  (Bearer <token>, JSON body)
 *   Node     →  validates JWT → checks biometricVerified flag
 *   Node     →  reads landRegistry.json → checks userId exists
 *   Node     →  calls OpenWeatherMap API → fetches current weather
 *   Node     →  appends claim + weather snapshot to auditLog.json
 *   Browser  ←  { claimId, status: "Pending", weather }
 */
app.post("/submit-claim", requireAuth, requireBiometric, async (req, res) => {
  const { userId, claimType, description, location, amount } = req.body;

  // Basic field validation
  if (!userId || !claimType || !description) {
    return res.status(400).json({ error: "userId, claimType, and description are required." });
  }

  // Ensure the authenticated user isn't spoofing a different userId
  if (req.user.userId !== userId) {
    return res.status(403).json({ error: "userId does not match the authenticated token." });
  }

  // ── Step 1: Validate userId against Land Registry ──────────────────────────
  const registry = readJSON(LAND_REGISTRY_PATH, { users: [] });
  const registeredUser = (registry.users ?? []).find((u) => u.id === userId);
  if (!registeredUser) {
    return res.status(404).json({ error: `User ID '${userId}' not found in the land registry.` });
  }

  // ── Step 2: Fetch weather context ─────────────────────────────────────────
  let weather;
  try {
    weather = await fetchWeather(location ?? WEATHER_CITY);
  } catch (err) {
    console.error("[weather] Fetch failed:", err.message);
    // Non-fatal: log the warning and continue with null weather
    weather = { condition: "unavailable", error: err.message };
  }

  // ── Step 3: Append to Audit Log ───────────────────────────────────────────
  const auditLog = readJSON(AUDIT_LOG_PATH, { claims: [] });
  if (!Array.isArray(auditLog.claims)) auditLog.claims = [];

  const claimId = `CLM-${Date.now()}-${Math.random().toString(36).slice(2, 7).toUpperCase()}`;
  const newClaim = {
    claimId,
    userId,
    userName: registeredUser.name ?? "Unknown",
    claimType,
    description,
    amount: amount ?? null,
    location: location ?? WEATHER_CITY,
    status: "Pending",
    biometricVerifiedAt: req.user.verifiedAt,
    weather,
    submittedAt: new Date().toISOString(),
  };

  auditLog.claims.push(newClaim);

  try {
    writeJSON(AUDIT_LOG_PATH, auditLog);
  } catch (err) {
    console.error("[audit] Failed to write auditLog.json:", err.message);
    return res.status(500).json({ error: "Failed to persist claim. Please retry." });
  }

  return res.status(201).json({
    message: "Claim submitted successfully.",
    claimId,
    status: "Pending",
    weather,
    submittedAt: newClaim.submittedAt,
  });
});

// ─── Route: GET /health ───────────────────────────────────────────────────────
app.get("/health", (_req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// ─── Global Error Handler ─────────────────────────────────────────────────────
// Catches multer errors (wrong file type, size exceeded) and any other unhandled throws
app.use((err, _req, res, _next) => {
  console.error("[ERROR]", err.message);
  if (err.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({ error: "Image file too large. Maximum size is 5 MB." });
  }
  res.status(400).json({ error: err.message || "Unexpected server error." });
});

// ─── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`[server] Insurance Claim API running on http://localhost:${PORT}`);
  console.log(`[server] Land registry : ${LAND_REGISTRY_PATH}`);
  console.log(`[server] Audit log     : ${AUDIT_LOG_PATH}`);
  console.log(`[server] Python script : ${PYTHON_SCRIPT}`);
});

module.exports = app; // for testing