/* contrast_check.cjs — asserts stone-rim visibility on both themes.

   Black stones on the dark board were nearly invisible (black rim on a near-black board);
   this pins the fix: the stroke (rim) of each stone colour must hit WCAG contrast ≥ 3:1
   against every board-background colour of its theme. Colours are extracted from
   webapp/ui.js source so the test follows the palette.

   Run: node scripts/contrast_check.cjs                                                    */
"use strict";
const fs = require("fs"), path = require("path");
const src = fs.readFileSync(path.join(__dirname, "..", "webapp", "ui.js"), "utf8");

const grab = (re, what) => {
  const m = src.match(re);
  if (!m) { console.error(`cannot find ${what} in ui.js`); process.exit(1); }
  return m[1];
};
// theme palettes
const darkB = grab(/dark:\s*{[^}]*bStroke:\s*"(#[0-9a-f]{6})"/i, "dark bStroke");
const darkW = grab(/dark:\s*{[^}]*wStroke:\s*"(#[0-9a-f]{6})"/i, "dark wStroke");
const lightB = grab(/light:\s*{[^}]*bStroke:\s*"(#[0-9a-f]{6})"/i, "light bStroke");
const lightW = grab(/light:\s*{[^}]*wStroke:\s*"(#[0-9a-f]{6})"/i, "light wStroke");
// board backgrounds: the dark radial gradient stops (bgs) + the light flat colour
const bgsStops = [...src.matchAll(/id="bgs"[^/]*?stop-color="(#[0-9a-f]{6})"[^/]*?stop-color="(#[0-9a-f]{6})"/gi)][0];
const darkBgs = bgsStops ? [bgsStops[1], bgsStops[2]] : ["#1a1f2c", "#0a0c12"];
const lightBg = grab(/light:\s*{\s*boardBg:\s*"(#[0-9a-f]{6})"/i, "light boardBg");

const lum = hex => {
  const c = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map(v => v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
};
const ratio = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };

let fails = 0;
const check = (label, rim, bgs) => {
  for (const bg of bgs) {
    const r = ratio(rim, bg);
    const ok = r >= 3.0;
    if (!ok) fails++;
    console.log(`${ok ? "ok  " : "FAIL"} ${label}: rim ${rim} vs board ${bg} → ${r.toFixed(2)}:1`);
  }
};
check("dark  · black stone", darkB, darkBgs);
check("dark  · white stone", darkW, darkBgs);
check("light · black stone", lightB, [lightBg]);
check("light · white stone", lightW, [lightBg]);

console.log(fails ? `\nFAIL — ${fails} rim/background pair(s) under 3:1` : "\nPASS — all stone rims ≥ 3:1 on both themes");
process.exit(fails ? 1 : 0);
