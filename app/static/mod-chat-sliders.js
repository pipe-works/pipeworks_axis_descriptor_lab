/**
 * mod-chat-sliders.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Slider-panel construction and JSON-sync helpers for the Chat Translation
 * page.
 *
 * This module owns the most DOM-heavy part of the chat page: rebuilding the
 * per-character axis slider panel from the current payload state and keeping
 * the JSON textarea view in sync with slider and label edits.
 *
 * Imports: mod-chat-state, mod-utils, mod-chat-server-mode
 */

import { clamp } from "./mod-utils.js";
import { chatState, charDom } from "./mod-chat-state.js";
import { applyActiveAxesIndicators } from "./mod-chat-server-mode.js";

/**
 * Serialise `chatState[ch].payload` back into the JSON textarea.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {void}
 */
export function syncJsonTextarea(ch) {
  const cd = charDom(ch);
  const payload = chatState[ch].payload;
  if (payload) cd.jsonTextarea.value = JSON.stringify(payload, null, 2);
}

/**
 * Update the JSON badge to indicate parse status.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @param {boolean} valid - True if the textarea JSON is currently valid.
 * @returns {void}
 */
export function setJsonBadge(ch, valid) {
  const cd = charDom(ch);
  if (valid) {
    cd.jsonBadge.textContent = "OK";
    cd.jsonBadge.className = "badge";
  } else {
    cd.jsonBadge.textContent = "ERR";
    cd.jsonBadge.className = "badge badge--err";
  }
}

/**
 * Rebuild the axis slider panel for character `ch` from `chatState[ch].payload`.
 *
 * @param {"a"|"b"} ch - Character identifier.
 * @returns {void}
 */
export function buildChatSliders(ch) {
  const cd = charDom(ch);
  const payload = chatState[ch].payload;
  const panel = cd.sliderPanel;

  if (!payload || typeof payload.axes !== "object" || payload.axes === null) {
    panel.textContent = "";
    const p = document.createElement("p");
    p.className = "placeholder-text";
    p.textContent = "No axes found in payload.";
    panel.appendChild(p);
    return;
  }

  const axes = payload.axes;
  const keys = Object.keys(axes);

  if (keys.length === 0) {
    panel.textContent = "";
    const p = document.createElement("p");
    p.className = "placeholder-text";
    p.textContent = "axes object is empty.";
    panel.appendChild(p);
    return;
  }

  if (chatState[ch].activeAxes === null) {
    chatState[ch].activeAxes = new Set(keys);
  }

  const fragment = document.createDocumentFragment();

  for (const axisKey of keys) {
    const axisVal = axes[axisKey];
    const score = clamp(parseFloat(axisVal.score) || 0, 0, 1);
    const label = String(axisVal.label || "");
    const isActive = chatState[ch].activeAxes.has(axisKey);
    const orig = chatState[ch].originalAxes && chatState[ch].originalAxes[axisKey];

    const row = document.createElement("div");
    row.className = "axis-row chat-axis-row";
    row.dataset.axis = axisKey;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "axis-enable-checkbox";
    checkbox.checked = isActive;
    checkbox.title = "Include this axis in the character profile";
    checkbox.setAttribute("aria-label", `Enable ${axisKey} in profile`);

    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        chatState[ch].activeAxes.add(axisKey);
      } else {
        chatState[ch].activeAxes.delete(axisKey);
      }
      row.classList.toggle("axis-row--disabled", !checkbox.checked);
    });

    row.classList.toggle("axis-row--disabled", !isActive);

    const nameEl = document.createElement("span");
    nameEl.className = "axis-name";
    nameEl.textContent = axisKey;
    nameEl.title = axisKey;

    const scoreDisplay = document.createElement("span");
    scoreDisplay.className = "axis-score";
    scoreDisplay.textContent = score.toFixed(3);

    if (orig && Math.abs(score - orig.score) > 0.0001) {
      scoreDisplay.classList.add("axis-modified");
    }

    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "range-input";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.005";
    slider.value = score.toFixed(3);
    slider.setAttribute("aria-label", `${axisKey} score`);

    const labelInput = document.createElement("input");
    labelInput.type = "text";
    labelInput.className = "axis-label-input";
    labelInput.value = label;
    labelInput.setAttribute("aria-label", `${axisKey} label`);
    labelInput.disabled = cd.autoLabel.checked;

    if (orig && label !== orig.label) {
      labelInput.classList.add("axis-modified");
    }

    slider.addEventListener("input", () => {
      const newScore = parseFloat(slider.value);
      scoreDisplay.textContent = newScore.toFixed(3);
      const origAxis = chatState[ch].originalAxes && chatState[ch].originalAxes[axisKey];
      if (origAxis) {
        scoreDisplay.classList.toggle("axis-modified", Math.abs(newScore - origAxis.score) > 0.0001);
      }
      chatState[ch].payload.axes[axisKey] = { ...chatState[ch].payload.axes[axisKey], score: newScore };
      syncJsonTextarea(ch);
    });

    labelInput.addEventListener("input", () => {
      const origAxis = chatState[ch].originalAxes && chatState[ch].originalAxes[axisKey];
      if (origAxis) {
        labelInput.classList.toggle("axis-modified", labelInput.value !== origAxis.label);
      }
      chatState[ch].payload.axes[axisKey] = { ...chatState[ch].payload.axes[axisKey], label: labelInput.value };
      syncJsonTextarea(ch);
    });

    row.appendChild(checkbox);
    row.appendChild(nameEl);
    row.appendChild(scoreDisplay);
    row.appendChild(slider);
    row.appendChild(labelInput);
    fragment.appendChild(row);
  }

  panel.textContent = "";
  panel.appendChild(fragment);

  if (chatState.worldConfig) applyActiveAxesIndicators();
}
