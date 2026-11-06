const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const context = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../space/live_tracking.js'), 'utf8') +
  '\nthis.LiveTracker = LiveTracker;', context);
const { LiveTracker } = context;
const cell = (score = 0.8, x = 0, cls = 1) => [x, 0, x + 100, 100, score, cls];
const update = (tracker, boxes, now, conf = 0.42, width = 1000) => tracker.update(boxes, now, conf, width, 1000);

test('one-frame false detections are hidden, persistent cells are confirmed', () => {
  const tracker = new LiveTracker();
  assert.equal(update(tracker, [cell()], 0).length, 0);
  assert.equal(update(tracker, [], 100).length, 0);
  assert.equal(update(tracker, [cell()], 200).length, 0);
  assert.equal(update(tracker, [cell()], 300).length, 1);
});

test('brief misses are bridged, but stale cells expire after 400 ms', () => {
  const tracker = new LiveTracker();
  update(tracker, [cell()], 0);
  update(tracker, [cell()], 100);
  assert.equal(update(tracker, [], 200).length, 1);
  assert.equal(update(tracker, [cell()], 300).length, 1);
  assert.equal(update(tracker, [], 700).length, 1);
  assert.equal(update(tracker, [], 701).length, 0);
});

test('threshold fluctuations do not repeatedly hide and rediscover a confirmed cell', () => {
  const tracker = new LiveTracker();
  update(tracker, [cell(0.6)], 0);
  update(tracker, [cell(0.6)], 100);
  const raw = [], smoothed = [];
  for (let i = 0; i < 50; i++) {
    const score = i % 2 ? 0.46 : 0.38;
    raw.push(score >= 0.42 ? 1 : 0);
    smoothed.push(update(tracker, [cell(score)], 200 + i * 100).length);
  }
  const changes = (values) => values.slice(1).filter((v, i) => v !== values[i]).length;
  assert.equal(changes(raw), 49);
  assert.equal(changes(smoothed), 0);
  for (let i = 0; i < 20; i++) update(tracker, [cell(0.32)], 5200 + i * 100);
  assert.equal(update(tracker, [cell(0.32)], 7200).length, 0, 'sustained low confidence must eventually hide the cell');
});

test('position and displayed score jitter are reduced', () => {
  const tracker = new LiveTracker();
  update(tracker, [cell(0.8)], 0);
  update(tracker, [cell(0.8)], 100);
  const displayedPositions = [], displayedScores = [];
  for (let i = 0; i < 30; i++) {
    const box = update(tracker, [cell(i % 2 ? 0.9 : 0.7, i % 2 ? 4 : -4)], 200 + i * 100)[0];
    displayedPositions.push(box[0]); displayedScores.push(box[4]);
  }
  assert(Math.max(...displayedPositions) - Math.min(...displayedPositions) < 6);
  assert(Math.max(...displayedScores) - Math.min(...displayedScores) < 0.12);
});

test('neighboring cells use separate one-to-one tracks', () => {
  const tracker = new LiveTracker();
  update(tracker, [cell(0.8, 0), cell(0.8, 90)], 0);
  const boxes = update(tracker, [cell(0.8, 88), cell(0.8, 2)], 100);
  assert.equal(boxes.length, 2);
  assert(boxes[0][0] < 10 && boxes[1][0] > 80);
});

test('a transient class flip is suppressed and persistent class evidence can change the label', () => {
  const tracker = new LiveTracker();
  update(tracker, [cell(0.9, 0, 1)], 0);
  update(tracker, [cell(0.9, 0, 1)], 100);
  assert.equal(update(tracker, [cell(0.9, 0, 0)], 200)[0][5], 1);
  assert.equal(update(tracker, [cell(0.9, 0, 1)], 300)[0][5], 1);
  let boxes;
  for (let i = 0; i < 20; i++) boxes = update(tracker, [cell(0.9, 0, 0)], 400 + i * 100);
  assert.equal(boxes.length, 1);
  assert.equal(boxes[0][5], 0);
});

test('slow inference still confirms a continuously detected cell', () => {
  const tracker = new LiveTracker();
  update(tracker, [cell()], 0);
  assert.equal(update(tracker, [cell()], 1500).length, 1);
  assert.equal(update(tracker, [], 3000).length, 0);
});

test('changing crop or resetting cannot inherit old detections', () => {
  const tracker = new LiveTracker();
  update(tracker, [cell()], 0);
  update(tracker, [cell()], 100);
  assert.equal(update(tracker, [cell()], 200, 0.42, 500).length, 0);
  tracker.reset();
  assert.equal(update(tracker, [cell()], 300).length, 0);
});

test('moving to a different field clears all held boxes', () => {
  const tracker = new LiveTracker();
  const oldField = Array.from({ length: 6 }, (_, i) => [i * 120, 0, i * 120 + 100, 100, 0.8, 1]);
  const newField = oldField.map((b) => [b[0], 500, b[2], 600, b[4], b[5]]);
  update(tracker, oldField, 0); update(tracker, oldField, 100);
  assert.equal(update(tracker, newField, 200).length, 0);
  assert.equal(update(tracker, newField, 300).length, 6);
});

test('WBC subtype probabilities are smoothed with their own track', () => {
  const tracker = new LiveTracker();
  const first = [...cell(0.9, 0, 0), [0.9, 0.1]];
  const other = [...cell(0.9, 0, 0), [0.1, 0.9]];
  update(tracker, [first], 0); update(tracker, [first], 100);
  let result = update(tracker, [other], 200)[0];
  assert(result[6][0] > result[6][1], 'One-frame subtype flip must not immediately change its label');
  for (let i = 0; i < 20; i++) result = update(tracker, [other], 300 + i * 100)[0];
  assert(result[6][1] > 0.8, 'Persistent subtype evidence must eventually change the label');
  assert(Math.abs(result[6][0] + result[6][1] - 1) < 1e-6);
  tracker.reset();
  assert.equal(update(tracker, [first], 3000).length, 0);
});
