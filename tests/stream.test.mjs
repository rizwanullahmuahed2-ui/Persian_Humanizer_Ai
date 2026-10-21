import test from 'node:test';
import assert from 'node:assert/strict';
import {consumeEvents} from '../frontrend/stream.mjs';

function stream(text, chunk = 1) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({start(controller) {
    for (let index = 0; index < bytes.length; index += chunk) controller.enqueue(bytes.slice(index, index + chunk));
    controller.close();
  }});
}

test('fragmented UTF-8 tokens retain Persian, spaces and escaped newlines', async () => {
  const values = [];
  await consumeEvents(stream('data: __TOKEN__"سلام\\n دنیا"\n\ndata: __TEXT__"سلام دنیا"\n\ndata: __DONE__\n\n'), data => values.push(data));
  assert.equal(JSON.parse(values[0].slice(9)), 'سلام\n دنیا');
  assert.equal(JSON.parse(values[1].slice(8)), 'سلام دنیا');
  assert.equal(values.at(-1), '__DONE__');
});
test('CRLF frames, trailing event and incomplete streams', async () => {
  const values = [];
  await consumeEvents(stream('data: __GENERATING__\r\n\r\ndata: __DONE__', 4), data => values.push(data));
  assert.deepEqual(values, ['__GENERATING__', '__DONE__']);
  await assert.rejects(consumeEvents(stream('data: __TOKEN__"partial"\n\n'), () => {}), /ارتباط/);
});
test('server error interrupts consumption', async () => {
  await assert.rejects(consumeEvents(stream('data: __ERROR__failed\n\n'), data => {
    if (data.startsWith('__ERROR__')) throw new Error(data.slice(9));
  }), /failed/);
});
