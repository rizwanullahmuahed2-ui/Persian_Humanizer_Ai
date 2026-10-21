// Tokens are JSON strings so whitespace survives SSE framing.
export async function consumeEvents(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "",
    complete = false;
  function dispatch(frame) {
    const data = frame
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).replace(/^ /, ""))
      .join("\n");
    if (!data) return;
    if (data === "__DONE__") complete = true;
    onEvent(data);
  }
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += done
        ? decoder.decode()
        : decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      let end;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        dispatch(buffer.slice(0, end));
        buffer = buffer.slice(end + 2);
      }
      if (done) break;
    }
    if (buffer.trim()) dispatch(buffer);
    if (!complete)
      throw new Error("ارتباط پیش از تکمیل نتیجه قطع شد. دوباره کوشش کن.");
  } finally {
    try {
      await reader.cancel();
    } catch {
      /* Request may already be aborted. */
    }
    reader.releaseLock();
  }
}
