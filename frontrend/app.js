import { consumeEvents } from "./stream.mjs";

const $ = (id) => document.getElementById(id);
const api =
  location.port === "5500" ? "http://127.0.0.1:8000" : location.origin;
const numerals = new Intl.NumberFormat("fa");
let controller = null,
  result = "",
  completed = false;
function message(value = "") {
  $("error").textContent = value;
  $("error").hidden = !value;
}
function count() {
  $("count").textContent =
    `${numerals.format([...$("input").value].length)} / ۲۰۰۰`;
}
function busy(value) {
  for (const id of ["submit", "sample", "clear", "input", "model"])
    $(id).disabled = value;
  document.querySelectorAll('[name="style"]').forEach((input) => {
    input.disabled = value;
  });
  $("cancel").hidden = !value;
  $("output").setAttribute("aria-busy", String(value));
  $("copy").disabled = $("download").disabled = value || !completed;
}
function modelNote() {
  $("model-note").textContent =
    $("model").value === "gpt-4.1-mini"
      ? "متن و نمونه‌های مرتبط برای پردازش به OpenAI فرستاده می‌شوند. مصرف API هزینه دارد."
      : "مدل محلی به Ollama نیاز دارد. در اولین اجرا، دریافت مدل‌های بازیابی ممکن است زمان ببرد.";
}
async function health() {
  $("connection").textContent = "در حال بررسی سرور…";
  try {
    const response = await fetch(`${api}/health`, {
      signal: AbortSignal.timeout(8000),
    });
    if (!response.ok) throw new Error();
    const info = await response.json();
    $("connection").textContent = "API در دسترس است";
    $("connection-note").textContent =
      "آماده‌بودن API، در دسترس بودن مدل را تضمین نمی‌کند.";
    const gpt = info.available_models?.find(
      (model) => model.id === "gpt-4.1-mini",
    );
    const option = $("model").querySelector('[value="gpt-4.1-mini"]');
    option.disabled = !gpt?.configured;
    option.textContent = gpt?.configured
      ? "GPT-4.1 mini · OpenAI"
      : "GPT-4.1 mini · تنظیم نشده";
    if (option.disabled && $("model").value === option.value)
      $("model").value = "llama3.1:8b";
    modelNote();
  } catch {
    $("connection").textContent = "سرور در دسترس نیست";
    $("connection-note").textContent =
      "بک‌اند را اجرا کن و این صفحه را از آدرس /ui/ باز کن؛ سپس دوباره بررسی کن.";
  }
}
$("input").addEventListener("input", count);
$("model").addEventListener("change", modelNote);
$("retry").addEventListener("click", health);
$("sample").addEventListener("click", () => {
  $("input").value =
    "استفاده از خدمات آنلاین می‌تواند باعث صرفه‌جویی در وقت شود، اما بررسی امنیت حساب و حفظ اطلاعات شخصی نیز ضروری است.";
  count();
  $("input").focus();
});
$("clear").addEventListener("click", () => {
  $("input").value = "";
  result = "";
  completed = false;
  $("output").textContent = "متن تازه را وارد کن.";
  $("result-note").textContent = "آماده برای شروع";
  $("progress").textContent = "";
  message();
  count();
  busy(false);
  $("input").focus();
});
$("cancel").addEventListener("click", () => controller?.abort());
$("rewrite-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (controller) return;
  message();
  const text = $("input").value.trim();
  if (!text) return message("اول متن را وارد کن.");
  if ([...text].length > 2000)
    return message("متن باید حداکثر ۲۰۰۰ کاراکتر باشد.");
  result = "";
  completed = false;
  controller = new AbortController();
  busy(true);
  $("output").textContent = "در انتظار مدل…";
  $("result-note").textContent = "در حال پردازش";
  $("progress").textContent = "در حال جست‌وجوی نمونه‌ها…";
  const timeout = setTimeout(() => controller?.abort("timeout"), 180000);
  try {
    const response = await fetch(`${api}/humanize/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        style: document.querySelector('[name="style"]:checked').value,
        model: $("model").value,
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(
        typeof payload.detail === "string"
          ? payload.detail
          : `درخواست پذیرفته نشد (${response.status}).`,
      );
    }
    await consumeEvents(response.body, (data) => {
      if (data === "__RETRIEVING__")
        $("progress").textContent = "در حال جست‌وجوی نمونه‌ها…";
      else if (data === "__GENERATING__") {
        $("progress").textContent = "مدل در حال بازنویسی است…";
        $("output").textContent = "";
      } else if (data.startsWith("__TOKEN__")) {
        result += JSON.parse(data.slice(9));
        $("output").textContent = result;
      } else if (data.startsWith("__TEXT__")) {
        result = JSON.parse(data.slice(8));
        $("output").textContent = result;
      } else if (data.startsWith("__ERROR__")) throw new Error(data.slice(9));
      // Rule-based evaluation is not a calibrated quality score.
    });
    if (!result.trim()) throw new Error("مدل متن قابل استفاده برنگرداند.");
    completed = true;
    $("progress").textContent = "تکمیل شد";
    $("result-note").textContent =
      "قبل از استفاده، معنا و جزئیات را مقایسه کن.";
  } catch (error) {
    const aborted = controller.signal.aborted;
    message(
      aborted
        ? controller.signal.reason === "timeout"
          ? "زمان انتظار تمام شد. اتصال مدل را بررسی کن."
          : "بازنویسی متوقف شد."
        : error instanceof TypeError
          ? "ارتباط با سرور ممکن نشد. سرور را بررسی کن."
          : error.message,
    );
    $("result-note").textContent = result
      ? "خروجی ناتمام؛ برای استفاده آماده نیست."
      : "نتیجه دریافت نشد";
    if (!result) $("output").textContent = "پس از بررسی اتصال، دوباره کوشش کن.";
    $("progress").textContent = "";
  } finally {
    clearTimeout(timeout);
    controller = null;
    busy(false);
  }
});
$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(result);
    $("result-note").textContent = "متن کپی شد.";
  } catch {
    message("کپی خودکار ممکن نشد؛ متن را انتخاب و کپی کن.");
  }
});
$("download").addEventListener("click", () => {
  const url = URL.createObjectURL(
    new Blob([result], { type: "text/plain;charset=utf-8" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "rewrite.txt";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
count();
health();
