function copyBibtex() {
  const el = document.getElementById("bibtex");
  const btn = document.getElementById("copybtn");
  navigator.clipboard.writeText(el.innerText).then(() => {
    const old = btn.innerText;
    btn.innerText = "Copied";
    setTimeout(() => { btn.innerText = old; }, 1200);
  }).catch(() => {
    alert("Copy failed. Please copy manually.");
  });
}

function highlightJsonBlocks() {
  document.querySelectorAll("pre.language-json").forEach((pre) => {
    const escaped = pre.textContent
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    pre.innerHTML = escaped.replace(
      /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|-?\b\d+(?:\.\d+)?\b/g,
      (match, str, colon, literal) => {
        if (colon) return `<span class="json-key">${str}</span>${colon}`;
        if (str) return `<span class="json-string">${str}</span>`;
        if (literal) return `<span class="json-literal">${literal}</span>`;
        return `<span class="json-number">${match}</span>`;
      }
    );
  });
}

document.addEventListener("DOMContentLoaded", highlightJsonBlocks);
