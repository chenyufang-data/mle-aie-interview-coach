// Home page (the launcher). Two jobs:
// 1. Old deep links: the mock report's rubric tie-in linked /?practice=<id>
//    when the practice setup was the home page; it now lives on practice.html.
// 2. Mark Coding drills "In the local app" when this server does not run
//    code (the online demo); the button still opens coding.html, which then
//    explains the feature and how to run it locally.
(function () {
  const params = new URLSearchParams(window.location.search);
  if (params.get("practice")) {
    window.location.replace("/practice.html" + window.location.search);
    return;
  }
  fetch("/api/meta", { headers: Account.headers() })
    .then((response) => (response.ok ? response.json() : null))
    .then((meta) => {
      if (meta && meta.code && !meta.code.local) {
        document.getElementById("codingBadge").hidden = false;
      }
    })
    .catch(() => {});
})();
