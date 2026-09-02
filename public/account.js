// Login-lite account chip, shared by every page. The access key is set once
// here, lives in this browser's localStorage, and every API call reads it at
// request time via Account.headers(). There are no passwords or sessions:
// the key is a bearer token the server looks up in users.json, so this file
// is only storage plus a small corner widget. The server stays the single
// authority on whether a key is valid (/api/meta echoes name + tier).
const ACCESS_KEY_STORAGE = "interviewCoachAccessKey";

const Account = {
  key() {
    try {
      return (
        localStorage.getItem(ACCESS_KEY_STORAGE) ||
        sessionStorage.getItem(ACCESS_KEY_STORAGE) || // pre-chip sessions
        ""
      ).trim();
    } catch (error) {
      return "";
    }
  },
  setKey(value) {
    try {
      const key = (value || "").trim();
      if (key) localStorage.setItem(ACCESS_KEY_STORAGE, key);
      else localStorage.removeItem(ACCESS_KEY_STORAGE);
      sessionStorage.removeItem(ACCESS_KEY_STORAGE);
    } catch (error) {
      // Storage blocked (private window): the key just won't persist.
    }
  },
  headers() {
    const key = Account.key();
    return key ? { "X-Access-Key": key } : {};
  },
  // Pages can react to a key change (e.g. refresh tier-dependent UI).
  onchange: null,
};

(function initAccountChip() {
  const mount = document.querySelector("#accountChip");
  if (!mount) return;

  mount.innerHTML = `
    <button type="button" class="account-chip" id="accountChipBtn">Free tier</button>
    <div class="account-pop" id="accountPop" hidden>
      <label for="accountKeyInput">Access key</label>
      <input id="accountKeyInput" type="password" placeholder="Paste your access key"
             autocomplete="off">
      <div class="account-pop-row">
        <button type="button" class="secondary-action" id="accountSave">Save</button>
        <button type="button" class="ghost-action" id="accountClear">Sign out</button>
      </div>
      <p class="muted-note">Stored only in this browser. Without a key you are
        on the free tier.</p>
    </div>`;

  const btn = mount.querySelector("#accountChipBtn");
  const pop = mount.querySelector("#accountPop");
  const input = mount.querySelector("#accountKeyInput");

  btn.addEventListener("click", () => {
    pop.hidden = !pop.hidden;
    if (!pop.hidden) input.focus();
  });
  document.addEventListener("click", (event) => {
    if (!mount.contains(event.target)) pop.hidden = true;
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") save();
  });
  mount.querySelector("#accountSave").addEventListener("click", save);
  mount.querySelector("#accountClear").addEventListener("click", () => {
    Account.setKey("");
    input.value = "";
    pop.hidden = true;
    refresh();
  });

  function save() {
    Account.setKey(input.value);
    input.value = ""; // never leave the key sitting in a visible-on-inspect field
    pop.hidden = true;
    refresh();
  }

  async function refresh() {
    let label = "Free tier";
    let title = "Instant local ML grading. Click to enter a paid access key.";
    try {
      const response = await fetch("/api/meta", { headers: Account.headers() });
      const meta = await response.json();
      const user = meta.user || {};
      if (!user.tiers_enabled) {
        // No users.json on this server: keys mean nothing, hide the chip.
        mount.hidden = true;
        if (Account.onchange) Account.onchange(meta);
        return;
      }
      mount.hidden = false;
      if (Account.key()) {
        if (user.tier === "paid") {
          label = `${user.name} · paid`;
          title = `${user.paid_left_today} of ${user.paid_quota} "Always Claude" gradings left today`
            + (user.paid_grader && user.paid_grader !== "claude"
              ? `; everyday grading by ${user.paid_grader}` : "");
        } else {
          label = "key not recognized";
          title = "The server did not recognize this key. Click to re-enter or sign out.";
        }
      }
      if (Account.onchange) Account.onchange(meta);
    } catch (error) {
      // Server unreachable: leave the default label.
    }
    btn.textContent = label;
    btn.title = title;
    btn.classList.toggle("account-chip-paid", label.endsWith("· paid"));
  }

  refresh();
})();
