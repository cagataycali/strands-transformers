/* Glass micro-interactions - scroll reveal + pointer-reactive sheen.
   Works with Material's navigation.instant (re-init on each page load). */
(function () {
  function reveal() {
    var els = document.querySelectorAll(
      ".md-typeset .grid.cards > ul > li, .md-typeset .grid.cards > ol > li, " +
      ".md-typeset > table, .md-typeset .result, .md-typeset .admonition, " +
      ".md-typeset details, .md-typeset h2"
    );
    if (!("IntersectionObserver" in window)) {
      els.forEach(function (e) { e.classList.add("st-in"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          en.target.classList.add("st-in");
          io.unobserve(en.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });
    els.forEach(function (e) { e.classList.add("st-reveal"); io.observe(e); });
  }

  // Pointer-reactive sheen on glass cards (CSS reads --mx/--my)
  function sheen() {
    document.querySelectorAll(".md-typeset .grid.cards > ul > li," +
      ".md-typeset .grid.cards > ol > li").forEach(function (card) {
      card.addEventListener("pointermove", function (ev) {
        var r = card.getBoundingClientRect();
        card.style.setProperty("--mx", ((ev.clientX - r.left) / r.width * 100) + "%");
        card.style.setProperty("--my", ((ev.clientY - r.top) / r.height * 100) + "%");
      });
    });
  }

  function init() { reveal(); sheen(); }

  if (window.document$) {        // Material instant-nav: re-run per page
    window.document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
