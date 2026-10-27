// pymdownx.arithmatex(generic) 가 남긴 \( \) · \[ \] 를 MathJax 가 렌더한다.
// Material 의 instant navigation 에서도 페이지를 바꿀 때마다 다시 그린다.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
