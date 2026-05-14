// subscribe to every "new page" event
document$.subscribe(() => {
  // locate all un-initialized Plotly containers
  document.querySelectorAll('[data-plotly-config]').forEach(el => {
    // only init once
    if (el.getAttribute('data-plotly-initialized')) return;
    el.setAttribute('data-plotly-initialized', 'true');

    // grab JSON config from the element
    const config = JSON.parse(el.getAttribute('data-plotly-config'));

    Plotly.newPlot(
      el.id,
      config.data,
      config.layout || {},
      config.options || {}
    );
  });
});


// subscribe to every new-page event
document$.subscribe(() => {
  // find any <canvas> with a JSON config
  document.querySelectorAll('canvas[data-chart-config]').forEach(canvas => {
    // don’t double-init
    if (canvas.chartjsInitialized) return;
    canvas.chartjsInitialized = true;

    // parse the Chart.js config
    const config = JSON.parse(canvas.getAttribute('data-chart-config'));
    // create the chart, then theme it
    const chart = new Chart(canvas.getContext('2d'), config);
    canvas.__chart = chart;
    applyChartTheme(chart);
  });
});

// Read current Material color scheme and return palette-aware chart colors.
function getChartThemeColors() {
  const scheme = document.body.getAttribute('data-md-color-scheme');
  const dark = scheme === 'slate';
  return {
    text:  dark ? '#e6edf3'              : '#0b1220',
    muted: dark ? 'rgba(230,237,243,0.7)' : 'rgba(11,18,32,0.65)',
    grid:  dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)',
  };
}

// Apply theme colors to a Chart.js instance and refresh it without animation.
function applyChartTheme(chart) {
  if (!chart || !chart.options) return;
  const c = getChartThemeColors();

  if (!chart.options.plugins) chart.options.plugins = {};
  if (!chart.options.plugins.legend) chart.options.plugins.legend = {};
  if (!chart.options.plugins.legend.labels) chart.options.plugins.legend.labels = {};
  chart.options.plugins.legend.labels.color = c.text;

  if (chart.options.plugins.title) chart.options.plugins.title.color = c.text;

  if (chart.options.scales) {
    Object.values(chart.options.scales).forEach(scale => {
      if (!scale) return;
      if (!scale.ticks) scale.ticks = {};
      scale.ticks.color = c.muted;
      if (scale.title) scale.title.color = c.text;
      if (!scale.grid) scale.grid = {};
      // Preserve `display: false` etc., only override color.
      scale.grid.color = c.grid;
      scale.grid.tickColor = c.grid;
    });
  }
  chart.update('none');
}

// Re-theme every chart whenever Material's palette attribute flips.
new MutationObserver(() => {
  document.querySelectorAll('canvas[data-chart-config]').forEach(canvas => {
    if (canvas.__chart) applyChartTheme(canvas.__chart);
  });
}).observe(document.body, { attributes: true, attributeFilter: ['data-md-color-scheme'] });

