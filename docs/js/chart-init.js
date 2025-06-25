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
    // create the chart
    new Chart(canvas.getContext('2d'), config);
  });
});
