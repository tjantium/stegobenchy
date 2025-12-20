# Visualization

StegoBenchy provides interactive visualizations for analyzing experiment results.

## Experiment Dashboard

Create a comprehensive dashboard:

```python
from src.viz import create_experiment_dashboard

# After running an experiment
fig = create_experiment_dashboard(results)
fig.show()  # Display in browser

# Or save to file
fig.write_html('dashboard.html')
```

## Individual Plots

### Metrics Over Time

```python
from src.viz import plot_metrics_over_time

fig = plot_metrics_over_time(results)
fig.show()
```

### Stego Success Rate

```python
from src.viz import plot_stego_success_rate

fig = plot_stego_success_rate(results)
fig.show()
```

### Entropy Distribution

```python
from src.viz import plot_entropy_distribution

fig = plot_entropy_distribution(results)
fig.show()
```

## Interactive Demo

Use the Streamlit demo for interactive visualizations:

```bash
streamlit run demo/app.py
```

The demo includes:
- Real-time metric visualization
- Interactive filtering
- Export capabilities
- Comparison views

## Custom Visualizations

Create custom plots with Plotly:

```python
import plotly.graph_objects as go

# Example: Custom scatter plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[r['metrics']['entropy'] for r in results],
    y=[r['metrics']['reasoning_accuracy'] for r in results],
    mode='markers'
))
fig.show()
```

## Export Options

Export visualizations in various formats:

```python
# HTML
fig.write_html('plot.html')

# PNG
fig.write_image('plot.png')

# PDF
fig.write_image('plot.pdf')
```

