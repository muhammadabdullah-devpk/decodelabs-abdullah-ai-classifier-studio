/**
 * DecodeLabs High-Performance Canvas Plotting Engine (chart.js)
 * Draws scatter plots, decision boundary grids, and ROC curves.
 */

class VisualizationEngine {
  /**
   * Colors mapping representing our design system classes.
   * Class 0: Cyber Pink
   * Class 1: Electric Teal
   */
  static get Colors() {
    return {
      class0: { main: '#ff007f', light: 'rgba(255, 0, 127, 0.12)', border: '#ff007f' },
      class1: { main: '#00f0ff', light: 'rgba(0, 240, 255, 0.12)', border: '#00f0ff' },
      mesh0: 'rgba(255, 0, 127, 0.08)',
      mesh1: 'rgba(0, 240, 255, 0.08)',
      grid: 'rgba(255, 255, 255, 0.05)',
      text: '#9ca3af',
      axis: '#374151'
    };
  }

  /**
   * Renders the 2D scatter plot and decision boundary.
   * @param {HTMLCanvasElement} canvas 
   * @param {Array<Array<number>>} X Features (must have at least 2 columns selected)
   * @param {Array<number>} y Labels
   * @param {Object} model Trained classifier instance (optional)
   * @param {Object} options Configuration flags
   */
  static drawScatterAndBoundary(canvas, X, y, model = null, options = {}) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#080711';
    ctx.fillRect(0, 0, width, height);

    if (!X || X.length === 0) {
      // Draw placeholder
      ctx.fillStyle = '#6b7280';
      ctx.font = '16px Outfit, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No data loaded to visualize', width / 2, height / 2);
      return;
    }

    const padding = 50;
    const graphWidth = width - 2 * padding;
    const graphHeight = height - 2 * padding;

    // Extract columns
    const xVals = X.map(row => row[0]);
    const yVals = X.map(row => row[1]);

    const xMin = Math.min(...xVals);
    const xMax = Math.max(...xVals);
    const yMin = Math.min(...yVals);
    const yMax = Math.max(...yVals);

    // Padding values to expand bounds slightly
    const xRange = (xMax - xMin) || 1;
    const yRange = (yMax - yMin) || 1;
    const padX = xRange * 0.1;
    const padY = yRange * 0.1;

    const xLimitMin = xMin - padX;
    const xLimitMax = xMax + padX;
    const yLimitMin = yMin - padY;
    const yLimitMax = yMax + padY;

    // Helper functions to map data values to canvas coordinates
    const mapX = (val) => padding + ((val - xLimitMin) / (xLimitMax - xLimitMin)) * graphWidth;
    const mapY = (val) => height - padding - ((val - yLimitMin) / (yLimitMax - yLimitMin)) * graphHeight;

    // Helper to map canvas coordinates back to data values (for decision boundary)
    const mapCanvasToData = (cx, cy) => {
      const xVal = xLimitMin + ((cx - padding) / graphWidth) * (xLimitMax - xLimitMin);
      const yVal = yLimitMin + ((height - padding - cy) / graphHeight) * (yLimitMax - yLimitMin);
      return [xVal, yVal];
    };

    // 1. Draw Decision Boundary Mesh (if model is provided)
    if (model) {
      const resolution = 60; // 60x60 grid points
      const stepX = graphWidth / resolution;
      const stepY = graphHeight / resolution;

      for (let px = 0; px < resolution; px++) {
        for (let py = 0; py < resolution; py++) {
          const cx = padding + px * stepX + stepX / 2;
          const cy = padding + py * stepY + stepY / 2;
          const dataPoint = mapCanvasToData(cx, cy);

          // Get prediction
          const prediction = model.predict([dataPoint])[0];

          ctx.fillStyle = prediction === 1 
            ? this.Colors.mesh1 
            : this.Colors.mesh0;
          
          ctx.fillRect(
            padding + px * stepX, 
            padding + py * stepY, 
            stepX + 0.5, 
            stepY + 0.5
          );
        }
      }
    }

    // 2. Draw Grid Lines
    ctx.strokeStyle = this.Colors.grid;
    ctx.lineWidth = 1;
    const gridCount = 5;

    for (let i = 0; i <= gridCount; i++) {
      // Vertical grids
      const gx = padding + (i / gridCount) * graphWidth;
      ctx.beginPath();
      ctx.moveTo(gx, padding);
      ctx.lineTo(gx, height - padding);
      ctx.stroke();

      // Horizontal grids
      const gy = padding + (i / gridCount) * graphHeight;
      ctx.beginPath();
      ctx.moveTo(padding, gy);
      ctx.lineTo(width - padding, gy);
      ctx.stroke();
    }

    // 3. Draw Axes Lines
    ctx.strokeStyle = this.Colors.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    // X axis
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    // Y axis
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();

    // Axes tick marks & labels
    ctx.fillStyle = this.Colors.text;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    // X Axis ticks
    for (let i = 0; i <= gridCount; i++) {
      const val = xLimitMin + (i / gridCount) * (xLimitMax - xLimitMin);
      const cx = padding + (i / gridCount) * graphWidth;
      ctx.fillText(val.toFixed(2), cx, height - padding + 8);
    }

    // Y Axis ticks
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= gridCount; i++) {
      const val = yLimitMin + (i / gridCount) * (yLimitMax - yLimitMin);
      const cy = height - padding - (i / gridCount) * graphHeight;
      ctx.fillText(val.toFixed(2), padding - 8, cy);
    }

    // Title label axis names
    ctx.font = '12px Outfit, sans-serif';
    ctx.fillStyle = varColor('--text-main', '#f3f4f6');
    ctx.textAlign = 'center';
    ctx.fillText(options.xLabel || 'Feature X', width / 2, height - 15);

    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(options.yLabel || 'Feature Y', 0, 0);
    ctx.restore();

    // 4. Plot Data Points
    X.forEach((row, idx) => {
      const cx = mapX(row[0]);
      const cy = mapY(row[1]);
      const label = y[idx];

      const isTestSample = options.testIndices && options.testIndices.includes(idx);

      ctx.beginPath();
      ctx.arc(cx, cy, isTestSample ? 6 : 5, 0, 2 * Math.PI);
      
      const pointColors = label === 1 ? this.Colors.class1 : this.Colors.class0;
      ctx.fillStyle = pointColors.main;
      ctx.strokeStyle = isTestSample ? '#ffffff' : 'rgba(0,0,0,0.5)';
      ctx.lineWidth = isTestSample ? 2 : 1;
      
      ctx.fill();
      ctx.stroke();

      // If it is a test sample, outline it with a white ring
      if (isTestSample) {
        ctx.beginPath();
        ctx.arc(cx, cy, 9, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });
  }

  /**
   * Renders the ROC Curve and AUC text on a canvas.
   * @param {HTMLCanvasElement} canvas 
   * @param {Array<Object>} points Array of { fpr, tpr }
   * @param {number} auc Area Under Curve score
   */
  static drawRocCurve(canvas, points, auc) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#080711';
    ctx.fillRect(0, 0, width, height);

    const padding = 45;
    const graphWidth = width - 2 * padding;
    const graphHeight = height - 2 * padding;

    // Helper maps
    const mapX = (fpr) => padding + fpr * graphWidth;
    const mapY = (tpr) => height - padding - tpr * graphHeight;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    const step = 5;
    for (let i = 0; i <= step; i++) {
      const ratio = i / step;
      const cx = padding + ratio * graphWidth;
      const cy = padding + ratio * graphHeight;
      
      ctx.beginPath(); ctx.moveTo(cx, padding); ctx.lineTo(cx, height - padding); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(padding, cy); ctx.lineTo(width - padding, cy); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = this.Colors.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();

    // Diagonal Random Guess Line
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(mapX(0), mapY(0));
    ctx.lineTo(mapX(1), mapY(1));
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash

    // Plot ROC Curve Line
    if (points && points.length > 0) {
      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(mapX(points[0].fpr), mapY(points[0].tpr));
      
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(mapX(points[i].fpr), mapY(points[i].tpr));
      }
      ctx.stroke();
      
      // Shadow glow effect
      ctx.shadowBlur = 10;
      ctx.shadowColor = 'rgba(0, 240, 255, 0.5)';
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
      ctx.lineWidth = 6;
      ctx.stroke();
      ctx.shadowBlur = 0; // Reset glow
    }

    // Axis labels
    ctx.fillStyle = this.Colors.text;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    for (let i = 0; i <= step; i++) {
      const val = i / step;
      ctx.fillText(val.toFixed(1), padding + val * graphWidth, height - padding + 8);
    }

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= step; i++) {
      const val = i / step;
      ctx.fillText(val.toFixed(1), padding - 8, height - padding - val * graphHeight);
    }

    ctx.fillStyle = '#f3f4f6';
    ctx.font = '11px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('False Positive Rate (FPR)', width / 2, height - 15);

    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('True Positive Rate (TPR)', 0, 0);
    ctx.restore();

    // Renders AUC text box overlay
    if (auc !== undefined) {
      ctx.fillStyle = 'rgba(15, 13, 34, 0.85)';
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
      ctx.lineWidth = 1;
      ctx.fillRect(width - padding - 140, padding + 15, 130, 40);
      ctx.strokeRect(width - padding - 140, padding + 15, 130, 40);

      ctx.fillStyle = '#00f0ff';
      ctx.font = 'bold 12px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`AUC = ${auc.toFixed(4)}`, width - padding - 75, padding + 35);
    }
  }
}

// Quick helper to read variables from style sheet
function varColor(variableName, fallback) {
  try {
    return getComputedStyle(document.documentElement).getPropertyValue(variableName).trim() || fallback;
  } catch (e) {
    return fallback;
  }
}
