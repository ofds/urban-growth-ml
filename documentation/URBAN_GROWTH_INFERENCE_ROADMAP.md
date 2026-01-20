---

## 🗺️ **Real-World System Paradigm Mapping**

Based on comprehensive investigation of established urban growth systems:

### **Most Common Paradigms in Practice:**
1. **Statistical Modeling** (6/8 systems) - UrbanSim, MOLAND, WhatIf?, LEAM, GeoMod
2. **Cellular Automata** (3/8 systems) - SLEUTH, MOLAND, LEAM  
3. **Machine Learning** (3/8 systems) - MIT Lab, LEAM, emerging in UrbanSim
4. **Optimization-Based** (3/8 systems) - CityEngine, WhatIf?, GeoMod

### **Key Real-World Examples:**

| System | Primary Paradigm | State Representation | Inverse Capabilities | Scale |
|--------|------------------|---------------------|---------------------|-------|
| **SLEUTH** | Cellular Automata | Raster Grid (30-100m) | Parameter Estimation | Regional (1000+ km²) |
| **UrbanSim** | Statistical Modeling | Relational DB (agents) | Model Calibration | Metropolitan (millions) |
| **CityEngine** | Procedural + Optimization | Geometric Primitives | Limited Pattern Fitting | City-scale |
| **MIT Lab** | Morphological + ML | Shape Elements | Pattern Decomposition | Research-scale |
| **MOLAND** | Statistical + CA Hybrid | Hybrid Raster/Vector | Historical Calibration | Regional |

### **Critical Insights:**
- **Hybrid approaches dominate** - Pure paradigms are rare in production systems
- **Statistical modeling is most common** - Foundation for UrbanSim, MOLAND, WhatIf?
- **Inverse capabilities vary** - Strong in SLEUTH/UrbanSim, moderate in CityEngine/GeoMod
- **No system uses our exact approach** - All differ from "backward peeling + heuristics"

### **Strategic Implications:**
- **Augment, don't replace** - Add statistical/ML elements to current search-based system
- **Statistical modeling first** - Most proven paradigm for urban inference
- **Hybrid combinations** - Follow MOLAND's statistical + CA approach

---

*Document Version: 1.2*
*Last Updated: January 18, 2026*
