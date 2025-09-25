# ImageCollage Coordinate System Standard

## 🎯 **MANDATORY CONVENTIONS**

### **1. Configuration Level**
```python
# ALWAYS: grid_size represents (width, height)
grid_size: Tuple[int, int] = (width, height)  # Width first, height second
```

### **2. Variable Extraction**
```python
# ✅ CORRECT - Use validation utilities
from ..utils.coordinate_validation import validate_grid_coordinates

width, height = validate_grid_coordinates(grid_size, "component_name")

# ❌ WRONG - Never extract directly without validation
grid_width, grid_height = grid_size  # FORBIDDEN - causes confusion
```

### **3. Array Operations**
```python
# ✅ CORRECT - NumPy arrays use (height, width) shape
individual = np.random.randint(0, n, size=(height, width))
target_tiles = np.zeros((height, width, tile_h, tile_w, 3))

# ✅ CORRECT - Array access uses (row, col) = (i, j) indexing
for i in range(height):      # i iterates through rows (height dimension)
    for j in range(width):   # j iterates through columns (width dimension)
        value = array[i, j]  # array[row, col]
```

### **4. Coordinate Variable Naming**
```python
# ✅ PREFERRED - Clear dimension names
height, width = validate_grid_coordinates(grid_size, context)
rows, cols = height, width

# ✅ ACCEPTABLE - When using grid_* prefix
grid_height, grid_width = height, width  # Only after validation

# ❌ FORBIDDEN - Direct extraction
grid_width, grid_height = grid_size  # Never do this
```

### **5. Array Shape Validation**
```python
# ✅ MANDATORY - Validate before any array operations
from ..utils.coordinate_validation import validate_individual_shape, ensure_coordinate_consistency

validate_individual_shape(individual, grid_size, context)
ensure_coordinate_consistency(grid_size, individual, target_array, context)
```

## 🔄 **Coordinate System Mappings**

### **Configuration → Array Shape**
```python
config_grid_size = (width, height)     # Configuration: (30, 40)
array_shape = (height, width)          # NumPy arrays: (40, 30)
```

### **Loop Iteration**
```python
# Standard nested loop pattern
for i in range(height):     # i = row index (0 to height-1)
    for j in range(width):  # j = column index (0 to width-1)
        element = array[i, j]  # array[row, col]
```

### **Coordinate Indexing**
```python
# Array indexing follows (row, col) convention
row = i      # First index = vertical position
col = j      # Second index = horizontal position
value = array[row, col] = array[i, j]
```

## 🚨 **FORBIDDEN PATTERNS**

### **❌ Direct Grid Size Extraction**
```python
# NEVER DO THIS - causes systematic confusion
grid_width, grid_height = self.grid_size
width, height = self.grid_size
w, h = grid_size
```

### **❌ Mixed Coordinate Systems**
```python
# NEVER MIX - choose one approach per component
grid_height, grid_width = individual.shape  # Gets from array
grid_width, grid_height = self.grid_size    # Gets from config
```

### **❌ Inconsistent Variable Usage**
```python
# NEVER USE - variables named opposite to their content
grid_width = height   # grid_width contains height value
grid_height = width   # grid_height contains width value
```

## ✅ **MANDATORY USAGE PATTERNS**

### **Component Initialization**
```python
def __init__(self, config):
    # Always use validation utilities
    self.width, self.height = validate_grid_coordinates(
        config.grid_size, self.__class__.__name__
    )

    # Log coordinate interpretation for debugging
    log_coordinate_interpretation(config.grid_size, self.__class__.__name__)
```

### **Array Creation**
```python
# Always use validated dimensions
width, height = validate_grid_coordinates(grid_size, context)
array = np.zeros((height, width, ...))  # NumPy convention: (rows, cols, ...)

# Validate result
validate_individual_shape(array, grid_size, context)
```

### **Cross-Component Operations**
```python
# Always validate compatibility
ensure_coordinate_consistency(
    config_grid_size, individual_array, target_array, component_name
)
```

## 🔧 **REQUIRED IMPORTS**

```python
# Every component handling coordinates must import:
from ..utils.coordinate_validation import (
    validate_grid_coordinates,
    validate_individual_shape,
    validate_array_compatibility,
    ensure_coordinate_consistency,
    log_coordinate_interpretation
)
```

## 📚 **CONVERSION UTILITIES**

```python
# Config to array shape
from ..utils.coordinate_validation import convert_config_to_array_shape
grid_size = (width, height)              # Config format
array_shape = convert_config_to_array_shape(grid_size)  # Returns (height, width)

# Array shape to config
from ..utils.coordinate_validation import convert_array_shape_to_config
array_shape = (height, width)           # Array format
grid_size = convert_array_shape_to_config(array_shape)  # Returns (width, height)
```

## 🎯 **ENFORCEMENT**

### **Code Review Checklist**
- [ ] No direct `grid_size` extraction without validation
- [ ] All coordinate extractions use `validate_grid_coordinates()`
- [ ] All array operations use validation functions
- [ ] All components log coordinate interpretation
- [ ] Cross-component operations use `ensure_coordinate_consistency()`

### **Automated Checks**
- Search for: `grid_width.*grid_height.*=.*grid_size` (FORBIDDEN)
- Search for: `width.*height.*=.*grid_size` without validation (FORBIDDEN)
- Verify: All coordinate extractions use validation utilities (REQUIRED)

---

**This standard MUST be followed by ALL components to prevent the recurring coordinate system errors.**