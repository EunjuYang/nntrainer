# Android Build Patch: Making OpenCL Optional

## Problem
The current build system requires OpenCL to be enabled when building for Android platform (`-Dplatform=android -Denable-opencl=true`). This patch makes OpenCL optional, allowing builds with `-Dplatform=android -Denable-opencl=false`.

## Solution Overview
The patch modifies three key files to conditionally include OpenCL dependencies based on the `enable-opencl` meson option:

1. **jni/Android.mk.in** - Android NDK build template
2. **jni/meson.build** - JNI meson configuration
3. **meson.build** - Main meson build configuration

## Changes Made

### 1. jni/Android.mk.in
- Wrapped OpenCL and CLBlast library declarations with conditional markers (`@MESON_OPENCL_BLOCK_START@` / `@MESON_OPENCL_BLOCK_END@`)
- Made OpenCL shared library dependency conditional using `@MESON_OPENCL_SHARED_LIBS@`
- Made CLBlast static library dependency conditional using `@MESON_OPENCL_STATIC_LIBS@`

### 2. jni/meson.build
Added conditional logic based on `get_option('enable-opencl')`:

**When OpenCL is enabled:**
- Sets proper paths for CLBLAST_ROOT and CL_ROOT
- Sets block markers to empty strings (includes the code)
- Configures OpenCL shared and static library dependencies

**When OpenCL is disabled:**
- Sets block markers to comment strings (comments out the code)
- Sets library dependencies to empty or commented values
- Sets dummy paths to avoid undefined variable errors

### 3. meson.build
- Added else block to handle Android platform when OpenCL is disabled
- Sets clblast_dep to dummy_dep for Android when OpenCL is disabled
- Initializes empty strings for clblast_root and opencl_root to avoid undefined variables

## Usage

### Building with OpenCL (existing behavior):
```bash
meson build -Dplatform=android -Denable-opencl=true
```

### Building without OpenCL (new capability):
```bash
meson build -Dplatform=android -Denable-opencl=false
```

## Files Modified
- `jni/Android.mk.in` - Added conditional compilation blocks
- `jni/meson.build` - Added OpenCL option handling logic
- `meson.build` - Added fallback for disabled OpenCL on Android

## Testing
After applying this patch, the build system should:
1. Successfully configure with `-Dplatform=android -Denable-opencl=false`
2. Generate Android.mk without OpenCL dependencies when OpenCL is disabled
3. Build Android libraries without requiring OpenCL libraries

## Patch Application
To apply the patch:
```bash
git apply android_opencl_optional.patch
```

Or manually apply the changes shown in the patch file.