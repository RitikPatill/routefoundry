---
title: RouteFoundry Offline Demo
emoji: 🏭
colorFrom: indigo
colorTo: orange
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
---

# RouteFoundry offline demo

This Space previews how a quality-loss budget and optimization objective affect an
explainable model-routing policy.

Every displayed observation is **synthetic, illustrative, and non-evidence**. The
fixture values are hand-designed to exercise the software; they are not measurements or
claims about real models. The app imports RouteFoundry's core `demo` and `report`
modules. It does not download a model, call an inference API, or require credentials.
Gradio serves the user interface over HTTP, but the demo makes no outbound model or
evaluation request.

For a standalone Space repository, install the released `routefoundry` package. When
run from the RouteFoundry source tree, `app.py` imports the package from `src/`.
