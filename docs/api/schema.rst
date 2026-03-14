Schema
======

Pydantic v2 models for request/response objects.

`app.schema` is now a package that re-exports the public model names for
backward-compatible imports such as `from app.schema import GenerateRequest`.
The models themselves are organized by domain submodule:

- ``app.schema.axis`` — shared axis primitives and payloads
- ``app.schema.generate`` — character-description generation and logging
- ``app.schema.save`` — save/export/import request and response models
- ``app.schema.analysis`` — signal-isolation and transformation-map models
- ``app.schema.chat`` — chat translation and chat save/import models
- ``app.schema.mud`` — mud-server proxy request and response models

.. automodule:: app.schema
   :members:
   :undoc-members:
   :show-inheritance:
