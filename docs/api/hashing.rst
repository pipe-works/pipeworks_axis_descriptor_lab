Hashing
=======

Normalisation and hash utilities for the Interpretive Provenance Chain (IPC).

See :doc:`/guides/ipc-and-hashing` for a comprehensive explanation of the IPC
framework and the design philosophy behind each hash function.

The lab now consumes hashing helpers from the external ``pipeworks_ipc``
package rather than a local ``app.hashing`` module. The runtime entry points
used throughout the app are:

- ``pipeworks_ipc.compute_input_hash``
- ``pipeworks_ipc.compute_system_prompt_hash``
- ``pipeworks_ipc.compute_output_hash``
- ``pipeworks_ipc.compute_ipc_id``
- ``pipeworks_ipc.payload_hash``

See the narrative guide for the canonical hashing rules and provenance model.
