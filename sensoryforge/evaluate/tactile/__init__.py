"""The tactile modality's evaluation pieces (report contract, sections 7 and 8.3).

``metrics`` is the tactile metric pack: it reads frames only and sits inside the
audit's import wall. The tactile declared adapter (A3) lives in its own module,
``declared``, outside the wall; this package therefore imports neither, so loading the
package pulls in nothing from the world engine.
"""
