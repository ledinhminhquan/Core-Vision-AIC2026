"""Team-facing fast web UI: the retrieval service + shared-account auth +
direct-to-pack export + a zero-framework SPA (round-36).

Streamlit reruns the whole script per click; competition operators found that
laggy. This layer serves a static single-page app from the SAME process as
the JSON service — every click is local JS, only search/export touch the
network — and gates everything behind ONE shared team login so the public
tunnel URL alone grants nothing.
"""
