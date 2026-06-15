"""Shared Koala brand marks for the build scripts (webapp / 3D / native).

A vector koala that reads from 16px (favicon) to header size. Light koala on the app's
teal→blue logo gradient; the head + ears + big nose silhouette survives tiny rasterization.
"""

# koala head, transparent bg — sits inside the .logo gradient square
LOGO_SVG = (
    '<svg viewBox="0 0 48 48" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" '
    'style="display:block">'
    '<circle cx="13" cy="15" r="9" fill="#eef1f6"/>'
    '<circle cx="35" cy="15" r="9" fill="#eef1f6"/>'
    '<circle cx="13" cy="15" r="4.3" fill="#d98fb8"/>'
    '<circle cx="35" cy="15" r="4.3" fill="#d98fb8"/>'
    '<circle cx="24" cy="26" r="14" fill="#eef1f6"/>'
    '<circle cx="18.4" cy="23.6" r="2.2" fill="#2b3038"/>'
    '<circle cx="29.6" cy="23.6" r="2.2" fill="#2b3038"/>'
    '<ellipse cx="24" cy="29.6" rx="4.7" ry="5.7" fill="#2b3038"/>'
    '</svg>'
)

# self-contained favicon (koala on the brand gradient) as a <link rel="icon"> data URI
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#00ffc2"/><stop offset="1" stop-color="#2b8cff"/>'
    '</linearGradient></defs>'
    '<rect width="48" height="48" rx="11" fill="url(#g)"/>'
    '<circle cx="13" cy="15" r="8.5" fill="#f3f5f9"/>'
    '<circle cx="35" cy="15" r="8.5" fill="#f3f5f9"/>'
    '<circle cx="13" cy="15" r="4" fill="#d98fb8"/>'
    '<circle cx="35" cy="15" r="4" fill="#d98fb8"/>'
    '<circle cx="24" cy="26" r="13.5" fill="#f3f5f9"/>'
    '<circle cx="18.6" cy="23.8" r="2.1" fill="#2b3038"/>'
    '<circle cx="29.4" cy="23.8" r="2.1" fill="#2b3038"/>'
    '<ellipse cx="24" cy="29.4" rx="4.5" ry="5.4" fill="#2b3038"/>'
    '</svg>'
)


def favicon_link() -> str:
    import base64
    b64 = base64.b64encode(_FAVICON_SVG.encode("utf-8")).decode("ascii")
    return f'<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,{b64}">\n'
