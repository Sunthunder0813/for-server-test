import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from app_detect import detect

app = Flask(__name__)

def decode_image(data):
    # Accepts base64 string or file upload
    if isinstance(data, str):
        img_bytes = base64.b64decode(data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    elif hasattr(data, 'read'):
        img_bytes = data.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return None

@app.route('/detect', methods=['POST'])
def detect_endpoint():
    # Accepts either a file upload or a base64-encoded image
    if 'image' in request.files:
        img = decode_image(request.files['image'])
    else:
        data = request.get_json()
        img = decode_image(data.get('image', '')) if data else None

    if img is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    results = detect([img])
    if not results:
        return jsonify({'success': False, 'error': 'Detection failed'}), 500

    res = results[0]
    # Convert numpy arrays to lists for JSON serialization
    return jsonify({
        'success': True,
        'boxes': res.xyxy.tolist(),
        'confidences': res.conf.tolist(),
        'classes': res.cls.tolist()
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
