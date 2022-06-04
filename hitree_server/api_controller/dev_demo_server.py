import io
from typing import Optional, List, Dict

import flask
import numpy as np
from PIL import Image
from flask import Flask, request, make_response, send_file, jsonify
from flask_cors import CORS
from hitree.api.ContactMatrixFacet import ContactMatrixFacet
from hitree.core.chunked_file import ChunkedFile
from hitree.core.common import LengthUnit
from hitree.core.contig_tree import ContigTree
from matplotlib import pyplot as plt
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
CORS(app)

chunked_file: Optional[ChunkedFile] = None

colormap = plt.get_cmap('Greens')

transport_dtype: str = 'uint8'

filename: Optional[str] = None
fasta_filename: Optional[str] = None


def get_contig_info(f: ChunkedFile):
    contig_names: Dict[int, str] = {}
    contig_name_to_id: Dict[str, int] = {}

    for contig_id, contig_name in enumerate(f.contig_names):
        contig_names[int(contig_id)] = str(contig_name)
        contig_name_to_id[str(contig_name)] = int(contig_id)

    contig_size: Dict[int, List[float]] = {}
    contig_direction: List[int] = []
    contig_ord_ids: List[int] = []

    def visit_node(n: ContigTree.Node):
        nonlocal contig_size
        for res, length in n.true_contig_descriptor().contig_length_at_resolution.items():
            if res not in contig_size.keys():
                contig_size[int(res)] = []
            contig_size[int(res)].append(float(length))
        contig_direction.append(int(n.true_direction().value))
        contig_ord_ids.append(int(n.contig_descriptor.contig_id))

    f.contig_tree.traverse(visit_node)
    return {
        'contig_size': contig_size,
        'contig_direction': contig_direction,
        'contig_ord_ids': contig_ord_ids,
        'contig_names': contig_names,
        'contig_name_to_id': contig_name_to_id,
    }


@app.get("/open")
def open_file():
    global chunked_file, filename, fasta_filename
    filename = request.args.get("filename")
    fasta_filename = request.args.get("fasta_filename")
    if filename is None or filename == "":
        return "Wrong filename specified", 404
    chunked_file = ContactMatrixFacet.get_file_descriptor(f"./data/{filename}")
    if fasta_filename is not None and fasta_filename != "":
        chunked_file.link_fasta(f"./data/{fasta_filename}")
    ContactMatrixFacet.open_file(chunked_file)

    resp = generate_contig_info()

    response = flask.jsonify(resp)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


def generate_contig_info():
    global chunked_file
    tile_size: int = int(max(chunked_file.dense_submatrix_size.values()))

    resolutions: List[int] = [int(r) for r in sorted(chunked_file.resolutions, reverse=True)]

    resp: dict = {
        'status': f"OK, File opened",
        'dtype': str(transport_dtype),
        'resolutions': resolutions,
        'pixel_resolutions': [float(r) / float(min(chunked_file.resolutions)) for r in resolutions],
        'tile_size': tile_size,
        'width': (int(chunked_file.contig_tree.root.subtree_length[
                          chunked_file.resolutions[0]]) if chunked_file.contig_tree.root is not None else 0),
        'height': (int(chunked_file.contig_tree.root.subtree_length[
                           chunked_file.resolutions[0]]) if chunked_file.contig_tree.root is not None else 0),
        'sizes': [int(chunked_file.contig_tree.root.subtree_length[r]) for r in
                  resolutions] if chunked_file.contig_tree.root is not None else [],
        'levels': len(chunked_file.resolutions),
        'contig_info': get_contig_info(chunked_file)
    }
    return resp


@app.get("/close")
def close_file():
    global chunked_file
    if chunked_file is not None:
        ContactMatrixFacet.close_file(chunked_file)

    response = make_response(f"File closed")
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.post("/reverse")
def reverse():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    req = request.get_json()

    chunked_file.reverse_contig_by_id(int(req['contigToRotate']))
    resp: dict = {'contig_info': get_contig_info(chunked_file)}

    response = make_response(jsonify(resp))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.post("/move")
def move():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    req = request.get_json()

    contig_id: int = int(req['contigToMove'])
    target_index: int = int(req['targetIndex'])

    chunked_file.move_contig_by_id(contig_id, target_index)
    resp: dict = {'contig_info': get_contig_info(chunked_file)}

    response = make_response(jsonify(resp))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.post("/save")
def save():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    chunked_file.save()

    resp: dict = {'contig_info': get_contig_info(chunked_file), 'result': "OK, file saved"}

    response = make_response(jsonify(resp))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.post("/get_fasta_for_assembly")
def get_fasta_for_assembly():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    buf = io.BytesIO()
    chunked_file.get_fasta_for_assembly(buf)
    buf.seek(0)

    response = make_response(send_file(buf, mimetype="text/plain"))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.get("/get_resolutions")
def get_resolutions():
    return flask.jsonify(chunked_file.resolutions)


@app.get("/get_tile")
def get_tile():
    if chunked_file is None:
        return "File is not opened yet", 400

    level: int = int(request.args.get("level"))
    row: int = int(request.args.get("row"))
    col: int = int(request.args.get("col"))
    tile_size: int = int(request.args.get("tile_size"))

    resolution: int = sorted(chunked_file.resolutions)[-level]
    x0: int = row * tile_size
    x1: int = (1 + row) * tile_size - 1
    y0: int = col * tile_size
    y1: int = (1 + col) * tile_size - 1

    dense_rect = ContactMatrixFacet.get_dense_submatrix(
        chunked_file,
        resolution,
        x0,
        y0,
        x1,
        y1,
        LengthUnit.PIXELS
    )

    padded_dense_rect: np.ndarray = np.zeros((tile_size, tile_size), dtype=dense_rect.dtype)
    padded_dense_rect[0:dense_rect.shape[0], 0:dense_rect.shape[1]] = dense_rect
    dense_rect: np.ndarray = np.log10(1 + padded_dense_rect)
    # dense_rect: np.ndarray = np.log10(1+dense_rect)

    colored_image: np.ndarray = colormap(dense_rect)
    image_matrix: np.ndarray = colored_image[:, :, :3] * 255
    image_matrix = image_matrix.astype(transport_dtype)

    image: Image = Image.fromarray(image_matrix)

    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)

    response = make_response(send_file(buf, mimetype="image/png"))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    response = make_response(f"Error: {e}")
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 500
    return response


if __name__ == '__main__':
    app.run(debug=True)
