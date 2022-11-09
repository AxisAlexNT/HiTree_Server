from argparse import ArgumentParser, Namespace
import argparse
from cmath import isclose
from collections import namedtuple
from filecmp import DEFAULT_IGNORES
import io
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from base64 import encodebytes
from urllib import response

import flask
import numpy as np
from PIL import Image
from flask import Flask, request, make_response, send_file, jsonify
from flask_cors import CORS
from hict.api.ContactMatrixFacet import ContactMatrixFacet
from hict.core.chunked_file import ChunkedFile
from hict.core.common import QueryLengthUnit, ContigDescriptor, ScaffoldDescriptor
from hict.core.contig_tree import ContigTree
from hict.core.scaffold_holder import ScaffoldHolder
from matplotlib import pyplot as plt
from werkzeug.exceptions import HTTPException
from readerwriterlock import rwlock

from hict_server.api_controller.dto.dto import AssemblyInfo, AssemblyInfoDTO, ContigDescriptorDTO, ContrastRangeSettings, ContrastRangeSettingsDTO, GetFastaForSelectionRequestDTO, GroupContigsIntoScaffoldRequestDTO, MoveSelectionRangeRequestDTO, NormalizationSettings, NormalizationSettingsDTO, OpenFileResponse, OpenFileResponseDTO, ReverseSelectionRangeRequestDTO, ScaffoldDescriptorDTO, UngroupContigsFromScaffoldRequestDTO

app = Flask(__name__)
CORS(app)

chunked_file: Optional[ChunkedFile] = None

colormap = plt.get_cmap('Greens')

transport_dtype: str = 'uint8'

filename: Optional[str] = None
fasta_filename: Optional[str] = None
data_path: Path = Path('./data')

chunked_file_lock: rwlock.RWLockWrite = rwlock.RWLockWrite()

DEFAULT_NORMALIZATION_SETTINGS: NormalizationSettings = NormalizationSettings(
    -1.0, 1.0, -1.0, 1.0, True
)

DEFAULT_CONTRAST_RANGE: ContrastRangeSettings = ContrastRangeSettings(
    0.0,
    1.0
)


currentNormalizationSettings: NormalizationSettings = DEFAULT_NORMALIZATION_SETTINGS
currentContrastRange: ContrastRangeSettings = DEFAULT_CONTRAST_RANGE

currentMinSignalValue: Dict[int, Union[float, int]] = dict()
currentMaxSignalValue: Dict[int, Union[float, int]] = dict()
minMaxLock: rwlock.RWLockWrite = rwlock.RWLockWrite()


currentTileVersion: int = 0
versionLock: rwlock.RWLockWrite = rwlock.RWLockWrite()

# def get_contig_info(f: ChunkedFile) -> ContigInfo.DTO:
#     contig_names: Dict[np.int64, str] = {}
#     contig_name_to_id: Dict[str, np.int64] = {}

#     for contig_id, contig_name in enumerate(f.contig_names):
#         contig_names[contig_id] = str(contig_name)
#         contig_name_to_id[str(contig_name)] = contig_id

#     contig_size: Dict[np.int64, List[np.int64]] = {}
#     contig_direction: List[ContigDirection] = []
#     contig_ord_ids: List[np.int64] = []
#     resolution_to_ord_contig_hide_type: Dict[int, List[ContigHideType]] = {}

#     def visit_node(n: ContigTree.Node):
#         nonlocal contig_size
#         for res, length in n.true_contig_descriptor().contig_length_at_resolution.items():
#             if res not in contig_size.keys():
#                 contig_size[res] = []
#                 resolution_to_ord_contig_hide_type[res] = []
#             contig_size[res].append(length)
#             resolution_to_ord_contig_hide_type[res].append(n.contig_descriptor.presence_in_resolution[res])
#         contig_direction.append(n.true_direction())
#         contig_ord_ids.append(n.contig_descriptor.contig_id)

#     f.contig_tree.traverse(visit_node)
#     contig_info: ContigInfo = ContigInfo(
#         contig_size,
#         contig_direction,
#         contig_ord_ids,
#         contig_names,
#         contig_name_to_id,
#         resolution_to_ord_contig_hide_type
#     )

#     return contig_info.to_dto()


def get_contig_descriptors(f: ChunkedFile) -> List[ContigDescriptor]:
    descriptors: List[ContigDescriptor] = []

    def visit_node(n: ContigTree.Node):
        descriptors.append(n.contig_descriptor)

    f.contig_tree.traverse(visit_node)

    return descriptors


def get_scaffold_descriptors(f: ChunkedFile) -> List[ScaffoldDescriptor]:
    scaffoldHolder: ScaffoldHolder = f.scaffold_holder
    descriptors: List[ScaffoldDescriptor] = []
    for scaffoldDescriptor in scaffoldHolder.scaffold_table.values():
        descriptors.append(scaffoldDescriptor)
    return list(filter(lambda sd: f.scaffold_housekeeping(sd.scaffold_id) is not None, descriptors))


def generate_assembly_info(f: ChunkedFile) -> AssemblyInfo:
    return AssemblyInfo(
        get_contig_descriptors(chunked_file),
        get_scaffold_descriptors(chunked_file)
    )


@app.post("/open")
def open_file():
    global chunked_file, filename, fasta_filename, currentTileVersion, currentNormalizationSettings, currentContrastRange
    req = request.get_json()
    filename = req["filename"]
    fasta_filename = req["fastaFilename"]
    app.logger.debug(
        f"/open: request={request} args={request.args} json={req}")
    app.logger.info(
        f"/open: Opening file {filename} and fasta file {fasta_filename}")
    if filename is None or filename == "":
        return "Wrong filename specified", 404
    # TODO: Fix this

    currentTileVersion = 0
    currentNormalizationSettings = DEFAULT_NORMALIZATION_SETTINGS
    currentContrastRange = DEFAULT_CONTRAST_RANGE
    chunked_file = ContactMatrixFacet.get_file_descriptor(
        str(data_path.joinpath(filename).resolve().absolute()))
    # chunked_file = ContactMatrixFacet.get_file_descriptor(filename)
    if fasta_filename is not None and fasta_filename != "":
        chunked_file.link_fasta(
            str(data_path.joinpath(fasta_filename).resolve().absolute()))
    with chunked_file_lock.gen_wlock() as cfl:
        ContactMatrixFacet.open_file(chunked_file)

    resp = generate_open_file_info()

    response = flask.jsonify(resp)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


def generate_open_file_info() -> OpenFileResponseDTO:
    global chunked_file
    tile_size: int = int(max(chunked_file.dense_submatrix_size.values()))

    resolutions: List[int] = [int(r) for r in sorted(
        chunked_file.resolutions, reverse=True)]
    with chunked_file_lock.gen_wlock() as cfl:
        response: OpenFileResponse = OpenFileResponse(
            "OK",
            transport_dtype,
            resolutions,
            [np.float64(r) / np.float64(min(chunked_file.resolutions))
             for r in resolutions],
            tile_size,
            generate_assembly_info(chunked_file),
            [chunked_file.contig_tree.root.subtree_length_bins[r] for r in
             resolutions] if chunked_file.contig_tree.root is not None else []
        )

    reponseDTO: OpenFileResponseDTO = OpenFileResponseDTO.fromEntity(response)

    return reponseDTO


@app.post("/close")
def close_file():
    global chunked_file
    if chunked_file is not None:
        with chunked_file_lock.gen_wlock() as cfl:
            ContactMatrixFacet.close_file(chunked_file)

    response = make_response(f"File closed")
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.post("/reverse_selection_range")
def reverse_selection_range():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    req = ReverseSelectionRangeRequestDTO(request.get_json()).toEntity()

    with chunked_file_lock.gen_wlock() as cfl:
        ContactMatrixFacet.reverse_selection_range(
            chunked_file, req.start_contig_id, req.end_contig_id)
        assemblyInfo: AssemblyInfo = generate_assembly_info(chunked_file)

    response = make_response(
        jsonify(AssemblyInfoDTO.fromEntity(assemblyInfo)))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.post("/get_fasta_for_selection")
def get_fasta_for_selection():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    req = GetFastaForSelectionRequestDTO(request.get_json()).toEntity()

    buf = io.BytesIO()
    with chunked_file_lock.gen_wlock() as cfl:
        chunked_file.get_fasta_for_selection(
            req.from_bp_x, req.to_bp_x,
            req.from_bp_y, req.to_bp_y,
            buf
        )
    buf.seek(0)

    response = make_response(send_file(buf, mimetype="text/plain"))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.post("/move_selection_range")
def move_selection_range():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    req = MoveSelectionRangeRequestDTO(request.get_json()).toEntity()

    with chunked_file_lock.gen_wlock() as cfl:
        ContactMatrixFacet.move_selection_range(
            chunked_file, req.start_contig_id, req.end_contig_id, req.target_start_order)
        assemblyInfo: AssemblyInfo = generate_assembly_info(chunked_file)

    response = make_response(
        jsonify(AssemblyInfoDTO.fromEntity(assemblyInfo)))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.post("/load_agp")
def load_agp():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    req = request.get_json()
    agp_filename = req["agpFilename"]

    with chunked_file_lock.gen_wlock() as cfl:
        ContactMatrixFacet.load_assembly_from_agp(
            chunked_file, data_path.joinpath(agp_filename).resolve().absolute())
        assemblyInfo: AssemblyInfo = generate_assembly_info(chunked_file)

    response = make_response(
        jsonify(AssemblyInfoDTO.fromEntity(assemblyInfo)))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.get("/get_assembly_info")
def get_assembly_info():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")
    with chunked_file_lock.gen_wlock() as cfl:
        assemblyInfo: AssemblyInfo = generate_assembly_info(chunked_file)

    response = make_response(
        jsonify(AssemblyInfoDTO.fromEntity(assemblyInfo)))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@ app.post("/move")
def move():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    req = request.get_json()

    contig_id: int = int(req['contigId'])
    target_ord: int = int(req['targetOrder'])

    chunked_file.move_contig_by_id(contig_id, target_ord)
    resp: dict = {'contig_info': get_contig_descriptors(chunked_file)}

    response = make_response(jsonify(resp))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@ app.post("/group_contigs_into_scaffold")
def group_contigs_into_scaffold():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    req = GroupContigsIntoScaffoldRequestDTO(request.get_json()).toEntity()

    with chunked_file_lock.gen_wlock() as cfl:
        ContactMatrixFacet.group_selection_range_into_scaffold(
            chunked_file, req.start_contig_id, req.end_contig_id, req.name, req.spacer_length)
        assemblyInfo: AssemblyInfo = generate_assembly_info(chunked_file)

    response = make_response(
        jsonify(AssemblyInfoDTO.fromEntity(assemblyInfo)))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@ app.post("/ungroup_contigs_from_scaffold")
def ungroup_contigs_from_scaffold():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    req = UngroupContigsFromScaffoldRequestDTO(request.get_json()).toEntity()

    with chunked_file_lock.gen_wlock() as cfl:
        ContactMatrixFacet.ungroup_selection_range(
            chunked_file, req.start_contig_id, req.end_contig_id)
        assemblyInfo: AssemblyInfo = generate_assembly_info(chunked_file)

    response = make_response(
        jsonify(AssemblyInfoDTO.fromEntity(assemblyInfo)))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@ app.post("/save")
def save():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    with chunked_file_lock.gen_wlock() as cfl:
        chunked_file.save()
        resp: dict = {
            'assembly': AssemblyInfoDTO.fromEntity(generate_assembly_info(chunked_file)),
            'result': "OK, file saved"
        }

    response = make_response(jsonify(resp))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@ app.post("/list_files")
def list_files():
    files = list(
        sorted(map(lambda p: str(p.relative_to(data_path)), data_path.rglob("*.hdf5"))))
    response = flask.jsonify(files)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@ app.post("/list_fasta_files")
def list_fasta_files():
    files = list(
        sorted(map(lambda p: str(p.relative_to(data_path)), data_path.rglob("*.fasta"))))
    response = flask.jsonify(files)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@ app.post("/list_agp_files")
def list_agp_files():
    files = list(
        sorted(map(lambda p: str(p.relative_to(data_path)), data_path.rglob("*.agp"))))
    response = flask.jsonify(files)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@ app.post("/link_fasta")
def link_fasta():
    global fasta_filename

    fasta_filename = str(request.get_json()['fastaFilename'])

    if chunked_file is not None and chunked_file.state == ChunkedFile.FileState.OPENED:
        chunked_file.link_fasta(
            str(data_path.joinpath(fasta_filename).resolve().absolute()))

    response = flask.jsonify("OK")
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@ app.post("/get_fasta_for_assembly")
def get_fasta_for_assembly():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    buf = io.BytesIO()
    with chunked_file_lock.gen_wlock() as cfl:
        chunked_file.get_fasta_for_assembly(buf)
    buf.seek(0)

    response = make_response(send_file(buf, mimetype="text/plain"))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response

@ app.post("/get_agp_for_assembly")
def get_agp_for_assembly():
    global chunked_file
    if chunked_file is None or chunked_file.state != ChunkedFile.FileState.OPENED:
        raise Exception("File is not opened?")

    buf = io.BytesIO()
    with chunked_file_lock.gen_wlock() as cfl:
        chunked_file.get_agp_for_assembly(buf)
    buf.seek(0)

    response = make_response(send_file(buf, mimetype="text/plain"))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@ app.get("/get_resolutions")
def get_resolutions():
    return flask.jsonify(chunked_file.resolutions)


@ app.post("/get_current_signal_range")
def get_current_signal_range():
    if chunked_file is None:
        return "File is not opened yet", 400
    version: int = int(request.get_json()["tileVersion"])

    actual_version: int
    with versionLock.gen_rlock():
        actual_version = currentTileVersion

    if version < actual_version:
        resp = make_response()
        resp.status_code = 204
        return resp
    elif version > actual_version:
        bumpVersion(version)

    with minMaxLock.gen_rlock():
        return make_response(jsonify({
            "lowerBounds": currentMinSignalValue,
            "upperBounds": currentMaxSignalValue
        }))


@ app.post("/set_contrast_range")
def set_contrast_range():
    global currentContrastRange
    currentContrastRange = ContrastRangeSettingsDTO(
        request.get_json()).toEntity()
    response = make_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@ app.post("/set_normalization")
def set_normalization():
    global currentNormalizationSettings
    currentNormalizationSettings = NormalizationSettingsDTO(
        request.get_json()).toEntity()
    response = make_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


def normalize_tile(
    raw_tile: np.ndarray,
    weights: Optional[Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    if currentNormalizationSettings.preLogEnabled():
        raw_tile = np.log1p(raw_tile) / \
            currentNormalizationSettings.preLogLnBase
    if currentNormalizationSettings.coolerBalanceEnabled():
        assert weights is not None, "Requested cooler balancing but didn't provide the weights?"
        raw_tile = ContactMatrixFacet.apply_cooler_balance_to_dense_matrix(
            raw_tile,
            weights[0],
            weights[1],
            inplace=True
        )
    if currentNormalizationSettings.postLogEnabled():
        raw_tile = np.log1p(raw_tile) / \
            currentNormalizationSettings.postLogLnBase
    return raw_tile


def contrast_tile(
    raw_tile: np.ndarray
) -> np.ndarray:
    lower_bound, upper_bound = (
        currentContrastRange.lowerSignalBound,
        currentContrastRange.upperSignalBound,
    )
    range = upper_bound - lower_bound
    if np.isclose(range, 0):
        range = 1.0
    raw_tile = raw_tile - currentContrastRange.lowerSignalBound
    raw_tile[raw_tile < 0] = 0
    raw_tile[raw_tile > range] = range
    raw_tile = raw_tile / range
    return raw_tile


@app.get("/get_tile")
def get_tile():
    if chunked_file is None:
        return "File is not opened yet", 400

    level: int = int(request.args.get("level"))
    row: int = int(request.args.get("row"))
    col: int = int(request.args.get("col"))
    tile_size: int = int(request.args.get("tile_size"))
    version: int = int(request.args.get("version"))

    actual_version: int
    with versionLock.gen_rlock():
        actual_version = currentTileVersion

    if version < actual_version:
        resp = make_response("Late query: current tile version is newer")
        resp.status_code = 204
        return resp
    elif version > actual_version:
        bumpVersion(version)

    resolution: int = sorted(chunked_file.resolutions)[-level]
    x0: int = row * tile_size
    x1: int = (1 + row) * tile_size
    y0: int = col * tile_size
    y1: int = (1 + col) * tile_size

    # with chunked_file_lock.gen_wlock():
    raw_dense_rect, row_weights, col_weights = ContactMatrixFacet.get_dense_submatrix(
        chunked_file,
        resolution,
        x0,
        y0,
        x1,
        y1,
        QueryLengthUnit.PIXELS,
        fetch_cooler_weights=currentNormalizationSettings.coolerBalanceEnabled()
    )

    dense_rect: np.ndarray = normalize_tile(
        raw_dense_rect, (row_weights, col_weights))

    if dense_rect.size > 0:
        minValue = int(dense_rect.min()) if (
            dense_rect.dtype in (np.int64, np.int32,
                                 np.int16, np.int8)
        ) else float(dense_rect.min())
        maxValue = int(dense_rect.max()) if (
            dense_rect.dtype in (np.int64, np.int32,
                                 np.int16, np.int8)
        )else float(dense_rect.max())
        with minMaxLock.gen_wlock():
            if level not in currentMinSignalValue.keys() or currentMinSignalValue[level] > minValue:
                currentMinSignalValue[level] = minValue
            if level not in currentMaxSignalValue.keys() or currentMaxSignalValue[level] < maxValue:
                currentMaxSignalValue[level] = maxValue

    dense_rect = contrast_tile(dense_rect)

    padded_dense_rect: np.ndarray = np.zeros(
        (tile_size, tile_size), dtype=dense_rect.dtype)
    padded_dense_rect[0:dense_rect.shape[0],
                      0: dense_rect.shape[1]] = dense_rect

    colored_image: np.ndarray = colormap(padded_dense_rect)
    image_matrix: np.ndarray = colored_image[:, :, : 3] * 255
    image_matrix = image_matrix.astype(transport_dtype)

    image: Image = Image.fromarray(image_matrix)

    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)

    ranges: Dict
    with minMaxLock.gen_rlock():
        ranges = {
            "lowerBounds": currentMinSignalValue,
            "upperBounds": currentMaxSignalValue
        }

    # response = make_response(send_file(buf, mimetype="image/png"))
    response = make_response(jsonify({
        "ranges": ranges,
        # encode as base64
        "image": "".join(("data:image/png;base64,", encodebytes(buf.getvalue()).decode('ascii')))
    }))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


def bumpVersion(version: int) -> None:
    global currentTileVersion
    with versionLock.gen_wlock():
        if version > currentTileVersion:
            currentTileVersion = version
            with minMaxLock.gen_wlock():
                currentMinSignalValue.clear()
                currentMaxSignalValue.clear()

# @app.errorhandler(Exception)
# def handle_exception(e):
#     if isinstance(e, HTTPException):
#         return e
#     response = make_response(f"Error: {e}")
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.status_code = 500
#     return response


def main():
    global data_path
    parser: ArgumentParser = argparse.ArgumentParser(
        description="Run development version of HiCT tile server.",
        epilog="Visit https://github.com/ctlab/HiCT for more info."
    )

    def dir_checker(arg_path: str) -> bool:
        if os.path.isdir(arg_path):
            return arg_path
        else:
            raise ValueError(
                f'Path {arg_path} does not point to any directory')
    parser.add_argument('--data-path', default='./data', type=dir_checker)
    parser.add_argument(
        '--log-level', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'], default='INFO', type=str)
    parser.add_argument('--verbose', action='store_true')
    arguments: Namespace = parser.parse_args()
    data_path = Path(os.path.abspath(arguments.data_path))
    log_level_str: str
    if arguments.verbose:
        log_level_str = 'DEBUG'
    else:
        log_level_str = arguments.log_level
    app.logger.setLevel(log_level_str)
    app.logger.info(f"Using '{data_path}' as data directory")
    app.run(debug=True)


if __name__ == '__main__':
    main()
