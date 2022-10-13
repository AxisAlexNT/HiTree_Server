from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from hict.core.common import ContigDescriptor, ScaffoldDescriptor, ScaffoldBorders


@dataclass
class ContigDescriptorDTO:
    contigId: int
    contigName: str
    contigDirection: int
    contigLengthBp: int
    contigLengthBins: Dict[int, int]
    scaffoldId: Optional[int]
    contigPresenceAtResolution: Dict[int, int]

    @staticmethod
    def fromEntity(descriptor: ContigDescriptor) -> 'ContigDescriptorDTO':
        contig_length_at_resolution: Dict[int, int] = dict()
        presence_in_resolution: Dict[int, int] = dict()

        for res, ctg_length in descriptor.contig_length_at_resolution.items():
            if res != 0:
                int_res: int = int(res)
                contig_length_at_resolution[int_res] = int(ctg_length)
                presence_in_resolution[int_res] = descriptor.presence_in_resolution[res].value

        return ContigDescriptorDTO(
            int(descriptor.contig_id),
            str(descriptor.contig_name),
            descriptor.direction.value,
            int(descriptor.contig_length_at_resolution[0]),
            contig_length_at_resolution,
            str(descriptor.scaffold_id) if descriptor.scaffold_id is not None else None,
            presence_in_resolution
        )


@dataclass
class ScaffoldBordersDTO:
    startContigId: int
    endContigId: int

    @staticmethod
    def fromEntity(borders: Optional[ScaffoldBorders]) -> Optional['ScaffoldBordersDTO']:
        return ScaffoldBordersDTO(
            int(borders.start_contig_id),
            int(borders.end_contig_id)
        ) if borders is not None else None


@dataclass
class ScaffoldDescriptorDTO:
    scaffoldId: int
    scaffoldName: str
    scaffoldBorders: Optional[ScaffoldBordersDTO]
    scaffoldDirection: int
    spacerLength: int

    @staticmethod
    def fromEntity(descriptor: ScaffoldDescriptor) -> 'ScaffoldDescriptorDTO':
        return ScaffoldDescriptorDTO(
            int(descriptor.scaffold_id),
            descriptor.scaffold_name,
            ScaffoldBordersDTO.fromEntity(descriptor.scaffold_borders),
            int(descriptor.scaffold_direction.value),
            int(descriptor.spacer_length)
        )


@dataclass
class AssemblyInfo:
    contigDescriptors: List[ContigDescriptor]
    scaffoldDescriptors: List[ScaffoldDescriptor]


@dataclass
class AssemblyInfoDTO:
    contigDescriptors: List[ContigDescriptorDTO]
    scaffoldDescriptors: List[ScaffoldDescriptorDTO]

    @staticmethod
    def fromEntity(assembly: AssemblyInfo) -> 'AssemblyInfoDTO':
        return AssemblyInfoDTO(
            [ContigDescriptorDTO.fromEntity(descriptor)
             for descriptor in assembly.contigDescriptors],
            [ScaffoldDescriptorDTO.fromEntity(descriptor)
             for descriptor in assembly.scaffoldDescriptors
             ]
        )


@dataclass
class GroupContigsIntoScaffoldRequest:
    start_contig_id: np.int64
    end_contig_id: np.int64
    name: Optional[str]
    spacer_length: Optional[int]


@dataclass
class GroupContigsIntoScaffoldRequestDTO:
    start_contig_id: int
    end_contig_id: int
    name: Optional[str]
    spacer_length: Optional[int]

    def __init__(self, request_json) -> None:
        self.start_contig_id: int = int(request_json['startContigId'])
        self.end_contig_id: int = int(request_json['endContigId'])
        self.name: Optional[str] = (
            request_json['scaffoldName'] if 'scaffoldName' in request_json.keys() else None)
        self.spacer_length: Optional[int] = int(
            request_json['spacerLength']) if 'spacerLength' in request_json.keys() else None

    def toEntity(self) -> GroupContigsIntoScaffoldRequest:
        return GroupContigsIntoScaffoldRequest(
            np.int64(self.start_contig_id),
            np.int64(self.end_contig_id),
            self.name if self.name != "" else None,
            self.spacer_length
        )


@dataclass
class UngroupContigsFromScaffoldRequest:
    start_contig_id: np.int64
    end_contig_id: np.int64


@dataclass
class UngroupContigsFromScaffoldRequestDTO:
    start_contig_id: int
    end_contig_id: int

    def __init__(self, request_json) -> None:
        self.start_contig_id: int = int(request_json['startContigId'])
        self.end_contig_id: int = int(request_json['endContigId'])

    def toEntity(self) -> UngroupContigsFromScaffoldRequest:
        return UngroupContigsFromScaffoldRequest(
            np.int64(self.start_contig_id),
            np.int64(self.end_contig_id),
        )


@dataclass
class ReverseSelectionRangeRequest:
    start_contig_id: np.int64
    end_contig_id: np.int64


@dataclass
class ReverseSelectionRangeRequestDTO:
    start_contig_id: int
    end_contig_id: int

    def __init__(self, request_json) -> None:
        self.start_contig_id: int = int(request_json['startContigId'])
        self.end_contig_id: int = int(request_json['endContigId'])

    def toEntity(self) -> ReverseSelectionRangeRequest:
        return ReverseSelectionRangeRequest(
            np.int64(self.start_contig_id),
            np.int64(self.end_contig_id),
        )


@dataclass
class MoveSelectionRangeRequest:
    start_contig_id: np.int64
    end_contig_id: np.int64
    target_start_order: np.int64


@dataclass
class MoveSelectionRangeRequestDTO:
    start_contig_id: int
    end_contig_id: int
    target_start_order: int

    def __init__(self, request_json) -> None:
        self.start_contig_id: int = int(request_json['startContigId'])
        self.end_contig_id: int = int(request_json['endContigId'])
        self.target_start_order: int = int(request_json['targetStartOrder'])

    def toEntity(self) -> MoveSelectionRangeRequest:
        return MoveSelectionRangeRequest(
            np.int64(self.start_contig_id),
            np.int64(self.end_contig_id),
            np.int64(self.target_start_order)
        )


@dataclass
class GetFastaForSelectionRequest:
    from_bp_x: np.int64
    from_bp_y: np.int64
    to_bp_x: np.int64
    to_bp_y: np.int64


@dataclass
class GetFastaForSelectionRequestDTO:
    from_bp_x: int
    from_bp_y: int
    to_bp_x:   int
    to_bp_y:   int

    def __init__(self, request_json) -> None:
        self.from_bp_x: int = int(request_json['fromBpX'])
        self.from_bp_y: int = int(request_json['fromBpY'])
        self.to_bp_x: int = int(request_json['toBpX'])
        self.to_bp_y: int = int(request_json['toBpY'])

    def toEntity(self) -> GetFastaForSelectionRequest:
        return GetFastaForSelectionRequest(
            np.int64(self.from_bp_x),
            np.int64(self.from_bp_y),
            np.int64(self.to_bp_x),
            np.int64(self.to_bp_y),
        )


@dataclass
class OpenFileResponse:
    status: str
    dtype: np.dtype
    resolutions: List[np.int64]
    pixelResolutions: List[np.float64]
    tileSize: int
    assemblyInfo: AssemblyInfo
    matrixSizesBins: List[np.int64]


@dataclass
class OpenFileResponseDTO:
    status: str
    dtype: str
    resolutions: List[int]
    pixelResolutions: List[float]
    tileSize: int
    assemblyInfo: AssemblyInfoDTO
    matrixSizesBins: List[int]

    @staticmethod
    def fromEntity(response: OpenFileResponse) -> 'OpenFileResponseDTO':
        return OpenFileResponseDTO(
            str(response.status),
            str(response.dtype),
            [int(r) for r in response.resolutions],
            [float(pr) for pr in response.pixelResolutions],
            int(response.tileSize),
            AssemblyInfoDTO.fromEntity(response.assemblyInfo),
            [int(sz) for sz in response.matrixSizesBins]
        )


@dataclass
class NormalizationSettings:
    preLogBase: np.float64
    preLogLnBase: np.float64
    postLogBase: np.float64
    postLogLnBase: np.float64
    applyCoolerWeights: bool

    def preLogEnabled(self) -> bool:
        return self.preLogBase > np.int64(1.0)

    def postLogEnabled(self) -> bool:
        return self.postLogBase > np.int64(1.0)

    def coolerBalanceEnabled(self) -> bool:
        return self.applyCoolerWeights


@dataclass
class NormalizationSettingsDTO:
    def __init__(self, request_json) -> None:
        self.preLogBase = request_json['preLogBase']
        self.postLogBase = request_json['postLogBase']
        self.applyCoolerWeights = request_json['applyCoolerWeights']

    def toEntity(self) -> NormalizationSettings:
        return NormalizationSettings(
            np.float64(self.preLogBase),
            np.log(np.float64(self.preLogBase)) if np.float64(
                self.preLogBase) > 1.0 else 1.0,
            np.float64(self.postLogBase),
            np.log(np.float64(self.postLogBase)) if np.float64(
                self.postLogBase) > 1.0 else 1.0,
            bool(self.applyCoolerWeights)
        )


@dataclass
class ContrastRangeSettings:
    lowerSignalBound: np.float64
    upperSignalBound: np.float64


@dataclass
class ContrastRangeSettingsDTO:
    def __init__(self, request_json) -> None:
        self.lowerSignalBound = request_json['lowerSignalBound']
        self.upperSignalBound = request_json['upperSignalBound']

    def toEntity(self) -> ContrastRangeSettings:
        return ContrastRangeSettings(
            np.float64(self.lowerSignalBound),
            np.float64(self.upperSignalBound)
        )
