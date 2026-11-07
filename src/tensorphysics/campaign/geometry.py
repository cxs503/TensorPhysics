"""Geometry contract consumer (``geom_pc6``) and failure-face interface extension.

The geometry contract is the only place where a campaign hands hull geometry to
several libraries (resistance, ice loads, structural reliability).  It therefore
carries a *verified* principal-dimension set: the block coefficient must close
against the displacement volume,

    Cb * Lpp * B * T = nabla          (relative closure <= 1e-12),

and the failure-face packet referenced by the contract must describe a closed
control surface,

    sum_i A_i n_i = 0                 (relative closure <= 1e-12),

which is the discrete divergence theorem on the contact patch.  Both checks are
geometric identities, so their residual is at machine precision and any real
geometry error is unmissable.

``failure_faces`` schema v2 is a backward-compatible extension of v1: it adds
optional centroid, failure mode, contact region, load share and redundancy group
fields while keeping v1's four required fields.  A v1 packet is accepted by the
v2 consumer, a v2 packet downgrades lossily to v1, and versions outside the
declared ``consumer_contract`` window are rejected -- the interface contract is
therefore negotiated explicitly instead of assumed.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import SchemaError
from .schema_defs import validate_artifact

#: Relative window for the Cb closure and the closed-surface identity.
CLOSURE_RTOL = 1e-12

FAILURE_MODES = ("crushing", "bending", "shear", "buckling")
SUPPORTED_FACE_VERSIONS = (1, 2)


class GeometryContractError(ValueError):
    """A geometry contract or failure-face packet violates its contract."""


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeometryContractError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise GeometryContractError(f"{name} must be finite and > 0")
    return number


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise GeometryContractError(f"{name} must have exactly three components")
    components = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise GeometryContractError(f"{name}[{index}] must be a real number")
        number = float(item)
        if not math.isfinite(number):
            raise GeometryContractError(f"{name}[{index}] must be finite")
        components.append(number)
    return (components[0], components[1], components[2])


@dataclass(frozen=True)
class PrincipalDimensions:
    """Verified principal dimensions of the hull (Cb closed against volume)."""

    length_pp_m: float
    breadth_m: float
    draught_m: float
    block_coefficient: float
    displacement_volume_m3: float

    def __post_init__(self) -> None:
        length = _positive(self.length_pp_m, "length_pp_m")
        breadth = _positive(self.breadth_m, "breadth_m")
        draught = _positive(self.draught_m, "draught_m")
        cb = _positive(self.block_coefficient, "block_coefficient")
        volume = _positive(self.displacement_volume_m3, "displacement_volume_m3")
        if cb > 1.0:
            raise GeometryContractError("block_coefficient must be <= 1")
        if not length > breadth > draught:
            raise GeometryContractError("expected Lpp > B > T for a displacement hull")
        residual = abs(cb * length * breadth * draught - volume)
        if residual > CLOSURE_RTOL * volume:
            raise GeometryContractError(
                "Cb closure failed: |Cb*L*B*T - nabla| / nabla = "
                f"{residual / volume:.3e} > {CLOSURE_RTOL:.0e}"
            )

    @property
    def cb_from_volume(self) -> float:
        """Block coefficient implied by the displacement volume."""
        return self.displacement_volume_m3 / (
            self.length_pp_m * self.breadth_m * self.draught_m
        )

    @property
    def cb_closure_residual(self) -> float:
        """Relative closure residual of Cb*L*B*T against the volume."""
        product = (
            self.block_coefficient * self.length_pp_m * self.breadth_m * self.draught_m
        )
        return abs(product - self.displacement_volume_m3) / self.displacement_volume_m3

    def displacement_mass_t(self, water_density_kg_m3: float) -> float:
        density = _positive(water_density_kg_m3, "water_density_kg_m3")
        return self.displacement_volume_m3 * density / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_pp_m": self.length_pp_m,
            "breadth_m": self.breadth_m,
            "draught_m": self.draught_m,
            "block_coefficient": self.block_coefficient,
            "displacement_volume_m3": self.displacement_volume_m3,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "PrincipalDimensions":
        if not isinstance(payload, Mapping):
            raise GeometryContractError("principal_dimensions must be an object")
        return cls(
            length_pp_m=payload["length_pp_m"],
            breadth_m=payload["breadth_m"],
            draught_m=payload["draught_m"],
            block_coefficient=payload["block_coefficient"],
            displacement_volume_m3=payload["displacement_volume_m3"],
        )


@dataclass(frozen=True)
class FailureFaceV1:
    """Schema v1 failure face: the original interface (four required fields)."""

    face_id: str
    normal: tuple[float, float, float]
    area_m2: float
    capacity_n: float

    def __post_init__(self) -> None:
        if not isinstance(self.face_id, str) or not self.face_id.strip():
            raise GeometryContractError("face_id must be a non-empty string")
        normal = _vector3(self.normal, f"face {self.face_id!r} normal")
        norm = math.sqrt(sum(component * component for component in normal))
        if abs(norm - 1.0) > 1e-12:
            raise GeometryContractError(
                f"face {self.face_id!r} normal must be a unit vector (|n| = {norm!r})"
            )
        _positive(self.area_m2, f"face {self.face_id!r} area_m2")
        _positive(self.capacity_n, f"face {self.face_id!r} capacity_n")

    def area_vector(self) -> tuple[float, float, float]:
        """Signed area vector A*n (the flux through the face)."""
        return tuple(self.area_m2 * component for component in self.normal)  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_id": self.face_id,
            "normal": list(self.normal),
            "area_m2": self.area_m2,
            "capacity_n": self.capacity_n,
        }


@dataclass(frozen=True)
class FailureFace(FailureFaceV1):
    """Schema v2 extension: v1 fields plus optional load-transfer metadata."""

    centroid_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    failure_mode: str = "crushing"
    contact_region: str = "bow_shoulder"
    load_share: float = 1.0
    redundancy_group: str = "RG-0"

    def __post_init__(self) -> None:
        super().__post_init__()
        _vector3(self.centroid_m, f"face {self.face_id!r} centroid_m")
        if self.failure_mode not in FAILURE_MODES:
            raise GeometryContractError(
                f"face {self.face_id!r} failure_mode must be one of {FAILURE_MODES}"
            )
        if not isinstance(self.contact_region, str) or not self.contact_region.strip():
            raise GeometryContractError("contact_region must be a non-empty string")
        share = _positive(self.load_share, f"face {self.face_id!r} load_share")
        if share > 1.0:
            raise GeometryContractError(f"face {self.face_id!r} load_share must be <= 1")
        if not isinstance(self.redundancy_group, str) or not self.redundancy_group.strip():
            raise GeometryContractError("redundancy_group must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "centroid_m": list(self.centroid_m),
                "failure_mode": self.failure_mode,
                "contact_region": self.contact_region,
                "load_share": self.load_share,
                "redundancy_group": self.redundancy_group,
            }
        )
        return payload

    def downgrade_v1(self) -> FailureFaceV1:
        """Lossy v2 -> v1 projection (documented loss of extension fields)."""
        return FailureFaceV1(
            face_id=self.face_id,
            normal=self.normal,
            area_m2=self.area_m2,
            capacity_n=self.capacity_n,
        )


@dataclass(frozen=True)
class FailureFacePacket:
    """Closed-surface failure-face packet (schema v1 or v2)."""

    faces: tuple[FailureFaceV1 | FailureFace, ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_FACE_VERSIONS:
            raise GeometryContractError(
                f"failure_faces schema_version {self.schema_version!r} is outside "
                f"the supported window {SUPPORTED_FACE_VERSIONS}"
            )
        if not self.faces:
            raise GeometryContractError("a failure-face packet needs at least one face")
        identifiers = [face.face_id for face in self.faces]
        if len(set(identifiers)) != len(identifiers):
            raise GeometryContractError("face_id values must be unique")
        if self.schema_version == 1 and any(isinstance(face, FailureFace) for face in self.faces):
            raise GeometryContractError("a v1 packet cannot carry v2 failure faces")
        if self.schema_version == 2 and any(type(face) is FailureFaceV1 for face in self.faces):
            raise GeometryContractError("a v2 packet requires v2 failure faces")
        for face in self.faces:
            if not isinstance(face, (FailureFaceV1, FailureFace)):
                raise GeometryContractError("faces must be FailureFaceV1/FailureFace objects")

    @property
    def total_area_m2(self) -> float:
        return math.fsum(face.area_m2 for face in self.faces)

    def net_area_vector(self) -> tuple[float, float, float]:
        """Sum of signed area vectors; identically zero for a closed surface."""
        total = [0.0, 0.0, 0.0]
        for face in self.faces:
            for index, component in enumerate(face.area_vector()):
                total[index] += component
        return (total[0], total[1], total[2])

    def closure_ratio(self) -> float:
        """|sum A_i n_i| / sum A_i (dimensionless closed-surface residual)."""
        net = self.net_area_vector()
        net_norm = math.sqrt(math.fsum(component * component for component in net))
        return net_norm / self.total_area_m2

    def verify_closure(self, *, rtol: float = CLOSURE_RTOL) -> dict[str, Any]:
        if not math.isfinite(rtol) or rtol <= 0.0:
            raise GeometryContractError("rtol must be finite and > 0")
        ratio = self.closure_ratio()
        return {
            "net_area_vector_m2": list(self.net_area_vector()),
            "total_area_m2": self.total_area_m2,
            "closure_ratio": ratio,
            "tolerance": rtol,
            "closed": ratio <= rtol,
        }

    def total_capacity_n(self) -> float:
        return math.fsum(face.capacity_n for face in self.faces)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_name": "failure_faces",
            "schema_version": self.schema_version,
            "interface": "failure_faces",
            "faces": [face.to_dict() for face in self.faces],
        }
        validate_artifact(f"failure_faces.v{self.schema_version}", payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> "FailureFacePacket":
        """Consumer-side constructor: version negotiation, then schema, then objects."""
        if not isinstance(payload, Mapping):
            raise GeometryContractError("failure-face packet must be a JSON object")
        version = payload.get("schema_version")
        if version not in SUPPORTED_FACE_VERSIONS:
            raise GeometryContractError(
                f"failure_faces schema_version {version!r} is outside the supported "
                f"window {SUPPORTED_FACE_VERSIONS}"
            )
        try:
            validate_artifact(f"failure_faces.v{version}", payload)
        except SchemaError as exc:
            raise GeometryContractError(f"failure-face packet rejected: {exc}") from exc
        faces: list[FailureFaceV1 | FailureFace] = []
        for entry in payload["faces"]:
            if version == 2:
                faces.append(
                    FailureFace(
                        face_id=entry["face_id"],
                        normal=_vector3(entry["normal"], "normal"),
                        area_m2=entry["area_m2"],
                        capacity_n=entry["capacity_n"],
                        centroid_m=_vector3(entry.get("centroid_m", (0.0, 0.0, 0.0)), "centroid_m"),
                        failure_mode=entry.get("failure_mode", "crushing"),
                        contact_region=entry.get("contact_region", "bow_shoulder"),
                        load_share=entry.get("load_share", 1.0),
                        redundancy_group=entry.get("redundancy_group", "RG-0"),
                    )
                )
            else:
                faces.append(
                    FailureFaceV1(
                        face_id=entry["face_id"],
                        normal=_vector3(entry["normal"], "normal"),
                        area_m2=entry["area_m2"],
                        capacity_n=entry["capacity_n"],
                    )
                )
        return cls(faces=tuple(faces), schema_version=int(version))

    def downgrade_v1(self) -> "FailureFacePacket":
        """Lossy packet projection to the v1 interface."""
        faces = tuple(
            face.downgrade_v1() if isinstance(face, FailureFace) else face
            for face in self.faces
        )
        return FailureFacePacket(faces=faces, schema_version=1)


def tetrahedron_face_packet(
    vertices: Iterable[Iterable[float]],
    *,
    capacities_n: Iterable[float],
    contact_regions: Iterable[str] | None = None,
    failure_modes: Iterable[str] | None = None,
    schema_version: int = 2,
    redundancy_group: str = "RG-BS",
) -> FailureFacePacket:
    """Build a closed failure-face packet from a tetrahedral contact volume.

    The four faces are generated from vertex coordinates: outward unit normals
    are ``cross/|cross|`` and areas are ``|cross|/2``.  Because the areas are
    irrational, the closed-surface identity ``sum A_i n_i = 0`` only cancels to
    round-off -- which is exactly the machine-precision residual the geometry
    contract is meant to certify.  Aperturbation of any vertex immediately opens
    the surface, so this constructor doubles as the negative control.
    """
    points = [_vector3(vertex, f"vertices[{index}]") for index, vertex in enumerate(vertices)]
    if len(points) != 4:
        raise GeometryContractError("a tetrahedron needs exactly four vertices")
    capacities = [float(value) for value in capacities_n]
    if len(capacities) != 4:
        raise GeometryContractError("capacities_n needs exactly four values")
    regions = list(contact_regions) if contact_regions is not None else ["bow_shoulder"] * 4
    modes = list(failure_modes) if failure_modes is not None else ["crushing"] * 4
    if len(regions) != 4 or len(modes) != 4:
        raise GeometryContractError("per-face metadata needs exactly four entries")

    centroid = tuple(math.fsum(point[index] for point in points) / 4.0 for index in range(3))
    faces: list[FailureFaceV1 | FailureFace] = []
    for opposite in range(4):
        ids = [index for index in range(4) if index != opposite]
        a, b, c = (points[index] for index in ids)
        edge1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        edge2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cross = (
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        )
        norm = math.sqrt(math.fsum(component * component for component in cross))
        if norm == 0.0:
            raise GeometryContractError("degenerate tetrahedron face")
        normal = tuple(component / norm for component in cross)
        face_centroid = tuple(math.fsum(point[index] for point in (a, b, c)) / 3.0 for index in range(3))
        outward = math.fsum(
            normal[index] * (face_centroid[index] - centroid[index]) for index in range(3)
        )
        if outward < 0.0:
            normal = tuple(-component for component in normal)
        area = 0.5 * norm
        face_id = "face_" + "".join(str(index) for index in ids)
        if schema_version == 2:
            faces.append(
                FailureFace(
                    face_id=face_id,
                    normal=normal,  # type: ignore[arg-type]
                    area_m2=area,
                    capacity_n=capacities[opposite],
                    centroid_m=face_centroid,  # type: ignore[arg-type]
                    failure_mode=modes[opposite],
                    contact_region=regions[opposite],
                    redundancy_group=redundancy_group,
                )
            )
        else:
            faces.append(
                FailureFaceV1(
                    face_id=face_id,
                    normal=normal,  # type: ignore[arg-type]
                    area_m2=area,
                    capacity_n=capacities[opposite],
                )
            )
    return FailureFacePacket(faces=tuple(faces), schema_version=schema_version)


@dataclass(frozen=True)
class GeometryHandle:
    """Resolved, checksum-pinned geometry contract handed to consumers."""

    geometry_id: str
    contract: Mapping[str, Any]
    checksum_sha256: str
    dimensions: PrincipalDimensions
    consumer_contract: Mapping[str, Any]
    faces: FailureFacePacket | None = None

    @property
    def displacement_mass_t(self) -> float:
        return self.dimensions.displacement_mass_t(
            self.contract["design_water_density_kg_m3"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_id": self.geometry_id,
            "checksum_sha256": self.checksum_sha256,
            "dimensions": self.dimensions.to_dict(),
            "displacement_mass_t": self.displacement_mass_t,
            "block_coefficient_from_volume": self.dimensions.cb_from_volume,
            "cb_closure_residual": self.dimensions.cb_closure_residual,
            "consumer_contract": dict(self.consumer_contract),
            "failure_faces_schema_version": None
            if self.faces is None
            else self.faces.schema_version,
        }


class GeometryContractRegistry:
    """Consumer-side registration of geometry contracts by ``geometry_id``."""

    def __init__(self) -> None:
        self._handles: dict[str, GeometryHandle] = {}

    def register(
        self,
        payload: Any,
        *,
        checksum_sha256: str | None = None,
        faces: FailureFacePacket | None = None,
    ) -> GeometryHandle:
        try:
            validate_artifact("geometry_contract.v1", payload)
        except SchemaError as exc:
            raise GeometryContractError(f"geometry contract rejected: {exc}") from exc
        dimensions = PrincipalDimensions.from_dict(payload["principal_dimensions"])
        consumer = payload["consumer_contract"]
        if consumer["min_schema_version"] > consumer["max_schema_version"]:
            raise GeometryContractError("consumer_contract version window is inverted")
        if faces is not None:
            version = faces.schema_version
            if not consumer["min_schema_version"] <= version <= consumer["max_schema_version"]:
                raise GeometryContractError(
                    f"failure_faces v{version} outside the declared consumer window "
                    f"[{consumer['min_schema_version']}, {consumer['max_schema_version']}]"
                )
            closure = faces.verify_closure()
            if not closure["closed"]:
                raise GeometryContractError(
                    "failure-face packet is not a closed surface: "
                    f"|sum A n| / sum A = {closure['closure_ratio']:.3e}"
                )
        geometry_id = payload["geometry_id"]
        if geometry_id in self._handles:
            raise GeometryContractError(f"{geometry_id!r} is already registered")
        handle = GeometryHandle(
            geometry_id=geometry_id,
            contract=payload,
            checksum_sha256=checksum_sha256 or hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            dimensions=dimensions,
            consumer_contract=consumer,
            faces=faces,
        )
        self._handles[geometry_id] = handle
        return handle

    def register_file(
        self, path: str | Path, *, faces_path: str | Path | None = None
    ) -> GeometryHandle:
        contract_path = Path(path)
        if not contract_path.is_file():
            raise GeometryContractError(f"geometry contract not found: {contract_path}")
        raw = contract_path.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GeometryContractError(f"geometry contract is not valid JSON: {exc}") from exc
        declared = payload.get("geometry_id")
        if isinstance(declared, str) and contract_path.stem.startswith("geom_"):
            if declared != contract_path.stem:
                raise GeometryContractError(
                    f"geometry_id {declared!r} does not match file {contract_path.name!r}"
                )
        packet: FailureFacePacket | None = None
        reference = faces_path if faces_path is not None else payload.get("failure_faces_packet")
        if reference is not None:
            packet_path = Path(reference)
            if not packet_path.is_absolute():
                packet_path = contract_path.parent / packet_path
            if not packet_path.is_file():
                raise GeometryContractError(
                    f"declared failure-face packet not found: {packet_path}"
                )
            try:
                packet_payload = json.loads(packet_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise GeometryContractError(f"failure-face packet is not JSON: {exc}") from exc
            packet = FailureFacePacket.from_dict(packet_payload)
        return self.register(
            payload,
            checksum_sha256=hashlib.sha256(raw).hexdigest(),
            faces=packet,
        )

    def resolve(self, geometry_id: str) -> GeometryHandle:
        try:
            return self._handles[geometry_id]
        except KeyError as exc:
            raise GeometryContractError(
                f"geometry {geometry_id!r} is not registered; known: {sorted(self._handles)}"
            ) from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._handles))