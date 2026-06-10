"""Reference-element SBP operator library.

Operators are indexed by ``(basis, quad_basis, op_type, selector)`` with
``basis`` and ``quad_basis`` matched up to permutation. The ``selector``
disambiguates multiple entries with the same first three keys. Operators with a
non-``None`` ``name`` can also be looked up directly by name.

Add new operators to ``OPERATOR_ENTRIES`` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .operators import Operator, canonical_basis_key


@dataclass(frozen=True)
class OperatorSpec:
    """Reference operator lookup key.

    Operators are selected either by unique name or by basis, quadrature basis,
    operator type, and selector. Basis lists are matched up to permutation
    during lookup.
    """

    basis: list[str] | tuple[str, ...] | str | None = None
    quad_basis: list[str] | tuple[str, ...] | None = None
    op_type: str | None = None
    selector: int = 0
    name: str | None = None

    def __post_init__(self) -> None:
        name = self.name
        basis = self.basis
        if (
            isinstance(basis, str)
            and self.quad_basis is None
            and self.op_type is None
            and name is None
        ):
            name = basis
            basis = None
            object.__setattr__(self, "name", name)
            object.__setattr__(self, "basis", None)

        if name is not None:
            if not isinstance(name, str):
                raise TypeError("name must be a string")
            if (
                basis is not None
                or self.quad_basis is not None
                or self.op_type is not None
            ):
                raise ValueError(
                    "OperatorSpec accepts either name or "
                    "basis/quad_basis/op_type, not both"
                )
            if self.selector != 0:
                raise ValueError(
                    "selector is only valid for basis/quad_basis/op_type lookup"
                )
            return

        if basis is None or self.quad_basis is None or self.op_type is None:
            raise ValueError(
                "OperatorSpec requires either name or basis, quad_basis, and op_type"
            )
        if not isinstance(self.selector, int) or isinstance(self.selector, bool):
            raise TypeError("selector must be an integer")
        if not isinstance(self.op_type, str):
            raise TypeError("op_type must be a string")

        object.__setattr__(self, "basis", _normalize_basis_for_spec(basis, "basis"))
        object.__setattr__(
            self,
            "quad_basis",
            _normalize_basis_for_spec(self.quad_basis, "quad_basis"),
        )


def _normalize_basis_for_spec(
    basis: list[str] | tuple[str, ...] | str,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(basis, (list, tuple)) or any(
        not isinstance(item, str) for item in basis
    ):
        raise TypeError(f"{field_name} must be a list or tuple of strings")
    return tuple(basis)


def _entry(
    *,
    name: str | None = None,
    basis: list[str],
    quad_basis: list[str],
    op_type: str,
    selector: int,
    interval: tuple[float, float] | list[float] | np.ndarray,
    nodes: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    tL: np.ndarray,
    tR: np.ndarray,
) -> dict[str, Any]:
    return {
        "name": name,
        "basis": basis,
        "quad_basis": quad_basis,
        "op_type": op_type,
        "selector": selector,
        # Keep the reference interval explicit in each tabulated operator.
        "interval": np.asarray(interval, dtype=float).copy(),
        "nodes": nodes,
        "D": D,
        "H": H,
        "tL": tL,
        "tR": tR,
    }


OPERATOR_ENTRIES: tuple[dict[str, Any], ...] = (
    _entry(
        name="LGLp2",
        basis=["1", "x", "x^2"],
        quad_basis=["1", "x", "x^2", "x^3"],
        op_type="closed",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-1.0, 0.0, 1.0]),
        D=np.array(
            [
                [-1.5, 2.0, -0.5],
                [-0.5, 0.0, 0.5],
                [0.5, -2.0, 1.5],
            ]
        ),
        H=np.array([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]),
        tL=np.array([1.0, 0.0, 0.0]),
        tR=np.array([0.0, 0.0, 1.0]),
    ),
    _entry(
        name="LGp2",
        basis=["1", "x", "x^2"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "x^5"],
        op_type="open",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)]),
        D=np.array(
            [
                [-np.sqrt(15.0)/2.0, 2.0*np.sqrt(15.0)/3.0, -np.sqrt(15.0)/6.0],
                [-np.sqrt(15.0)/6.0, 0.0, np.sqrt(15.0)/6.0],
                [np.sqrt(15.0)/6.0, -2.0*np.sqrt(15.0)/3.0, np.sqrt(15.0)/2.0],
            ]
        ),
        H=np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]),
        tL=np.array([(5.0 + np.sqrt(15.0))/6.0, -2.0/3.0, (5.0 - np.sqrt(15.0))/6.0]),
        tR=np.array([(5.0 - np.sqrt(15.0))/6.0, -2.0/3.0, (5.0 + np.sqrt(15.0))/6.0]),
    ),
    _entry(
        name="RadauRp2",
        basis=["1", "x", "x^2"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4"],
        op_type="half-open-right",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-1.0, -0.2898979485566356, 0.6898979485566356]),
        D=np.array([
                        [-2.0, 2.4288690166235205, -0.4288690166235206],
                        [-0.816496580927726, 0.3876275643042055, 0.4288690166235206],
                        [0.816496580927726, -2.4288690166235205, 1.6123724356957945],
                    ]),
        H=np.array([2.0/9.0, 1.0249716523768433, 0.7528061254009346]),
        tL=np.array([1.0, 0.0, 0.0]),
        tR=np.array([1.0/3.0, -0.8914115380582557, 1.5580782047249224]),
    ),
    _entry(
        name="LGp3",
        basis=["1", "x", "x^2, x^3"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "x^5", "x^6", "x^7"],
        op_type="open",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]),
        D=np.array([
                        [-3.3320002363522816, 4.8601544156851961, -2.1087823484951791, 0.5806281691622645],
                        [-0.7575576147992339, -0.3844143922232086, 1.4706702312807167, -0.3286982242582743],
                        [0.3286982242582743, -1.4706702312807167, 0.3844143922232086, 0.7575576147992339],
                        [-0.5806281691622645, 2.1087823484951791, -4.8601544156851961, 3.3320002363522816],
                    ]),
        H=np.array([0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539]),
        tL=np.array([1.5267881254572668, -0.8136324494869273, 0.4007615203116504, -0.1139171962819899]),
        tR=np.array([-0.1139171962819899, 0.4007615203116504, -0.8136324494869273, 1.5267881254572668]),
    ),
    _entry(
        name="LGLp3",
        basis=["1", "x", "x^2, x^3"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "x^5"],
        op_type="closed",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-1.0, -0.4472135954999579, 0.4472135954999579, 1.0]),
        D=np.array([
                        [-3.0, 4.0450849718747373, -1.545084971874737, 0.5],
                        [-0.8090169943749475, 0.0, 1.1180339887498949, -0.3090169943749474],
                        [0.3090169943749474, -1.1180339887498949, 0.0, 0.8090169943749475],
                        [-0.5, 1.545084971874737, -4.0450849718747373, 3.0],
                    ]),
        H=np.array([1.0/6.0, 5.0/6.0, 5.0/6.0, 1.0/6.0]),
        tL=np.array([1.0, 0.0, 0.0, 0.0]),
        tR=np.array([0.0, 0.0, 0.0, 1.0]),
    ),
    _entry(
        name="RadauRp3",
        basis=["1", "x", "x^2, x^3"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "x^5", "x^6"],
        op_type="half-open-right",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-1.0, -0.5753189235216941, 0.1810662711185306, 0.8228240809745921]),
        D=np.array([
                        [-3.75, 4.7935967957376917, -1.3502656444120271, 0.3066688486743356],
                        [-1.1566785437711786, 0.3173960475776094, 1.0356811085889206, -0.1963986123953511],
                        [0.5309238658498484, -1.6876714615931911, 0.6105500144473459, 0.5461975812959968],
                        [-0.9813881792215268, 2.6047041188063171, -4.4453698775598349, 2.8220539379750447],
                    ]),
        H=np.array([0.125, 0.6576886399601195, 0.7763869376863438, 0.4409244223535368]),
        tL=np.array([1.0, 0.0, 0.0, 0.0]),
        tR=np.array([-0.25, 0.6461389554268266, -0.9736765952010225, 1.5775376397741958]),
    ),
    _entry(
        name=None, # unique LGp2(minimal) + e^x
        basis=["1", "x", "x^2, e^x"],
        quad_basis=["1", "x", "x^2", "x^3", "e^x", "x e^x", "x^2 e^x", "e^{2x}"],
        op_type="open",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-0.8425308036500644, -0.2778651467476725, 0.4000735152781082, 0.8782490795710493]),
        D=np.array([
                        [-2.9478805654298261, 4.2613237652942573, -1.7857948348726012, 0.4723516350081702],
                        [-0.7341010664266771, -0.3372939061271039, 1.3659765982566983, -0.2945816257029173],
                        [0.3569871824360631, -1.5840051273286317, 0.4537967147767736, 0.7732212301157952],
                        [-0.6895028653396812, 2.4911552531989805, -5.6330301446394557, 3.8313777567801566],
                    ]),
        H=np.array([0.3906821775619106, 0.6813432335603375, 0.6200850567800854, 0.3078895320976664]),
        tL=np.array([1.5224487539548868, -0.7982997281563335, 0.3793074691855646, -0.1034564949841178]),
        tR=np.array([-0.1203387361977243, 0.4214933150748657, -0.8406302870334269, 1.5394757081562855]),
    ),
     _entry(
        name=None, # unique LGp2(full) + e^x, optimized for x^3 and e^2x with default weights
        basis=["1", "x", "x^2, e^x"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "x^5", "e^x", "x e^x", "x^2 e^x", "e^{2x}"],
        op_type="open",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-0.8979508515330211, -0.5059974223007856, 0.0452992590334787, 0.5703118766229123, 0.914142595134012]),
        D=np.array([
                        [-3.9721395897622651, 5.1643767926805504, -1.3279874676202836, 0.051080465412216, 0.0846697992897825],
                        [-1.0703760867141752, -0.2950143551479512, 1.7589618737725057, -0.4934961004930791, 0.0999246685826997],
                        [0.2959454525599143, -1.4244908465166628, -0.0085688565299916, 1.4371625562923216, -0.3000483058055815],
                        [-0.0615316481991663, 0.4439666780373332, -1.8825348853205899, 0.3392196635849275, 1.1608801918974954],
                        [-0.3560577675657881, 0.4797463809162799, 1.2229898008161979, -6.1038281691756096, 4.7571497550089195],
                    ]),
        H=np.array([0.2566531800429939, 0.5019355539130103, 0.5684202219909265, 0.4552203975749141, 0.2177706464781553]),
        tL=np.array([1.429373156123237, -0.5567429898242584, 0.1178461793671307, 0.0356221741386054, -0.0260985198047147]),
        tR=np.array([-0.0646769268857643, 0.1175005071967839, 0.064391764453675, -0.5568737375160367, 1.4396583927513422]),
    ),
     _entry(
        name=None, # unique LGp2(full) + e^x, optimized for x^3, x^4, and e^2x
        basis=["1", "x", "x^2, e^x"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "x^5", "e^x", "x e^x", "x^2 e^x", "e^{2x}"],
        op_type="open",
        selector=1,
        interval=(-1.0, 1.0),
        nodes=np.array([-0.8979508515330211, -0.5059974223007856, 0.0452992590334787, 0.5703118766229123, 0.914142595134012]),
        D=np.array([
                        [-4.4110421625764404, 6.3689870224972349, -2.8044670357917458, 1.1197654230372898, -0.2732432471663383],
                        [-0.9752321266605998, -0.556146083884323, 2.079028607839025, -0.7251623636142712, 0.1775119663201688],
                        [0.3058160939734251, -1.4515817690996149, 0.0246362356123278, 1.4131285071567834, -0.2919990676429213],
                        [-0.2053448362035758, 0.8386757733654404, -2.3663261567197034, 0.6893907499446551, 1.0436044696131839],
                        [0.3386461486077487, -1.4269351834013297, 3.5599916816258612, -7.7953644489873444, 5.3236618021550646],
                    ]),
        H=np.array([0.2566531800429939, 0.5019355539130103, 0.5684202219909265, 0.4552203975749141, 0.2177706464781553]),
        tL=np.array([1.5052032448063586, -0.7648659451532743, 0.3729405523059333, -0.1490166959218304, 0.0357388439628128]),
        tR=np.array([0.037693653546111, -0.1634653746612822, 0.4087690357473458, -0.8061361164294821, 1.5231388017973073]),
    ),
    _entry(
        name=None, # unique RadauRp2 + sqrt(1-x) (right-open)
        basis=["1", "x", "x^2", "sqrt(1-x)"],
        quad_basis=["1", "x", "x^2", "x^3", "1/sqrt(1-x)", "x/sqrt(1-x)", "x^2/sqrt(1-x)"],
        op_type="half-open-right",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-1.0, -0.2597302979996616, 0.6508370907900723, 0.9833265234153543]),
        D=np.array([
                        [-2.0245897836466971, 2.6457101870376456, -0.8221177929890279, 0.2009973895966339],
                        [-0.69336683794337, 0.1029803506448331, 0.7433692278800592, -0.1529827405805379],
                        [0.5025301566375941, -1.7473900039872259, 0.5276067701407721, 0.7172530772062751],
                        [-3.4821275601923078, 10.3663120245466516, -20.992303039148382, 14.1081185748064755],
                    ]),
        H=np.array([0.2408833006290572, 1.0230257924979749, 0.653324877371268, 0.0827660295017]),
        tL=np.array([1.0, 0.0, 0.0, 0.0]),
        tR=np.array([-0.1569084477988031, 0.4590240839654271, -0.8302994982564673, 1.5281838620898434]),
    ),
    _entry(
        name=None, # unique LGp2 + sqrt(1-x) (open)
        basis=["1", "x", "x^2", "sqrt(1-x)"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "1/sqrt(1-x)", "x/sqrt(1-x)", "x^2/sqrt(1-x)"],
        op_type="open",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-0.7515992105845385, 0.0524697085059223, 0.7570646167026761, 0.9887923741992989]),
        D=np.array([
                        [-1.974781067551294, 2.9346324867448725, -1.3415335496316325, 0.3816821304377079],
                        [-0.5317071821838553, -0.3711107102544478, 1.1774941331259428, -0.2746762406866828],
                        [0.3800209562871011, -1.8637005802006288, 0.3609400179235328, 1.1227396059861507],
                        [-2.5956137736855003, 10.6815842253018332, -27.9812186362110751, 19.8952481846157276],
                    ]),
        H=np.array([0.605073006769668, 0.8717326365880017, 0.4673490752457602, 0.0558452813965701]),
        tL=np.array([1.5478503754839972, -0.8635391726922363, 0.4465494657445304, -0.1308606685362913]),
        tR=np.array([-0.0778931856826916, 0.314135746042504, -0.7326502525408887, 1.4964076921810763]),
    ),
    _entry(
        name='SQRTp1', 
        basis=["1", "sqrt(1-x)", "x"],
        quad_basis=["1", "x", "1/sqrt(1-x)", "x/sqrt(1-x)", "x^2/sqrt(1-x)"],
        op_type="half-open-right",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-1.0, 0.1680816411546915, 0.9519183588453085]),
        D=np.array([
                        [-1.0, 1.2144345083117603, -0.2144345083117603],
                        [-0.6329931618554521, 0.3005102572168219, 0.3324829046386302],
                        [2.6329931618554521, -7.8324829046386304, 5.1994897427831779],
                    ]),
        H=np.array([0.4444444444444444, 1.322108831729595, 0.2334467238259604]),
        tL=np.array([1.0, 0.0, 0.0]),
        tR=np.array([0.3333333333333333, -0.8914115380582557, 1.5580782047249224]),
    ),
    _entry(
        name='SQRTp1.5', 
        basis=["1", "sqrt(1-x)", "x", "x sqrt(1-x)"],
        quad_basis=["1", "x", "x^2", "x^3", "1/sqrt(1-x)", "x/sqrt(1-x)", "x^2/sqrt(1-x)"],
        op_type="half-open-right",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-1.0, -0.2597302979994637, 0.6508370907903757, 0.9833265234154601]),
        D=np.array([
                        [-1.9160251471689218, 2.4281182497627749, -0.6562177267888575, 0.1441246241950045],
                        [-0.7616180198710388, 0.239773577047239, 0.6390730595560477, -0.1172286167322477],
                        [0.6725623479205176, -2.0881790039852417, 0.7874367912713751, 0.6281798647933488],
                        [-5.7518582752105436, 14.9154462490812492, -24.4607296659169968, 15.2971416920462921],
                    ]),
        H=np.array([0.2408833006291285, 1.0230257924982096, 0.6533248773711968, 0.0827660295014652]),
        tL=np.array([1.0, 0.0, 0.0, 0.0]),
        tR=np.array([-0.2773500981126146, 0.7004206645707738, -1.0143490968053748, 1.5912785303472157]),
    ),
    _entry(
        name='SQRTp2alt', 
        basis=["1", "sqrt(1-x)", "x", "x sqrt(1-x)", "x^2"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "1/sqrt(1-x)", "x/sqrt(1-x)", "x^2/sqrt(1-x)", "x^3/sqrt(1-x)"],
        op_type="half-open-right",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-1.0, -0.4893557343503926, 0.3062895017761031, 0.8419071261901024, 0.9932151597979015]),
        D=np.array([
                        [-3.045544725589981, 3.9570103518906672, -1.2607836999993693, 0.4622504992590605, -0.1129324255603776],
                        [-0.9744347701124999, 0.1985493056143163, 1.0103846095071245, -0.3051019643488509, 0.0706028193399098],
                        [0.4981540856631933, -1.6211539083866608, 0.4028838607225658, 0.8842627831861973, -0.1641468211852955],
                        [-0.9306520102122409, 2.4944186725060846, -4.5057611546704424, 1.6652130995769427, 1.2767813927996563],
                        [10.7134711302001264, -27.198691160288277, 39.4113138581198328, -60.1614315250974414, 37.2353376970657592],
                    ]),
        H=np.array([0.1563564219528015, 0.757781645625746, 0.7321925765285159, 0.3195563944608329, 0.0341129614321037]),
        tL=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        tR=np.array([0.2182178902359924, -0.5485563226256093, 0.7680996966855425, -1.0316292881745417, 1.593868023878616]),
    ),
    _entry(
        name='SQRTp2', 
        basis=["1", "sqrt(1-x)", "x", "x sqrt(1-x)", "x^2"],
        quad_basis=["1", "x", "x^2", "x^3", "1/sqrt(1-x)", "x/sqrt(1-x)", "x^2/sqrt(1-x)", "x^3/sqrt(1-x)", "x^4/sqrt(1-x)"],
        op_type="half-open-right",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-1.0, -0.4800261819876617, 0.3188444144949214, 0.8467158915995232, 0.9934782215722294]),
        D=np.array([
                        [-3.0, 3.9182394717123468, -1.2805599683378841, 0.482416677291199, -0.1200961806656619],
                        [-0.9493020088016044, 0.168915930706207, 1.0229096450108441, -0.3173492517096948, 0.0748256847942481],
                        [0.4823147391848416, -1.5902099322644938, 0.367023342860243, 0.9136612522643081, -0.1727894020448986],
                        [-0.8948679698043009, 2.4297450416369215, -4.4997796262421046, 1.6309583727155783, 1.3339441816939057],
                        [10.2507441283099521, -26.3610966018599129, 39.1572952852142464, -61.3800451653822563, 38.3331023537179689],
                    ]),
        H=np.array([0.16, 0.7676917205342506, 0.7279159014700353, 0.3115657885567628, 0.0328265894389513]),
        tL=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        tR=np.array([0.2, -0.5092648848477427, 0.7309748661597875, -1.0081178814983729, 1.5864079001863283]),
    ),
    _entry(
        name='SQRTp2.5', 
        basis=["1", "sqrt(1-x)", "x", "x sqrt(1-x)", "x^2", "x^2 sqrt(1-x)"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "1/sqrt(1-x)", "x/sqrt(1-x)", "x^2/sqrt(1-x)", "x^3/sqrt(1-x)", "x^4/sqrt(1-x)", "x^5/sqrt(1-x)"],
        op_type="half-open-right",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-1.0, -0.6252779830714585, 0.0326588889773979, 0.6163561311775322, 0.9215813726840237, 0.9968303505630832]),
        D=np.array([
                        [-4.375, 5.7781771029033422, -2.0235762748576067, 0.9014328703870168, -0.3799908258909944, 0.0989571274582418],
                        [-1.2358294546511095, 0.153819840423574, 1.4571313065669798, -0.5324246300124638, 0.2110160495994185, -0.0537131119263988],
                        [0.4788606107228112, -1.612205997088292, 0.2584403755317691, 1.1477229483177616, -0.3573529630196436, 0.0845350255355938],
                        [-0.5011709990172168, 1.3840219770146109, -2.6964978027128974, 0.6516460194381163, 1.430871207669556, -0.268870402392169],
                        [1.2914514408223265, -3.3531483275530065, 5.1323092940258652, -8.7468760599271462, 3.1880180584220343, 2.4882455942099262],
                        [-17.2078983747363168, 43.6710238565418862, -62.1194932959274695, 84.0952000614779536, -127.3119079535405547, 78.8730757061845082],
                    ]),
        H=np.array([0.1111111111111111, 0.5762898483542528, 0.6751388966474732, 0.4563056180404707, 0.1651041162463372, 0.016050409600355]),
        tL=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        tR=np.array([-0.1666666666666667, 0.4210577454734133, -0.5907336963229323, 0.7711676077783899, -1.0260164756115118, 1.5911914853493074]),
    ),
    _entry(
        name=None, 
        basis=["1", "x", "x^2", "(1-x)^{3/2}"],
        quad_basis=["1", "sqrt(1-x)", "x", "x^2", "x^3", "x sqrt(1-x)", "x^2 sqrt(1-x)"],
        op_type="half-open-right",
        selector=0,
        interval=(-1.0, 1.0),
        nodes=np.array([-1.0, -0.3689393849936914, 0.5, 0.9403679564222629]),
        D=np.array([
                        [-2.4750000000000001, 3.3077430354649988, -1.2, 0.3672569645350013],
                        [-0.765653411439156, 0.044147760148184, 0.9715656022950613, -0.2500599510040893],
                        [0.4125, -1.4423468910411352, 0.2, 0.8298468910411353],
                        [-1.109346588560844, 3.2250599510040892, -6.9715656022950609, 4.8558522398518162],
                    ]),
        H=np.array([0.2, 0.9008669984965653, 0.7111111111111111, 0.1880218903923236]),
        tL=np.array([1.0, 0.0, 0.0, 0.0]),
        tR=np.array([-0.1, 0.2820328355884853, -0.5333333333333333, 1.351300497744848]),
    ),
)


def operator_lookup_key(
    basis: list[str] | tuple[str, ...],
    quad_basis: list[str] | tuple[str, ...],
    op_type: str,
    selector: int,
) -> tuple[tuple[str, ...], tuple[str, ...], str, int]:
    return (
        canonical_basis_key(list(basis)),
        canonical_basis_key(list(quad_basis)),
        op_type,
        selector,
    )


_OPERATORS: list[Operator] | None = None
_OPERATOR_INDEX: dict[
    tuple[tuple[str, ...], tuple[str, ...], str, int], Operator
] | None = None
_OPERATOR_NAME_INDEX: dict[str, Operator] | None = None


def all_operators() -> list[Operator]:
    global _OPERATORS
    if _OPERATORS is None:
        _OPERATORS = [Operator(**entry) for entry in OPERATOR_ENTRIES]
    return list(_OPERATORS)


def _operator_index() -> dict[tuple[tuple[str, ...], tuple[str, ...], str, int], Operator]:
    global _OPERATOR_INDEX
    if _OPERATOR_INDEX is None:
        index: dict[tuple[tuple[str, ...], tuple[str, ...], str, int], Operator] = {}
        for operator in all_operators():
            key = operator_lookup_key(
                operator.basis, operator.quad_basis, operator.op_type, operator.selector
            )
            if key in index:
                raise ValueError(f"Duplicate operator entry for {key}")
            index[key] = operator
        _OPERATOR_INDEX = index
    return _OPERATOR_INDEX


def _operator_name_index() -> dict[str, Operator]:
    global _OPERATOR_NAME_INDEX
    if _OPERATOR_NAME_INDEX is None:
        index: dict[str, Operator] = {}
        for operator in all_operators():
            if operator.name is None:
                continue
            if operator.name in index:
                raise ValueError(f"Duplicate operator name {operator.name!r}")
            index[operator.name] = operator
        _OPERATOR_NAME_INDEX = index
    return _OPERATOR_NAME_INDEX


def operator_names() -> list[str]:
    """Return sorted names for operators that have a non-None name."""
    return sorted(_operator_name_index())


def get_operator_by_name(name: str) -> Operator:
    """Look up a built-in reference operator by unique name."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")

    try:
        return _operator_name_index()[name]
    except KeyError as exc:
        raise KeyError(
            f"No operator named {name!r}. Available names: {operator_names()}"
        ) from exc


def selectors_for(
    basis: list[str] | tuple[str, ...],
    quad_basis: list[str] | tuple[str, ...],
    op_type: str,
) -> list[int]:
    """Return sorted selector indices available for ``(basis, quad_basis, op_type)``."""
    key = operator_lookup_key(basis, quad_basis, op_type, 0)[:3]
    selectors = [
        op.selector for op_key, op in _operator_index().items() if op_key[:3] == key
    ]
    return sorted(selectors)


def get_operator(
    basis: list[str] | tuple[str, ...] | str | None = None,
    quad_basis: list[str] | tuple[str, ...] | None = None,
    op_type: str | None = None,
    selector: int = 0,
    *,
    name: str | None = None,
) -> Operator:
    """Look up a built-in reference operator.

    Operators can be looked up by unique ``name`` or by exactly
    ``(basis, quad_basis, op_type, selector)``, with ``basis`` and
    ``quad_basis`` matched up to permutation.
    """
    if (
        isinstance(basis, str)
        and quad_basis is None
        and op_type is None
        and name is None
    ):
        if selector != 0:
            raise ValueError(
                "selector is only valid for basis/quad_basis/op_type lookup"
            )
        return get_operator_by_name(basis)

    if name is not None:
        if basis is not None or quad_basis is not None or op_type is not None:
            raise ValueError(
                "get_operator accepts either name or basis/quad_basis/op_type, not both"
            )
        if selector != 0:
            raise ValueError(
                "selector is only valid for basis/quad_basis/op_type lookup"
            )
        return get_operator_by_name(name)

    if basis is None or quad_basis is None or op_type is None:
        raise TypeError(
            "get_operator requires either a name or basis, quad_basis, and op_type"
        )

    key = operator_lookup_key(basis, quad_basis, op_type, selector)
    try:
        return _operator_index()[key]
    except KeyError as exc:
        available = selectors_for(basis, quad_basis, op_type)
        raise KeyError(
            f"No operator for basis={list(basis)}, quad_basis={list(quad_basis)}, "
            f"op_type={op_type}, selector={selector}. Available selectors: {available}"
        ) from exc


def operator_from_spec(spec: OperatorSpec) -> Operator:
    if spec.name is not None:
        return get_operator_by_name(spec.name)
    assert spec.basis is not None
    assert spec.quad_basis is not None
    assert spec.op_type is not None
    return get_operator(spec.basis, spec.quad_basis, spec.op_type, spec.selector)
