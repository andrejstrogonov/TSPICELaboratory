"""
Модуль графовой нейросети (GNN) для ускоренного физического моделирования полупроводниковых приборов.
Реализует конвейер A-8 (G1-G6): кодирование тетраэдрального меша в граф, Message Passing (C^inf),
физически-информированную функцию потерь (PINN), активное обучение с FEM-оракулом,
вычисление якобиана через autograd и экспорт в ONNX (opset 17) для Rust ort::Session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATConv, MessagePassing, SAGEConv, global_add_pool, global_mean_pool
    HAS_PYG = True
except ImportError as exc:
    HAS_PYG = False
    _PYG_IMPORT_ERROR = exc


# =====================================================================
# Константы и справочники кодирования (G1)
# =====================================================================

MATERIAL_TYPES: Tuple[str, ...] = (
    "channel_body",
    "gate_oxide",
    "buried_oxide",
    "barrier",
    "metal_contact",
)

BOUNDARY_TYPES: Tuple[str, ...] = (
    "interior",
    "dirichlet",
    "robin_gate",
    "neumann",
)

DEVICE_FAMILIES: Tuple[str, ...] = (
    "planar_mosfet",
    "fd_soi",
    "finfet",
    "gaa_nanosheet",
    "gaa_nanowire",
    "tfet",
    "hemt",
    "memristor",
    "memtransistor",
    "cad_custom",
)

Q_ELECTRON = 1.602176634e-19
EPS_0 = 8.8541878128e-12
K_B = 1.380649e-23


@dataclass
class OperatingPoint:
    """Глобальный контекст рабочей точки u (G1)."""
    vg: float
    vd: float
    vb: float = 0.0
    temperature_k: float = 300.0
    device_family: str = "planar_mosfet"
    vdd_scale: float = 1.2

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        family_vec = [1.0 if self.device_family == f else 0.0 for f in DEVICE_FAMILIES]
        vdd = max(abs(self.vdd_scale), 1e-3)
        # Признаки рабочей точки + индикатор режима (подпорог / линейный / насыщение)
        is_sat = 1.0 if abs(self.vd) >= max(abs(self.vg) - 0.35, 0.1) else 0.0
        base = [
            self.vg / vdd,
            self.vd / vdd,
            self.vb / vdd,
            (self.temperature_k - 300.0) / 100.0,
            is_sat,
        ]
        return torch.tensor(base + family_vec, dtype=torch.float32, device=device)


@dataclass
class QualityGateReport:
    """Отчёт проверки гладкости и монотонности ВАХ (НФТ.5.2)."""
    passed: bool
    is_monotonic: bool
    max_second_derivative_jump: float
    spurious_extrema_count: int
    details: List[str] = field(default_factory=list)


# =====================================================================
# G1 · Графовое кодирование меша / геометрии
# =====================================================================

class GraphMeshEncoder:
    """
    Преобразует тетраэдральную сетку (или облако точек геометрии) в объект torch_geometric.data.Data.
    """

    def __init__(self, k_neighbors: int = 8) -> None:
        if not HAS_PYG:
            raise ImportError(
                f"Для работы GraphMeshEncoder требуется torch_geometric: {_PYG_IMPORT_ERROR}"
            )
        self.k_neighbors = k_neighbors

    def encode(
        self,
        coords: np.ndarray,
        materials: List[str],
        eps_r: np.ndarray,
        mobility: np.ndarray,
        nd_plus: np.ndarray,
        na_minus: np.ndarray,
        boundary_types: List[str],
        contact_potentials: np.ndarray,
        op_point: OperatingPoint,
        tetrahedra: Optional[np.ndarray] = None,
        node_volumes: Optional[np.ndarray] = None,
        targets: Optional[Dict[str, np.ndarray]] = None,
    ) -> Data:
        """
        Формирует граф G = (V, E, u):
          - x: [N, D_node]
          - edge_index: [2, E]
          - edge_attr: [E, 2] -> (длина ребра d_ij, проводимость перехода g_ij = eps_ij * A_ij / d_ij)
          - u: [1, D_global]
        """
        num_nodes = coords.shape[0]
        if num_nodes == 0:
            raise ValueError("Пустой набор узлов сетки для графового кодирования.")

        # 1. Нормированные координаты (x, y, z)
        coord_min = coords.min(axis=0, keepdims=True)
        coord_span = np.maximum(coords.max(axis=0, keepdims=True) - coord_min, 1e-12)
        coords_norm = (coords - coord_min) / coord_span

        # 2. One-hot кодирование материалов и типов границ
        mat_idx = {m: i for i, m in enumerate(MATERIAL_TYPES)}
        mat_one_hot = np.zeros((num_nodes, len(MATERIAL_TYPES)), dtype=np.float32)
        for i, m in enumerate(materials):
            mat_one_hot[i, mat_idx.get(m, 0)] = 1.0

        bc_idx = {b: i for i, b in enumerate(BOUNDARY_TYPES)}
        bc_one_hot = np.zeros((num_nodes, len(BOUNDARY_TYPES)), dtype=np.float32)
        for i, b in enumerate(boundary_types):
            bc_one_hot[i, bc_idx.get(b, 0)] = 1.0

        # 3. Логарифмическое масштабирование легирования и физических свойств
        log_nd = np.log10(np.maximum(nd_plus, 1.0)) / 22.0
        log_na = np.log10(np.maximum(na_minus, 1.0)) / 22.0
        eps_norm = eps_r / 25.0
        mob_norm = np.log10(np.maximum(mobility, 1e-3)) / 4.0
        v_contact_norm = contact_potentials / max(abs(op_point.vdd_scale), 1e-3)

        node_features = np.concatenate(
            [
                coords_norm.astype(np.float32),
                mat_one_hot,
                eps_norm[:, None].astype(np.float32),
                mob_norm[:, None].astype(np.float32),
                log_nd[:, None].astype(np.float32),
                log_na[:, None].astype(np.float32),
                bc_one_hot,
                v_contact_norm[:, None].astype(np.float32),
            ],
            axis=1,
        )

        # 4. Построение рёбер E (из тетраэдров или по k-NN)
        if tetrahedra is not None and len(tetrahedra) > 0:
            edge_pairs = self._edges_from_tetrahedra(tetrahedra)
        else:
            edge_pairs = self._edges_from_knn(coords, self.k_neighbors)

        src = edge_pairs[:, 0]
        dst = edge_pairs[:, 1]
        diff = coords[dst] - coords[src]
        dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)

        # Оценка эффективного сечения A_ij и проводимости перехода g_ij = eps_ij * A_ij / d_ij
        eps_ij = 0.5 * (eps_r[src] + eps_r[dst]) * EPS_0
        effective_area = dist ** 2
        conductance_ij = (eps_ij * effective_area) / dist

        dist_norm = dist / np.maximum(dist.max(), 1e-12)
        cond_norm = np.log10(np.maximum(conductance_ij, 1e-30)) / 20.0
        edge_attr = np.stack([dist_norm, cond_norm], axis=1).astype(np.float32)

        # 5. Объёмы ячеек Вороного/контрольных объёмов для PINN-интегралов
        if node_volumes is None:
            mean_spacing = float(np.median(dist))
            node_volumes = np.full((num_nodes,), mean_spacing ** 3, dtype=np.float32)

        data = Data(
            x=torch.from_numpy(node_features),
            edge_index=torch.from_numpy(edge_pairs.T.astype(np.int64)),
            edge_attr=torch.from_numpy(edge_attr),
            pos=torch.from_numpy(coords.astype(np.float32)),
            eps_r=torch.from_numpy(eps_r.astype(np.float32)),
            net_doping=torch.from_numpy((nd_plus - na_minus).astype(np.float32)),
            node_volume=torch.from_numpy(node_volumes.astype(np.float32)),
            bc_mask=torch.from_numpy(bc_one_hot),
            bc_potential=torch.from_numpy(contact_potentials.astype(np.float32)),
            u=op_point.to_tensor().unsqueeze(0),
        )

        if targets is not None:
            # Узловые таргеты: [psi (В), log10(n), log10(p), T (К)]
            psi_t = targets["psi"]
            log_n_t = np.log10(np.maximum(targets["n"], 1.0))
            log_p_t = np.log10(np.maximum(targets["p"], 1.0))
            temp_t = targets.get("T", np.full_like(psi_t, op_point.temperature_k))
            data.y_node = torch.from_numpy(
                np.stack([psi_t, log_n_t, log_p_t, temp_t], axis=1).astype(np.float32)
            )
            # Интегральные таргеты: [Id, Qg, Qd, Qs]
            if all(k in targets for k in ("Id", "Qg", "Qd", "Qs")):
                data.y_integral = torch.tensor(
                    [[targets["Id"], targets["Qg"], targets["Qd"], targets["Qs"]]],
                    dtype=torch.float32,
                )

        return data

    @staticmethod
    def _edges_from_tetrahedra(tetrahedra: np.ndarray) -> np.ndarray:
        local_edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        pairs = []
        for a, b in local_edges:
            pairs.append(tetrahedra[:, [a, b]])
            pairs.append(tetrahedra[:, [b, a]])
        all_edges = np.vstack(pairs)
        return np.unique(all_edges, axis=0)

    @staticmethod
    def _edges_from_knn(coords: np.ndarray, k: int) -> np.ndarray:
        n = coords.shape[0]
        k_eff = min(k, max(n - 1, 1))
        diff = coords[:, None, :] - coords[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_mat, np.inf)
        neighbors = np.argpartition(dist_mat, kth=k_eff - 1, axis=1)[:, :k_eff]
        src = np.repeat(np.arange(n), k_eff)
        dst = neighbors.reshape(-1)
        edges = np.stack([src, dst], axis=1)
        reverse_edges = np.stack([dst, src], axis=1)
        return np.unique(np.vstack([edges, reverse_edges]), axis=0)


# =====================================================================
# G2 · Архитектура графовой сети (C^inf гладкость, Message Passing)
# =====================================================================

class SmoothPhysicsMPBlock(MessagePassing):
    """
    Слой передачи сообщений:
      m_ij = MLP_msg(h_i, h_j, e_ij, u_node)
      h_i' = h_i + MLP_upd(h_i, sum_{j in N(i)} m_ij, u_node)
    Использует SiLU/tanh и LayerNorm для C^inf-непрерывности производных dI/dV.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        global_dim: int,
        dropout: float = 0.05,
        activation: Literal["silu", "tanh"] = "silu",
    ) -> None:
        super().__init__(aggr="add")
        act_cls = nn.SiLU if activation == "silu" else nn.Tanh

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim + global_dim, hidden_dim),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            act_cls(),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + global_dim, hidden_dim),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u_node: torch.Tensor,
    ) -> torch.Tensor:
        agg_msg = self.propagate(edge_index, h=h, edge_attr=edge_attr, u_node=u_node)
        upd_in = torch.cat([h, agg_msg, u_node], dim=-1)
        h_next = self.norm(h + self.upd_mlp(upd_in))
        return h_next

    def message(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        edge_attr: torch.Tensor,
        u_node_i: torch.Tensor,
    ) -> torch.Tensor:
        msg_in = torch.cat([h_i, h_j, edge_attr, u_node_i], dim=-1)
        return self.msg_mlp(msg_in)


class SemiconductorGNN(nn.Module):
    """
    Универсальный GNN-суррогат для полупроводниковых приборов (A-8):
      - Узловой декодер: [psi (В), log10(n), log10(p), T (К)]
      - Интегральный декодер (permutation-invariant readout): [Id, Qg, Qd, Qs]
    """

    def __init__(
        self,
        node_in_dim: int = 18,
        edge_in_dim: int = 2,
        global_in_dim: int = 5 + len(DEVICE_FAMILIES),
        hidden_dim: int = 128,
        num_layers: int = 6,
        backbone: Literal["physics_mp", "sage", "gat"] = "physics_mp",
        dropout: float = 0.05,
        activation: Literal["silu", "tanh"] = "silu",
    ) -> None:
        super().__init__()
        if not HAS_PYG:
            raise ImportError(
                f"Для создания SemiconductorGNN требуется torch_geometric: {_PYG_IMPORT_ERROR}"
            )

        self.backbone_type = backbone
        self.dropout_rate = dropout
        act_cls = nn.SiLU if activation == "silu" else nn.Tanh

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim + global_in_dim, hidden_dim),
            act_cls(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            act_cls(),
        )

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if backbone == "physics_mp":
                self.layers.append(
                    SmoothPhysicsMPBlock(
                        hidden_dim=hidden_dim,
                        edge_dim=edge_in_dim,
                        global_dim=global_in_dim,
                        dropout=dropout,
                        activation=activation,
                    )
                )
            elif backbone == "sage":
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif backbone == "gat":
                self.layers.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=edge_in_dim))
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                raise ValueError(f"Неподдерживаемый backbone GNN: {backbone}")

        self.act = act_cls()

        # Узловой декодер: [psi, log10(n), log10(p), T]
        self.field_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_cls(),
            nn.Linear(hidden_dim // 2, 4),
        )

        # Внимание для permutation-invariant pooling
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Интегральный декодер: [Id, Qg, Qd, Qs]
        self.integral_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + global_in_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_cls(),
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        u_node = u[batch]
        h = self.node_encoder(torch.cat([x, u_node], dim=-1))

        for idx, layer in enumerate(self.layers):
            if self.backbone_type == "physics_mp":
                h = layer(h, edge_index, edge_attr, u_node)
            elif self.backbone_type == "sage":
                h_res = self.act(layer(h, edge_index))
                h_res = F.dropout(h_res, p=self.dropout_rate, training=self.training)
                h = self.norms[idx](h + h_res)
            else:  # gat
                h_res = self.act(layer(h, edge_index, edge_attr=edge_attr))
                h_res = F.dropout(h_res, p=self.dropout_rate, training=self.training)
                h = self.norms[idx](h + h_res)

        node_fields = self.field_head(h)

        # Permutation-invariant readout: attention-weighted sum + mean pool
        weights = torch.sigmoid(self.attn_gate(h))
        pooled_attn = global_add_pool(h * weights, batch) / (global_add_pool(weights, batch) + 1e-8)
        pooled_mean = global_mean_pool(h, batch)
        graph_repr = torch.cat([pooled_attn, pooled_mean, u], dim=-1)
        integral_outputs = self.integral_head(graph_repr)

        return node_fields, integral_outputs

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        data: Data,
        mc_samples: int = 12,
    ) -> Dict[str, torch.Tensor]:
        """
        Оценка эпистемической неуверенности через MC-Dropout (G4).
        """
        was_training = self.training
        self.train()  # Активируем dropout для стохастических проходов

        batch = getattr(data, "batch", None)
        field_preds = []
        integral_preds = []

        for _ in range(mc_samples):
            f_out, i_out = self.forward(data.x, data.edge_index, data.edge_attr, data.u, batch)
            field_preds.append(f_out)
            integral_preds.append(i_out)

        if not was_training:
            self.eval()

        fields_stack = torch.stack(field_preds, dim=0)
        integrals_stack = torch.stack(integral_preds, dim=0)

        rel_uncertainty = integrals_stack.std(dim=0) / (integrals_stack.mean(dim=0).abs() + 1e-9)

        return {
            "node_fields_mean": fields_stack.mean(dim=0),
            "node_fields_std": fields_stack.std(dim=0),
            "integral_mean": integrals_stack.mean(dim=0),
            "integral_std": integrals_stack.std(dim=0),
            "relative_uncertainty": rel_uncertainty.max(),
        }


# =====================================================================
# G3 · Физически-информированная функция потерь (PINN)
# =====================================================================

class PhysicsInformedLoss(nn.Module):
    """
    L_total = alpha*L_data + beta*L_poisson + gamma*L_continuity + delta*L_BC + zeta*L_charge
    """

    def __init__(
        self,
        alpha_data: float = 1.0,
        beta_poisson: float = 0.1,
        gamma_continuity: float = 0.05,
        delta_bc: float = 0.5,
        zeta_charge: float = 0.2,
    ) -> None:
        super().__init__()
        self.alpha = alpha_data
        self.beta = beta_poisson
        self.gamma = gamma_continuity
        self.delta = delta_bc
        self.zeta = zeta_charge

    def forward(
        self,
        pred_fields: torch.Tensor,
        pred_integrals: torch.Tensor,
        data: Data,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        psi = pred_fields[:, 0]
        log_n = torch.clamp(pred_fields[:, 1], 0.0, 24.0)
        log_p = torch.clamp(pred_fields[:, 2], 0.0, 24.0)

        n = torch.pow(10.0, log_n)
        p = torch.pow(10.0, log_p)

        # 1. L_data: ошибка по эталонным узловым и интегральным данным
        l_data = torch.tensor(0.0, device=psi.device)
        if hasattr(data, "y_node") and data.y_node is not None:
            l_data = l_data + F.mse_loss(pred_fields, data.y_node)
        if hasattr(data, "y_integral") and data.y_integral is not None:
            l_data = l_data + F.mse_loss(pred_integrals, data.y_integral)

        # 2. L_poisson: невязка графового лапласиана div(eps * grad(psi)) + rho = 0
        src, dst = data.edge_index[0], data.edge_index[1]
        dist = torch.clamp(torch.norm(data.pos[dst] - data.pos[src], dim=-1), min=1e-12)
        eps_edge = 0.5 * (data.eps_r[src] + data.eps_r[dst]) * EPS_0
        flux = eps_edge * (psi[dst] - psi[src]) / (dist ** 2)

        div_flux = torch.zeros_like(psi)
        div_flux.index_add_(0, src, flux)

        rho = Q_ELECTRON * (p - n + data.net_doping)
        # Масштабируем невязку для численной устойчивости (Тихонов-нормировка)
        poisson_res = (div_flux + rho) * 1e-12
        l_poisson = torch.mean(poisson_res ** 2)

        # 3. L_continuity: стационарный баланс рекомбинации SRH + Auger
        ni = 1.0e10  # Собственная концентрация Si при 300 К [см^-3]
        tau_srh = 1.0e-7
        c_auger = 1.0e-30
        r_srh = (n * p - ni ** 2) / (tau_srh * (n + p + 2.0 * ni))
        r_auger = c_auger * (n + p) * (n * p - ni ** 2)
        recomb_norm = (r_srh + r_auger) * 1e-22
        l_continuity = torch.mean(recomb_norm ** 2)

        # 4. L_BC: соблюдение граничных условий Дирихле на контактах
        dirichlet_mask = data.bc_mask[:, 1] > 0.5
        if torch.any(dirichlet_mask):
            l_bc = F.mse_loss(psi[dirichlet_mask], data.bc_potential[dirichlet_mask])
        else:
            l_bc = torch.tensor(0.0, device=psi.device)

        # 5. L_charge: сохранение заряда Уорда-Даттона (Qg + Qd + Qs == 0)
        q_sum = pred_integrals[:, 1] + pred_integrals[:, 2] + pred_integrals[:, 3]
        l_charge = torch.mean((q_sum * 1e15) ** 2)

        total = (
            self.alpha * l_data
            + self.beta * l_poisson
            + self.gamma * l_continuity
            + self.delta * l_bc
            + self.zeta * l_charge
        )

        metrics = {
            "total": float(total.detach().item()),
            "l_data": float(l_data.detach().item()),
            "l_poisson": float(l_poisson.detach().item()),
            "l_continuity": float(l_continuity.detach().item()),
            "l_bc": float(l_bc.detach().item()),
            "l_charge": float(l_charge.detach().item()),
        }
        return total, metrics


# =====================================================================
# G4 · Обучение и активное обучение с FEM-оракулом
# =====================================================================

class GNNTrainer:
    """
    Обучающий цикл с AdamW, CosineAnnealingLR, AMP и активным пополнением выборки.
    """

    def __init__(
        self,
        model: SemiconductorGNN,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        uncertainty_threshold: float = 0.03,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = PhysicsInformedLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        self.uncertainty_threshold = uncertainty_threshold
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_epoch(self, dataset: List[Data], batch_size: int = 8) -> Dict[str, float]:
        self.model.train()
        accum: Dict[str, float] = {}
        num_batches = 0

        for start in range(0, len(dataset), batch_size):
            batch_list = dataset[start : start + batch_size]
            batch = Batch.from_data_list(batch_list).to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred_fields, pred_integrals = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch
                )
                loss, metrics = self.criterion(pred_fields, pred_integrals, batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            num_batches += 1

        self.scheduler.step()
        return {k: v / max(num_batches, 1) for k, v in accum.items()}

    def active_learning_step(
        self,
        candidate_graphs: List[Data],
        dataset: List[Data],
        fem_oracle_fn: Callable[[Data], Data],
    ) -> int:
        """
        Проверяет неуверенность на кандидатах и при превышении порога
        вызывает точный FEM-решатель для дообучения (ФТ.5.4).
        """
        added = 0
        for cand in candidate_graphs:
            cand_dev = cand.clone().to(self.device)
            stats = self.model.predict_with_uncertainty(cand_dev, mc_samples=10)
            if float(stats["relative_uncertainty"].item()) > self.uncertainty_threshold:
                labeled_graph = fem_oracle_fn(cand)
                dataset.append(labeled_graph)
                added += 1
        return added


# =====================================================================
# G5 · Инференс, якобиан через autograd и гейты качества (НФТ.5.2)
# =====================================================================

def compute_iv_jacobian(
    model: SemiconductorGNN,
    base_graph: Data,
    vg: float,
    vd: float,
    vb: float = 0.0,
    temperature_k: float = 300.0,
    device_family: str = "planar_mosfet",
    vdd_scale: float = 1.2,
) -> Dict[str, float]:
    """
    Вычисляет Id, Qg и точные частные производные gm = dId/dVg, gds = dId/dVd,
    Cgg = dQg/dVg методом автоматического дифференцирования (backprop по графу).
    """
    model.eval()
    dev = next(model.parameters()).device
    graph = base_graph.clone().to(dev)

    v_vec = torch.tensor([vg, vd, vb], dtype=torch.float32, device=dev, requires_grad=True)
    vdd = max(abs(vdd_scale), 1e-3)
    is_sat = 1.0 if abs(vd) >= max(abs(vg) - 0.35, 0.1) else 0.0
    family_vec = torch.tensor(
        [1.0 if device_family == f else 0.0 for f in DEVICE_FAMILIES],
        dtype=torch.float32,
        device=dev,
    )
    extra = torch.tensor([(temperature_k - 300.0) / 100.0, is_sat], dtype=torch.float32, device=dev)
    u = torch.cat([v_vec / vdd, extra, family_vec], dim=0).unsqueeze(0)

    _, integrals = model(graph.x, graph.edge_index, graph.edge_attr, u, batch=None)
    id_val = integrals[0, 0]
    qg_val = integrals[0, 1]

    grad_id = torch.autograd.grad(id_val, v_vec, retain_graph=True)[0]
    grad_qg = torch.autograd.grad(qg_val, v_vec, retain_graph=False)[0]

    return {
        "Id": float(id_val.detach().item()),
        "Qg": float(qg_val.detach().item()),
        "gm": float(grad_id[0].detach().item()),
        "gds": float(grad_id[1].detach().item()),
        "gmb": float(grad_id[2].detach().item()),
        "Cgg": float(grad_qg[0].detach().item()),
        "Cgd": float(-grad_qg[1].detach().item()),
    }


def check_iv_quality_gates(
    vg_grid: np.ndarray,
    id_curve: np.ndarray,
    max_d2_jump_ratio: float = 5.0,
) -> QualityGateReport:
    """
    Проверка качества ВАХ перед допуском суррогата в схемный решатель (НФТ.5.2):
      1. Монотонность Id(Vg)
      2. C^2-гладкость (отсутствие резких осцилляций 2-й производной / эффекта Рунге)
      3. Отсутствие ложных локальных экстремумов
    """
    details: List[str] = []
    if len(vg_grid) < 5:
        return QualityGateReport(False, False, float("inf"), 0, ["Недостаточно точек для проверки ВАХ"])

    d1 = np.gradient(id_curve, vg_grid)
    is_monotonic = bool(np.all(d1 >= -1e-12))
    if not is_monotonic:
        details.append("Нарушена монотонность ВАХ (обнаружены участки отрицательной крутизны gm < 0).")

    # Подсчёт смены знака первой производной (ложные локальные экстремумы)
    signs = np.sign(d1)
    non_zero_signs = signs[signs != 0]
    spurious_extrema = int(np.sum(np.diff(non_zero_signs) != 0)) if len(non_zero_signs) > 1 else 0
    if spurious_extrema > 0:
        details.append(f"Обнаружены ложные локальные экстремумы ({spurious_extrema}) — эффект Рунге.")

    d2 = np.gradient(d1, vg_grid)
    d2_jumps = np.abs(np.diff(d2))
    median_jump = float(np.median(d2_jumps)) + 1e-15
    max_jump_ratio = float(np.max(d2_jumps) / median_jump)
    if max_jump_ratio > max_d2_jump_ratio:
        details.append(
            f"Нарушена C^2-гладкость: скачок второй производной в {max_jump_ratio:.2f} раз выше медианы."
        )

    passed = is_monotonic and spurious_extrema == 0 and (max_jump_ratio <= max_d2_jump_ratio)
    return QualityGateReport(
        passed=passed,
        is_monotonic=is_monotonic,
        max_second_derivative_jump=max_jump_ratio,
        spurious_extrema_count=spurious_extrema,
        details=details,
    )


# =====================================================================
# G6 · Гибридное сопряжение с физическим решателем и экспорт ONNX
# =====================================================================

class OnnxExportableGNN(nn.Module):
    """
    Обёртка над SemiconductorGNN со стандартными тензорными входами
    для экспорта в ONNX (opset 17) и загрузки в Rust ort::Session.
    """

    def __init__(self, model: SemiconductorGNN) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, edge_index, edge_attr, u, batch=None)


def export_gnn_to_onnx(
    model: SemiconductorGNN,
    sample_graph: Data,
    onnx_path: str,
    opset_version: int = 17,
) -> None:
    """
    Экспортирует обученный GNN-суррогат в ONNX с динамическим числом узлов и рёбер
    для инференса в Rust (tspice_solver::surrogate_onnx).
    """
    model.eval()
    wrapper = OnnxExportableGNN(model).cpu()
    g = sample_graph.clone().cpu()

    torch.onnx.export(
        wrapper,
        (g.x, g.edge_index, g.edge_attr, g.u),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["node_features", "edge_index", "edge_attr", "global_context"],
        output_names=["node_fields", "integral_outputs"],
        dynamic_axes={
            "node_features": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "edge_attr": {0: "num_edges"},
            "node_fields": {0: "num_nodes"},
        },
    )


@torch.no_grad()
def prepare_warm_start(
    model: SemiconductorGNN,
    graph: Data,
) -> Dict[str, np.ndarray]:
    """
    Формирует начальное приближение (psi_0, n_0, p_0, T_0) для передачи
    в решатели Newton / 3D FEM / Quantum-Poisson (ускорение на 3-5 итераций).
    """
    model.eval()
    dev = next(model.parameters()).device
    g = graph.to(dev)
    node_fields, integral_outputs = model(g.x, g.edge_index, g.edge_attr, g.u, batch=None)

    fields_np = node_fields.cpu().numpy()
    integrals_np = integral_outputs.squeeze(0).cpu().numpy()

    return {
        "psi_0": fields_np[:, 0],
        "n_0": np.power(10.0, np.clip(fields_np[:, 1], 0.0, 24.0)),
        "p_0": np.power(10.0, np.clip(fields_np[:, 2], 0.0, 24.0)),
        "T_0": fields_np[:, 3],
        "Id_est": float(integrals_np[0]),
        "Qg_est": float(integrals_np[1]),
        "Qd_est": float(integrals_np[2]),
        "Qs_est": float(integrals_np[3]),
    }
