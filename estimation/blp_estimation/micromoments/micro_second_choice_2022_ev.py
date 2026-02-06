import numpy as np
import pandas as pd
import pyblp


def build_second_choice_ev_moments_2022(product_data: pd.DataFrame):
    """
    2022 EV second-choice micro-moments (all vehicles)
    Moments:
      1) P(second is EV | first is EV)
      2) P(second is EV in same class | first is EV)
    """

    TARGETS_2022_EV = {
        "ev_given_ev": 0.52,
        "same_class_ev_given_ev": 0.33,
    }

    # ---------- 2. Build 2022 maps (keys are strings) ----------

    prod2022 = product_data.loc[product_data["market_ids"] == 2022].copy()
    prod2022_ids_str = prod2022["clustering_ids"].astype(str).to_numpy()

    ev_vals_2022 = pd.to_numeric(prod2022["ev"], errors="coerce").fillna(0.0)
    ev_map_2022 = dict(zip(prod2022_ids_str, ev_vals_2022))

    vt_vals_2022 = (
        prod2022["vehicle_type"].fillna("").str.lower().str.strip()
    )
    vt_map_2022 = dict(zip(prod2022_ids_str, vt_vals_2022))

    # ---------- 3. MicroDataset for 2022 second-choice EV moments ----------

    def _all_inside_pairs_weights_2022(t, products, agents):
        J = products.clustering_ids.size
        I = agents.size
        W = np.ones((J, J), dtype=np.float64)
        np.fill_diagonal(W, 0.0)
        W_full = np.zeros((J, J + 1), dtype=np.float64)
        W_full[:, 1:] = W
        return np.broadcast_to(W_full, (I, J, J + 1))

    def build_second_choice_dataset_2022():
        return pyblp.MicroDataset(
            name="SecondChoice2022_all",
            observations=1000,
            market_ids=[2022],
            compute_weights=_all_inside_pairs_weights_2022,
        )

    d2022 = build_second_choice_dataset_2022()

    # ---------- 4. MicroParts for EV second-choice behavior in 2022 ----------

    def make_ev_second_choice_parts_2022(dataset):
        tag = dataset.name

        def _get_ev_vectors(products):
            pids = np.asarray(products.clustering_ids)
            v_ev, v_suv, v_truck, v_van, v_car = [], [], [], [], []

            for pid in pids:
                pid_str = str(pid.item()) if isinstance(pid, np.ndarray) else str(pid)
                ev = float(ev_map_2022.get(pid_str, 0.0))
                vt = vt_map_2022.get(pid_str, "")

                is_suv = float(vt == "suv")
                is_truck = float(vt == "truck")
                is_van = float(vt == "van")
                is_car = float(vt == "car")

                v_ev.append(ev)
                v_suv.append(ev * is_suv)
                v_truck.append(ev * is_truck)
                v_van.append(ev * is_van)
                v_car.append(ev * is_car)

            return (
                np.array(v_ev, dtype=np.float64),
                np.array(v_suv, dtype=np.float64),
                np.array(v_truck, dtype=np.float64),
                np.array(v_van, dtype=np.float64),
                np.array(v_car, dtype=np.float64),
            )

        def _values_num_ev_ev(t, products, agents):
            v_ev, _, _, _, _ = _get_ev_vectors(products)
            J = v_ev.size
            I = agents.size
            mat = np.outer(v_ev, v_ev)
            out = np.zeros((I, J, J + 1), dtype=np.float64)
            out[:, :, 1:] = np.broadcast_to(mat, (I, J, J))
            return out

        def _values_den_first_ev(t, products, agents):
            v_ev, _, _, _, _ = _get_ev_vectors(products)
            J = v_ev.size
            I = agents.size
            mat = v_ev.reshape(J, 1) @ np.ones((1, J))
            out = np.zeros((I, J, J + 1), dtype=np.float64)
            out[:, :, 1:] = np.broadcast_to(mat, (I, J, J))
            return out

        def _values_num_sameclass_ev_ev(t, products, agents):
            v_ev, v_suv, v_truck, v_van, v_car = _get_ev_vectors(products)
            J = v_ev.size
            I = agents.size
            M = (
                np.outer(v_suv, v_suv)
                + np.outer(v_truck, v_truck)
                + np.outer(v_van, v_van)
                + np.outer(v_car, v_car)
            )
            out = np.zeros((I, J, J + 1), dtype=np.float64)
            out[:, :, 1:] = np.broadcast_to(M, (I, J, J))
            return out

        num_ev_given_ev_2022 = pyblp.MicroPart(
            name=f"[{tag}] E[1{{j EV}} * 1{{k EV}}]",
            dataset=dataset,
            compute_values=_values_num_ev_ev,
        )

        den_first_ev_2022 = pyblp.MicroPart(
            name=f"[{tag}] E[1{{j EV}}]",
            dataset=dataset,
            compute_values=_values_den_first_ev,
        )

        num_sameclass_ev_given_ev_2022 = pyblp.MicroPart(
            name=f"[{tag}] E[1{{j EV, same class}} * 1{{k EV, same class}}]",
            dataset=dataset,
            compute_values=_values_num_sameclass_ev_ev,
        )

        return (
            num_ev_given_ev_2022,
            den_first_ev_2022,
            num_sameclass_ev_given_ev_2022,
        )

    (
        num_ev_given_ev_2022,
        den_first_ev_2022,
        num_sameclass_ev_given_ev_2022,
    ) = make_ev_second_choice_parts_2022(d2022)

    # ---------- 5. Ratio moments & gradients ----------

    def ratio(v):
        num, den = v[0], v[1] + 1e-16
        return num / den

    def ratio_grad(v):
        num, den = v[0], v[1] + 1e-16
        return np.array([1.0 / den, -num / (den**2)], dtype=np.float64)

    # ---------- 6. Build 2022 EV micro-moments ----------

    ev_moments_2022 = [
        pyblp.MicroMoment(
            name="P(second is EV | first is EV) | 2022 (all vehicles)",
            value=TARGETS_2022_EV["ev_given_ev"],
            parts=[num_ev_given_ev_2022, den_first_ev_2022],
            compute_value=ratio,
            compute_gradient=ratio_grad,
        ),
        pyblp.MicroMoment(
            name="P(second is EV in same class | first is EV) | 2022 (all vehicles)",
            value=TARGETS_2022_EV["same_class_ev_given_ev"],
            parts=[num_sameclass_ev_given_ev_2022, den_first_ev_2022],
            compute_value=ratio,
            compute_gradient=ratio_grad,
        ),
    ]

    return ev_moments_2022
