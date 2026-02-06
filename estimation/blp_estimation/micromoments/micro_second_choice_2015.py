import numpy as np
import pandas as pd
import pyblp


def build_second_choice_moments_2015(
    product_data: pd.DataFrame,
    include_mpg: int = 1,
    include_hp: int = 1,
    include_size: int = 1,
    include_eu: int = 1,
    include_us: int = 1,
    include_hyb_in_mpg: int = 0,
    include_ev_in_lux: int = 1,
):
    """
    Build 2015 second-choice micro-moments:
      - corr(size_first, size_second)
      - corr(hp_first, hp_second)
      - corr(mpg_first, mpg_second) (ICE+hybrid depending on flag)
      - type/brand: van, truck, suv, euro, us, luxury.
    """

    # ---------- 1. Correlation helper functions ----------

    def _corr_value(v):
        m12, m1, m2, m1sq, m2sq = v
        A = m12 - m1 * m2
        B1 = m1sq - m1**2
        B2 = m2sq - m2**2
        return A / (np.sqrt(max(B1 * B2, 0.0)) + 1e-16)

    def _corr_grad(v):
        m12, m1, m2, m1sq, m2sq = v
        A = m12 - m1 * m2
        B1 = m1sq - m1**2
        B2 = m2sq - m2**2

        D = np.sqrt(max(B1 * B2, 0.0)) + 1e-16
        D3 = D**3 + 1e-32

        dm12 = 1.0 / D
        dm1 = (-m2 / D) + (A * m1 * B2) / D3
        dm2 = (-m1 / D) + (A * m2 * B1) / D3
        dm1sq = -(A * B2) / (2.0 * D3)
        dm2sq = -(A * B1) / (2.0 * D3)

        return np.array([dm12, dm1, dm2, dm1sq, dm2sq], dtype=np.float64)

    # ---------- 2. Ratio helper for discrete second-choice moments ----------

    def ratio(v):
        num, den = v[0], v[1] + 1e-16
        return num / den

    def ratio_grad(v):
        num, den = v[0], v[1] + 1e-16
        return np.array([1.0 / den, -num / (den**2)], dtype=np.float64)

    # ---------- 3. Attribute maps for 2015 ----------

    prod2015 = product_data.loc[product_data["market_ids"] == 2015].copy()
    prod2015_ids_str = prod2015["clustering_ids"].astype(str).to_numpy()

    size_map_2015 = dict(
        zip(prod2015_ids_str, pd.to_numeric(prod2015["size"], errors="coerce"))
    )
    hp_map_2015 = dict(
        zip(prod2015_ids_str, pd.to_numeric(prod2015["hp"], errors="coerce"))
    )
    mpg_map_2015 = dict(
        zip(prod2015_ids_str, pd.to_numeric(prod2015["mpg"], errors="coerce"))
    )
    price_map_2015 = dict(
        zip(prod2015_ids_str, pd.to_numeric(prod2015["prices"], errors="coerce"))
    )

    attr_maps_2015 = {
        "size": size_map_2015,
        "hp": hp_map_2015,
        "mpg": mpg_map_2015,
        "price": price_map_2015,
    }

    vt_vals_2015 = (
        prod2015["vehicle_type"].fillna("").str.lower().str.strip()
    )
    vt_map_2015 = dict(zip(prod2015_ids_str, vt_vals_2015))

    euro_vals_2015 = (
        pd.to_numeric(prod2015["euro_brand"], errors="coerce").fillna(0.0)
    )
    us_vals_2015 = (
        pd.to_numeric(prod2015["us_brand"], errors="coerce").fillna(0.0)
    )

    euro_map_2015 = dict(zip(prod2015_ids_str, euro_vals_2015))
    us_map_2015 = dict(zip(prod2015_ids_str, us_vals_2015))

    if include_ev_in_lux == 1:
        luxury_brands = {
            "porsche",
            "maserati",
            "lotus",
            "audi",
            "bmw",
            "mercedesbenz",
            "lexus",
            "infiniti",
            "cadillac",
            "lincoln",
            "jaguar",
            "landrover",
            "tesla",
            "rivian",
            "lucidmotors",
            "polestar",
            "acura",
            "volvo",
            "alpharomeo",
            "genesis",
        }
    else:
        luxury_brands = {
            "porsche",
            "maserati",
            "lotus",
            "audi",
            "bmw",
            "mercedesbenz",
            "lexus",
            "infiniti",
            "cadillac",
            "lincoln",
            "jaguar",
            "landrover",
            "acura",
            "volvo",
            "genesis",
            "alpharomeo",
        }

    luxury_vals_2015 = prod2015["firm_ids"].isin(luxury_brands).astype(float)
    luxury_map_2015 = dict(zip(prod2015_ids_str, luxury_vals_2015))

    # ---------- 4. Pair-weight matrices ----------

    def _all_inside_pairs_weights(t, products, agents):
        J = products.clustering_ids.size
        I = agents.size
        W = np.ones((J, J), dtype=np.float64)
        np.fill_diagonal(W, 0.0)
        return np.broadcast_to(W, (I, J, J))

    def _icehyb_pairs_weights(t, products, agents):
        J = products.clustering_ids.size
        I = agents.size

        is_ev = np.asarray(products.ev, dtype=int)
        is_hyb = np.asarray(products.hybrid, dtype=int)

        if include_hyb_in_mpg == 1:
            is_non_ev = 1 - is_ev
        else:
            is_non_ev = 1 - is_ev - is_hyb

        W = np.zeros((J, J), dtype=np.float64)

        for j in range(J):
            if is_non_ev[j]:
                valid_k = (is_non_ev == 1)
                valid_k[j] = False
                W[j, valid_k] = 1.0

        return np.broadcast_to(W, (I, J, J))

    # datasets
    def build_second_choice_dataset_2015():
        return pyblp.MicroDataset(
            name="SecondChoice2015_all",
            observations=1000,
            market_ids=[2015],
            compute_weights=_all_inside_pairs_weights,
        )

    def build_second_choice_dataset_2015_icehyb():
        def compute_weights(t, products, agents):
            J = products.clustering_ids.size
            I = agents.size
            is_ev = np.asarray(products.ev, dtype=int)
            is_hyb = np.asarray(products.hybrid, dtype=int)
            if include_hyb_in_mpg == 1:
                is_non_ev = 1 - is_ev
            else:
                is_non_ev = 1 - is_ev - is_hyb

            W = np.zeros((J, J), dtype=np.float64)
            for j in range(J):
                if is_non_ev[j]:
                    valid_k = np.where(
                        (is_non_ev == 1) & (np.arange(J) != j)
                    )[0]
                    if valid_k.size > 0:
                        W[j, valid_k] = 1.0 / valid_k.size
            return np.broadcast_to(W[None, :, :], (I, J, J))

        return pyblp.MicroDataset(
            name="SecondChoice2015_ice_hybrid",
            observations=1000,
            market_ids=[2015],
            compute_weights=compute_weights,
        )

    d2015 = build_second_choice_dataset_2015()
    d2015_icehyb = build_second_choice_dataset_2015_icehyb()

    # ---------- 5. MicroParts for corr(x_first, x_second) ----------

    def make_second_choice_parts_2015(dataset, attr_name, attr_label):
        attr_map = attr_maps_2015[attr_name]
        tag = dataset.name

        def _get_x(products):
            pids = np.asarray(products.clustering_ids)
            x_vals = []
            for pid in pids:
                if isinstance(pid, np.ndarray):
                    pid_key = str(pid.item())
                else:
                    pid_key = str(pid)
                if pid_key not in attr_map:
                    raise KeyError(
                        f"product_id {pid_key!r} not in attr_map_2015[{attr_name}]"
                    )
                x_vals.append(attr_map[pid_key])
            return np.array(x_vals, dtype=np.float64)

        def _values_x1x2(t, products, agents):
            x = _get_x(products)
            J = x.size
            I = agents.size
            mat = np.outer(x, x)
            return np.broadcast_to(mat, (I, J, J))

        def _values_x1(t, products, agents):
            x = _get_x(products)
            J = x.size
            I = agents.size
            mat = x.reshape(J, 1) @ np.ones((1, J))
            return np.broadcast_to(mat, (I, J, J))

        def _values_x2(t, products, agents):
            x = _get_x(products)
            J = x.size
            I = agents.size
            mat = np.ones((J, 1)) @ x.reshape(1, J)
            return np.broadcast_to(mat, (I, J, J))

        def _values_x1sq(t, products, agents):
            x2 = _get_x(products) ** 2
            J = x2.size
            I = agents.size
            mat = x2.reshape(J, 1) @ np.ones((1, J))
            return np.broadcast_to(mat, (I, J, J))

        def _values_x2sq(t, products, agents):
            x2 = _get_x(products) ** 2
            J = x2.size
            I = agents.size
            mat = np.ones((J, 1)) @ x2.reshape(1, J)
            return np.broadcast_to(mat, (I, J, J))

        return (
            pyblp.MicroPart(
                f"[{tag}] E[x1*x2] ({attr_label})",
                dataset,
                compute_values=_values_x1x2,
            ),
            pyblp.MicroPart(
                f"[{tag}] E[x1] ({attr_label})",
                dataset,
                compute_values=_values_x1,
            ),
            pyblp.MicroPart(
                f"[{tag}] E[x2] ({attr_label})",
                dataset,
                compute_values=_values_x2,
            ),
            pyblp.MicroPart(
                f"[{tag}] E[x1^2] ({attr_label})",
                dataset,
                compute_values=_values_x1sq,
            ),
            pyblp.MicroPart(
                f"[{tag}] E[x2^2] ({attr_label})",
                dataset,
                compute_values=_values_x2sq,
            ),
        )

    # ---------- 6. Type/brand second-choice parts ----------

    def make_type_brand_second_choice_parts_2015(dataset):
        tag = dataset.name

        def _get_vectors(products):
            pids = np.asarray(products.clustering_ids)

            v_van, v_truck, v_suv, v_euro, v_us, v_lux = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for pid in pids:
                pid_str = str(pid.item()) if isinstance(pid, np.ndarray) else str(pid)

                vt = vt_map_2015.get(pid_str, "")
                euro = float(euro_map_2015.get(pid_str, 0.0))
                usv = float(us_map_2015.get(pid_str, 0.0))
                lux = float(luxury_map_2015.get(pid_str, 0.0))

                v_van.append(1.0 if vt == "van" else 0.0)
                v_truck.append(1.0 if vt == "truck" else 0.0)
                v_suv.append(1.0 if vt == "suv" else 0.0)
                v_euro.append(euro)
                v_us.append(usv)
                v_lux.append(lux)

            return (
                np.array(v_van, dtype=np.float64),
                np.array(v_truck, dtype=np.float64),
                np.array(v_suv, dtype=np.float64),
                np.array(v_euro, dtype=np.float64),
                np.array(v_us, dtype=np.float64),
                np.array(v_lux, dtype=np.float64),
            )

        def _build_pair(name_prefix, vec_index):
            def _values_num(t, products, agents):
                v_van, v_truck, v_suv, v_euro, v_us, v_lux = _get_vectors(
                    products
                )
                vs = [v_van, v_truck, v_suv, v_euro, v_us, v_lux][vec_index]
                J = vs.size
                I = agents.size
                mat = np.outer(vs, vs)
                return np.broadcast_to(mat, (I, J, J))

            def _values_den(t, products, agents):
                v_van, v_truck, v_suv, v_euro, v_us, v_lux = _get_vectors(
                    products
                )
                vs = [v_van, v_truck, v_suv, v_euro, v_us, v_lux][vec_index]
                J = vs.size
                I = agents.size
                mat = vs.reshape(J, 1) @ np.ones((1, J))
                return np.broadcast_to(mat, (I, J, J))

            num = pyblp.MicroPart(
                name=f"[{tag}] {name_prefix} numerator",
                dataset=dataset,
                compute_values=_values_num,
            )
            den = pyblp.MicroPart(
                name=f"[{tag}] {name_prefix} denominator",
                dataset=dataset,
                compute_values=_values_den,
            )
            return num, den

        num_van_2015, den_van_2015 = _build_pair(
            "E[1{j van} * 1{k van}]", 0
        )
        num_truck_2015, den_truck_2015 = _build_pair(
            "E[1{j truck} * 1{k truck}]", 1
        )
        num_suv_2015, den_suv_2015 = _build_pair(
            "E[1{j suv} * 1{k suv}]", 2
        )
        num_euro_2015, den_euro_2015 = _build_pair(
            "E[1{j euro} * 1{k euro}]", 3
        )
        num_us_2015, den_us_2015 = _build_pair(
            "E[1{j us} * 1{k us}]", 4
        )
        num_lux_2015, den_lux_2015 = _build_pair(
            "E[1{j luxury} * 1{k luxury}]", 5
        )

        return (
            num_van_2015,
            den_van_2015,
            num_truck_2015,
            den_truck_2015,
            num_suv_2015,
            den_suv_2015,
            num_euro_2015,
            den_euro_2015,
            num_us_2015,
            den_us_2015,
            num_lux_2015,
            den_lux_2015,
        )

    # ---------- 7. Targets ----------

    TARGETS_2015 = {
        "size": 0.782,
        "hp": 0.674,
        "mpg": 0.611,
        "price": 0.860,
        "van": 0.720,
        "truck": 0.872,
        "suv": 0.690,
        "euro": 0.413,
        "us": 0.464,
        "luxury": 0.550,
    }

    # continuous attributes
    size_parts = make_second_choice_parts_2015(d2015, "size", "size")
    hp_parts = make_second_choice_parts_2015(d2015, "hp", "hp")
    mpg_parts = make_second_choice_parts_2015(
        d2015_icehyb, "mpg", "mpg"
    )
    price_parts = make_second_choice_parts_2015(
        d2015, "price", "price"
    )

    (
        num_van_2015,
        den_van_2015,
        num_truck_2015,
        den_truck_2015,
        num_suv_2015,
        den_suv_2015,
        num_euro_2015,
        den_euro_2015,
        num_us_2015,
        den_us_2015,
        num_lux_2015,
        den_lux_2015,
    ) = make_type_brand_second_choice_parts_2015(d2015)

    # optional lists
    if include_mpg == 1:
        mpg_moments_list = [
            pyblp.MicroMoment(
                "corr(MPG_first, MPG_second) | 2015 (ICE+hybrid only)",
                TARGETS_2015["mpg"],
                mpg_parts,
                _corr_value,
                _corr_grad,
            )
        ]
    else:
        mpg_moments_list = []

    if include_hp == 1:
        hp_moments_list = [
            pyblp.MicroMoment(
                "corr(HP_first, HP_second) | 2015 (all vehicles)",
                TARGETS_2015["hp"],
                hp_parts,
                _corr_value,
                _corr_grad,
            )
        ]
    else:
        hp_moments_list = []

    if include_size == 1:
        size_moments_list = [
            pyblp.MicroMoment(
                "corr(size_first, size_second) | 2015 (all vehicles)",
                TARGETS_2015["size"],
                size_parts,
                _corr_value,
                _corr_grad,
            )
        ]
    else:
        size_moments_list = []

    if include_eu == 1:
        eu_moments_list = [
            pyblp.MicroMoment(
                "P(second is Euro-brand | first is Euro-brand) | 2015",
                TARGETS_2015["euro"],
                [num_euro_2015, den_euro_2015],
                ratio,
                ratio_grad,
            )
        ]
    else:
        eu_moments_list = []

    if include_us == 1:
        us_moments_list = [
            pyblp.MicroMoment(
                "P(second is US-brand | first is US-brand) | 2015",
                TARGETS_2015["us"],
                [num_us_2015, den_us_2015],
                ratio,
                ratio_grad,
            )
        ]
    else:
        us_moments_list = []

    # core type/brand moments
    second_choice_moments_2015 = [
        pyblp.MicroMoment(
            "P(second is van | first is van) | 2015",
            TARGETS_2015["van"],
            [num_van_2015, den_van_2015],
            ratio,
            ratio_grad,
        ),
        pyblp.MicroMoment(
            "P(second is truck | first is truck) | 2015",
            TARGETS_2015["truck"],
            [num_truck_2015, den_truck_2015],
            ratio,
            ratio_grad,
        ),
        pyblp.MicroMoment(
            "P(second is SUV | first is SUV) | 2015",
            TARGETS_2015["suv"],
            [num_suv_2015, den_suv_2015],
            ratio,
            ratio_grad,
        ),
        pyblp.MicroMoment(
            "P(second is luxury | first is luxury) | 2015",
            TARGETS_2015["luxury"],
            [num_lux_2015, den_lux_2015],
            ratio,
            ratio_grad,
        ),
    ]

    second_choice_moments_2015 += mpg_moments_list
    second_choice_moments_2015 += size_moments_list
    second_choice_moments_2015 += hp_moments_list
    second_choice_moments_2015 += eu_moments_list
    second_choice_moments_2015 += us_moments_list

    return second_choice_moments_2015
