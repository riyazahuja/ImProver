theorem AffineSubspace.isClosed_direction_iff (s : AffineSubspace 𝕜 Q) :
    IsClosed (s.direction : Set W) ↔ IsClosed (s : Set Q) := by
  /-
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    s : AffineSubspace 𝕜 Q
    ⊢ Iff (IsClosed ↑s.direction) (IsClosed ↑s)
  -/
  rcases s.eq_bot_or_nonempty with (rfl | ⟨x, hx⟩); · simp [isClosed_singleton]
                                                      /-
                                                        🎉 no goals
                                                      -/
  rw [← (IsometryEquiv.vaddConst x).toHomeomorph.symm.isClosed_image,
    AffineSubspace.coe_direction_eq_vsub_set_right hx]
  /-
    case inr.intro
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    s : AffineSubspace 𝕜 Q
    x : Q
    hx : Membership.mem (↑s) x
    ⊢ Iff (IsClosed (Set.image (fun x_1 => VSub.vsub x_1 x) ↑s)) (IsClosed (Set.im …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem dist_center_homothety (p₁ p₂ : P) (c : 𝕜) :
    dist p₁ (homothety p₁ c p₂) = ‖c‖ * dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : SeminormedAddCommGroup V
    inst✝³ : PseudoMetricSpace P
    inst✝² : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 V
    p₁ p₂ : P
    c : 𝕜
    ⊢ Eq (Dist.dist p₁ ((AffineMap.homothety p₁ c) p₂)) (HMul.hMul (Norm.norm c) ( …
  -/
  simp [homothety_def, norm_smul, ← dist_eq_norm_vsub, dist_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_center_homothety (p₁ p₂ : P) (c : 𝕜) :
    nndist p₁ (homothety p₁ c p₂) = ‖c‖₊ * nndist p₁ p₂ :=
  NNReal.eq <| dist_center_homothety _ _ _


@[simp]
theorem dist_homothety_center (p₁ p₂ : P) (c : 𝕜) :
                                                         /-
                                                           V : Type u_1
                                                           P : Type u_2
                                                           inst✝⁴ : SeminormedAddCommGroup V
                                                           inst✝³ : PseudoMetricSpace P
                                                           inst✝² : NormedAddTorsor V P
                                                           𝕜 : Type u_5
                                                           inst✝¹ : NormedField 𝕜
                                                           inst✝ : NormedSpace 𝕜 V
                                                           p₁ p₂ : P
                                                           c : 𝕜
                                                           ⊢ Eq (Dist.dist ((AffineMap.homothety p₁ c) p₂) p₁) (HMul.hMul (Norm.norm c) ( …
                                                         -/
    dist (homothety p₁ c p₂) p₁ = ‖c‖ * dist p₁ p₂ := by rw [dist_comm, dist_center_homothety]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem nndist_homothety_center (p₁ p₂ : P) (c : 𝕜) :
    nndist (homothety p₁ c p₂) p₁ = ‖c‖₊ * nndist p₁ p₂ :=
  NNReal.eq <| dist_homothety_center _ _ _


@[simp]
theorem dist_lineMap_lineMap (p₁ p₂ : P) (c₁ c₂ : 𝕜) :
    dist (lineMap p₁ p₂ c₁) (lineMap p₁ p₂ c₂) = dist c₁ c₂ * dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : SeminormedAddCommGroup V
    inst✝³ : PseudoMetricSpace P
    inst✝² : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 V
    p₁ p₂ : P
    c₁ c₂ : 𝕜
    ⊢ Eq (Dist.dist ((AffineMap.lineMap p₁ p₂) c₁) ((AffineMap.lineMap p₁ p₂) c₂)) …
  -/
  rw [dist_comm p₁ p₂]
  simp only [lineMap_apply, dist_eq_norm_vsub, vadd_vsub_vadd_cancel_right,
    ← sub_smul, norm_smul, vsub_eq_sub]


@[simp]
theorem nndist_lineMap_lineMap (p₁ p₂ : P) (c₁ c₂ : 𝕜) :
    nndist (lineMap p₁ p₂ c₁) (lineMap p₁ p₂ c₂) = nndist c₁ c₂ * nndist p₁ p₂ :=
  NNReal.eq <| dist_lineMap_lineMap _ _ _ _


theorem lipschitzWith_lineMap (p₁ p₂ : P) : LipschitzWith (nndist p₁ p₂) (lineMap p₁ p₂ : 𝕜 → P) :=
  LipschitzWith.of_dist_le_mul fun c₁ c₂ =>
    ((dist_lineMap_lineMap p₁ p₂ c₁ c₂).trans (mul_comm _ _)).le


@[simp]
theorem dist_lineMap_left (p₁ p₂ : P) (c : 𝕜) : dist (lineMap p₁ p₂ c) p₁ = ‖c‖ * dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : SeminormedAddCommGroup V
    inst✝³ : PseudoMetricSpace P
    inst✝² : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 V
    p₁ p₂ : P
    c : 𝕜
    ⊢ Eq (Dist.dist ((AffineMap.lineMap p₁ p₂) c) p₁) (HMul.hMul (Norm.norm c) (Di …
  -/
  simpa only [lineMap_apply_zero, dist_zero_right] using dist_lineMap_lineMap p₁ p₂ c 0
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_lineMap_left (p₁ p₂ : P) (c : 𝕜) :
    nndist (lineMap p₁ p₂ c) p₁ = ‖c‖₊ * nndist p₁ p₂ :=
  NNReal.eq <| dist_lineMap_left _ _ _


@[simp]
theorem dist_left_lineMap (p₁ p₂ : P) (c : 𝕜) : dist p₁ (lineMap p₁ p₂ c) = ‖c‖ * dist p₁ p₂ :=
  (dist_comm _ _).trans (dist_lineMap_left _ _ _)


@[simp]
theorem nndist_left_lineMap (p₁ p₂ : P) (c : 𝕜) :
    nndist p₁ (lineMap p₁ p₂ c) = ‖c‖₊ * nndist p₁ p₂ :=
  NNReal.eq <| dist_left_lineMap _ _ _


@[simp]
theorem dist_lineMap_right (p₁ p₂ : P) (c : 𝕜) :
    dist (lineMap p₁ p₂ c) p₂ = ‖1 - c‖ * dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : SeminormedAddCommGroup V
    inst✝³ : PseudoMetricSpace P
    inst✝² : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 V
    p₁ p₂ : P
    c : 𝕜
    ⊢ Eq (Dist.dist ((AffineMap.lineMap p₁ p₂) c) p₂) (HMul.hMul (Norm.norm (HSub. …
  -/
  simpa only [lineMap_apply_one, dist_eq_norm'] using dist_lineMap_lineMap p₁ p₂ c 1
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_lineMap_right (p₁ p₂ : P) (c : 𝕜) :
    nndist (lineMap p₁ p₂ c) p₂ = ‖1 - c‖₊ * nndist p₁ p₂ :=
  NNReal.eq <| dist_lineMap_right _ _ _


@[simp]
theorem dist_right_lineMap (p₁ p₂ : P) (c : 𝕜) : dist p₂ (lineMap p₁ p₂ c) = ‖1 - c‖ * dist p₁ p₂ :=
  (dist_comm _ _).trans (dist_lineMap_right _ _ _)


@[simp]
theorem nndist_right_lineMap (p₁ p₂ : P) (c : 𝕜) :
    nndist p₂ (lineMap p₁ p₂ c) = ‖1 - c‖₊ * nndist p₁ p₂ :=
  NNReal.eq <| dist_right_lineMap _ _ _


@[simp]
theorem dist_homothety_self (p₁ p₂ : P) (c : 𝕜) :
    dist (homothety p₁ c p₂) p₂ = ‖1 - c‖ * dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : SeminormedAddCommGroup V
    inst✝³ : PseudoMetricSpace P
    inst✝² : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 V
    p₁ p₂ : P
    c : 𝕜
    ⊢ Eq (Dist.dist ((AffineMap.homothety p₁ c) p₂) p₂) (HMul.hMul (Norm.norm (HSu …
  -/
  rw [homothety_eq_lineMap, dist_lineMap_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_homothety_self (p₁ p₂ : P) (c : 𝕜) :
    nndist (homothety p₁ c p₂) p₂ = ‖1 - c‖₊ * nndist p₁ p₂ :=
  NNReal.eq <| dist_homothety_self _ _ _


@[simp]
theorem dist_self_homothety (p₁ p₂ : P) (c : 𝕜) :
                                                             /-
                                                               V : Type u_1
                                                               P : Type u_2
                                                               inst✝⁴ : SeminormedAddCommGroup V
                                                               inst✝³ : PseudoMetricSpace P
                                                               inst✝² : NormedAddTorsor V P
                                                               𝕜 : Type u_5
                                                               inst✝¹ : NormedField 𝕜
                                                               inst✝ : NormedSpace 𝕜 V
                                                               p₁ p₂ : P
                                                               c : 𝕜
                                                               ⊢ Eq (Dist.dist p₂ ((AffineMap.homothety p₁ c) p₂)) (HMul.hMul (Norm.norm (HSu …
                                                             -/
    dist p₂ (homothety p₁ c p₂) = ‖1 - c‖ * dist p₁ p₂ := by rw [dist_comm, dist_homothety_self]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem nndist_self_homothety (p₁ p₂ : P) (c : 𝕜) :
    nndist p₂ (homothety p₁ c p₂) = ‖1 - c‖₊ * nndist p₁ p₂ :=
  NNReal.eq <| dist_self_homothety _ _ _


@[simp]
theorem dist_left_midpoint (p₁ p₂ : P) : dist p₁ (midpoint 𝕜 p₁ p₂) = ‖(2 : 𝕜)‖⁻¹ * dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : SeminormedAddCommGroup V
    inst✝⁴ : PseudoMetricSpace P
    inst✝³ : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 V
    inst✝ : Invertible 2
    p₁ p₂ : P
    ⊢ Eq (Dist.dist p₁ (midpoint 𝕜 p₁ p₂)) (HMul.hMul (Inv.inv (Norm.norm 2)) (Dis …
  -/
  rw [midpoint, dist_comm, dist_lineMap_left, invOf_eq_inv, ← norm_inv]
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_left_midpoint (p₁ p₂ : P) :
    nndist p₁ (midpoint 𝕜 p₁ p₂) = ‖(2 : 𝕜)‖₊⁻¹ * nndist p₁ p₂ :=
  NNReal.eq <| dist_left_midpoint _ _


@[simp]
theorem dist_midpoint_left (p₁ p₂ : P) : dist (midpoint 𝕜 p₁ p₂) p₁ = ‖(2 : 𝕜)‖⁻¹ * dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : SeminormedAddCommGroup V
    inst✝⁴ : PseudoMetricSpace P
    inst✝³ : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 V
    inst✝ : Invertible 2
    p₁ p₂ : P
    ⊢ Eq (Dist.dist (midpoint 𝕜 p₁ p₂) p₁) (HMul.hMul (Inv.inv (Norm.norm 2)) (Dis …
  -/
  rw [dist_comm, dist_left_midpoint]
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_midpoint_left (p₁ p₂ : P) :
    nndist (midpoint 𝕜 p₁ p₂) p₁ = ‖(2 : 𝕜)‖₊⁻¹ * nndist p₁ p₂ :=
  NNReal.eq <| dist_midpoint_left _ _


@[simp]
theorem dist_midpoint_right (p₁ p₂ : P) :
    dist (midpoint 𝕜 p₁ p₂) p₂ = ‖(2 : 𝕜)‖⁻¹ * dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : SeminormedAddCommGroup V
    inst✝⁴ : PseudoMetricSpace P
    inst✝³ : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 V
    inst✝ : Invertible 2
    p₁ p₂ : P
    ⊢ Eq (Dist.dist (midpoint 𝕜 p₁ p₂) p₂) (HMul.hMul (Inv.inv (Norm.norm 2)) (Dis …
  -/
  rw [midpoint_comm, dist_midpoint_left, dist_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_midpoint_right (p₁ p₂ : P) :
    nndist (midpoint 𝕜 p₁ p₂) p₂ = ‖(2 : 𝕜)‖₊⁻¹ * nndist p₁ p₂ :=
  NNReal.eq <| dist_midpoint_right _ _


@[simp]
theorem dist_right_midpoint (p₁ p₂ : P) :
    dist p₂ (midpoint 𝕜 p₁ p₂) = ‖(2 : 𝕜)‖⁻¹ * dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : SeminormedAddCommGroup V
    inst✝⁴ : PseudoMetricSpace P
    inst✝³ : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 V
    inst✝ : Invertible 2
    p₁ p₂ : P
    ⊢ Eq (Dist.dist p₂ (midpoint 𝕜 p₁ p₂)) (HMul.hMul (Inv.inv (Norm.norm 2)) (Dis …
  -/
  rw [dist_comm, dist_midpoint_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_right_midpoint (p₁ p₂ : P) :
    nndist p₂ (midpoint 𝕜 p₁ p₂) = ‖(2 : 𝕜)‖₊⁻¹ * nndist p₁ p₂ :=
  NNReal.eq <| dist_right_midpoint _ _


theorem dist_midpoint_midpoint_le' (p₁ p₂ p₃ p₄ : P) :
    dist (midpoint 𝕜 p₁ p₂) (midpoint 𝕜 p₃ p₄) ≤ (dist p₁ p₃ + dist p₂ p₄) / ‖(2 : 𝕜)‖ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : SeminormedAddCommGroup V
    inst✝⁴ : PseudoMetricSpace P
    inst✝³ : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 V
    inst✝ : Invertible 2
    p₁ p₂ p₃ p₄ : P
    ⊢ LE.le (Dist.dist (midpoint 𝕜 p₁ p₂) (midpoint 𝕜 p₃ p₄)) (HDiv.hDiv (HAdd.hAd …
  -/
  rw [dist_eq_norm_vsub V, dist_eq_norm_vsub V, dist_eq_norm_vsub V, midpoint_vsub_midpoint]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : SeminormedAddCommGroup V
    inst✝⁴ : PseudoMetricSpace P
    inst✝³ : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 V
    inst✝ : Invertible 2
    p₁ p₂ p₃ p₄ : P
    ⊢ LE.le (Norm.norm (midpoint 𝕜 (VSub.vsub p₁ p₃) (VSub.vsub p₂ p₄))) (HDiv.hDi …
  -/
  rw [midpoint_eq_smul_add, norm_smul, invOf_eq_inv, norm_inv, ← div_eq_inv_mul]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : SeminormedAddCommGroup V
    inst✝⁴ : PseudoMetricSpace P
    inst✝³ : NormedAddTorsor V P
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 V
    inst✝ : Invertible 2
    p₁ p₂ p₃ p₄ : P
    ⊢ LE.le (HDiv.hDiv (Norm.norm (HAdd.hAdd (VSub.vsub p₁ p₃) (VSub.vsub p₂ p₄))) …
  -/
  exact div_le_div_of_nonneg_right (norm_add_le _ _) (norm_nonneg _)
  /-
    🎉 no goals
  -/


theorem nndist_midpoint_midpoint_le' (p₁ p₂ p₃ p₄ : P) :
    nndist (midpoint 𝕜 p₁ p₂) (midpoint 𝕜 p₃ p₄) ≤ (nndist p₁ p₃ + nndist p₂ p₄) / ‖(2 : 𝕜)‖₊ :=
  dist_midpoint_midpoint_le' _ _ _ _


@[simp] theorem dist_pointReflection_left (p q : P) :
    dist (Equiv.pointReflection p q) p = dist p q := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p q : P
    ⊢ Eq (Dist.dist ((Equiv.pointReflection p) q) p) (Dist.dist p q)
  -/
  simp [dist_eq_norm_vsub V, Equiv.pointReflection_vsub_left (G := V)]
  /-
    🎉 no goals
  -/


@[simp] theorem dist_left_pointReflection (p q : P) :
    dist p (Equiv.pointReflection p q) = dist p q :=
  (dist_comm _ _).trans (dist_pointReflection_left _ _)


variable (𝕜) in
theorem dist_pointReflection_right (p q : P) :
    dist (Equiv.pointReflection p q) q = ‖(2 : 𝕜)‖ * dist p q := by
  simp [dist_eq_norm_vsub V, Equiv.pointReflection_vsub_right (G := V), ← Nat.cast_smul_eq_nsmul 𝕜,
    norm_smul]


variable (𝕜) in
theorem dist_right_pointReflection (p q : P) :
    dist q (Equiv.pointReflection p q) = ‖(2 : 𝕜)‖ * dist p q :=
  (dist_comm _ _).trans (dist_pointReflection_right 𝕜 _ _)


theorem antilipschitzWith_lineMap {p₁ p₂ : Q} (h : p₁ ≠ p₂) :
    AntilipschitzWith (nndist p₁ p₂)⁻¹ (lineMap p₁ p₂ : 𝕜 → Q) :=
  AntilipschitzWith.of_le_mul_dist fun c₁ c₂ => by
    rw [dist_lineMap_lineMap, NNReal.coe_inv, ← dist_nndist, mul_left_comm,
      inv_mul_cancel₀ (dist_ne_zero.2 h), mul_one]


theorem eventually_homothety_mem_of_mem_interior (x : Q) {s : Set Q} {y : Q} (hy : y ∈ interior s) :
    ∀ᶠ δ in 𝓝 (1 : 𝕜), homothety x δ y ∈ s := by
  /-
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    x : Q
    s : Set Q
    y : Q
    hy : Membership.mem (interior s) y
    ⊢ Filter.Eventually (fun δ => Membership.mem s ((AffineMap.homothety x δ) y))  …
  -/
  rw [(NormedAddCommGroup.nhds_basis_norm_lt (1 : 𝕜)).eventually_iff]
  /-
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    x : Q
    s : Set Q
    y : Q
    hy : Membership.mem (interior s) y
    ⊢ Exists fun i => And (LT.lt 0 i) (∀ ⦃x_1 : 𝕜⦄, Membership.mem (setOf fun y => …
  -/
  rcases eq_or_ne y x with h | h
    /-
      case inl
      W : Type u_3
      Q : Type u_4
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : MetricSpace Q
      inst✝² : NormedAddTorsor W Q
      𝕜 : Type u_5
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedSpace 𝕜 W
      x : Q
      s : Set Q
      y : Q
      hy : Membership.mem (interior s) y
      h : Eq y x
      ⊢ Exists fun i => And (LT.lt 0 i) (∀ ⦃x_1 : 𝕜⦄, Membership.mem (setOf fun y => …
    -/
  · use 1
    /-
      case h
      W : Type u_3
      Q : Type u_4
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : MetricSpace Q
      inst✝² : NormedAddTorsor W Q
      𝕜 : Type u_5
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedSpace 𝕜 W
      x : Q
      s : Set Q
      y : Q
      hy : Membership.mem (interior s) y
      h : Eq y x
      ⊢ And (LT.lt 0 1) (∀ ⦃x_1 : 𝕜⦄, Membership.mem (setOf fun y => LT.lt (Norm.nor …
    -/
    simp [h.symm, interior_subset hy]
    /-
      🎉 no goals
    -/
  /-
    case inr
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    x : Q
    s : Set Q
    y : Q
    hy : Membership.mem (interior s) y
    h : Ne y x
    ⊢ Exists fun i => And (LT.lt 0 i) (∀ ⦃x_1 : 𝕜⦄, Membership.mem (setOf fun y => …
  -/
  have hxy : 0 < ‖y -ᵥ x‖ := by rwa [norm_pos_iff, vsub_ne_zero]
  /-
    case inr
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    x : Q
    s : Set Q
    y : Q
    hy : Membership.mem (interior s) y
    h : Ne y x
    hxy : LT.lt 0 (Norm.norm (VSub.vsub y x))
    ⊢ Exists fun i => And (LT.lt 0 i) (∀ ⦃x_1 : 𝕜⦄, Membership.mem (setOf fun y => …
  -/
  obtain ⟨u, hu₁, hu₂, hu₃⟩ := mem_interior.mp hy
  /-
    case inr.intro.intro.intro
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    x : Q
    s : Set Q
    y : Q
    hy : Membership.mem (interior s) y
    h : Ne y x
    hxy : LT.lt 0 (Norm.norm (VSub.vsub y x))
    u : Set Q
    hu₁ : HasSubset.Subset u s
    hu₂ : IsOpen u
    hu₃ : Membership.mem u y
    ⊢ Exists fun i => And (LT.lt 0 i) (∀ ⦃x_1 : 𝕜⦄, Membership.mem (setOf fun y => …
  -/
  obtain ⟨ε, hε, hyε⟩ := Metric.isOpen_iff.mp hu₂ y hu₃
  /-
    case inr.intro.intro.intro.intro.intro
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    x : Q
    s : Set Q
    y : Q
    hy : Membership.mem (interior s) y
    h : Ne y x
    hxy : LT.lt 0 (Norm.norm (VSub.vsub y x))
    u : Set Q
    hu₁ : HasSubset.Subset u s
    hu₂ : IsOpen u
    hu₃ : Membership.mem u y
    ε : Real
    hε : GT.gt ε 0
    hyε : HasSubset.Subset (Metric.ball y ε) u
    ⊢ Exists fun i => And (LT.lt 0 i) (∀ ⦃x_1 : 𝕜⦄, Membership.mem (setOf fun y => …
  -/
  refine ⟨ε / ‖y -ᵥ x‖, div_pos hε hxy, fun δ (hδ : ‖δ - 1‖ < ε / ‖y -ᵥ x‖) => hu₁ (hyε ?_)⟩
  /-
    case inr.intro.intro.intro.intro.intro
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    x : Q
    s : Set Q
    y : Q
    hy : Membership.mem (interior s) y
    h : Ne y x
    hxy : LT.lt 0 (Norm.norm (VSub.vsub y x))
    u : Set Q
    hu₁ : HasSubset.Subset u s
    hu₂ : IsOpen u
    hu₃ : Membership.mem u y
    ε : Real
    hε : GT.gt ε 0
    hyε : HasSubset.Subset (Metric.ball y ε) u
    δ : 𝕜
    hδ : LT.lt (Norm.norm (HSub.hSub δ 1)) (HDiv.hDiv ε (Norm.norm (VSub.vsub y x)))
    ⊢ Membership.mem (Metric.ball y ε) ((AffineMap.homothety x δ) y)
  -/
  rw [lt_div_iff₀ hxy, ← norm_smul, sub_smul, one_smul] at hδ
  /-
    case inr.intro.intro.intro.intro.intro
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    x : Q
    s : Set Q
    y : Q
    hy : Membership.mem (interior s) y
    h : Ne y x
    hxy : LT.lt 0 (Norm.norm (VSub.vsub y x))
    u : Set Q
    hu₁ : HasSubset.Subset u s
    hu₂ : IsOpen u
    hu₃ : Membership.mem u y
    ε : Real
    hε : GT.gt ε 0
    hyε : HasSubset.Subset (Metric.ball y ε) u
    δ : 𝕜
    hδ : LT.lt (Norm.norm (HSub.hSub (HSMul.hSMul δ (VSub.vsub y x)) (VSub.vsub y  …
    ⊢ Membership.mem (Metric.ball y ε) ((AffineMap.homothety x δ) y)
  -/
  rwa [homothety_apply, Metric.mem_ball, dist_eq_norm_vsub W, vadd_vsub_eq_sub_vsub]
  /-
    🎉 no goals
  -/


theorem eventually_homothety_image_subset_of_finite_subset_interior (x : Q) {s : Set Q} {t : Set Q}
    (ht : t.Finite) (h : t ⊆ interior s) : ∀ᶠ δ in 𝓝 (1 : 𝕜), homothety x δ '' t ⊆ s := by
  suffices ∀ y ∈ t, ∀ᶠ δ in 𝓝 (1 : 𝕜), homothety x δ y ∈ s by
    simp_rw [Set.image_subset_iff]
    exact (Filter.eventually_all_finite ht).mpr this
  /-
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    x : Q
    s t : Set Q
    ht : t.Finite
    h : HasSubset.Subset t (interior s)
    ⊢ ∀ (y : Q), Membership.mem t y → Filter.Eventually (fun δ => Membership.mem s …
  -/
  intro y hy
  /-
    W : Type u_3
    Q : Type u_4
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : MetricSpace Q
    inst✝² : NormedAddTorsor W Q
    𝕜 : Type u_5
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 W
    x : Q
    s t : Set Q
    ht : t.Finite
    h : HasSubset.Subset t (interior s)
    y : Q
    hy : Membership.mem t y
    ⊢ Filter.Eventually (fun δ => Membership.mem s ((AffineMap.homothety x δ) y))  …
  -/
  exact eventually_homothety_mem_of_mem_interior 𝕜 x (h hy)
  /-
    🎉 no goals
  -/


theorem dist_midpoint_midpoint_le (p₁ p₂ p₃ p₄ : V) :
    dist (midpoint ℝ p₁ p₂) (midpoint ℝ p₃ p₄) ≤ (dist p₁ p₃ + dist p₂ p₄) / 2 := by
  /-
    V : Type u_1
    inst✝¹ : SeminormedAddCommGroup V
    inst✝ : NormedSpace Real V
    p₁ p₂ p₃ p₄ : V
    ⊢ LE.le (Dist.dist (midpoint Real p₁ p₂) (midpoint Real p₃ p₄)) (HDiv.hDiv (HA …
  -/
  simpa using dist_midpoint_midpoint_le' (𝕜 := ℝ) p₁ p₂ p₃ p₄
  /-
    🎉 no goals
  -/


theorem nndist_midpoint_midpoint_le (p₁ p₂ p₃ p₄ : V) :
    nndist (midpoint ℝ p₁ p₂) (midpoint ℝ p₃ p₄) ≤ (nndist p₁ p₃ + nndist p₂ p₄) / 2 :=
  dist_midpoint_midpoint_le _ _ _ _


/-- A continuous map between two normed affine spaces is an affine map provided that
it sends midpoints to midpoints. -/
def AffineMap.ofMapMidpoint (f : P → Q) (h : ∀ x y, f (midpoint ℝ x y) = midpoint ℝ (f x) (f y))
    (hfc : Continuous f) : P →ᵃ[ℝ] Q :=
  let c := Classical.arbitrary P
  AffineMap.mk' f (↑((AddMonoidHom.ofMapMidpoint ℝ ℝ
                                                                                  /-
                                                                                    V : Type u_1
                                                                                    P : Type u_2
                                                                                    W : Type u_3
                                                                                    Q : Type u_4
                                                                                    inst✝⁷ : SeminormedAddCommGroup V
                                                                                    inst✝⁶ : PseudoMetricSpace P
                                                                                    inst✝⁵ : NormedAddTorsor V P
                                                                                    inst✝⁴ : NormedAddCommGroup W
                                                                                    inst✝³ : MetricSpace Q
                                                                                    inst✝² : NormedAddTorsor W Q
                                                                                    inst✝¹ : NormedSpace Real V
                                                                                    inst✝ : NormedSpace Real W
                                                                                    f : P → Q
                                                                                    h : ∀ (x y : P), Eq (f (midpoint Real x y)) (midpoint Real (f x) (f y))
                                                                                    hfc : Continuous f
                                                                                    c : P := Classical.arbitrary P
                                                                                    ⊢ Eq (Function.comp (⇑(AffineEquiv.vaddConst Real (f c)).symm) (Function.comp  …
                                                                                  -/
    ((AffineEquiv.vaddConst ℝ (f <| c)).symm ∘ f ∘ AffineEquiv.vaddConst ℝ c) (by simp)
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
    fun x y => by -- Porting note: was `by simp [h]`
      simp only [c, Function.comp_apply, AffineEquiv.vaddConst_apply,
        AffineEquiv.vaddConst_symm_apply]
      conv_lhs => rw [(midpoint_self ℝ (Classical.arbitrary P)).symm, midpoint_vadd_midpoint, h, h,
          midpoint_vsub_midpoint]).toRealLinearMap <| by
        /-
          V : Type u_1
          P : Type u_2
          W : Type u_3
          Q : Type u_4
          inst✝⁷ : SeminormedAddCommGroup V
          inst✝⁶ : PseudoMetricSpace P
          inst✝⁵ : NormedAddTorsor V P
          inst✝⁴ : NormedAddCommGroup W
          inst✝³ : MetricSpace Q
          inst✝² : NormedAddTorsor W Q
          inst✝¹ : NormedSpace Real V
          inst✝ : NormedSpace Real W
          f : P → Q
          h : ∀ (x y : P), Eq (f (midpoint Real x y)) (midpoint Real (f x) (f y))
          hfc : Continuous f
          c : P := Classical.arbitrary P
          ⊢ Continuous ⇑(AddMonoidHom.ofMapMidpoint Real Real (Function.comp (⇑(AffineEq …
        -/
        apply_rules [Continuous.vadd, Continuous.vsub, continuous_const, hfc.comp, continuous_id]))
        /-
          🎉 no goals
        -/
                  /-
                    V : Type u_1
                    P : Type u_2
                    W : Type u_3
                    Q : Type u_4
                    inst✝⁷ : SeminormedAddCommGroup V
                    inst✝⁶ : PseudoMetricSpace P
                    inst✝⁵ : NormedAddTorsor V P
                    inst✝⁴ : NormedAddCommGroup W
                    inst✝³ : MetricSpace Q
                    inst✝² : NormedAddTorsor W Q
                    inst✝¹ : NormedSpace Real V
                    inst✝ : NormedSpace Real W
                    f : P → Q
                    h : ∀ (x y : P), Eq (f (midpoint Real x y)) (midpoint Real (f x) (f y))
                    hfc : Continuous f
                    c : P := Classical.arbitrary P
                    p : P
                    ⊢ Eq (f p) (HVAdd.hVAdd (↑((AddMonoidHom.ofMapMidpoint Real Real (Function.com …
                  -/
    c fun p => by simp
                  /-
                    🎉 no goals
                  -/


/-- Scaling by an element `k` of the scalar ring as a `DilationEquiv` with ratio `‖k‖₊`, mapping
from a normed space to a normed torsor over that space sending `0` to `c`. -/
@[simps]
def DilationEquiv.smulTorsor (c : P) {k : 𝕜} (hk : k ≠ 0) : E ≃ᵈ P where
  toFun := (k • · +ᵥ c)
  invFun := k⁻¹ • (· -ᵥ c)
                   /-
                     𝕜 : Type u_1
                     E : Type u_2
                     inst✝⁵ : NormedDivisionRing 𝕜
                     inst✝⁴ : SeminormedAddCommGroup E
                     inst✝³ : Module 𝕜 E
                     inst✝² : BoundedSMul 𝕜 E
                     P : Type u_3
                     inst✝¹ : PseudoMetricSpace P
                     inst✝ : NormedAddTorsor E P
                     c : P
                     k : 𝕜
                     hk : Ne k 0
                     x : E
                     ⊢ Eq (HSMul.hSMul (Inv.inv k) (fun x => VSub.vsub x c) ((fun x => HVAdd.hVAdd  …
                   -/
  left_inv x := by simp [inv_smul_smul₀ hk]
                   /-
                     🎉 no goals
                   -/
                    /-
                      𝕜 : Type u_1
                      E : Type u_2
                      inst✝⁵ : NormedDivisionRing 𝕜
                      inst✝⁴ : SeminormedAddCommGroup E
                      inst✝³ : Module 𝕜 E
                      inst✝² : BoundedSMul 𝕜 E
                      P : Type u_3
                      inst✝¹ : PseudoMetricSpace P
                      inst✝ : NormedAddTorsor E P
                      c : P
                      k : 𝕜
                      hk : Ne k 0
                      p : P
                      ⊢ Eq ((fun x => HVAdd.hVAdd (HSMul.hSMul k x) c) (HSMul.hSMul (Inv.inv k) (fun …
                    -/
  right_inv p := by simp [smul_inv_smul₀ hk]
                    /-
                      🎉 no goals
                    -/
  edist_eq' := ⟨‖k‖₊, nnnorm_ne_zero_iff.mpr hk, fun x y ↦ by
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : NormedDivisionRing 𝕜
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : BoundedSMul 𝕜 E
      P : Type u_3
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor E P
      c : P
      k : 𝕜
      hk : Ne k 0
      x y : E
      ⊢ Eq (EDist.edist ({ toFun := fun x => HVAdd.hVAdd (HSMul.hSMul k x) c, invFun …
    -/
    rw [show edist (k • x +ᵥ c) (k • y +ᵥ c) = _ from (IsometryEquiv.vaddConst c).isometry ..]
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : NormedDivisionRing 𝕜
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : BoundedSMul 𝕜 E
      P : Type u_3
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor E P
      c : P
      k : 𝕜
      hk : Ne k 0
      x y : E
      ⊢ Eq (EDist.edist (HSMul.hSMul k x) (HSMul.hSMul k y)) (HMul.hMul (↑(NNNorm.nn …
    -/
    exact edist_smul₀ ..⟩
    /-
      🎉 no goals
    -/


@[simp]
lemma DilationEquiv.smulTorsor_ratio {c : P} {k : 𝕜} (hk : k ≠ 0) {x y : E}
    (h : dist x y ≠ 0) : ratio (smulTorsor c hk) = ‖k‖₊ :=
                                                  /-
                                                    𝕜 : Type u_1
                                                    E : Type u_2
                                                    inst✝⁵ : NormedDivisionRing 𝕜
                                                    inst✝⁴ : SeminormedAddCommGroup E
                                                    inst✝³ : Module 𝕜 E
                                                    inst✝² : BoundedSMul 𝕜 E
                                                    P : Type u_3
                                                    inst✝¹ : PseudoMetricSpace P
                                                    inst✝ : NormedAddTorsor E P
                                                    c : P
                                                    k : 𝕜
                                                    hk : Ne k 0
                                                    x y : E
                                                    h : Ne (Dist.dist x y) 0
                                                    ⊢ Eq (Dist.dist ((DilationEquiv.smulTorsor c hk) x) ((DilationEquiv.smulTorsor …
                                                  -/
  Eq.symm <| ratio_unique_of_dist_ne_zero h <| by simp [dist_eq_norm, ← smul_sub, norm_smul]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
lemma DilationEquiv.smulTorsor_preimage_ball {c : P} {k : 𝕜} (hk : k ≠ 0) :
    smulTorsor c hk ⁻¹' (Metric.ball c ‖k‖) = Metric.ball (0 : E) 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : NormedDivisionRing 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : BoundedSMul 𝕜 E
    P : Type u_3
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor E P
    c : P
    k : 𝕜
    hk : Ne k 0
    ⊢ Eq (Set.preimage (⇑(DilationEquiv.smulTorsor c hk)) (Metric.ball c (Norm.nor …
  -/
  aesop (add simp norm_smul)
  /-
    🎉 no goals
  -/


