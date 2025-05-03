@[to_additive (attr := simp) comap_norm_atTop]
lemma comap_norm_atTop' : comap norm atTop = cobounded E := by
  /-
    E : Type u_2
    inst✝ : SeminormedGroup E
    ⊢ Eq (Filter.comap Norm.norm Filter.atTop) (Bornology.cobounded E)
  -/
  simpa only [dist_one_right] using comap_dist_right_atTop (1 : E)
  /-
    🎉 no goals
  -/


@[to_additive Filter.HasBasis.cobounded_of_norm]
lemma Filter.HasBasis.cobounded_of_norm' {ι : Sort*} {p : ι → Prop} {s : ι → Set ℝ}
    (h : HasBasis atTop p s) : HasBasis (cobounded E) p fun i ↦ norm ⁻¹' s i :=
  comap_norm_atTop' (E := E) ▸ h.comap _


@[to_additive Filter.hasBasis_cobounded_norm]
lemma Filter.hasBasis_cobounded_norm' : HasBasis (cobounded E) (fun _ ↦ True) ({x | · ≤ ‖x‖}) :=
  atTop_basis.cobounded_of_norm'


@[to_additive (attr := simp) tendsto_norm_atTop_iff_cobounded]
lemma tendsto_norm_atTop_iff_cobounded' {f : α → E} {l : Filter α} :
    Tendsto (‖f ·‖) l atTop ↔ Tendsto f l (cobounded E) := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : SeminormedGroup E
    f : α → E
    l : Filter α
    ⊢ Iff (Filter.Tendsto (fun x => Norm.norm (f x)) l Filter.atTop) (Filter.Tends …
  -/
  rw [← comap_norm_atTop', tendsto_comap_iff]; rfl
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive tendsto_norm_cobounded_atTop]
lemma tendsto_norm_cobounded_atTop' : Tendsto norm (cobounded E) atTop :=
  tendsto_norm_atTop_iff_cobounded'.2 tendsto_id


@[to_additive eventually_cobounded_le_norm]
lemma eventually_cobounded_le_norm' (a : ℝ) : ∀ᶠ x in cobounded E, a ≤ ‖x‖ :=
  tendsto_norm_cobounded_atTop'.eventually_ge_atTop a


@[to_additive tendsto_norm_cocompact_atTop]
lemma tendsto_norm_cocompact_atTop' [ProperSpace E] : Tendsto norm (cocompact E) atTop :=
  cobounded_eq_cocompact (α := E) ▸ tendsto_norm_cobounded_atTop'


@[to_additive (attr := simp)]
lemma Filter.inv_cobounded : (cobounded E)⁻¹ = cobounded E := by
  /-
    E : Type u_2
    inst✝ : SeminormedGroup E
    ⊢ Eq (Inv.inv (Bornology.cobounded E)) (Bornology.cobounded E)
  -/
  simp only [← comap_norm_atTop', ← Filter.comap_inv, comap_comap, Function.comp_def, norm_inv']
  /-
    🎉 no goals
  -/


/-- In a (semi)normed group, inversion `x ↦ x⁻¹` tends to infinity at infinity. -/
@[to_additive "In a (semi)normed group, negation `x ↦ -x` tends to infinity at infinity."]
theorem Filter.tendsto_inv_cobounded : Tendsto Inv.inv (cobounded E) (cobounded E) :=
  inv_cobounded.le


@[to_additive isBounded_iff_forall_norm_le]
lemma isBounded_iff_forall_norm_le' : Bornology.IsBounded s ↔ ∃ C, ∀ x ∈ s, ‖x‖ ≤ C := by
  /-
    E : Type u_2
    inst✝ : SeminormedGroup E
    s : Set E
    ⊢ Iff (Bornology.IsBounded s) (Exists fun C => ∀ (x : E), Membership.mem s x → …
  -/
  simpa only [Set.subset_def, mem_closedBall_one_iff] using isBounded_iff_subset_closedBall (1 : E)
  /-
    🎉 no goals
  -/


alias ⟨Bornology.IsBounded.exists_norm_le', _⟩ := isBounded_iff_forall_norm_le'


alias ⟨Bornology.IsBounded.exists_norm_le, _⟩ := isBounded_iff_forall_norm_le


@[to_additive exists_pos_norm_le]
lemma Bornology.IsBounded.exists_pos_norm_le' (hs : IsBounded s) : ∃ R > 0, ∀ x ∈ s, ‖x‖ ≤ R :=
  let ⟨R₀, hR₀⟩ := hs.exists_norm_le'
                /-
                  E : Type u_2
                  inst✝ : SeminormedGroup E
                  s : Set E
                  hs : Bornology.IsBounded s
                  R₀ : Real
                  hR₀ : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm x) R₀
                  ⊢ GT.gt (Max.max R₀ 1) 0
                -/
  ⟨max R₀ 1, by positivity, fun x hx => (hR₀ x hx).trans <| le_max_left _ _⟩
                /-
                  🎉 no goals
                -/


@[to_additive Bornology.IsBounded.exists_pos_norm_lt]
lemma Bornology.IsBounded.exists_pos_norm_lt' (hs : IsBounded s) : ∃ R > 0, ∀ x ∈ s, ‖x‖ < R :=
  let ⟨R, hR₀, hR⟩ := hs.exists_pos_norm_le'
             /-
               E : Type u_2
               inst✝ : SeminormedGroup E
               s : Set E
               hs : Bornology.IsBounded s
               R : Real
               hR₀ : GT.gt R 0
               hR : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm x) R
               ⊢ GT.gt (HAdd.hAdd R 1) 0
             -/
  ⟨R + 1, by positivity, fun x hx ↦ (hR x hx).trans_lt (lt_add_one _)⟩
             /-
               🎉 no goals
             -/


@[to_additive]
lemma NormedCommGroup.cauchySeq_iff [Nonempty α] [SemilatticeSup α] {u : α → E} :
    CauchySeq u ↔ ∀ ε > 0, ∃ N, ∀ m, N ≤ m → ∀ n, N ≤ n → ‖u m / u n‖ < ε := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : SeminormedGroup E
    inst✝¹ : Nonempty α
    inst✝ : SemilatticeSup α
    u : α → E
    ⊢ Iff (CauchySeq u) (∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (m : α), LE.l …
  -/
  simp [Metric.cauchySeq_iff, dist_eq_norm_div]
  /-
    🎉 no goals
  -/


@[to_additive IsCompact.exists_bound_of_continuousOn]
lemma IsCompact.exists_bound_of_continuousOn' [TopologicalSpace α] {s : Set α} (hs : IsCompact s)
    {f : α → E} (hf : ContinuousOn f s) : ∃ C, ∀ x ∈ s, ‖f x‖ ≤ C :=
  (isBounded_iff_forall_norm_le'.1 (hs.image_of_continuousOn hf).isBounded).imp fun _C hC _x hx =>
    hC _ <| Set.mem_image_of_mem _ hx


@[to_additive]
lemma HasCompactMulSupport.exists_bound_of_continuous [TopologicalSpace α]
    {f : α → E} (hf : HasCompactMulSupport f) (h'f : Continuous f) : ∃ C, ∀ x, ‖f x‖ ≤ C := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedGroup E
    inst✝ : TopologicalSpace α
    f : α → E
    hf : HasCompactMulSupport f
    h'f : Continuous f
    ⊢ Exists fun C => ∀ (x : α), LE.le (Norm.norm (f x)) C
  -/
  simpa using (hf.isCompact_range h'f).isBounded.exists_norm_le'
  /-
    🎉 no goals
  -/


/-- A helper lemma used to prove that the (scalar or usual) product of a function that tends to one
and a bounded function tends to one. This lemma is formulated for any binary operation
`op : E → F → G` with an estimate `‖op x y‖ ≤ A * ‖x‖ * ‖y‖` for some constant A instead of
multiplication so that it can be applied to `(*)`, `flip (*)`, `(•)`, and `flip (•)`. -/
@[to_additive "A helper lemma used to prove that the (scalar or usual) product of a function that
tends to zero and a bounded function tends to zero. This lemma is formulated for any binary
operation `op : E → F → G` with an estimate `‖op x y‖ ≤ A * ‖x‖ * ‖y‖` for some constant A instead
of multiplication so that it can be applied to `(*)`, `flip (*)`, `(•)`, and `flip (•)`."]
lemma Filter.Tendsto.op_one_isBoundedUnder_le' {f : α → E} {g : α → F} {l : Filter α}
    (hf : Tendsto f l (𝓝 1)) (hg : IsBoundedUnder (· ≤ ·) l (Norm.norm ∘ g)) (op : E → F → G)
    (h_op : ∃ A, ∀ x y, ‖op x y‖ ≤ A * ‖x‖ * ‖y‖) : Tendsto (fun x => op (f x) (g x)) l (𝓝 1) := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝² : SeminormedGroup E
    inst✝¹ : SeminormedGroup F
    inst✝ : SeminormedGroup G
    f : α → E
    g : α → F
    l : Filter α
    hf : Filter.Tendsto f l (nhds 1)
    hg : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Function.comp Norm.no …
    op : E → F → G
    h_op : Exists fun A => ∀ (x : E) (y : F), LE.le (Norm.norm (op x y)) (HMul.hMu …
    ⊢ Filter.Tendsto (fun x => op (f x) (g x)) l (nhds 1)
  -/
  cases' h_op with A h_op
  /-
    case intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝² : SeminormedGroup E
    inst✝¹ : SeminormedGroup F
    inst✝ : SeminormedGroup G
    f : α → E
    g : α → F
    l : Filter α
    hf : Filter.Tendsto f l (nhds 1)
    hg : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Function.comp Norm.no …
    op : E → F → G
    A : Real
    h_op : ∀ (x : E) (y : F), LE.le (Norm.norm (op x y)) (HMul.hMul (HMul.hMul A ( …
    ⊢ Filter.Tendsto (fun x => op (f x) (g x)) l (nhds 1)
  -/
  rcases hg with ⟨C, hC⟩; rw [eventually_map] at hC
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝² : SeminormedGroup E
    inst✝¹ : SeminormedGroup F
    inst✝ : SeminormedGroup G
    f : α → E
    g : α → F
    l : Filter α
    hf : Filter.Tendsto f l (nhds 1)
    op : E → F → G
    A : Real
    h_op : ∀ (x : E) (y : F), LE.le (Norm.norm (op x y)) (HMul.hMul (HMul.hMul A ( …
    C : Real
    hC : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Function.comp Nor …
    ⊢ Filter.Tendsto (fun x => op (f x) (g x)) l (nhds 1)
  -/
  rw [NormedCommGroup.tendsto_nhds_one] at hf ⊢
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝² : SeminormedGroup E
    inst✝¹ : SeminormedGroup F
    inst✝ : SeminormedGroup G
    f : α → E
    g : α → F
    l : Filter α
    hf : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Norm.norm (f …
    op : E → F → G
    A : Real
    h_op : ∀ (x : E) (y : F), LE.le (Norm.norm (op x y)) (HMul.hMul (HMul.hMul A ( …
    C : Real
    hC : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Function.comp Nor …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Norm.norm (op ( …
  -/
  intro ε ε₀
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝² : SeminormedGroup E
    inst✝¹ : SeminormedGroup F
    inst✝ : SeminormedGroup G
    f : α → E
    g : α → F
    l : Filter α
    hf : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Norm.norm (f …
    op : E → F → G
    A : Real
    h_op : ∀ (x : E) (y : F), LE.le (Norm.norm (op x y)) (HMul.hMul (HMul.hMul A ( …
    C : Real
    hC : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Function.comp Nor …
    ε : Real
    ε₀ : GT.gt ε 0
    ⊢ Filter.Eventually (fun x => LT.lt (Norm.norm (op (f x) (g x))) ε) l
  -/
  rcases exists_pos_mul_lt ε₀ (A * C) with ⟨δ, δ₀, hδ⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝² : SeminormedGroup E
    inst✝¹ : SeminormedGroup F
    inst✝ : SeminormedGroup G
    f : α → E
    g : α → F
    l : Filter α
    hf : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Norm.norm (f …
    op : E → F → G
    A : Real
    h_op : ∀ (x : E) (y : F), LE.le (Norm.norm (op x y)) (HMul.hMul (HMul.hMul A ( …
    C : Real
    hC : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Function.comp Nor …
    ε : Real
    ε₀ : GT.gt ε 0
    δ : Real
    δ₀ : LT.lt 0 δ
    hδ : LT.lt (HMul.hMul (HMul.hMul A C) δ) ε
    ⊢ Filter.Eventually (fun x => LT.lt (Norm.norm (op (f x) (g x))) ε) l
  -/
  filter_upwards [hf δ δ₀, hC] with i hf hg
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝² : SeminormedGroup E
    inst✝¹ : SeminormedGroup F
    inst✝ : SeminormedGroup G
    f : α → E
    g : α → F
    l : Filter α
    hf✝ : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Norm.norm ( …
    op : E → F → G
    A : Real
    h_op : ∀ (x : E) (y : F), LE.le (Norm.norm (op x y)) (HMul.hMul (HMul.hMul A ( …
    C : Real
    hC : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Function.comp Nor …
    ε : Real
    ε₀ : GT.gt ε 0
    δ : Real
    δ₀ : LT.lt 0 δ
    hδ : LT.lt (HMul.hMul (HMul.hMul A C) δ) ε
    i : α
    hf : LT.lt (Norm.norm (f i)) δ
    hg : LE.le (Function.comp Norm.norm g i) C
    ⊢ LT.lt (Norm.norm (op (f i) (g i))) ε
  -/
  refine (h_op _ _).trans_lt ?_
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝² : SeminormedGroup E
    inst✝¹ : SeminormedGroup F
    inst✝ : SeminormedGroup G
    f : α → E
    g : α → F
    l : Filter α
    hf✝ : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Norm.norm ( …
    op : E → F → G
    A : Real
    h_op : ∀ (x : E) (y : F), LE.le (Norm.norm (op x y)) (HMul.hMul (HMul.hMul A ( …
    C : Real
    hC : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Function.comp Nor …
    ε : Real
    ε₀ : GT.gt ε 0
    δ : Real
    δ₀ : LT.lt 0 δ
    hδ : LT.lt (HMul.hMul (HMul.hMul A C) δ) ε
    i : α
    hf : LT.lt (Norm.norm (f i)) δ
    hg : LE.le (Function.comp Norm.norm g i) C
    ⊢ LT.lt (HMul.hMul (HMul.hMul A (Norm.norm (f i))) (Norm.norm (g i))) ε
  -/
  rcases le_total A 0 with hA | hA
  · exact (mul_nonpos_of_nonpos_of_nonneg (mul_nonpos_of_nonpos_of_nonneg hA <| norm_nonneg' _) <|
      norm_nonneg' _).trans_lt ε₀
  calc
    A * ‖f i‖ * ‖g i‖ ≤ A * δ * C := by gcongr; exact hg
    _ = A * C * δ := mul_right_comm _ _ _
    _ < ε := hδ


/-- A helper lemma used to prove that the (scalar or usual) product of a function that tends to one
and a bounded function tends to one. This lemma is formulated for any binary operation
`op : E → F → G` with an estimate `‖op x y‖ ≤ ‖x‖ * ‖y‖` instead of multiplication so that it
can be applied to `(*)`, `flip (*)`, `(•)`, and `flip (•)`. -/
@[to_additive "A helper lemma used to prove that the (scalar or usual) product of a function that
tends to zero and a bounded function tends to zero. This lemma is formulated for any binary
operation `op : E → F → G` with an estimate `‖op x y‖ ≤ ‖x‖ * ‖y‖` instead of multiplication so
that it can be applied to `(*)`, `flip (*)`, `(•)`, and `flip (•)`."]
theorem Filter.Tendsto.op_one_isBoundedUnder_le {f : α → E} {g : α → F} {l : Filter α}
    (hf : Tendsto f l (𝓝 1)) (hg : IsBoundedUnder (· ≤ ·) l (Norm.norm ∘ g)) (op : E → F → G)
    (h_op : ∀ x y, ‖op x y‖ ≤ ‖x‖ * ‖y‖) : Tendsto (fun x => op (f x) (g x)) l (𝓝 1) :=
  hf.op_one_isBoundedUnder_le' hg op ⟨1, fun x y => (one_mul ‖x‖).symm ▸ h_op x y⟩


lemma Continuous.bounded_above_of_compact_support (hf : Continuous f) (h : HasCompactSupport f) :
    ∃ C, ∀ x, ‖f x‖ ≤ C := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddGroup E
    inst✝ : TopologicalSpace α
    f : α → E
    hf : Continuous f
    h : HasCompactSupport f
    ⊢ Exists fun C => ∀ (x : α), LE.le (Norm.norm (f x)) C
  -/
  simpa [bddAbove_def] using hf.norm.bddAbove_range_of_hasCompactSupport h.norm
  /-
    🎉 no goals
  -/


@[to_additive]
lemma HasCompactMulSupport.exists_pos_le_norm [One E] (hf : HasCompactMulSupport f) :
    ∃ R : ℝ, 0 < R ∧ ∀ x : α, R ≤ ‖x‖ → f x = 1 := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddGroup α
    f : α → E
    inst✝ : One E
    hf : HasCompactMulSupport f
    ⊢ Exists fun R => And (LT.lt 0 R) (∀ (x : α), LE.le R (Norm.norm x) → Eq (f x) …
  -/
  obtain ⟨K, ⟨hK1, hK2⟩⟩ := exists_compact_iff_hasCompactMulSupport.mpr hf
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddGroup α
    f : α → E
    inst✝ : One E
    hf : HasCompactMulSupport f
    K : Set α
    hK1 : IsCompact K
    hK2 : ∀ (x : α), Not (Membership.mem K x) → Eq (f x) 1
    ⊢ Exists fun R => And (LT.lt 0 R) (∀ (x : α), LE.le R (Norm.norm x) → Eq (f x) …
  -/
  obtain ⟨S, hS, hS'⟩ := hK1.isBounded.exists_pos_norm_le
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddGroup α
    f : α → E
    inst✝ : One E
    hf : HasCompactMulSupport f
    K : Set α
    hK1 : IsCompact K
    hK2 : ∀ (x : α), Not (Membership.mem K x) → Eq (f x) 1
    S : Real
    hS : GT.gt S 0
    hS' : ∀ (x : α), Membership.mem K x → LE.le (Norm.norm x) S
    ⊢ Exists fun R => And (LT.lt 0 R) (∀ (x : α), LE.le R (Norm.norm x) → Eq (f x) …
  -/
  refine ⟨S + 1, by positivity, fun x hx => hK2 x ((mt <| hS' x) ?_)⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddGroup α
    f : α → E
    inst✝ : One E
    hf : HasCompactMulSupport f
    K : Set α
    hK1 : IsCompact K
    hK2 : ∀ (x : α), Not (Membership.mem K x) → Eq (f x) 1
    S : Real
    hS : GT.gt S 0
    hS' : ∀ (x : α), Membership.mem K x → LE.le (Norm.norm x) S
    x : α
    hx : LE.le (HAdd.hAdd S 1) (Norm.norm x)
    ⊢ Not (LE.le (Norm.norm x) S)
  -/
  contrapose! hx
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddGroup α
    f : α → E
    inst✝ : One E
    hf : HasCompactMulSupport f
    K : Set α
    hK1 : IsCompact K
    hK2 : ∀ (x : α), Not (Membership.mem K x) → Eq (f x) 1
    S : Real
    hS : GT.gt S 0
    hS' : ∀ (x : α), Membership.mem K x → LE.le (Norm.norm x) S
    x : α
    hx : LE.le (Norm.norm x) S
    ⊢ LT.lt (Norm.norm x) (HAdd.hAdd S 1)
  -/
  exact lt_add_of_le_of_pos hx zero_lt_one
  /-
    🎉 no goals
  -/


