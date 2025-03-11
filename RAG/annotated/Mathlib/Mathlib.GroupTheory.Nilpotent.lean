/-- If `H` is a normal subgroup of `G`, then the set `{x : G | ∀ y : G, x*y*x⁻¹*y⁻¹ ∈ H}`
is a subgroup of `G` (because it is the preimage in `G` of the centre of the
quotient group `G/H`.)
-/
def upperCentralSeriesStep : Subgroup G where
  carrier := { x : G | ∀ y : G, x * y * x⁻¹ * y⁻¹ ∈ H }
                   /-
                     G : Type u_1
                     inst✝¹ : Group G
                     H : Subgroup G
                     inst✝ : H.Normal
                     y : G
                     ⊢ Membership.mem H (HMul.hMul (HMul.hMul (HMul.hMul 1 y) (Inv.inv 1)) (Inv.inv …
                   -/
  one_mem' y := by simp [Subgroup.one_mem]
    /-
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      inst✝ : H.Normal
      a b : G
      ha : Membership.mem (setOf fun x => ∀ (y : G), Membership.mem H (HMul.hMul (HM …
      hb : Membership.mem (setOf fun x => ∀ (y : G), Membership.mem H (HMul.hMul (HM …
      y : G
      ⊢ Membership.mem H (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul a b) y) (Inv.in …
    -/
                   /-
                     🎉 no goals
                   -/
    /-
      case h.e'_5
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      inst✝ : H.Normal
      a b : G
      ha : Membership.mem (setOf fun x => ∀ (y : G), Membership.mem H (HMul.hMul (HM …
      hb : Membership.mem (setOf fun x => ∀ (y : G), Membership.mem H (HMul.hMul (HM …
      y : G
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul a b) y) (Inv.inv (HMul.hMul a …
    -/
  mul_mem' {a b ha hb y} := by
    /-
      🎉 no goals
    -/
    convert Subgroup.mul_mem _ (ha (b * y * b⁻¹)) (hb y) using 1
    group
  inv_mem' {x hx y} := by
    /-
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      inst✝ : H.Normal
      x : G
      hx : Membership.mem { carrier := setOf fun x => ∀ (y : G), Membership.mem H (H …
      y : G
      ⊢ Membership.mem H (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv x) y) (Inv.inv (I …
    -/
    specialize hx y⁻¹
    /-
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      inst✝ : H.Normal
      x y : G
      hx : Membership.mem H (HMul.hMul (HMul.hMul (HMul.hMul x (Inv.inv y)) (Inv.inv …
      ⊢ Membership.mem H (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv x) y) (Inv.inv (I …
    -/
    rw [mul_assoc, inv_inv] at hx ⊢
    /-
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      inst✝ : H.Normal
      x y : G
      hx : Membership.mem H (HMul.hMul (HMul.hMul x (Inv.inv y)) (HMul.hMul (Inv.inv …
      ⊢ Membership.mem H (HMul.hMul (HMul.hMul (Inv.inv x) y) (HMul.hMul x (Inv.inv  …
    -/
    exact Subgroup.Normal.mem_comm inferInstance hx
    /-
      🎉 no goals
    -/


theorem mem_upperCentralSeriesStep (x : G) :
    x ∈ upperCentralSeriesStep H ↔ ∀ y, x * y * x⁻¹ * y⁻¹ ∈ H := Iff.rfl


/-- The proof that `upperCentralSeriesStep H` is the preimage of the centre of `G/H` under
the canonical surjection. -/
theorem upperCentralSeriesStep_eq_comap_center :
    upperCentralSeriesStep H = Subgroup.comap (mk' H) (center (G ⧸ H)) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    ⊢ Eq (upperCentralSeriesStep H) (Subgroup.comap (QuotientGroup.mk' H) (Subgrou …
  -/
  ext
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    x✝ : G
    ⊢ Iff (Membership.mem (upperCentralSeriesStep H) x✝) (Membership.mem (Subgroup …
  -/
  rw [mem_comap, mem_center_iff, forall_mk]
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    x✝ : G
    ⊢ Iff (Membership.mem (upperCentralSeriesStep H) x✝) (∀ (x : G), Eq (HMul.hMul …
  -/
  apply forall_congr'
  /-
    case h.h
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    x✝ : G
    ⊢ ∀ (a : G), Iff (Membership.mem H (HMul.hMul (HMul.hMul (HMul.hMul x✝ a) (Inv …
  -/
  intro y
  rw [coe_mk', ← QuotientGroup.mk_mul, ← QuotientGroup.mk_mul, eq_comm, eq_iff_div_mem,
    div_eq_mul_inv, mul_inv_rev, mul_assoc]


instance : Normal (upperCentralSeriesStep H) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    ⊢ (upperCentralSeriesStep H).Normal
  -/
  rw [upperCentralSeriesStep_eq_comap_center]
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    ⊢ (Subgroup.comap (QuotientGroup.mk' H) (Subgroup.center (HasQuotient.Quotient …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- An auxiliary type-theoretic definition defining both the upper central series of
a group, and a proof that it is normal, all in one go. -/
def upperCentralSeriesAux : ℕ → Σ'H : Subgroup G, Normal H
  | 0 => ⟨⊥, inferInstance⟩
  | n + 1 =>
    let un := upperCentralSeriesAux n
    let _un_normal := un.2
    ⟨upperCentralSeriesStep un.1, inferInstance⟩


/-- `upperCentralSeries G n` is the `n`th term in the upper central series of `G`. -/
def upperCentralSeries (n : ℕ) : Subgroup G :=
  (upperCentralSeriesAux G n).1


instance upperCentralSeries_normal (n : ℕ) : Normal (upperCentralSeries G n) :=
  (upperCentralSeriesAux G n).2


@[simp]
theorem upperCentralSeries_zero : upperCentralSeries G 0 = ⊥ := rfl


@[simp]
theorem upperCentralSeries_one : upperCentralSeries G 1 = center G := by
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ Eq (upperCentralSeries G 1) (Subgroup.center G)
  -/
  ext
  simp only [upperCentralSeries, upperCentralSeriesAux, upperCentralSeriesStep,
    Subgroup.mem_center_iff, mem_mk, mem_bot, Set.mem_setOf_eq]
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    x✝ : G
    ⊢ Iff (∀ (y : G), Eq (HMul.hMul (HMul.hMul (HMul.hMul x✝ y) (Inv.inv x✝)) (Inv …
  -/
  exact forall_congr' fun y => by rw [mul_inv_eq_one, mul_inv_eq_iff_eq_mul, eq_comm]
  /-
    🎉 no goals
  -/


/-- The `n+1`st term of the upper central series `H i` has underlying set equal to the `x` such
that `⁅x,G⁆ ⊆ H n`-/
theorem mem_upperCentralSeries_succ_iff {n : ℕ} {x : G} :
    x ∈ upperCentralSeries G (n + 1) ↔ ∀ y : G, x * y * x⁻¹ * y⁻¹ ∈ upperCentralSeries G n :=
  Iff.rfl


@[simp] lemma comap_upperCentralSeries {H : Type*} [Group H] (e : H ≃* G) :
    ∀ n, (upperCentralSeries G n).comap e = upperCentralSeries H n
            /-
              G : Type u_1
              inst✝¹ : Group G
              H : Type u_2
              inst✝ : Group H
              e : MulEquiv H G
              ⊢ Eq (Subgroup.comap (↑e) (upperCentralSeries G 0)) (upperCentralSeries H 0)
            -/
  | 0 => by simpa [MonoidHom.ker_eq_bot_iff] using e.injective
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      e : MulEquiv H G
      n : Nat
      ⊢ Eq (Subgroup.comap (↑e) (upperCentralSeries G (HAdd.hAdd n 1))) (upperCentra …
    -/
    ext
    simp [mem_upperCentralSeries_succ_iff, ← comap_upperCentralSeries e n,
      ← e.toEquiv.forall_congr_right]


variable (G) in
-- `IsNilpotent` is already defined in the root namespace (for elements of rings).
-- TODO: Rename it to `IsNilpotentElement`?
/-- A group `G` is nilpotent if its upper central series is eventually `G`. -/
@[mk_iff]
class IsNilpotent (G : Type*) [Group G] : Prop where
  nilpotent' : ∃ n : ℕ, upperCentralSeries G n = ⊤

-- Porting note: add lemma since infer kinds are unsupported in the definition of `IsNilpotent`

lemma IsNilpotent.nilpotent (G : Type*) [Group G] [IsNilpotent G] :
    ∃ n : ℕ, upperCentralSeries G n = ⊤ := Group.IsNilpotent.nilpotent'


lemma isNilpotent_congr {H : Type*} [Group H] (e : G ≃* H) : IsNilpotent G ↔ IsNilpotent H := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    e : MulEquiv G H
    ⊢ Iff (Group.IsNilpotent G) (Group.IsNilpotent H)
  -/
  simp_rw [isNilpotent_iff]
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    e : MulEquiv G H
    ⊢ Iff (Exists fun n => Eq (upperCentralSeries G n) Top.top) (Exists fun n => E …
  -/
  refine exists_congr fun n ↦ ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      e : MulEquiv G H
      n : Nat
      h : Eq (upperCentralSeries G n) Top.top
      ⊢ Eq (upperCentralSeries H n) Top.top
    -/
  · simp [← Subgroup.comap_top e.symm.toMonoidHom, ← h]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      e : MulEquiv G H
      n : Nat
      h : Eq (upperCentralSeries H n) Top.top
      ⊢ Eq (upperCentralSeries G n) Top.top
    -/
  · simp [← Subgroup.comap_top e.toMonoidHom, ← h]
    /-
      🎉 no goals
    -/


@[simp] lemma isNilpotent_top : IsNilpotent (⊤ : Subgroup G) ↔ IsNilpotent G :=
  isNilpotent_congr Subgroup.topEquiv


variable (G) in
/-- A group `G` is virtually nilpotent if it has a nilpotent cofinite subgroup `N`. -/
def IsVirtuallyNilpotent : Prop := ∃ N : Subgroup G, IsNilpotent N ∧ FiniteIndex N


lemma IsNilpotent.isVirtuallyNilpotent (hG : IsNilpotent G) : IsVirtuallyNilpotent G :=
         /-
           G : Type u_1
           inst✝ : Group G
           hG : Group.IsNilpotent G
           ⊢ Group.IsNilpotent (Subtype fun x => Membership.mem Top.top x)
         -/
  ⟨⊤, by simpa, inferInstance⟩
         /-
           🎉 no goals
         -/


/-- A sequence of subgroups of `G` is an ascending central series if `H 0` is trivial and
  `⁅H (n + 1), G⁆ ⊆ H n` for all `n`. Note that we do not require that `H n = G` for some `n`. -/
def IsAscendingCentralSeries (H : ℕ → Subgroup G) : Prop :=
  H 0 = ⊥ ∧ ∀ (x : G) (n : ℕ), x ∈ H (n + 1) → ∀ g, x * g * x⁻¹ * g⁻¹ ∈ H n


/-- A sequence of subgroups of `G` is a descending central series if `H 0` is `G` and
  `⁅H n, G⁆ ⊆ H (n + 1)` for all `n`. Note that we do not require that `H n = {1}` for some `n`. -/
def IsDescendingCentralSeries (H : ℕ → Subgroup G) :=
  H 0 = ⊤ ∧ ∀ (x : G) (n : ℕ), x ∈ H n → ∀ g, x * g * x⁻¹ * g⁻¹ ∈ H (n + 1)


/-- Any ascending central series for a group is bounded above by the upper central series. -/
theorem ascending_central_series_le_upper (H : ℕ → Subgroup G) (hH : IsAscendingCentralSeries H) :
    ∀ n : ℕ, H n ≤ upperCentralSeries G n
  | 0 => hH.1.symm ▸ le_refl ⊥
  | n + 1 => by
    /-
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      n : Nat
      ⊢ LE.le (H (HAdd.hAdd n 1)) (upperCentralSeries G (HAdd.hAdd n 1))
    -/
    intro x hx
    /-
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      n : Nat
      x : G
      hx : Membership.mem (H (HAdd.hAdd n 1)) x
      ⊢ Membership.mem (upperCentralSeries G (HAdd.hAdd n 1)) x
    -/
    rw [mem_upperCentralSeries_succ_iff]
    /-
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      n : Nat
      x : G
      hx : Membership.mem (H (HAdd.hAdd n 1)) x
      ⊢ ∀ (y : G), Membership.mem (upperCentralSeries G n) (HMul.hMul (HMul.hMul (HM …
    -/
    exact fun y => ascending_central_series_le_upper H hH n (hH.2 x n hx y)
    /-
      🎉 no goals
    -/


/-- The upper central series of a group is an ascending central series. -/
theorem upperCentralSeries_isAscendingCentralSeries :
    IsAscendingCentralSeries (upperCentralSeries G) :=
  ⟨rfl, fun _x _n h => h⟩


theorem upperCentralSeries_mono : Monotone (upperCentralSeries G) := by
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ Monotone (upperCentralSeries G)
  -/
  refine monotone_nat_of_le_succ ?_
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ ∀ (n : Nat), LE.le (upperCentralSeries G n) (upperCentralSeries G (HAdd.hAdd …
  -/
  intro n x hx y
  /-
    G : Type u_1
    inst✝ : Group G
    n : Nat
    x : G
    hx : Membership.mem (upperCentralSeries G n) x
    y : G
    ⊢ Membership.mem (upperCentralSeriesAux G n).fst (HMul.hMul (HMul.hMul (HMul.h …
  -/
  rw [mul_assoc, mul_assoc, ← mul_assoc y x⁻¹ y⁻¹]
  /-
    G : Type u_1
    inst✝ : Group G
    n : Nat
    x : G
    hx : Membership.mem (upperCentralSeries G n) x
    y : G
    ⊢ Membership.mem (upperCentralSeriesAux G n).fst (HMul.hMul x (HMul.hMul (HMul …
  -/
  exact mul_mem hx (Normal.conj_mem (upperCentralSeries_normal G n) x⁻¹ (inv_mem hx) y)
  /-
    🎉 no goals
  -/


/-- A group `G` is nilpotent iff there exists an ascending central series which reaches `G` in
  finitely many steps. -/
theorem nilpotent_iff_finite_ascending_central_series :
    IsNilpotent G ↔ ∃ n : ℕ, ∃ H : ℕ → Subgroup G, IsAscendingCentralSeries H ∧ H n = ⊤ := by
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ Iff (Group.IsNilpotent G) (Exists fun n => Exists fun H => And (IsAscendingC …
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      ⊢ Group.IsNilpotent G → Exists fun n => Exists fun H => And (IsAscendingCentra …
    -/
  · rintro ⟨n, nH⟩
    /-
      case mp.mk.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      nH : Eq (upperCentralSeries G n) Top.top
      ⊢ Exists fun n => Exists fun H => And (IsAscendingCentralSeries H) (Eq (H n) T …
    -/
    exact ⟨_, _, upperCentralSeries_isAscendingCentralSeries G, nH⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      ⊢ (Exists fun n => Exists fun H => And (IsAscendingCentralSeries H) (Eq (H n)  …
    -/
  · rintro ⟨n, H, hH, hn⟩
    /-
      case mpr.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ Group.IsNilpotent G
    -/
    use n
    /-
      case h
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ Eq (upperCentralSeries G n) Top.top
    -/
    rw [eq_top_iff, ← hn]
    /-
      case h
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ LE.le (H n) (upperCentralSeries G n)
    -/
    exact ascending_central_series_le_upper H hH n
    /-
      🎉 no goals
    -/


theorem is_descending_rev_series_of_is_ascending {H : ℕ → Subgroup G} {n : ℕ} (hn : H n = ⊤)
    (hasc : IsAscendingCentralSeries H) : IsDescendingCentralSeries fun m : ℕ => H (n - m) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Nat → Subgroup G
    n : Nat
    hn : Eq (H n) Top.top
    hasc : IsAscendingCentralSeries H
    ⊢ IsDescendingCentralSeries fun m => H (HSub.hSub n m)
  -/
  cases' hasc with h0 hH
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    H : Nat → Subgroup G
    n : Nat
    hn : Eq (H n) Top.top
    h0 : Eq (H 0) Bot.bot
    hH : ∀ (x : G) (n : Nat), Membership.mem (H (HAdd.hAdd n 1)) x → ∀ (g : G), Me …
    ⊢ IsDescendingCentralSeries fun m => H (HSub.hSub n m)
  -/
  refine ⟨hn, fun x m hx g => ?_⟩
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    H : Nat → Subgroup G
    n : Nat
    hn : Eq (H n) Top.top
    h0 : Eq (H 0) Bot.bot
    hH : ∀ (x : G) (n : Nat), Membership.mem (H (HAdd.hAdd n 1)) x → ∀ (g : G), Me …
    x : G
    m : Nat
    hx : Membership.mem ((fun m => H (HSub.hSub n m)) m) x
    g : G
    ⊢ Membership.mem ((fun m => H (HSub.hSub n m)) (HAdd.hAdd m 1)) (HMul.hMul (HM …
  -/
  dsimp at hx
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    H : Nat → Subgroup G
    n : Nat
    hn : Eq (H n) Top.top
    h0 : Eq (H 0) Bot.bot
    hH : ∀ (x : G) (n : Nat), Membership.mem (H (HAdd.hAdd n 1)) x → ∀ (g : G), Me …
    x : G
    m : Nat
    hx : Membership.mem (H (HSub.hSub n m)) x
    g : G
    ⊢ Membership.mem ((fun m => H (HSub.hSub n m)) (HAdd.hAdd m 1)) (HMul.hMul (HM …
  -/
  by_cases hm : n ≤ m
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Top.top
      h0 : Eq (H 0) Bot.bot
      hH : ∀ (x : G) (n : Nat), Membership.mem (H (HAdd.hAdd n 1)) x → ∀ (g : G), Me …
      x : G
      m : Nat
      hx : Membership.mem (H (HSub.hSub n m)) x
      g : G
      hm : LE.le n m
      ⊢ Membership.mem ((fun m => H (HSub.hSub n m)) (HAdd.hAdd m 1)) (HMul.hMul (HM …
    -/
  · rw [tsub_eq_zero_of_le hm, h0, Subgroup.mem_bot] at hx
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Top.top
      h0 : Eq (H 0) Bot.bot
      hH : ∀ (x : G) (n : Nat), Membership.mem (H (HAdd.hAdd n 1)) x → ∀ (g : G), Me …
      x : G
      m : Nat
      hx : Eq x 1
      g : G
      hm : LE.le n m
      ⊢ Membership.mem ((fun m => H (HSub.hSub n m)) (HAdd.hAdd m 1)) (HMul.hMul (HM …
    -/
    subst hx
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Top.top
      h0 : Eq (H 0) Bot.bot
      hH : ∀ (x : G) (n : Nat), Membership.mem (H (HAdd.hAdd n 1)) x → ∀ (g : G), Me …
      m : Nat
      g : G
      hm : LE.le n m
      ⊢ Membership.mem ((fun m => H (HSub.hSub n m)) (HAdd.hAdd m 1)) (HMul.hMul (HM …
    -/
    rw [show (1 : G) * g * (1⁻¹ : G) * g⁻¹ = 1 by group]
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Top.top
      h0 : Eq (H 0) Bot.bot
      hH : ∀ (x : G) (n : Nat), Membership.mem (H (HAdd.hAdd n 1)) x → ∀ (g : G), Me …
      m : Nat
      g : G
      hm : LE.le n m
      ⊢ Membership.mem ((fun m => H (HSub.hSub n m)) (HAdd.hAdd m 1)) 1
    -/
    exact Subgroup.one_mem _
    /-
      🎉 no goals
    -/
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Top.top
      h0 : Eq (H 0) Bot.bot
      hH : ∀ (x : G) (n : Nat), Membership.mem (H (HAdd.hAdd n 1)) x → ∀ (g : G), Me …
      x : G
      m : Nat
      hx : Membership.mem (H (HSub.hSub n m)) x
      g : G
      hm : Not (LE.le n m)
      ⊢ Membership.mem ((fun m => H (HSub.hSub n m)) (HAdd.hAdd m 1)) (HMul.hMul (HM …
    -/
  · push_neg at hm
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Top.top
      h0 : Eq (H 0) Bot.bot
      hH : ∀ (x : G) (n : Nat), Membership.mem (H (HAdd.hAdd n 1)) x → ∀ (g : G), Me …
      x : G
      m : Nat
      hx : Membership.mem (H (HSub.hSub n m)) x
      g : G
      hm : LT.lt m n
      ⊢ Membership.mem ((fun m => H (HSub.hSub n m)) (HAdd.hAdd m 1)) (HMul.hMul (HM …
    -/
    apply hH
    /-
      case neg.a
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Top.top
      h0 : Eq (H 0) Bot.bot
      hH : ∀ (x : G) (n : Nat), Membership.mem (H (HAdd.hAdd n 1)) x → ∀ (g : G), Me …
      x : G
      m : Nat
      hx : Membership.mem (H (HSub.hSub n m)) x
      g : G
      hm : LT.lt m n
      ⊢ Membership.mem (H (HAdd.hAdd (HSub.hSub n (HAdd.hAdd m 1)) 1)) x
    -/
    convert hx using 1
    /-
      case h.e'_4
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Top.top
      h0 : Eq (H 0) Bot.bot
      hH : ∀ (x : G) (n : Nat), Membership.mem (H (HAdd.hAdd n 1)) x → ∀ (g : G), Me …
      x : G
      m : Nat
      hx : Membership.mem (H (HSub.hSub n m)) x
      g : G
      hm : LT.lt m n
      ⊢ Eq (H (HAdd.hAdd (HSub.hSub n (HAdd.hAdd m 1)) 1)) (H (HSub.hSub n m))
    -/
    rw [tsub_add_eq_add_tsub (Nat.succ_le_of_lt hm), Nat.succ_eq_add_one, Nat.add_sub_add_right]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-25")]
alias is_decending_rev_series_of_is_ascending := is_descending_rev_series_of_is_ascending


theorem is_ascending_rev_series_of_is_descending {H : ℕ → Subgroup G} {n : ℕ} (hn : H n = ⊥)
    (hdesc : IsDescendingCentralSeries H) : IsAscendingCentralSeries fun m : ℕ => H (n - m) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Nat → Subgroup G
    n : Nat
    hn : Eq (H n) Bot.bot
    hdesc : IsDescendingCentralSeries H
    ⊢ IsAscendingCentralSeries fun m => H (HSub.hSub n m)
  -/
  cases' hdesc with h0 hH
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    H : Nat → Subgroup G
    n : Nat
    hn : Eq (H n) Bot.bot
    h0 : Eq (H 0) Top.top
    hH : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
    ⊢ IsAscendingCentralSeries fun m => H (HSub.hSub n m)
  -/
  refine ⟨hn, fun x m hx g => ?_⟩
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    H : Nat → Subgroup G
    n : Nat
    hn : Eq (H n) Bot.bot
    h0 : Eq (H 0) Top.top
    hH : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
    x : G
    m : Nat
    hx : Membership.mem ((fun m => H (HSub.hSub n m)) (HAdd.hAdd m 1)) x
    g : G
    ⊢ Membership.mem ((fun m => H (HSub.hSub n m)) m) (HMul.hMul (HMul.hMul (HMul. …
  -/
  dsimp only at hx ⊢
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    H : Nat → Subgroup G
    n : Nat
    hn : Eq (H n) Bot.bot
    h0 : Eq (H 0) Top.top
    hH : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
    x : G
    m : Nat
    hx : Membership.mem (H (HSub.hSub n (HAdd.hAdd m 1))) x
    g : G
    ⊢ Membership.mem (H (HSub.hSub n m)) (HMul.hMul (HMul.hMul (HMul.hMul x g) (In …
  -/
  by_cases hm : n ≤ m
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Bot.bot
      h0 : Eq (H 0) Top.top
      hH : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
      x : G
      m : Nat
      hx : Membership.mem (H (HSub.hSub n (HAdd.hAdd m 1))) x
      g : G
      hm : LE.le n m
      ⊢ Membership.mem (H (HSub.hSub n m)) (HMul.hMul (HMul.hMul (HMul.hMul x g) (In …
    -/
  · have hnm : n - m = 0 := tsub_eq_zero_iff_le.mpr hm
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Bot.bot
      h0 : Eq (H 0) Top.top
      hH : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
      x : G
      m : Nat
      hx : Membership.mem (H (HSub.hSub n (HAdd.hAdd m 1))) x
      g : G
      hm : LE.le n m
      hnm : Eq (HSub.hSub n m) 0
      ⊢ Membership.mem (H (HSub.hSub n m)) (HMul.hMul (HMul.hMul (HMul.hMul x g) (In …
    -/
    rw [hnm, h0]
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Bot.bot
      h0 : Eq (H 0) Top.top
      hH : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
      x : G
      m : Nat
      hx : Membership.mem (H (HSub.hSub n (HAdd.hAdd m 1))) x
      g : G
      hm : LE.le n m
      hnm : Eq (HSub.hSub n m) 0
      ⊢ Membership.mem Top.top (HMul.hMul (HMul.hMul (HMul.hMul x g) (Inv.inv x)) (I …
    -/
    exact mem_top _
    /-
      🎉 no goals
    -/
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Bot.bot
      h0 : Eq (H 0) Top.top
      hH : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
      x : G
      m : Nat
      hx : Membership.mem (H (HSub.hSub n (HAdd.hAdd m 1))) x
      g : G
      hm : Not (LE.le n m)
      ⊢ Membership.mem (H (HSub.hSub n m)) (HMul.hMul (HMul.hMul (HMul.hMul x g) (In …
    -/
  · push_neg at hm
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Bot.bot
      h0 : Eq (H 0) Top.top
      hH : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
      x : G
      m : Nat
      hx : Membership.mem (H (HSub.hSub n (HAdd.hAdd m 1))) x
      g : G
      hm : LT.lt m n
      ⊢ Membership.mem (H (HSub.hSub n m)) (HMul.hMul (HMul.hMul (HMul.hMul x g) (In …
    -/
    convert hH x _ hx g using 1
    /-
      case h.e'_4
      G : Type u_1
      inst✝ : Group G
      H : Nat → Subgroup G
      n : Nat
      hn : Eq (H n) Bot.bot
      h0 : Eq (H 0) Top.top
      hH : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
      x : G
      m : Nat
      hx : Membership.mem (H (HSub.hSub n (HAdd.hAdd m 1))) x
      g : G
      hm : LT.lt m n
      ⊢ Eq (H (HSub.hSub n m)) (H (HAdd.hAdd (HSub.hSub n (HAdd.hAdd m 1)) 1))
    -/
    rw [tsub_add_eq_add_tsub (Nat.succ_le_of_lt hm), Nat.succ_eq_add_one, Nat.add_sub_add_right]
    /-
      🎉 no goals
    -/


/-- A group `G` is nilpotent iff there exists a descending central series which reaches the
  trivial group in a finite time. -/
theorem nilpotent_iff_finite_descending_central_series :
    IsNilpotent G ↔ ∃ n : ℕ, ∃ H : ℕ → Subgroup G, IsDescendingCentralSeries H ∧ H n = ⊥ := by
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ Iff (Group.IsNilpotent G) (Exists fun n => Exists fun H => And (IsDescending …
  -/
  rw [nilpotent_iff_finite_ascending_central_series]
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ Iff (Exists fun n => Exists fun H => And (IsAscendingCentralSeries H) (Eq (H …
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      ⊢ (Exists fun n => Exists fun H => And (IsAscendingCentralSeries H) (Eq (H n)  …
    -/
  · rintro ⟨n, H, hH, hn⟩
    /-
      case mp.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ Exists fun n => Exists fun H => And (IsDescendingCentralSeries H) (Eq (H n)  …
    -/
    refine ⟨n, fun m => H (n - m), is_descending_rev_series_of_is_ascending G hn hH, ?_⟩
    /-
      case mp.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ Eq ((fun m => H (HSub.hSub n m)) n) Bot.bot
    -/
    dsimp only
    /-
      case mp.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ Eq (H (HSub.hSub n n)) Bot.bot
    -/
    rw [tsub_self]
    /-
      case mp.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ Eq (H 0) Bot.bot
    -/
    exact hH.1
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      ⊢ (Exists fun n => Exists fun H => And (IsDescendingCentralSeries H) (Eq (H n) …
    -/
  · rintro ⟨n, H, hH, hn⟩
    /-
      case mpr.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hH : IsDescendingCentralSeries H
      hn : Eq (H n) Bot.bot
      ⊢ Exists fun n => Exists fun H => And (IsAscendingCentralSeries H) (Eq (H n) T …
    -/
    refine ⟨n, fun m => H (n - m), is_ascending_rev_series_of_is_descending G hn hH, ?_⟩
    /-
      case mpr.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hH : IsDescendingCentralSeries H
      hn : Eq (H n) Bot.bot
      ⊢ Eq ((fun m => H (HSub.hSub n m)) n) Top.top
    -/
    dsimp only
    /-
      case mpr.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hH : IsDescendingCentralSeries H
      hn : Eq (H n) Bot.bot
      ⊢ Eq (H (HSub.hSub n n)) Top.top
    -/
    rw [tsub_self]
    /-
      case mpr.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hH : IsDescendingCentralSeries H
      hn : Eq (H n) Bot.bot
      ⊢ Eq (H 0) Top.top
    -/
    exact hH.1
    /-
      🎉 no goals
    -/


/-- The lower central series of a group `G` is a sequence `H n` of subgroups of `G`, defined
  by `H 0` is all of `G` and for `n≥1`, `H (n + 1) = ⁅H n, G⁆` -/
def lowerCentralSeries (G : Type*) [Group G] : ℕ → Subgroup G
  | 0 => ⊤
  | n + 1 => ⁅lowerCentralSeries G n, ⊤⁆


@[simp]
theorem lowerCentralSeries_zero : lowerCentralSeries G 0 = ⊤ := rfl


@[simp]
theorem lowerCentralSeries_one : lowerCentralSeries G 1 = commutator G := rfl


theorem mem_lowerCentralSeries_succ_iff (n : ℕ) (q : G) :
    q ∈ lowerCentralSeries G (n + 1) ↔
    q ∈ closure { x | ∃ p ∈ lowerCentralSeries G n,
                        ∃ q ∈ (⊤ : Subgroup G), p * q * p⁻¹ * q⁻¹ = x } := Iff.rfl


theorem lowerCentralSeries_succ (n : ℕ) :
    lowerCentralSeries G (n + 1) =
      closure { x | ∃ p ∈ lowerCentralSeries G n, ∃ q ∈ (⊤ : Subgroup G), p * q * p⁻¹ * q⁻¹ = x } :=
  rfl


instance lowerCentralSeries_normal (n : ℕ) : Normal (lowerCentralSeries G n) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    n : Nat
    ⊢ (lowerCentralSeries G n).Normal
  -/
  induction' n with d hd
    /-
      case zero
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      inst✝ : H.Normal
      ⊢ (lowerCentralSeries G 0).Normal
    -/
  · exact (⊤ : Subgroup G).normal_of_characteristic
    /-
      🎉 no goals
    -/
    /-
      case succ
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      inst✝ : H.Normal
      d : Nat
      hd : (lowerCentralSeries G d).Normal
      ⊢ (lowerCentralSeries G (HAdd.hAdd d 1)).Normal
    -/
  · exact @Subgroup.commutator_normal _ _ (lowerCentralSeries G d) ⊤ hd _
    /-
      🎉 no goals
    -/


theorem lowerCentralSeries_antitone : Antitone (lowerCentralSeries G) := by
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ Antitone (lowerCentralSeries G)
  -/
  refine antitone_nat_of_succ_le fun n x hx => ?_
  simp only [mem_lowerCentralSeries_succ_iff, exists_prop, mem_top, exists_true_left,
    true_and] at hx
  refine
    closure_induction ?_ (Subgroup.one_mem _) (fun _ _ _ _ ↦ mul_mem) (fun _ _ ↦ inv_mem) hx
  /-
    G : Type u_1
    inst✝ : Group G
    n : Nat
    x : G
    hx : Membership.mem (Subgroup.closure (setOf fun x => Exists fun p => And (Mem …
    ⊢ ∀ (x : G), Membership.mem (setOf fun x => Exists fun p => And (Membership.me …
  -/
  rintro y ⟨z, hz, a, ha⟩
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    n : Nat
    x : G
    hx : Membership.mem (Subgroup.closure (setOf fun x => Exists fun p => And (Mem …
    y z : G
    hz : Membership.mem (lowerCentralSeries G n) z
    a : G
    ha : Eq (HMul.hMul (HMul.hMul (HMul.hMul z a) (Inv.inv z)) (Inv.inv a)) y
    ⊢ Membership.mem (lowerCentralSeries G n) y
  -/
  rw [← ha, mul_assoc, mul_assoc, ← mul_assoc a z⁻¹ a⁻¹]
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    n : Nat
    x : G
    hx : Membership.mem (Subgroup.closure (setOf fun x => Exists fun p => And (Mem …
    y z : G
    hz : Membership.mem (lowerCentralSeries G n) z
    a : G
    ha : Eq (HMul.hMul (HMul.hMul (HMul.hMul z a) (Inv.inv z)) (Inv.inv a)) y
    ⊢ Membership.mem (lowerCentralSeries G n) (HMul.hMul z (HMul.hMul (HMul.hMul a …
  -/
  exact mul_mem hz (Normal.conj_mem (lowerCentralSeries_normal n) z⁻¹ (inv_mem hz) a)
  /-
    🎉 no goals
  -/



/-- The lower central series of a group is a descending central series. -/
theorem lowerCentralSeries_isDescendingCentralSeries :
    IsDescendingCentralSeries (lowerCentralSeries G) := by
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ IsDescendingCentralSeries (lowerCentralSeries G)
  -/
  constructor
    /-
      case left
      G : Type u_1
      inst✝ : Group G
      ⊢ Eq (lowerCentralSeries G 0) Top.top
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case right
    G : Type u_1
    inst✝ : Group G
    ⊢ ∀ (x : G) (n : Nat), Membership.mem (lowerCentralSeries G n) x → ∀ (g : G),  …
  -/
  intro x n hxn g
  /-
    case right
    G : Type u_1
    inst✝ : Group G
    x : G
    n : Nat
    hxn : Membership.mem (lowerCentralSeries G n) x
    g : G
    ⊢ Membership.mem (lowerCentralSeries G (HAdd.hAdd n 1)) (HMul.hMul (HMul.hMul  …
  -/
  exact commutator_mem_commutator hxn (mem_top g)
  /-
    🎉 no goals
  -/


/-- Any descending central series for a group is bounded below by the lower central series. -/
theorem descending_central_series_ge_lower (H : ℕ → Subgroup G) (hH : IsDescendingCentralSeries H) :
    ∀ n : ℕ, lowerCentralSeries G n ≤ H n
  | 0 => hH.1.symm ▸ le_refl ⊤
  | n + 1 => commutator_le.mpr fun x hx q _ =>
      hH.2 x n (descending_central_series_ge_lower H hH n hx) q


/-- A group is nilpotent if and only if its lower central series eventually reaches
  the trivial subgroup. -/
theorem nilpotent_iff_lowerCentralSeries : IsNilpotent G ↔ ∃ n, lowerCentralSeries G n = ⊥ := by
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ Iff (Group.IsNilpotent G) (Exists fun n => Eq (lowerCentralSeries G n) Bot.b …
  -/
  rw [nilpotent_iff_finite_descending_central_series]
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ Iff (Exists fun n => Exists fun H => And (IsDescendingCentralSeries H) (Eq ( …
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      ⊢ (Exists fun n => Exists fun H => And (IsDescendingCentralSeries H) (Eq (H n) …
    -/
  · rintro ⟨n, H, ⟨h0, hs⟩, hn⟩
    /-
      case mp.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hn : Eq (H n) Bot.bot
      h0 : Eq (H 0) Top.top
      hs : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
      ⊢ Exists fun n => Eq (lowerCentralSeries G n) Bot.bot
    -/
    use n
    /-
      case h
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hn : Eq (H n) Bot.bot
      h0 : Eq (H 0) Top.top
      hs : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
      ⊢ Eq (lowerCentralSeries G n) Bot.bot
    -/
    rw [eq_bot_iff, ← hn]
    /-
      case h
      G : Type u_1
      inst✝ : Group G
      n : Nat
      H : Nat → Subgroup G
      hn : Eq (H n) Bot.bot
      h0 : Eq (H 0) Top.top
      hs : ∀ (x : G) (n : Nat), Membership.mem (H n) x → ∀ (g : G), Membership.mem ( …
      ⊢ LE.le (lowerCentralSeries G n) (H n)
    -/
    exact descending_central_series_ge_lower H ⟨h0, hs⟩ n
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      ⊢ (Exists fun n => Eq (lowerCentralSeries G n) Bot.bot) → Exists fun n => Exis …
    -/
  · rintro ⟨n, hn⟩
    /-
      case mpr.intro
      G : Type u_1
      inst✝ : Group G
      n : Nat
      hn : Eq (lowerCentralSeries G n) Bot.bot
      ⊢ Exists fun n => Exists fun H => And (IsDescendingCentralSeries H) (Eq (H n)  …
    -/
    exact ⟨n, lowerCentralSeries G, lowerCentralSeries_isDescendingCentralSeries, hn⟩
    /-
      🎉 no goals
    -/


/-- The nilpotency class of a nilpotent group is the smallest natural `n` such that
the `n`'th term of the upper central series is `G`. -/
noncomputable def Group.nilpotencyClass : ℕ := Nat.find (IsNilpotent.nilpotent G)


@[simp]
theorem upperCentralSeries_nilpotencyClass : upperCentralSeries G (Group.nilpotencyClass G) = ⊤ :=
  Nat.find_spec (IsNilpotent.nilpotent G)


theorem upperCentralSeries_eq_top_iff_nilpotencyClass_le {n : ℕ} :
    upperCentralSeries G n = ⊤ ↔ Group.nilpotencyClass G ≤ n := by
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Group.IsNilpotent G
    n : Nat
    ⊢ Iff (Eq (upperCentralSeries G n) Top.top) (LE.le (Group.nilpotencyClass G) n)
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      ⊢ Eq (upperCentralSeries G n) Top.top → LE.le (Group.nilpotencyClass G) n
    -/
  · intro h
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      h : Eq (upperCentralSeries G n) Top.top
      ⊢ LE.le (Group.nilpotencyClass G) n
    -/
    exact Nat.find_le h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      ⊢ LE.le (Group.nilpotencyClass G) n → Eq (upperCentralSeries G n) Top.top
    -/
  · intro h
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      h : LE.le (Group.nilpotencyClass G) n
      ⊢ Eq (upperCentralSeries G n) Top.top
    -/
    rw [eq_top_iff, ← upperCentralSeries_nilpotencyClass]
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      h : LE.le (Group.nilpotencyClass G) n
      ⊢ LE.le (upperCentralSeries G (Group.nilpotencyClass G)) (upperCentralSeries G …
    -/
    exact upperCentralSeries_mono _ h
    /-
      🎉 no goals
    -/


/-- The nilpotency class of a nilpotent `G` is equal to the smallest `n` for which an ascending
central series reaches `G` in its `n`'th term. -/
theorem least_ascending_central_series_length_eq_nilpotencyClass :
    Nat.find ((nilpotent_iff_finite_ascending_central_series G).mp hG) =
    Group.nilpotencyClass G := by
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Group.IsNilpotent G
    ⊢ Eq (Nat.find ⋯) (Group.nilpotencyClass G)
  -/
  refine le_antisymm (Nat.find_mono ?_) (Nat.find_mono ?_)
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      ⊢ ∀ (n : Nat), Eq (upperCentralSeries G n) Top.top → Exists fun H => And (IsAs …
    -/
  · intro n hn
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      hn : Eq (upperCentralSeries G n) Top.top
      ⊢ Exists fun H => And (IsAscendingCentralSeries H) (Eq (H n) Top.top)
    -/
    exact ⟨upperCentralSeries G, upperCentralSeries_isAscendingCentralSeries G, hn⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      ⊢ ∀ (n : Nat), (Exists fun H => And (IsAscendingCentralSeries H) (Eq (H n) Top …
    -/
  · rintro n ⟨H, ⟨hH, hn⟩⟩
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ Eq (upperCentralSeries G n) Top.top
    -/
    rw [← top_le_iff, ← hn]
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ LE.le (H n) (upperCentralSeries G n)
    -/
    exact ascending_central_series_le_upper H hH n
    /-
      🎉 no goals
    -/


/-- The nilpotency class of a nilpotent `G` is equal to the smallest `n` for which the descending
central series reaches `⊥` in its `n`'th term. -/
theorem least_descending_central_series_length_eq_nilpotencyClass :
    Nat.find ((nilpotent_iff_finite_descending_central_series G).mp hG) =
    Group.nilpotencyClass G := by
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Group.IsNilpotent G
    ⊢ Eq (Nat.find ⋯) (Group.nilpotencyClass G)
  -/
  rw [← least_ascending_central_series_length_eq_nilpotencyClass]
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Group.IsNilpotent G
    ⊢ Eq (Nat.find ⋯) (Nat.find ⋯)
  -/
  refine le_antisymm (Nat.find_mono ?_) (Nat.find_mono ?_)
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      ⊢ ∀ (n : Nat), (Exists fun H => And (IsAscendingCentralSeries H) (Eq (H n) Top …
    -/
  · rintro n ⟨H, ⟨hH, hn⟩⟩
    /-
      case refine_1.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ Exists fun H => And (IsDescendingCentralSeries H) (Eq (H n) Bot.bot)
    -/
    refine ⟨fun m => H (n - m), is_descending_rev_series_of_is_ascending G hn hH, ?_⟩
    /-
      case refine_1.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ Eq ((fun m => H (HSub.hSub n m)) n) Bot.bot
    -/
    dsimp only
    /-
      case refine_1.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ Eq (H (HSub.hSub n n)) Bot.bot
    -/
    rw [tsub_self]
    /-
      case refine_1.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsAscendingCentralSeries H
      hn : Eq (H n) Top.top
      ⊢ Eq (H 0) Bot.bot
    -/
    exact hH.1
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      ⊢ ∀ (n : Nat), (Exists fun H => And (IsDescendingCentralSeries H) (Eq (H n) Bo …
    -/
  · rintro n ⟨H, ⟨hH, hn⟩⟩
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsDescendingCentralSeries H
      hn : Eq (H n) Bot.bot
      ⊢ Exists fun H => And (IsAscendingCentralSeries H) (Eq (H n) Top.top)
    -/
    refine ⟨fun m => H (n - m), is_ascending_rev_series_of_is_descending G hn hH, ?_⟩
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsDescendingCentralSeries H
      hn : Eq (H n) Bot.bot
      ⊢ Eq ((fun m => H (HSub.hSub n m)) n) Top.top
    -/
    dsimp only
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsDescendingCentralSeries H
      hn : Eq (H n) Bot.bot
      ⊢ Eq (H (HSub.hSub n n)) Top.top
    -/
    rw [tsub_self]
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsDescendingCentralSeries H
      hn : Eq (H n) Bot.bot
      ⊢ Eq (H 0) Top.top
    -/
    exact hH.1
    /-
      🎉 no goals
    -/


/-- The nilpotency class of a nilpotent `G` is equal to the length of the lower central series. -/
theorem lowerCentralSeries_length_eq_nilpotencyClass :
    Nat.find (nilpotent_iff_lowerCentralSeries.mp hG) = Group.nilpotencyClass (G := G) := by
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Group.IsNilpotent G
    ⊢ Eq (Nat.find ⋯) (Group.nilpotencyClass G)
  -/
  rw [← least_descending_central_series_length_eq_nilpotencyClass]
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Group.IsNilpotent G
    ⊢ Eq (Nat.find ⋯) (Nat.find ⋯)
  -/
  refine le_antisymm (Nat.find_mono ?_) (Nat.find_mono ?_)
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      ⊢ ∀ (n : Nat), (Exists fun H => And (IsDescendingCentralSeries H) (Eq (H n) Bo …
    -/
  · rintro n ⟨H, ⟨hH, hn⟩⟩
    /-
      case refine_1.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsDescendingCentralSeries H
      hn : Eq (H n) Bot.bot
      ⊢ Eq (lowerCentralSeries G n) Bot.bot
    -/
    rw [← le_bot_iff, ← hn]
    /-
      case refine_1.intro.intro
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      H : Nat → Subgroup G
      hH : IsDescendingCentralSeries H
      hn : Eq (H n) Bot.bot
      ⊢ LE.le (lowerCentralSeries G n) (H n)
    -/
    exact descending_central_series_ge_lower H hH n
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      ⊢ ∀ (n : Nat), Eq (lowerCentralSeries G n) Bot.bot → Exists fun H => And (IsDe …
    -/
  · rintro n h
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      h : Eq (lowerCentralSeries G n) Bot.bot
      ⊢ Exists fun H => And (IsDescendingCentralSeries H) (Eq (H n) Bot.bot)
    -/
    exact ⟨lowerCentralSeries G, ⟨lowerCentralSeries_isDescendingCentralSeries, h⟩⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem lowerCentralSeries_nilpotencyClass :
    lowerCentralSeries G (Group.nilpotencyClass G) = ⊥ := by
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Group.IsNilpotent G
    ⊢ Eq (lowerCentralSeries G (Group.nilpotencyClass G)) Bot.bot
  -/
  rw [← lowerCentralSeries_length_eq_nilpotencyClass]
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Group.IsNilpotent G
    ⊢ Eq (lowerCentralSeries G (Nat.find ⋯)) Bot.bot
  -/
  exact Nat.find_spec (nilpotent_iff_lowerCentralSeries.mp hG)
  /-
    🎉 no goals
  -/


theorem lowerCentralSeries_eq_bot_iff_nilpotencyClass_le {n : ℕ} :
    lowerCentralSeries G n = ⊥ ↔ Group.nilpotencyClass G ≤ n := by
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Group.IsNilpotent G
    n : Nat
    ⊢ Iff (Eq (lowerCentralSeries G n) Bot.bot) (LE.le (Group.nilpotencyClass G) n)
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      ⊢ Eq (lowerCentralSeries G n) Bot.bot → LE.le (Group.nilpotencyClass G) n
    -/
  · intro h
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      h : Eq (lowerCentralSeries G n) Bot.bot
      ⊢ LE.le (Group.nilpotencyClass G) n
    -/
    rw [← lowerCentralSeries_length_eq_nilpotencyClass]
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      h : Eq (lowerCentralSeries G n) Bot.bot
      ⊢ LE.le (Nat.find ⋯) n
    -/
    exact Nat.find_le h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      ⊢ LE.le (Group.nilpotencyClass G) n → Eq (lowerCentralSeries G n) Bot.bot
    -/
  · intro h
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      h : LE.le (Group.nilpotencyClass G) n
      ⊢ Eq (lowerCentralSeries G n) Bot.bot
    -/
    rw [eq_bot_iff, ← lowerCentralSeries_nilpotencyClass]
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      hG : Group.IsNilpotent G
      n : Nat
      h : LE.le (Group.nilpotencyClass G) n
      ⊢ LE.le (lowerCentralSeries G n) (lowerCentralSeries G (Group.nilpotencyClass  …
    -/
    exact lowerCentralSeries_antitone h
    /-
      🎉 no goals
    -/


theorem lowerCentralSeries_map_subtype_le (H : Subgroup G) (n : ℕ) :
    (lowerCentralSeries H n).map H.subtype ≤ lowerCentralSeries G n := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    n : Nat
    ⊢ LE.le (Subgroup.map H.subtype (lowerCentralSeries (Subtype fun x => Membersh …
  -/
  induction' n with d hd
    /-
      case zero
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      ⊢ LE.le (Subgroup.map H.subtype (lowerCentralSeries (Subtype fun x => Membersh …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      d : Nat
      hd : LE.le (Subgroup.map H.subtype (lowerCentralSeries (Subtype fun x => Membe …
      ⊢ LE.le (Subgroup.map H.subtype (lowerCentralSeries (Subtype fun x => Membersh …
    -/
  · rw [lowerCentralSeries_succ, lowerCentralSeries_succ, MonoidHom.map_closure]
    /-
      case succ
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      d : Nat
      hd : LE.le (Subgroup.map H.subtype (lowerCentralSeries (Subtype fun x => Membe …
      ⊢ LE.le (Subgroup.closure (Set.image (⇑H.subtype) (setOf fun x => Exists fun p …
    -/
    apply Subgroup.closure_mono
    /-
      case succ.h'
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      d : Nat
      hd : LE.le (Subgroup.map H.subtype (lowerCentralSeries (Subtype fun x => Membe …
      ⊢ HasSubset.Subset (Set.image (⇑H.subtype) (setOf fun x => Exists fun p => And …
    -/
    rintro x1 ⟨x2, ⟨x3, hx3, x4, _hx4, rfl⟩, rfl⟩
    /-
      case succ.h'.intro.intro.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      d : Nat
      hd : LE.le (Subgroup.map H.subtype (lowerCentralSeries (Subtype fun x => Membe …
      x3 : Subtype fun x => Membership.mem H x
      hx3 : Membership.mem (lowerCentralSeries (Subtype fun x => Membership.mem H x) …
      x4 : Subtype fun x => Membership.mem H x
      _hx4 : Membership.mem Top.top x4
      ⊢ Membership.mem (setOf fun x => Exists fun p => And (Membership.mem (lowerCen …
    -/
    exact ⟨x3, hd (mem_map.mpr ⟨x3, hx3, rfl⟩), x4, by simp⟩
    /-
      🎉 no goals
    -/


/-- A subgroup of a nilpotent group is nilpotent -/
instance Subgroup.isNilpotent (H : Subgroup G) [hG : IsNilpotent G] : IsNilpotent H := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H✝ : Subgroup G
    inst✝ : H✝.Normal
    H : Subgroup G
    hG : Group.IsNilpotent G
    ⊢ Group.IsNilpotent (Subtype fun x => Membership.mem H x)
  -/
  rw [nilpotent_iff_lowerCentralSeries] at *
  /-
    G : Type u_1
    inst✝¹ : Group G
    H✝ : Subgroup G
    inst✝ : H✝.Normal
    H : Subgroup G
    hG : Exists fun n => Eq (lowerCentralSeries G n) Bot.bot
    ⊢ Exists fun n => Eq (lowerCentralSeries (Subtype fun x => Membership.mem H x) …
  -/
  rcases hG with ⟨n, hG⟩
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    H✝ : Subgroup G
    inst✝ : H✝.Normal
    H : Subgroup G
    n : Nat
    hG : Eq (lowerCentralSeries G n) Bot.bot
    ⊢ Exists fun n => Eq (lowerCentralSeries (Subtype fun x => Membership.mem H x) …
  -/
  use n
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    H✝ : Subgroup G
    inst✝ : H✝.Normal
    H : Subgroup G
    n : Nat
    hG : Eq (lowerCentralSeries G n) Bot.bot
    ⊢ Eq (lowerCentralSeries (Subtype fun x => Membership.mem H x) n) Bot.bot
  -/
  have := lowerCentralSeries_map_subtype_le H n
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    H✝ : Subgroup G
    inst✝ : H✝.Normal
    H : Subgroup G
    n : Nat
    hG : Eq (lowerCentralSeries G n) Bot.bot
    this : LE.le (Subgroup.map H.subtype (lowerCentralSeries (Subtype fun x => Mem …
    ⊢ Eq (lowerCentralSeries (Subtype fun x => Membership.mem H x) n) Bot.bot
  -/
  simp only [hG, SetLike.le_def, mem_map, forall_apply_eq_imp_iff₂, exists_imp] at this
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    H✝ : Subgroup G
    inst✝ : H✝.Normal
    H : Subgroup G
    n : Nat
    hG : Eq (lowerCentralSeries G n) Bot.bot
    this : ∀ ⦃x : G⦄ (x_1 : Subtype fun x => Membership.mem H x), And (Membership. …
    ⊢ Eq (lowerCentralSeries (Subtype fun x => Membership.mem H x) n) Bot.bot
  -/
  exact eq_bot_iff.mpr fun x hx => Subtype.ext (this x ⟨hx, rfl⟩)
  /-
    🎉 no goals
  -/


/-- The nilpotency class of a subgroup is less or equal to the nilpotency class of the group -/
theorem Subgroup.nilpotencyClass_le (H : Subgroup G) [hG : IsNilpotent G] :
    Group.nilpotencyClass H ≤ Group.nilpotencyClass G := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    hG : Group.IsNilpotent G
    ⊢ LE.le (Group.nilpotencyClass (Subtype fun x => Membership.mem H x)) (Group.n …
  -/
  repeat rw [← lowerCentralSeries_length_eq_nilpotencyClass]
  --- Porting note: Lean needs to be told that predicates are decidable
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    hG : Group.IsNilpotent G
    ⊢ LE.le (Nat.find ⋯) (Nat.find ⋯)
  -/
  refine @Nat.find_mono _ _ (Classical.decPred _) (Classical.decPred _) ?_ _ _
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    hG : Group.IsNilpotent G
    ⊢ ∀ (n : Nat), Eq (lowerCentralSeries G n) Bot.bot → Eq (lowerCentralSeries (S …
  -/
  intro n hG
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    hG✝ : Group.IsNilpotent G
    n : Nat
    hG : Eq (lowerCentralSeries G n) Bot.bot
    ⊢ Eq (lowerCentralSeries (Subtype fun x => Membership.mem H x) n) Bot.bot
  -/
  have := lowerCentralSeries_map_subtype_le H n
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    hG✝ : Group.IsNilpotent G
    n : Nat
    hG : Eq (lowerCentralSeries G n) Bot.bot
    this : LE.le (Subgroup.map H.subtype (lowerCentralSeries (Subtype fun x => Mem …
    ⊢ Eq (lowerCentralSeries (Subtype fun x => Membership.mem H x) n) Bot.bot
  -/
  simp only [hG, SetLike.le_def, mem_map, forall_apply_eq_imp_iff₂, exists_imp] at this
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    hG✝ : Group.IsNilpotent G
    n : Nat
    hG : Eq (lowerCentralSeries G n) Bot.bot
    this : ∀ ⦃x : G⦄ (x_1 : Subtype fun x => Membership.mem H x), And (Membership. …
    ⊢ Eq (lowerCentralSeries (Subtype fun x => Membership.mem H x) n) Bot.bot
  -/
  exact eq_bot_iff.mpr fun x hx => Subtype.ext (this x ⟨hx, rfl⟩)
  /-
    🎉 no goals
  -/


instance (priority := 100) Group.isNilpotent_of_subsingleton [Subsingleton G] : IsNilpotent G :=
  nilpotent_iff_lowerCentralSeries.2 ⟨0, Subsingleton.elim ⊤ ⊥⟩


theorem upperCentralSeries.map {H : Type*} [Group H] {f : G →* H} (h : Function.Surjective f)
    (n : ℕ) : Subgroup.map f (upperCentralSeries G n) ≤ upperCentralSeries H n := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    h : Function.Surjective ⇑f
    n : Nat
    ⊢ LE.le (Subgroup.map f (upperCentralSeries G n)) (upperCentralSeries H n)
  -/
  induction' n with d hd
    /-
      case zero
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      f : MonoidHom G H
      h : Function.Surjective ⇑f
      ⊢ LE.le (Subgroup.map f (upperCentralSeries G 0)) (upperCentralSeries H 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      f : MonoidHom G H
      h : Function.Surjective ⇑f
      d : Nat
      hd : LE.le (Subgroup.map f (upperCentralSeries G d)) (upperCentralSeries H d)
      ⊢ LE.le (Subgroup.map f (upperCentralSeries G (HAdd.hAdd d 1))) (upperCentralS …
    -/
  · rintro _ ⟨x, hx : x ∈ upperCentralSeries G d.succ, rfl⟩ y'
    /-
      case succ.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      f : MonoidHom G H
      h : Function.Surjective ⇑f
      d : Nat
      hd : LE.le (Subgroup.map f (upperCentralSeries G d)) (upperCentralSeries H d)
      x : G
      hx : Membership.mem (upperCentralSeries G d.succ) x
      y' : H
      ⊢ Membership.mem (upperCentralSeriesAux H d).fst (HMul.hMul (HMul.hMul (HMul.h …
    -/
    rcases h y' with ⟨y, rfl⟩
    /-
      case succ.intro.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      f : MonoidHom G H
      h : Function.Surjective ⇑f
      d : Nat
      hd : LE.le (Subgroup.map f (upperCentralSeries G d)) (upperCentralSeries H d)
      x : G
      hx : Membership.mem (upperCentralSeries G d.succ) x
      y : G
      ⊢ Membership.mem (upperCentralSeriesAux H d).fst (HMul.hMul (HMul.hMul (HMul.h …
    -/
    simpa using hd (mem_map_of_mem f (hx y))
    /-
      🎉 no goals
    -/


theorem lowerCentralSeries.map {H : Type*} [Group H] (f : G →* H) (n : ℕ) :
    Subgroup.map f (lowerCentralSeries G n) ≤ lowerCentralSeries H n := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    n : Nat
    ⊢ LE.le (Subgroup.map f (lowerCentralSeries G n)) (lowerCentralSeries H n)
  -/
  induction' n with d hd
    /-
      case zero
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      f : MonoidHom G H
      ⊢ LE.le (Subgroup.map f (lowerCentralSeries G 0)) (lowerCentralSeries H 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      f : MonoidHom G H
      d : Nat
      hd : LE.le (Subgroup.map f (lowerCentralSeries G d)) (lowerCentralSeries H d)
      ⊢ LE.le (Subgroup.map f (lowerCentralSeries G (HAdd.hAdd d 1))) (lowerCentralS …
    -/
  · rintro a ⟨x, hx : x ∈ lowerCentralSeries G d.succ, rfl⟩
    refine closure_induction (hx := hx) ?_ (by simp [f.map_one, Subgroup.one_mem _])
      (fun y z _ _ hy hz => by simp [MonoidHom.map_mul, Subgroup.mul_mem _ hy hz]) (fun y _ hy => by
        rw [f.map_inv]; exact Subgroup.inv_mem _ hy)
    /-
      case succ.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      f : MonoidHom G H
      d : Nat
      hd : LE.le (Subgroup.map f (lowerCentralSeries G d)) (lowerCentralSeries H d)
      x : G
      hx : Membership.mem (lowerCentralSeries G d.succ) x
      ⊢ ∀ (x : G), Membership.mem (setOf fun g => Exists fun g₁ => And (Membership.m …
    -/
    rintro a ⟨y, hy, z, ⟨-, rfl⟩⟩
    /-
      case succ.intro.intro.intro.intro.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      f : MonoidHom G H
      d : Nat
      hd : LE.le (Subgroup.map f (lowerCentralSeries G d)) (lowerCentralSeries H d)
      x : G
      hx : Membership.mem (lowerCentralSeries G d.succ) x
      y : G
      hy : Membership.mem (lowerCentralSeries G d) y
      z : G
      ⊢ Membership.mem (lowerCentralSeries H (HAdd.hAdd d 1)) (f (Bracket.bracket y  …
    -/
    apply mem_closure.mpr
    /-
      case succ.intro.intro.intro.intro.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      f : MonoidHom G H
      d : Nat
      hd : LE.le (Subgroup.map f (lowerCentralSeries G d)) (lowerCentralSeries H d)
      x : G
      hx : Membership.mem (lowerCentralSeries G d.succ) x
      y : G
      hy : Membership.mem (lowerCentralSeries G d) y
      z : G
      ⊢ ∀ (K : Subgroup H), HasSubset.Subset (setOf fun g => Exists fun g₁ => And (M …
    -/
    exact fun K hK => hK ⟨f y, hd (mem_map_of_mem f hy), by simp [commutatorElement_def]⟩
    /-
      🎉 no goals
    -/


theorem lowerCentralSeries_succ_eq_bot {n : ℕ} (h : lowerCentralSeries G n ≤ center G) :
    lowerCentralSeries G (n + 1) = ⊥ := by
  /-
    G : Type u_1
    inst✝ : Group G
    n : Nat
    h : LE.le (lowerCentralSeries G n) (Subgroup.center G)
    ⊢ Eq (lowerCentralSeries G (HAdd.hAdd n 1)) Bot.bot
  -/
  rw [lowerCentralSeries_succ, closure_eq_bot_iff, Set.subset_singleton_iff]
  /-
    G : Type u_1
    inst✝ : Group G
    n : Nat
    h : LE.le (lowerCentralSeries G n) (Subgroup.center G)
    ⊢ ∀ (y : G), Membership.mem (setOf fun x => Exists fun p => And (Membership.me …
  -/
  rintro x ⟨y, hy1, z, ⟨⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    n : Nat
    h : LE.le (lowerCentralSeries G n) (Subgroup.center G)
    y : G
    hy1 : Membership.mem (lowerCentralSeries G n) y
    z : G
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul y z) (Inv.inv y)) (Inv.inv z)) 1
  -/
  rw [mul_assoc, ← mul_inv_rev, mul_inv_eq_one, eq_comm]
  /-
    case intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    n : Nat
    h : LE.le (lowerCentralSeries G n) (Subgroup.center G)
    y : G
    hy1 : Membership.mem (lowerCentralSeries G n) y
    z : G
    ⊢ Eq (HMul.hMul z y) (HMul.hMul y z)
  -/
  exact mem_center_iff.mp (h hy1) z
  /-
    🎉 no goals
  -/


/-- The preimage of a nilpotent group is nilpotent if the kernel of the homomorphism is contained
in the center -/
theorem isNilpotent_of_ker_le_center {H : Type*} [Group H] (f : G →* H) (hf1 : f.ker ≤ center G)
    (hH : IsNilpotent H) : IsNilpotent G := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    hH : Group.IsNilpotent H
    ⊢ Group.IsNilpotent G
  -/
  rw [nilpotent_iff_lowerCentralSeries] at *
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    hH : Exists fun n => Eq (lowerCentralSeries H n) Bot.bot
    ⊢ Exists fun n => Eq (lowerCentralSeries G n) Bot.bot
  -/
  rcases hH with ⟨n, hn⟩
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    n : Nat
    hn : Eq (lowerCentralSeries H n) Bot.bot
    ⊢ Exists fun n => Eq (lowerCentralSeries G n) Bot.bot
  -/
  use n + 1
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    n : Nat
    hn : Eq (lowerCentralSeries H n) Bot.bot
    ⊢ Eq (lowerCentralSeries G (HAdd.hAdd n 1)) Bot.bot
  -/
  refine lowerCentralSeries_succ_eq_bot (le_trans ((Subgroup.map_eq_bot_iff _).mp ?_) hf1)
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    n : Nat
    hn : Eq (lowerCentralSeries H n) Bot.bot
    ⊢ Eq (Subgroup.map f (lowerCentralSeries G n)) Bot.bot
  -/
  exact eq_bot_iff.mpr (hn ▸ lowerCentralSeries.map f n)
  /-
    🎉 no goals
  -/


theorem nilpotencyClass_le_of_ker_le_center {H : Type*} [Group H] (f : G →* H)
    (hf1 : f.ker ≤ center G) (hH : IsNilpotent H) :
    Group.nilpotencyClass (hG := isNilpotent_of_ker_le_center f hf1 hH) ≤
      Group.nilpotencyClass H + 1 := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    hH : Group.IsNilpotent H
    ⊢ LE.le (Group.nilpotencyClass G) (HAdd.hAdd (Group.nilpotencyClass H) 1)
  -/
  haveI : IsNilpotent G := isNilpotent_of_ker_le_center f hf1 hH
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    hH : Group.IsNilpotent H
    this : Group.IsNilpotent G
    ⊢ LE.le (Group.nilpotencyClass G) (HAdd.hAdd (Group.nilpotencyClass H) 1)
  -/
  rw [← lowerCentralSeries_length_eq_nilpotencyClass]
  -- Porting note: Lean needs to be told that predicates are decidable
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    hH : Group.IsNilpotent H
    this : Group.IsNilpotent G
    ⊢ LE.le (Nat.find ⋯) (HAdd.hAdd (Group.nilpotencyClass H) 1)
  -/
  refine @Nat.find_min' _ (Classical.decPred _) _ _ ?_
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    hH : Group.IsNilpotent H
    this : Group.IsNilpotent G
    ⊢ Eq (lowerCentralSeries G (HAdd.hAdd (Group.nilpotencyClass H) 1)) Bot.bot
  -/
  refine lowerCentralSeries_succ_eq_bot (le_trans ((Subgroup.map_eq_bot_iff _).mp ?_) hf1)
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    hH : Group.IsNilpotent H
    this : Group.IsNilpotent G
    ⊢ Eq (Subgroup.map f (lowerCentralSeries G (Group.nilpotencyClass H))) Bot.bot
  -/
  rw [eq_bot_iff]
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    hH : Group.IsNilpotent H
    this : Group.IsNilpotent G
    ⊢ LE.le (Subgroup.map f (lowerCentralSeries G (Group.nilpotencyClass H))) Bot. …
  -/
  apply le_trans (lowerCentralSeries.map f _)
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    hf1 : LE.le f.ker (Subgroup.center G)
    hH : Group.IsNilpotent H
    this : Group.IsNilpotent G
    ⊢ LE.le (lowerCentralSeries H (Group.nilpotencyClass H)) Bot.bot
  -/
  simp only [lowerCentralSeries_nilpotencyClass, le_bot_iff]
  /-
    🎉 no goals
  -/


/-- The range of a surjective homomorphism from a nilpotent group is nilpotent -/
theorem nilpotent_of_surjective {G' : Type*} [Group G'] [h : IsNilpotent G] (f : G →* G')
    (hf : Function.Surjective f) : IsNilpotent G' := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    G' : Type u_2
    inst✝ : Group G'
    h : Group.IsNilpotent G
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    ⊢ Group.IsNilpotent G'
  -/
  rcases h with ⟨n, hn⟩
  /-
    case mk.intro
    G : Type u_1
    inst✝¹ : Group G
    G' : Type u_2
    inst✝ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    n : Nat
    hn : Eq (upperCentralSeries G n) Top.top
    ⊢ Group.IsNilpotent G'
  -/
  use n
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    G' : Type u_2
    inst✝ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    n : Nat
    hn : Eq (upperCentralSeries G n) Top.top
    ⊢ Eq (upperCentralSeries G' n) Top.top
  -/
  apply eq_top_iff.mpr
  calc
    ⊤ = f.range := symm (f.range_eq_top_of_surjective hf)
    _ = Subgroup.map f ⊤ := MonoidHom.range_eq_map _
    _ = Subgroup.map f (upperCentralSeries G n) := by rw [hn]
    _ ≤ upperCentralSeries G' n := upperCentralSeries.map hf n


/-- The nilpotency class of the range of a surjective homomorphism from a
nilpotent group is less or equal the nilpotency class of the domain -/
theorem nilpotencyClass_le_of_surjective {G' : Type*} [Group G'] (f : G →* G')
    (hf : Function.Surjective f) [h : IsNilpotent G] :
    Group.nilpotencyClass (hG := nilpotent_of_surjective _ hf) ≤ Group.nilpotencyClass G := by
  -- Porting note: Lean needs to be told that predicates are decidable
  /-
    G : Type u_1
    inst✝¹ : Group G
    G' : Type u_2
    inst✝ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    h : Group.IsNilpotent G
    ⊢ LE.le (Group.nilpotencyClass G') (Group.nilpotencyClass G)
  -/
  refine @Nat.find_mono _ _ (Classical.decPred _) (Classical.decPred _) ?_ _ _
  /-
    G : Type u_1
    inst✝¹ : Group G
    G' : Type u_2
    inst✝ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    h : Group.IsNilpotent G
    ⊢ ∀ (n : Nat), Eq (upperCentralSeries G n) Top.top → Eq (upperCentralSeries G' …
  -/
  intro n hn
  /-
    G : Type u_1
    inst✝¹ : Group G
    G' : Type u_2
    inst✝ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    h : Group.IsNilpotent G
    n : Nat
    hn : Eq (upperCentralSeries G n) Top.top
    ⊢ Eq (upperCentralSeries G' n) Top.top
  -/
  rw [eq_top_iff]
  calc
    ⊤ = f.range := symm (f.range_eq_top_of_surjective hf)
    _ = Subgroup.map f ⊤ := MonoidHom.range_eq_map _
    _ = Subgroup.map f (upperCentralSeries G n) := by rw [hn]
    _ ≤ upperCentralSeries G' n := upperCentralSeries.map hf n


/-- Nilpotency respects isomorphisms -/
theorem nilpotent_of_mulEquiv {G' : Type*} [Group G'] [_h : IsNilpotent G] (f : G ≃* G') :
    IsNilpotent G' :=
  nilpotent_of_surjective f.toMonoidHom (MulEquiv.surjective f)


/-- A quotient of a nilpotent group is nilpotent -/
instance nilpotent_quotient_of_nilpotent (H : Subgroup G) [H.Normal] [_h : IsNilpotent G] :
    IsNilpotent (G ⧸ H) :=
  nilpotent_of_surjective (QuotientGroup.mk' H) QuotientGroup.mk_surjective


/-- The nilpotency class of a quotient of `G` is less or equal the nilpotency class of `G` -/
theorem nilpotencyClass_quotient_le (H : Subgroup G) [H.Normal] [_h : IsNilpotent G] :
    Group.nilpotencyClass (G ⧸ H) ≤ Group.nilpotencyClass G :=
  nilpotencyClass_le_of_surjective (QuotientGroup.mk' H) QuotientGroup.mk_surjective

-- This technical lemma helps with rewriting the subgroup, which occurs in indices

private theorem comap_center_subst {H₁ H₂ : Subgroup G} [Normal H₁] [Normal H₂] (h : H₁ = H₂) :
                                                                              /-
                                                                                G : Type u_1
                                                                                inst✝² : Group G
                                                                                H₁ H₂ : Subgroup G
                                                                                inst✝¹ : H₁.Normal
                                                                                inst✝ : H₂.Normal
                                                                                h : Eq H₁ H₂
                                                                                ⊢ Eq (Subgroup.comap (QuotientGroup.mk' H₁) (Subgroup.center (HasQuotient.Quot …
                                                                              -/
    comap (mk' H₁) (center (G ⧸ H₁)) = comap (mk' H₂) (center (G ⧸ H₂)) := by subst h; rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


theorem comap_upperCentralSeries_quotient_center (n : ℕ) :
    comap (mk' (center G)) (upperCentralSeries (G ⧸ center G) n) = upperCentralSeries G n.succ := by
  /-
    G : Type u_1
    inst✝ : Group G
    n : Nat
    ⊢ Eq (Subgroup.comap (QuotientGroup.mk' (Subgroup.center G)) (upperCentralSeri …
  -/
  induction' n with n ih
  · simp only [upperCentralSeries_zero, MonoidHom.comap_bot, ker_mk',
      (upperCentralSeries_one G).symm]
    /-
      case succ
      G : Type u_1
      inst✝ : Group G
      n : Nat
      ih : Eq (Subgroup.comap (QuotientGroup.mk' (Subgroup.center G)) (upperCentralS …
      ⊢ Eq (Subgroup.comap (QuotientGroup.mk' (Subgroup.center G)) (upperCentralSeri …
    -/
  · let Hn := upperCentralSeries (G ⧸ center G) n
    calc
      comap (mk' (center G)) (upperCentralSeriesStep Hn) =
          comap (mk' (center G)) (comap (mk' Hn) (center ((G ⧸ center G) ⧸ Hn))) := by
        rw [upperCentralSeriesStep_eq_comap_center]
      _ = comap (mk' (comap (mk' (center G)) Hn)) (center (G ⧸ comap (mk' (center G)) Hn)) :=
        QuotientGroup.comap_comap_center
      _ = comap (mk' (upperCentralSeries G n.succ)) (center (G ⧸ upperCentralSeries G n.succ)) :=
        (comap_center_subst ih)
      _ = upperCentralSeriesStep (upperCentralSeries G n.succ) :=
        symm (upperCentralSeriesStep_eq_comap_center _)


theorem nilpotencyClass_zero_iff_subsingleton [IsNilpotent G] :
    Group.nilpotencyClass G = 0 ↔ Subsingleton G := by
  -- Porting note: Lean needs to be told that predicates are decidable
  rw [Group.nilpotencyClass, @Nat.find_eq_zero _ (Classical.decPred _), upperCentralSeries_zero,
    subsingleton_iff_bot_eq_top, Subgroup.subsingleton_iff]


/-- Quotienting the `center G` reduces the nilpotency class by 1 -/
theorem nilpotencyClass_quotient_center [hH : IsNilpotent G] :
    Group.nilpotencyClass (G ⧸ center G) = Group.nilpotencyClass G - 1 := by
  /-
    G : Type u_1
    inst✝ : Group G
    hH : Group.IsNilpotent G
    ⊢ Eq (Group.nilpotencyClass (HasQuotient.Quotient G (Subgroup.center G))) (HSu …
  -/
  generalize hn : Group.nilpotencyClass G = n
  /-
    G : Type u_1
    inst✝ : Group G
    hH : Group.IsNilpotent G
    n : Nat
    hn : Eq (Group.nilpotencyClass G) n
    ⊢ Eq (Group.nilpotencyClass (HasQuotient.Quotient G (Subgroup.center G))) (HSu …
  -/
  rcases n with (rfl | n)
    /-
      case zero
      G : Type u_1
      inst✝ : Group G
      hH : Group.IsNilpotent G
      hn : Eq (Group.nilpotencyClass G) 0
      ⊢ Eq (Group.nilpotencyClass (HasQuotient.Quotient G (Subgroup.center G))) (HSu …
    -/
  · simp [nilpotencyClass_zero_iff_subsingleton] at *
    /-
      case zero
      G : Type u_1
      inst✝ : Group G
      hH : Group.IsNilpotent G
      hn : Subsingleton G
      ⊢ Subsingleton (HasQuotient.Quotient G (Subgroup.center G))
    -/
    exact Quotient.instSubsingletonQuotient (leftRel (center G))
    /-
      🎉 no goals
    -/
    /-
      case succ
      G : Type u_1
      inst✝ : Group G
      hH : Group.IsNilpotent G
      n : Nat
      hn : Eq (Group.nilpotencyClass G) (HAdd.hAdd n 1)
      ⊢ Eq (Group.nilpotencyClass (HasQuotient.Quotient G (Subgroup.center G))) (HSu …
    -/
  · suffices Group.nilpotencyClass (G ⧸ center G) = n by simpa
    /-
      case succ
      G : Type u_1
      inst✝ : Group G
      hH : Group.IsNilpotent G
      n : Nat
      hn : Eq (Group.nilpotencyClass G) (HAdd.hAdd n 1)
      ⊢ Eq (Group.nilpotencyClass (HasQuotient.Quotient G (Subgroup.center G))) n
    -/
    apply le_antisymm
      /-
        case succ.a
        G : Type u_1
        inst✝ : Group G
        hH : Group.IsNilpotent G
        n : Nat
        hn : Eq (Group.nilpotencyClass G) (HAdd.hAdd n 1)
        ⊢ LE.le (Group.nilpotencyClass (HasQuotient.Quotient G (Subgroup.center G))) n
      -/
    · apply upperCentralSeries_eq_top_iff_nilpotencyClass_le.mp
      /-
        case succ.a
        G : Type u_1
        inst✝ : Group G
        hH : Group.IsNilpotent G
        n : Nat
        hn : Eq (Group.nilpotencyClass G) (HAdd.hAdd n 1)
        ⊢ Eq (upperCentralSeries (HasQuotient.Quotient G (Subgroup.center G)) n) Top.top
      -/
      apply comap_injective (f := (mk' (center G))) Quot.mk_surjective
      /-
        case succ.a.a
        G : Type u_1
        inst✝ : Group G
        hH : Group.IsNilpotent G
        n : Nat
        hn : Eq (Group.nilpotencyClass G) (HAdd.hAdd n 1)
        ⊢ Eq (Subgroup.comap (QuotientGroup.mk' (Subgroup.center G)) (upperCentralSeri …
      -/
      rw [comap_upperCentralSeries_quotient_center, comap_top, Nat.succ_eq_add_one, ← hn]
      /-
        case succ.a.a
        G : Type u_1
        inst✝ : Group G
        hH : Group.IsNilpotent G
        n : Nat
        hn : Eq (Group.nilpotencyClass G) (HAdd.hAdd n 1)
        ⊢ Eq (upperCentralSeries G (Group.nilpotencyClass G)) Top.top
      -/
      exact upperCentralSeries_nilpotencyClass
      /-
        🎉 no goals
      -/
      /-
        case succ.a
        G : Type u_1
        inst✝ : Group G
        hH : Group.IsNilpotent G
        n : Nat
        hn : Eq (Group.nilpotencyClass G) (HAdd.hAdd n 1)
        ⊢ LE.le n (Group.nilpotencyClass (HasQuotient.Quotient G (Subgroup.center G)))
      -/
    · apply le_of_add_le_add_right
      calc
        n + 1 = Group.nilpotencyClass G := hn.symm
        _ ≤ Group.nilpotencyClass (G ⧸ center G) + 1 :=
          nilpotencyClass_le_of_ker_le_center _ (le_of_eq (ker_mk' _)) _


/-- The nilpotency class of a non-trivial group is one more than its quotient by the center -/
theorem nilpotencyClass_eq_quotient_center_plus_one [hH : IsNilpotent G] [Nontrivial G] :
    Group.nilpotencyClass G = Group.nilpotencyClass (G ⧸ center G) + 1 := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    hH : Group.IsNilpotent G
    inst✝ : Nontrivial G
    ⊢ Eq (Group.nilpotencyClass G) (HAdd.hAdd (Group.nilpotencyClass (HasQuotient. …
  -/
  rw [nilpotencyClass_quotient_center]
  /-
    G : Type u_1
    inst✝¹ : Group G
    hH : Group.IsNilpotent G
    inst✝ : Nontrivial G
    ⊢ Eq (Group.nilpotencyClass G) (HAdd.hAdd (HSub.hSub (Group.nilpotencyClass G) …
  -/
  rcases h : Group.nilpotencyClass G with ⟨⟩
    /-
      case zero
      G : Type u_1
      inst✝¹ : Group G
      hH : Group.IsNilpotent G
      inst✝ : Nontrivial G
      h : Eq (Group.nilpotencyClass G) 0
      ⊢ Eq 0 (HAdd.hAdd (HSub.hSub 0 1) 1)
    -/
  · exfalso
    /-
      case zero
      G : Type u_1
      inst✝¹ : Group G
      hH : Group.IsNilpotent G
      inst✝ : Nontrivial G
      h : Eq (Group.nilpotencyClass G) 0
      ⊢ False
    -/
    rw [nilpotencyClass_zero_iff_subsingleton] at h
    /-
      case zero
      G : Type u_1
      inst✝¹ : Group G
      hH : Group.IsNilpotent G
      inst✝ : Nontrivial G
      h : Subsingleton G
      ⊢ False
    -/
    apply false_of_nontrivial_of_subsingleton G
    /-
      🎉 no goals
    -/
    /-
      case succ
      G : Type u_1
      inst✝¹ : Group G
      hH : Group.IsNilpotent G
      inst✝ : Nontrivial G
      n✝ : Nat
      h : Eq (Group.nilpotencyClass G) (HAdd.hAdd n✝ 1)
      ⊢ Eq (HAdd.hAdd n✝ 1) (HAdd.hAdd (HSub.hSub (HAdd.hAdd n✝ 1) 1) 1)
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- If the quotient by `center G` is nilpotent, then so is G. -/
theorem of_quotient_center_nilpotent (h : IsNilpotent (G ⧸ center G)) : IsNilpotent G := by
  /-
    G : Type u_1
    inst✝ : Group G
    h : Group.IsNilpotent (HasQuotient.Quotient G (Subgroup.center G))
    ⊢ Group.IsNilpotent G
  -/
  obtain ⟨n, hn⟩ := h.nilpotent
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    h : Group.IsNilpotent (HasQuotient.Quotient G (Subgroup.center G))
    n : Nat
    hn : Eq (upperCentralSeries (HasQuotient.Quotient G (Subgroup.center G)) n) To …
    ⊢ Group.IsNilpotent G
  -/
  use n.succ
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    h : Group.IsNilpotent (HasQuotient.Quotient G (Subgroup.center G))
    n : Nat
    hn : Eq (upperCentralSeries (HasQuotient.Quotient G (Subgroup.center G)) n) To …
    ⊢ Eq (upperCentralSeries G n.succ) Top.top
  -/
  simp [← comap_upperCentralSeries_quotient_center, hn]
  /-
    🎉 no goals
  -/


/-- A custom induction principle for nilpotent groups. The base case is a trivial group
(`subsingleton G`), and in the induction step, one can assume the hypothesis for
the group quotiented by its center. -/
@[elab_as_elim]
theorem nilpotent_center_quotient_ind {P : ∀ (G) [Group G] [IsNilpotent G], Prop}
    (G : Type*) [Group G] [IsNilpotent G]
    (hbase : ∀ (G) [Group G] [Subsingleton G], P G)
    (hstep : ∀ (G) [Group G] [IsNilpotent G], P (G ⧸ center G) → P G) : P G := by
  /-
    P : (G : Type u_2) → [inst : Group G] → [inst : Group.IsNilpotent G] → Prop
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : Group.IsNilpotent G
    hbase : ∀ (G : Type u_2) [inst : Group G] [inst_1 : Subsingleton G], P G
    hstep : ∀ (G : Type u_2) [inst : Group G] [inst_1 : Group.IsNilpotent G], P (H …
    ⊢ P G
  -/
  obtain ⟨n, h⟩ : ∃ n, Group.nilpotencyClass G = n := ⟨_, rfl⟩
  /-
    case intro
    P : (G : Type u_2) → [inst : Group G] → [inst : Group.IsNilpotent G] → Prop
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : Group.IsNilpotent G
    hbase : ∀ (G : Type u_2) [inst : Group G] [inst_1 : Subsingleton G], P G
    hstep : ∀ (G : Type u_2) [inst : Group G] [inst_1 : Group.IsNilpotent G], P (H …
    n : Nat
    h : Eq (Group.nilpotencyClass G) n
    ⊢ P G
  -/
  induction' n with n ih generalizing G
    /-
      case intro.zero
      P : (G : Type u_2) → [inst : Group G] → [inst : Group.IsNilpotent G] → Prop
      hbase : ∀ (G : Type u_2) [inst : Group G] [inst_1 : Subsingleton G], P G
      hstep : ∀ (G : Type u_2) [inst : Group G] [inst_1 : Group.IsNilpotent G], P (H …
      G : Type u_2
      inst✝¹ : Group G
      inst✝ : Group.IsNilpotent G
      h : Eq (Group.nilpotencyClass G) 0
      ⊢ P G
    -/
  · haveI := nilpotencyClass_zero_iff_subsingleton.mp h
    /-
      case intro.zero
      P : (G : Type u_2) → [inst : Group G] → [inst : Group.IsNilpotent G] → Prop
      hbase : ∀ (G : Type u_2) [inst : Group G] [inst_1 : Subsingleton G], P G
      hstep : ∀ (G : Type u_2) [inst : Group G] [inst_1 : Group.IsNilpotent G], P (H …
      G : Type u_2
      inst✝¹ : Group G
      inst✝ : Group.IsNilpotent G
      h : Eq (Group.nilpotencyClass G) 0
      this : Subsingleton G
      ⊢ P G
    -/
    exact hbase _
    /-
      🎉 no goals
    -/
  · have hn : Group.nilpotencyClass (G ⧸ center G) = n := by
      simp [nilpotencyClass_quotient_center, h]
    /-
      case intro.succ
      P : (G : Type u_2) → [inst : Group G] → [inst : Group.IsNilpotent G] → Prop
      hbase : ∀ (G : Type u_2) [inst : Group G] [inst_1 : Subsingleton G], P G
      hstep : ∀ (G : Type u_2) [inst : Group G] [inst_1 : Group.IsNilpotent G], P (H …
      n : Nat
      ih : ∀ (G : Type u_2) [inst : Group G] [inst_1 : Group.IsNilpotent G], Eq (Gro …
      G : Type u_2
      inst✝¹ : Group G
      inst✝ : Group.IsNilpotent G
      h : Eq (Group.nilpotencyClass G) (HAdd.hAdd n 1)
      hn : Eq (Group.nilpotencyClass (HasQuotient.Quotient G (Subgroup.center G))) n
      ⊢ P G
    -/
    exact hstep _ (ih _ hn)
    /-
      🎉 no goals
    -/


theorem derived_le_lower_central (n : ℕ) : derivedSeries G n ≤ lowerCentralSeries G n := by
  /-
    G : Type u_1
    inst✝ : Group G
    n : Nat
    ⊢ LE.le (derivedSeries G n) (lowerCentralSeries G n)
  -/
  induction' n with i ih
    /-
      case zero
      G : Type u_1
      inst✝ : Group G
      ⊢ LE.le (derivedSeries G 0) (lowerCentralSeries G 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      G : Type u_1
      inst✝ : Group G
      i : Nat
      ih : LE.le (derivedSeries G i) (lowerCentralSeries G i)
      ⊢ LE.le (derivedSeries G (HAdd.hAdd i 1)) (lowerCentralSeries G (HAdd.hAdd i 1))
    -/
  · apply commutator_mono ih
    /-
      case succ
      G : Type u_1
      inst✝ : Group G
      i : Nat
      ih : LE.le (derivedSeries G i) (lowerCentralSeries G i)
      ⊢ LE.le (derivedSeries G i) Top.top
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Abelian groups are nilpotent -/
instance (priority := 100) CommGroup.isNilpotent {G : Type*} [CommGroup G] : IsNilpotent G := by
  /-
    G✝ : Type u_1
    inst✝² : Group G✝
    H : Subgroup G✝
    inst✝¹ : H.Normal
    G : Type u_2
    inst✝ : CommGroup G
    ⊢ Group.IsNilpotent G
  -/
  use 1
  /-
    case h
    G✝ : Type u_1
    inst✝² : Group G✝
    H : Subgroup G✝
    inst✝¹ : H.Normal
    G : Type u_2
    inst✝ : CommGroup G
    ⊢ Eq (upperCentralSeries G 1) Top.top
  -/
  rw [upperCentralSeries_one]
  /-
    case h
    G✝ : Type u_1
    inst✝² : Group G✝
    H : Subgroup G✝
    inst✝¹ : H.Normal
    G : Type u_2
    inst✝ : CommGroup G
    ⊢ Eq (Subgroup.center G) Top.top
  -/
  apply CommGroup.center_eq_top
  /-
    🎉 no goals
  -/


/-- Abelian groups have nilpotency class at most one -/
theorem CommGroup.nilpotencyClass_le_one {G : Type*} [CommGroup G] :
    Group.nilpotencyClass G ≤ 1 := by
  /-
    G : Type u_2
    inst✝ : CommGroup G
    ⊢ LE.le (Group.nilpotencyClass G) 1
  -/
  rw [← upperCentralSeries_eq_top_iff_nilpotencyClass_le, upperCentralSeries_one]
  /-
    G : Type u_2
    inst✝ : CommGroup G
    ⊢ Eq (Subgroup.center G) Top.top
  -/
  apply CommGroup.center_eq_top
  /-
    🎉 no goals
  -/


/-- Groups with nilpotency class at most one are abelian -/
def commGroupOfNilpotencyClass [IsNilpotent G] (h : Group.nilpotencyClass G ≤ 1) : CommGroup G :=
  Group.commGroupOfCenterEqTop <| by
    /-
      G : Type u_1
      inst✝² : Group G
      H : Subgroup G
      inst✝¹ : H.Normal
      inst✝ : Group.IsNilpotent G
      h : LE.le (Group.nilpotencyClass G) 1
      ⊢ Eq (Subgroup.center G) Top.top
    -/
    rw [← upperCentralSeries_one]
    /-
      G : Type u_1
      inst✝² : Group G
      H : Subgroup G
      inst✝¹ : H.Normal
      inst✝ : Group.IsNilpotent G
      h : LE.le (Group.nilpotencyClass G) 1
      ⊢ Eq (upperCentralSeries G 1) Top.top
    -/
    exact upperCentralSeries_eq_top_iff_nilpotencyClass_le.mpr h
    /-
      🎉 no goals
    -/


theorem lowerCentralSeries_prod (n : ℕ) :
    lowerCentralSeries (G₁ × G₂) n = (lowerCentralSeries G₁ n).prod (lowerCentralSeries G₂ n) := by
  /-
    G₁ : Type u_2
    G₂ : Type u_3
    inst✝¹ : Group G₁
    inst✝ : Group G₂
    n : Nat
    ⊢ Eq (lowerCentralSeries (Prod G₁ G₂) n) ((lowerCentralSeries G₁ n).prod (lowe …
  -/
  induction' n with n ih
    /-
      case zero
      G₁ : Type u_2
      G₂ : Type u_3
      inst✝¹ : Group G₁
      inst✝ : Group G₂
      ⊢ Eq (lowerCentralSeries (Prod G₁ G₂) 0) ((lowerCentralSeries G₁ 0).prod (lowe …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · calc
      lowerCentralSeries (G₁ × G₂) n.succ = ⁅lowerCentralSeries (G₁ × G₂) n, ⊤⁆ := rfl
      _ = ⁅(lowerCentralSeries G₁ n).prod (lowerCentralSeries G₂ n), ⊤⁆ := by rw [ih]
      _ = ⁅(lowerCentralSeries G₁ n).prod (lowerCentralSeries G₂ n), (⊤ : Subgroup G₁).prod ⊤⁆ := by
        simp
      _ = ⁅lowerCentralSeries G₁ n, (⊤ : Subgroup G₁)⁆.prod ⁅lowerCentralSeries G₂ n, ⊤⁆ :=
        (commutator_prod_prod _ _ _ _)
      _ = (lowerCentralSeries G₁ n.succ).prod (lowerCentralSeries G₂ n.succ) := rfl


/-- Products of nilpotent groups are nilpotent -/
instance isNilpotent_prod [IsNilpotent G₁] [IsNilpotent G₂] : IsNilpotent (G₁ × G₂) := by
  /-
    G : Type u_1
    inst✝⁵ : Group G
    H : Subgroup G
    inst✝⁴ : H.Normal
    G₁ : Type u_2
    G₂ : Type u_3
    inst✝³ : Group G₁
    inst✝² : Group G₂
    inst✝¹ : Group.IsNilpotent G₁
    inst✝ : Group.IsNilpotent G₂
    ⊢ Group.IsNilpotent (Prod G₁ G₂)
  -/
  rw [nilpotent_iff_lowerCentralSeries]
  /-
    G : Type u_1
    inst✝⁵ : Group G
    H : Subgroup G
    inst✝⁴ : H.Normal
    G₁ : Type u_2
    G₂ : Type u_3
    inst✝³ : Group G₁
    inst✝² : Group G₂
    inst✝¹ : Group.IsNilpotent G₁
    inst✝ : Group.IsNilpotent G₂
    ⊢ Exists fun n => Eq (lowerCentralSeries (Prod G₁ G₂) n) Bot.bot
  -/
  refine ⟨max (Group.nilpotencyClass G₁) (Group.nilpotencyClass G₂), ?_⟩
  rw [lowerCentralSeries_prod,
    lowerCentralSeries_eq_bot_iff_nilpotencyClass_le.mpr (le_max_left _ _),
    lowerCentralSeries_eq_bot_iff_nilpotencyClass_le.mpr (le_max_right _ _), bot_prod_bot]


/-- The nilpotency class of a product is the max of the nilpotency classes of the factors -/
theorem nilpotencyClass_prod [IsNilpotent G₁] [IsNilpotent G₂] :
    Group.nilpotencyClass (G₁ × G₂) =
    max (Group.nilpotencyClass G₁) (Group.nilpotencyClass G₂) := by
  /-
    G₁ : Type u_2
    G₂ : Type u_3
    inst✝³ : Group G₁
    inst✝² : Group G₂
    inst✝¹ : Group.IsNilpotent G₁
    inst✝ : Group.IsNilpotent G₂
    ⊢ Eq (Group.nilpotencyClass (Prod G₁ G₂)) (Max.max (Group.nilpotencyClass G₁)  …
  -/
  refine eq_of_forall_ge_iff fun k => ?_
  simp only [max_le_iff, ← lowerCentralSeries_eq_bot_iff_nilpotencyClass_le,
    lowerCentralSeries_prod, prod_eq_bot_iff]


theorem lowerCentralSeries_pi_le (n : ℕ) :
    lowerCentralSeries (∀ i, Gs i) n ≤ Subgroup.pi Set.univ
      fun i => lowerCentralSeries (Gs i) n := by
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝ : (i : η) → Group (Gs i)
    n : Nat
    ⊢ LE.le (lowerCentralSeries ((i : η) → Gs i) n) (Subgroup.pi Set.univ fun i => …
  -/
  let pi := fun f : ∀ i, Subgroup (Gs i) => Subgroup.pi Set.univ f
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝ : (i : η) → Group (Gs i)
    n : Nat
    pi : ((i : η) → Subgroup (Gs i)) → Subgroup ((i : η) → Gs i) := fun f => Subgr …
    ⊢ LE.le (lowerCentralSeries ((i : η) → Gs i) n) (Subgroup.pi Set.univ fun i => …
  -/
  induction' n with n ih
    /-
      case zero
      η : Type u_2
      Gs : η → Type u_3
      inst✝ : (i : η) → Group (Gs i)
      pi : ((i : η) → Subgroup (Gs i)) → Subgroup ((i : η) → Gs i) := fun f => Subgr …
      ⊢ LE.le (lowerCentralSeries ((i : η) → Gs i) 0) (Subgroup.pi Set.univ fun i => …
    -/
  · simp [pi_top]
    /-
      🎉 no goals
    -/
  · calc
      lowerCentralSeries (∀ i, Gs i) n.succ = ⁅lowerCentralSeries (∀ i, Gs i) n, ⊤⁆ := rfl
      _ ≤ ⁅pi fun i => lowerCentralSeries (Gs i) n, ⊤⁆ := commutator_mono ih (le_refl _)
      _ = ⁅pi fun i => lowerCentralSeries (Gs i) n, pi fun i => ⊤⁆ := by simp [pi, pi_top]
      _ ≤ pi fun i => ⁅lowerCentralSeries (Gs i) n, ⊤⁆ := commutator_pi_pi_le _ _
      _ = pi fun i => lowerCentralSeries (Gs i) n.succ := rfl


/-- products of nilpotent groups are nilpotent if their nilpotency class is bounded -/
theorem isNilpotent_pi_of_bounded_class [∀ i, IsNilpotent (Gs i)] (n : ℕ)
    (h : ∀ i, Group.nilpotencyClass (Gs i) ≤ n) : IsNilpotent (∀ i, Gs i) := by
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝¹ : (i : η) → Group (Gs i)
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    n : Nat
    h : ∀ (i : η), LE.le (Group.nilpotencyClass (Gs i)) n
    ⊢ Group.IsNilpotent ((i : η) → Gs i)
  -/
  rw [nilpotent_iff_lowerCentralSeries]
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝¹ : (i : η) → Group (Gs i)
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    n : Nat
    h : ∀ (i : η), LE.le (Group.nilpotencyClass (Gs i)) n
    ⊢ Exists fun n => Eq (lowerCentralSeries ((i : η) → Gs i) n) Bot.bot
  -/
  refine ⟨n, ?_⟩
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝¹ : (i : η) → Group (Gs i)
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    n : Nat
    h : ∀ (i : η), LE.le (Group.nilpotencyClass (Gs i)) n
    ⊢ Eq (lowerCentralSeries ((i : η) → Gs i) n) Bot.bot
  -/
  rw [eq_bot_iff]
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝¹ : (i : η) → Group (Gs i)
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    n : Nat
    h : ∀ (i : η), LE.le (Group.nilpotencyClass (Gs i)) n
    ⊢ LE.le (lowerCentralSeries ((i : η) → Gs i) n) Bot.bot
  -/
  apply le_trans (lowerCentralSeries_pi_le _)
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝¹ : (i : η) → Group (Gs i)
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    n : Nat
    h : ∀ (i : η), LE.le (Group.nilpotencyClass (Gs i)) n
    ⊢ LE.le (Subgroup.pi Set.univ fun i => lowerCentralSeries (Gs i) n) Bot.bot
  -/
  rw [← eq_bot_iff, pi_eq_bot_iff]
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝¹ : (i : η) → Group (Gs i)
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    n : Nat
    h : ∀ (i : η), LE.le (Group.nilpotencyClass (Gs i)) n
    ⊢ ∀ (i : η), Eq (lowerCentralSeries (Gs i) n) Bot.bot
  -/
  intro i
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝¹ : (i : η) → Group (Gs i)
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    n : Nat
    h : ∀ (i : η), LE.le (Group.nilpotencyClass (Gs i)) n
    i : η
    ⊢ Eq (lowerCentralSeries (Gs i) n) Bot.bot
  -/
  apply lowerCentralSeries_eq_bot_iff_nilpotencyClass_le.mpr (h i)
  /-
    🎉 no goals
  -/


theorem lowerCentralSeries_pi_of_finite [Finite η] (n : ℕ) :
    lowerCentralSeries (∀ i, Gs i) n = Subgroup.pi Set.univ
      fun i => lowerCentralSeries (Gs i) n := by
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝¹ : (i : η) → Group (Gs i)
    inst✝ : Finite η
    n : Nat
    ⊢ Eq (lowerCentralSeries ((i : η) → Gs i) n) (Subgroup.pi Set.univ fun i => lo …
  -/
  let pi := fun f : ∀ i, Subgroup (Gs i) => Subgroup.pi Set.univ f
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝¹ : (i : η) → Group (Gs i)
    inst✝ : Finite η
    n : Nat
    pi : ((i : η) → Subgroup (Gs i)) → Subgroup ((i : η) → Gs i) := fun f => Subgr …
    ⊢ Eq (lowerCentralSeries ((i : η) → Gs i) n) (Subgroup.pi Set.univ fun i => lo …
  -/
  induction' n with n ih
    /-
      case zero
      η : Type u_2
      Gs : η → Type u_3
      inst✝¹ : (i : η) → Group (Gs i)
      inst✝ : Finite η
      pi : ((i : η) → Subgroup (Gs i)) → Subgroup ((i : η) → Gs i) := fun f => Subgr …
      ⊢ Eq (lowerCentralSeries ((i : η) → Gs i) 0) (Subgroup.pi Set.univ fun i => lo …
    -/
  · simp [pi_top]
    /-
      🎉 no goals
    -/
  · calc
      lowerCentralSeries (∀ i, Gs i) n.succ = ⁅lowerCentralSeries (∀ i, Gs i) n, ⊤⁆ := rfl
      _ = ⁅pi fun i => lowerCentralSeries (Gs i) n, ⊤⁆ := by rw [ih]
      _ = ⁅pi fun i => lowerCentralSeries (Gs i) n, pi fun i => ⊤⁆ := by simp [pi, pi_top]
      _ = pi fun i => ⁅lowerCentralSeries (Gs i) n, ⊤⁆ := commutator_pi_pi_of_finite _ _
      _ = pi fun i => lowerCentralSeries (Gs i) n.succ := rfl


/-- n-ary products of nilpotent groups are nilpotent -/
instance isNilpotent_pi [Finite η] [∀ i, IsNilpotent (Gs i)] : IsNilpotent (∀ i, Gs i) := by
  /-
    G : Type u_1
    inst✝⁴ : Group G
    H : Subgroup G
    inst✝³ : H.Normal
    η : Type u_2
    Gs : η → Type u_3
    inst✝² : (i : η) → Group (Gs i)
    inst✝¹ : Finite η
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    ⊢ Group.IsNilpotent ((i : η) → Gs i)
  -/
  cases nonempty_fintype η
  /-
    case intro
    G : Type u_1
    inst✝⁴ : Group G
    H : Subgroup G
    inst✝³ : H.Normal
    η : Type u_2
    Gs : η → Type u_3
    inst✝² : (i : η) → Group (Gs i)
    inst✝¹ : Finite η
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    val✝ : Fintype η
    ⊢ Group.IsNilpotent ((i : η) → Gs i)
  -/
  rw [nilpotent_iff_lowerCentralSeries]
  /-
    case intro
    G : Type u_1
    inst✝⁴ : Group G
    H : Subgroup G
    inst✝³ : H.Normal
    η : Type u_2
    Gs : η → Type u_3
    inst✝² : (i : η) → Group (Gs i)
    inst✝¹ : Finite η
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    val✝ : Fintype η
    ⊢ Exists fun n => Eq (lowerCentralSeries ((i : η) → Gs i) n) Bot.bot
  -/
  refine ⟨Finset.univ.sup fun i => Group.nilpotencyClass (Gs i), ?_⟩
  /-
    case intro
    G : Type u_1
    inst✝⁴ : Group G
    H : Subgroup G
    inst✝³ : H.Normal
    η : Type u_2
    Gs : η → Type u_3
    inst✝² : (i : η) → Group (Gs i)
    inst✝¹ : Finite η
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    val✝ : Fintype η
    ⊢ Eq (lowerCentralSeries ((i : η) → Gs i) (Finset.univ.sup fun i => Group.nilp …
  -/
  rw [lowerCentralSeries_pi_of_finite, pi_eq_bot_iff]
  /-
    case intro
    G : Type u_1
    inst✝⁴ : Group G
    H : Subgroup G
    inst✝³ : H.Normal
    η : Type u_2
    Gs : η → Type u_3
    inst✝² : (i : η) → Group (Gs i)
    inst✝¹ : Finite η
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    val✝ : Fintype η
    ⊢ ∀ (i : η), Eq (lowerCentralSeries (Gs i) (Finset.univ.sup fun i => Group.nil …
  -/
  intro i
  /-
    case intro
    G : Type u_1
    inst✝⁴ : Group G
    H : Subgroup G
    inst✝³ : H.Normal
    η : Type u_2
    Gs : η → Type u_3
    inst✝² : (i : η) → Group (Gs i)
    inst✝¹ : Finite η
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    val✝ : Fintype η
    i : η
    ⊢ Eq (lowerCentralSeries (Gs i) (Finset.univ.sup fun i => Group.nilpotencyClas …
  -/
  rw [lowerCentralSeries_eq_bot_iff_nilpotencyClass_le]
  /-
    case intro
    G : Type u_1
    inst✝⁴ : Group G
    H : Subgroup G
    inst✝³ : H.Normal
    η : Type u_2
    Gs : η → Type u_3
    inst✝² : (i : η) → Group (Gs i)
    inst✝¹ : Finite η
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    val✝ : Fintype η
    i : η
    ⊢ LE.le (Group.nilpotencyClass (Gs i)) (Finset.univ.sup fun i => Group.nilpote …
  -/
  exact Finset.le_sup (f := fun i => Group.nilpotencyClass (Gs i)) (Finset.mem_univ i)
  /-
    🎉 no goals
  -/


/-- The nilpotency class of an n-ary product is the sup of the nilpotency classes of the factors -/
theorem nilpotencyClass_pi [Fintype η] [∀ i, IsNilpotent (Gs i)] :
    Group.nilpotencyClass (∀ i, Gs i) = Finset.univ.sup fun i => Group.nilpotencyClass (Gs i) := by
  /-
    η : Type u_2
    Gs : η → Type u_3
    inst✝² : (i : η) → Group (Gs i)
    inst✝¹ : Fintype η
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    ⊢ Eq (Group.nilpotencyClass ((i : η) → Gs i)) (Finset.univ.sup fun i => Group. …
  -/
  apply eq_of_forall_ge_iff
  /-
    case H
    η : Type u_2
    Gs : η → Type u_3
    inst✝² : (i : η) → Group (Gs i)
    inst✝¹ : Fintype η
    inst✝ : ∀ (i : η), Group.IsNilpotent (Gs i)
    ⊢ ∀ (c : Nat), Iff (LE.le (Group.nilpotencyClass ((i : η) → Gs i)) c) (LE.le ( …
  -/
  intro k
  simp only [Finset.sup_le_iff, ← lowerCentralSeries_eq_bot_iff_nilpotencyClass_le,
    lowerCentralSeries_pi_of_finite, pi_eq_bot_iff, Finset.mem_univ, true_imp_iff]


/-- A nilpotent subgroup is solvable -/
instance (priority := 100) IsNilpotent.to_isSolvable [h : IsNilpotent G] : IsSolvable G := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    h : Group.IsNilpotent G
    ⊢ IsSolvable G
  -/
  obtain ⟨n, hn⟩ := nilpotent_iff_lowerCentralSeries.1 h
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    h : Group.IsNilpotent G
    n : Nat
    hn : Eq (lowerCentralSeries G n) Bot.bot
    ⊢ IsSolvable G
  -/
  use n
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    h : Group.IsNilpotent G
    n : Nat
    hn : Eq (lowerCentralSeries G n) Bot.bot
    ⊢ Eq (derivedSeries G n) Bot.bot
  -/
  rw [eq_bot_iff, ← hn]
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    h : Group.IsNilpotent G
    n : Nat
    hn : Eq (lowerCentralSeries G n) Bot.bot
    ⊢ LE.le (derivedSeries G n) (lowerCentralSeries G n)
  -/
  exact derived_le_lower_central n
  /-
    🎉 no goals
  -/


theorem normalizerCondition_of_isNilpotent [h : IsNilpotent G] : NormalizerCondition G := by
  -- roughly based on https://groupprops.subwiki.org/wiki/Nilpotent_implies_normalizer_condition
  /-
    G : Type u_1
    inst✝ : Group G
    h : Group.IsNilpotent G
    ⊢ NormalizerCondition G
  -/
  rw [normalizerCondition_iff_only_full_group_self_normalizing]
  /-
    G : Type u_1
    inst✝ : Group G
    h : Group.IsNilpotent G
    ⊢ ∀ (H : Subgroup G), Eq H.normalizer H → Eq H Top.top
  -/
  apply @nilpotent_center_quotient_ind _ G _ _ <;> clear! G
    /-
      case hbase
      ⊢ ∀ (G : Type u_1) [inst : Group G] [inst_1 : Subsingleton G] (H : Subgroup G) …
    -/
  · intro G _ _ H _
    /-
      case hbase
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Subsingleton G
      H : Subgroup G
      a✝ : Eq H.normalizer H
      ⊢ Eq H Top.top
    -/
    exact @Subsingleton.elim _ Unique.instSubsingleton _ _
    /-
      🎉 no goals
    -/
    /-
      case hstep
      ⊢ ∀ (G : Type u_1) [inst : Group G] [inst_1 : Group.IsNilpotent G], (∀ (H : Su …
    -/
  · intro G _ _ ih H hH
    /-
      case hstep
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Group.IsNilpotent G
      ih : ∀ (H : Subgroup (HasQuotient.Quotient G (Subgroup.center G))), Eq H.norma …
      H : Subgroup G
      hH : Eq H.normalizer H
      ⊢ Eq H Top.top
    -/
    have hch : center G ≤ H := Subgroup.center_le_normalizer.trans (le_of_eq hH)
    /-
      case hstep
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Group.IsNilpotent G
      ih : ∀ (H : Subgroup (HasQuotient.Quotient G (Subgroup.center G))), Eq H.norma …
      H : Subgroup G
      hH : Eq H.normalizer H
      hch : LE.le (Subgroup.center G) H
      ⊢ Eq H Top.top
    -/
    have hkh : (mk' (center G)).ker ≤ H := by simpa using hch
    /-
      case hstep
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Group.IsNilpotent G
      ih : ∀ (H : Subgroup (HasQuotient.Quotient G (Subgroup.center G))), Eq H.norma …
      H : Subgroup G
      hH : Eq H.normalizer H
      hch : LE.le (Subgroup.center G) H
      hkh : LE.le (QuotientGroup.mk' (Subgroup.center G)).ker H
      ⊢ Eq H Top.top
    -/
    have hsur : Function.Surjective (mk' (center G)) := Quot.mk_surjective
    /-
      case hstep
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Group.IsNilpotent G
      ih : ∀ (H : Subgroup (HasQuotient.Quotient G (Subgroup.center G))), Eq H.norma …
      H : Subgroup G
      hH : Eq H.normalizer H
      hch : LE.le (Subgroup.center G) H
      hkh : LE.le (QuotientGroup.mk' (Subgroup.center G)).ker H
      hsur : Function.Surjective ⇑(QuotientGroup.mk' (Subgroup.center G))
      ⊢ Eq H Top.top
    -/
    let H' := H.map (mk' (center G))
    have hH' : H'.normalizer = H' := by
      apply comap_injective hsur
      rw [comap_normalizer_eq_of_surjective _ hsur, comap_map_eq_self hkh]
      exact hH
    /-
      case hstep
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Group.IsNilpotent G
      ih : ∀ (H : Subgroup (HasQuotient.Quotient G (Subgroup.center G))), Eq H.norma …
      H : Subgroup G
      hH : Eq H.normalizer H
      hch : LE.le (Subgroup.center G) H
      hkh : LE.le (QuotientGroup.mk' (Subgroup.center G)).ker H
      hsur : Function.Surjective ⇑(QuotientGroup.mk' (Subgroup.center G))
      H' : Subgroup (HasQuotient.Quotient G (Subgroup.center G)) := Subgroup.map (Qu …
      hH' : Eq H'.normalizer H'
      ⊢ Eq H Top.top
    -/
    apply map_injective_of_ker_le (mk' (center G)) hkh le_top
    /-
      case hstep
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Group.IsNilpotent G
      ih : ∀ (H : Subgroup (HasQuotient.Quotient G (Subgroup.center G))), Eq H.norma …
      H : Subgroup G
      hH : Eq H.normalizer H
      hch : LE.le (Subgroup.center G) H
      hkh : LE.le (QuotientGroup.mk' (Subgroup.center G)).ker H
      hsur : Function.Surjective ⇑(QuotientGroup.mk' (Subgroup.center G))
      H' : Subgroup (HasQuotient.Quotient G (Subgroup.center G)) := Subgroup.map (Qu …
      hH' : Eq H'.normalizer H'
      ⊢ Eq (Subgroup.map (QuotientGroup.mk' (Subgroup.center G)) H) (Subgroup.map (Q …
    -/
    exact (ih H' hH').trans (symm (map_top_of_surjective _ hsur))
    /-
      🎉 no goals
    -/


/-- A p-group is nilpotent -/
theorem IsPGroup.isNilpotent [Finite G] {p : ℕ} [hp : Fact (Nat.Prime p)] (h : IsPGroup p G) :
    IsNilpotent G := by
  /-
    G : Type u_1
    hG : Group G
    inst✝ : Finite G
    p : Nat
    hp : Fact (Nat.Prime p)
    h : IsPGroup p G
    ⊢ Group.IsNilpotent G
  -/
  cases' nonempty_fintype G
  classical
    revert hG
    apply @Fintype.induction_subsingleton_or_nontrivial _ G _
    · intro _ _ _ _
      infer_instance
    · intro G _ _ ih _ h
      have hcq : Fintype.card (G ⧸ center G) < Fintype.card G := by
        simp only [← Nat.card_eq_fintype_card]
        rw [card_eq_card_quotient_mul_card_subgroup (center G)]
        simp only [Nat.card_eq_fintype_card]
        apply lt_mul_of_one_lt_right
        · exact Fintype.card_pos_iff.mpr One.instNonempty
        · simp only [← Nat.card_eq_fintype_card]
          exact (Subgroup.one_lt_card_iff_ne_bot _).mpr (ne_of_gt h.bot_lt_center)
      have hnq : IsNilpotent (G ⧸ center G) := ih _ hcq (h.to_quotient (center G))
      exact of_quotient_center_nilpotent hnq


/-- If a finite group is the direct product of its Sylow groups, it is nilpotent -/
theorem isNilpotent_of_product_of_sylow_group
    (e : (∀ p : (Nat.card G).primeFactors, ∀ P : Sylow p G, (↑P : Subgroup G)) ≃* G) :
    IsNilpotent G := by
  classical
    let ps := (Nat.card G).primeFactors
    have : ∀ (p : ps) (P : Sylow p G), IsNilpotent (↑P : Subgroup G) := by
      intro p P
      haveI : Fact (Nat.Prime ↑p) := Fact.mk <| Nat.prime_of_mem_primeFactors p.2
      exact P.isPGroup'.isNilpotent
    exact nilpotent_of_mulEquiv e


/-- A finite group is nilpotent iff the normalizer condition holds, and iff all maximal groups are
normal and iff all Sylow groups are normal and iff the group is the direct product of its Sylow
groups. -/
theorem isNilpotent_of_finite_tfae :
    List.TFAE
      [IsNilpotent G, NormalizerCondition G, ∀ H : Subgroup G, IsCoatom H → H.Normal,
        ∀ (p : ℕ) (_hp : Fact p.Prime) (P : Sylow p G), (↑P : Subgroup G).Normal,
        Nonempty
          ((∀ p : (Nat.card G).primeFactors, ∀ P : Sylow p G, (↑P : Subgroup G)) ≃* G)] := by
  /-
    G : Type u_1
    hG : Group G
    inst✝ : Finite G
    ⊢ (List.cons (Group.IsNilpotent G) (List.cons (NormalizerCondition G) (List.co …
  -/
  tfae_have 1 → 2 := @normalizerCondition_of_isNilpotent _ _
  tfae_have 2 → 3
  | h, H => NormalizerCondition.normal_of_coatom H h
  tfae_have 3 → 4
  | h, p, _, P => Sylow.normal_of_all_max_subgroups_normal h _
  tfae_have 4 → 5
  | h => Nonempty.intro (Sylow.directProductOfNormal fun {p hp hP} => h p hp hP)
  tfae_have 5 → 1
  | ⟨e⟩ => isNilpotent_of_product_of_sylow_group e
  /-
    G : Type u_1
    hG : Group G
    inst✝ : Finite G
    tfae_1_to_2 : Group.IsNilpotent G → NormalizerCondition G
    tfae_2_to_3 : NormalizerCondition G → ∀ (H : Subgroup G), IsCoatom H → H.Normal
    tfae_3_to_4 : (∀ (H : Subgroup G), IsCoatom H → H.Normal) → ∀ (p : Nat), Fact  …
    tfae_4_to_5 : (∀ (p : Nat), Fact (Nat.Prime p) → ∀ (P : Sylow p G), (↑P).Norma …
    tfae_5_to_1 : Nonempty (MulEquiv ((p : Subtype fun x => Membership.mem (Nat.ca …
    ⊢ (List.cons (Group.IsNilpotent G) (List.cons (NormalizerCondition G) (List.co …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-05")] alias isNilpotent_of_finite_tFAE := isNilpotent_of_finite_tfae


