/-- A generalisation of the lower central series. The zeroth term is a specified Lie submodule of
a Lie module. In the case when we specify the top ideal `⊤` of the Lie algebra, regarded as a Lie
module over itself, we get the usual lower central series of a Lie algebra.

It can be more convenient to work with this generalisation when considering the lower central series
of a Lie submodule, regarded as a Lie module in its own right, since it provides a type-theoretic
expression of the fact that the terms of the Lie submodule's lower central series are also Lie
submodules of the enclosing Lie module.

See also `LieSubmodule.lowerCentralSeries_eq_lcs_comap` and
`LieSubmodule.lowerCentralSeries_map_eq_lcs` below, as well as `LieSubmodule.ucs`. -/
def lcs : LieSubmodule R L M → LieSubmodule R L M :=
  (fun N => ⁅(⊤ : LieIdeal R L), N⁆)^[k]


@[simp]
theorem lcs_zero (N : LieSubmodule R L M) : N.lcs 0 = N :=
  rfl


@[simp]
theorem lcs_succ : N.lcs (k + 1) = ⁅(⊤ : LieIdeal R L), N.lcs k⁆ :=
  Function.iterate_succ_apply' (fun N' => ⁅⊤, N'⁆) k N


@[simp]
lemma lcs_sup {N₁ N₂ : LieSubmodule R L M} {k : ℕ} :
    (N₁ ⊔ N₂).lcs k = N₁.lcs k ⊔ N₂.lcs k := by
  induction k with
  | zero => simp
  | succ k ih => simp only [LieSubmodule.lcs_succ, ih, LieSubmodule.lie_sup]


/-- The lower central series of Lie submodules of a Lie module. -/
def lowerCentralSeries : LieSubmodule R L M :=
  (⊤ : LieSubmodule R L M).lcs k


@[simp]
theorem lowerCentralSeries_zero : lowerCentralSeries R L M 0 = ⊤ :=
  rfl


@[simp]
theorem lowerCentralSeries_succ :
    lowerCentralSeries R L M (k + 1) = ⁅(⊤ : LieIdeal R L), lowerCentralSeries R L M k⁆ :=
  (⊤ : LieSubmodule R L M).lcs_succ k


theorem lcs_le_self : N.lcs k ≤ N := by
  induction k with
  | zero => simp
  | succ k ih =>
    simp only [lcs_succ]
    exact (LieSubmodule.mono_lie_right ⊤ ih).trans (N.lie_le_right ⊤)


theorem lowerCentralSeries_eq_lcs_comap : lowerCentralSeries R L N k = (N.lcs k).comap N.incl := by
  induction k with
  | zero => simp
  | succ k ih =>
    simp only [lcs_succ, lowerCentralSeries_succ] at ih ⊢
    have : N.lcs k ≤ N.incl.range := by
      rw [N.range_incl]
      apply lcs_le_self
    rw [ih, LieSubmodule.comap_bracket_eq _ N.incl _ N.ker_incl this]


theorem lowerCentralSeries_map_eq_lcs : (lowerCentralSeries R L N k).map N.incl = N.lcs k := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    k : Nat
    N : LieSubmodule R L M
    inst✝ : LieModule R L M
    ⊢ Eq (LieSubmodule.map N.incl (LieModule.lowerCentralSeries R L (Subtype fun x …
  -/
  rw [lowerCentralSeries_eq_lcs_comap, LieSubmodule.map_comap_incl, inf_eq_right]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    k : Nat
    N : LieSubmodule R L M
    inst✝ : LieModule R L M
    ⊢ LE.le (LieSubmodule.lcs k N) N
  -/
  apply lcs_le_self
  /-
    🎉 no goals
  -/


theorem antitone_lowerCentralSeries : Antitone <| lowerCentralSeries R L M := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    ⊢ Antitone (LieModule.lowerCentralSeries R L M)
  -/
  intro l k
  induction k generalizing l with
  | zero => exact fun h ↦ (Nat.le_zero.mp h).symm ▸ le_rfl
  | succ k ih =>
    intro h
    rcases Nat.of_le_succ h with (hk | hk)
    · rw [lowerCentralSeries_succ]
      exact (LieSubmodule.mono_lie_right ⊤ (ih hk)).trans (LieSubmodule.lie_le_right _ _)
    · exact hk.symm ▸ le_rfl


theorem eventually_iInf_lowerCentralSeries_eq [IsArtinian R M] :
    ∀ᶠ l in Filter.atTop, ⨅ k, lowerCentralSeries R L M k = lowerCentralSeries R L M l := by
  have h_wf : WellFoundedGT (LieSubmodule R L M)ᵒᵈ :=
    LieSubmodule.wellFoundedLT_of_isArtinian R L M
  obtain ⟨n, hn : ∀ m, n ≤ m → lowerCentralSeries R L M n = lowerCentralSeries R L M m⟩ :=
    WellFounded.monotone_chain_condition.mp h_wf.wf ⟨_, antitone_lowerCentralSeries R L M⟩
  /-
    case intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : IsArtinian R M
    h_wf : WellFoundedGT (OrderDual (LieSubmodule R L M))
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LieModule.lowerCentralSeries R L M n) (LieMo …
    ⊢ Filter.Eventually (fun l => Eq (iInf fun k => LieModule.lowerCentralSeries R …
  -/
  refine Filter.eventually_atTop.mpr ⟨n, fun l hl ↦ le_antisymm (iInf_le _ _) (le_iInf fun m ↦ ?_)⟩
  /-
    case intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : IsArtinian R M
    h_wf : WellFoundedGT (OrderDual (LieSubmodule R L M))
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LieModule.lowerCentralSeries R L M n) (LieMo …
    l : Nat
    hl : GE.ge l n
    m : Nat
    ⊢ LE.le (LieModule.lowerCentralSeries R L M l) (LieModule.lowerCentralSeries R …
  -/
  rcases le_or_lt l m with h | h
    /-
      case intro.inl
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : IsArtinian R M
      h_wf : WellFoundedGT (OrderDual (LieSubmodule R L M))
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq (LieModule.lowerCentralSeries R L M n) (LieMo …
      l : Nat
      hl : GE.ge l n
      m : Nat
      h : LE.le l m
      ⊢ LE.le (LieModule.lowerCentralSeries R L M l) (LieModule.lowerCentralSeries R …
    -/
  · rw [← hn _ hl, ← hn _ (hl.trans h)]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : IsArtinian R M
      h_wf : WellFoundedGT (OrderDual (LieSubmodule R L M))
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq (LieModule.lowerCentralSeries R L M n) (LieMo …
      l : Nat
      hl : GE.ge l n
      m : Nat
      h : LT.lt m l
      ⊢ LE.le (LieModule.lowerCentralSeries R L M l) (LieModule.lowerCentralSeries R …
    -/
  · exact antitone_lowerCentralSeries R L M (le_of_lt h)
    /-
      🎉 no goals
    -/


theorem trivial_iff_lower_central_eq_bot : IsTrivial L M ↔ lowerCentralSeries R L M 1 = ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    ⊢ Iff (LieModule.IsTrivial L M) (Eq (LieModule.lowerCentralSeries R L M 1) Bot …
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u
      L : Type v
      M : Type w
      inst✝⁵ : CommRing R
      inst✝⁴ : LieRing L
      inst✝³ : LieAlgebra R L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      h : LieModule.IsTrivial L M
      ⊢ Eq (LieModule.lowerCentralSeries R L M 1) Bot.bot
    -/
  · erw [eq_bot_iff, LieSubmodule.lieSpan_le]; rintro m ⟨x, n, hn⟩; rw [← hn, h.trivial]; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
    /-
      case mpr
      R : Type u
      L : Type v
      M : Type w
      inst✝⁵ : CommRing R
      inst✝⁴ : LieRing L
      inst✝³ : LieAlgebra R L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      h : Eq (LieModule.lowerCentralSeries R L M 1) Bot.bot
      ⊢ LieModule.IsTrivial L M
    -/
  · rw [LieSubmodule.eq_bot_iff] at h; apply IsTrivial.mk; intro x m; apply h
    /-
      case mpr.trivial.a
      R : Type u
      L : Type v
      M : Type w
      inst✝⁵ : CommRing R
      inst✝⁴ : LieRing L
      inst✝³ : LieAlgebra R L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      h : ∀ (m : M), Membership.mem (LieModule.lowerCentralSeries R L M 1) m → Eq m 0
      x : L
      m : M
      ⊢ Membership.mem (LieModule.lowerCentralSeries R L M 1) (Bracket.bracket x m)
    -/
    apply LieSubmodule.subset_lieSpan
    -- Porting note: was `use x, m; rfl`
    simp only [LieSubmodule.top_coe, Subtype.exists, LieSubmodule.mem_top, exists_prop, true_and,
      Set.mem_setOf]
    /-
      case mpr.trivial.a.a
      R : Type u
      L : Type v
      M : Type w
      inst✝⁵ : CommRing R
      inst✝⁴ : LieRing L
      inst✝³ : LieAlgebra R L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      h : ∀ (m : M), Membership.mem (LieModule.lowerCentralSeries R L M 1) m → Eq m 0
      x : L
      m : M
      ⊢ Exists fun a => Exists fun a_1 => Eq (Bracket.bracket a a_1) (Bracket.bracke …
    -/
    exact ⟨x, m, rfl⟩
    /-
      🎉 no goals
    -/


theorem iterate_toEnd_mem_lowerCentralSeries (x : L) (m : M) (k : ℕ) :
    (toEnd R L M x)^[k] m ∈ lowerCentralSeries R L M k := by
  induction k with
  | zero => simp only [Function.iterate_zero, lowerCentralSeries_zero, LieSubmodule.mem_top]
  | succ k ih =>
    simp only [lowerCentralSeries_succ, Function.comp_apply, Function.iterate_succ',
      toEnd_apply_apply]
    exact LieSubmodule.lie_mem_lie (LieSubmodule.mem_top x) ih


theorem iterate_toEnd_mem_lowerCentralSeries₂ (x y : L) (m : M) (k : ℕ) :
    (toEnd R L M x ∘ₗ toEnd R L M y)^[k] m ∈
      lowerCentralSeries R L M (2 * k) := by
  induction k with
  | zero => simp
  | succ k ih =>
    have hk : 2 * k.succ = (2 * k + 1) + 1 := rfl
    simp only [lowerCentralSeries_succ, Function.comp_apply, Function.iterate_succ', hk,
      toEnd_apply_apply, LinearMap.coe_comp, toEnd_apply_apply]
    refine LieSubmodule.lie_mem_lie (LieSubmodule.mem_top x) ?_
    exact LieSubmodule.lie_mem_lie (LieSubmodule.mem_top y) ih


theorem map_lowerCentralSeries_le (f : M →ₗ⁅R,L⁆ M₂) :
    (lowerCentralSeries R L M k).map f ≤ lowerCentralSeries R L M₂ k := by
  induction k with
  | zero => simp only [lowerCentralSeries_zero, le_top]
  | succ k ih =>
    simp only [LieModule.lowerCentralSeries_succ, LieSubmodule.map_bracket_eq]
    exact LieSubmodule.mono_lie_right ⊤ ih


lemma map_lowerCentralSeries_eq {f : M →ₗ⁅R,L⁆ M₂} (hf : Function.Surjective f) :
    (lowerCentralSeries R L M k).map f = lowerCentralSeries R L M₂ k := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    k : Nat
    M₂ : Type w₁
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : Module R M₂
    inst✝² : LieRingModule L M₂
    inst✝¹ : LieModule R L M₂
    inst✝ : LieModule R L M
    f : LieModuleHom R L M M₂
    hf : Function.Surjective ⇑f
    ⊢ Eq (LieSubmodule.map f (LieModule.lowerCentralSeries R L M k)) (LieModule.lo …
  -/
  apply le_antisymm (map_lowerCentralSeries_le k f)
  induction k with
  | zero =>
    rwa [lowerCentralSeries_zero, lowerCentralSeries_zero, top_le_iff, f.map_top,
      f.range_eq_top]
  | succ =>
    simp only [lowerCentralSeries_succ, LieSubmodule.map_bracket_eq]
    apply LieSubmodule.mono_lie_right
    assumption


theorem derivedSeries_le_lowerCentralSeries (k : ℕ) :
    derivedSeries R L k ≤ lowerCentralSeries R L L k := by
  induction k with
  | zero => rw [derivedSeries_def, derivedSeriesOfIdeal_zero, lowerCentralSeries_zero]
  | succ k h =>
    have h' : derivedSeries R L k ≤ ⊤ := by simp only [le_top]
    rw [derivedSeries_def, derivedSeriesOfIdeal_succ, lowerCentralSeries_succ]
    exact LieSubmodule.mono_lie h' h


/-- A Lie module is nilpotent if its lower central series reaches 0 (in a finite number of
steps). -/
class IsNilpotent : Prop where
  nilpotent : ∃ k, lowerCentralSeries R L M k = ⊥


theorem exists_lowerCentralSeries_eq_bot_of_isNilpotent [IsNilpotent R L M] :
    ∃ k, lowerCentralSeries R L M k = ⊥ :=
  IsNilpotent.nilpotent


@[simp] lemma iInf_lowerCentralSeries_eq_bot_of_isNilpotent [IsNilpotent R L M] :
    ⨅ k, lowerCentralSeries R L M k = ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    ⊢ Eq (iInf fun k => LieModule.lowerCentralSeries R L M k) Bot.bot
  -/
  obtain ⟨k, hk⟩ := exists_lowerCentralSeries_eq_bot_of_isNilpotent R L M
  /-
    case intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    ⊢ Eq (iInf fun k => LieModule.lowerCentralSeries R L M k) Bot.bot
  -/
  rw [eq_bot_iff, ← hk]
  /-
    case intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    ⊢ LE.le (iInf fun k => LieModule.lowerCentralSeries R L M k) (LieModule.lowerC …
  -/
  exact iInf_le _ _
  /-
    🎉 no goals
  -/


/-- See also `LieModule.isNilpotent_iff_exists_ucs_eq_top`. -/
theorem isNilpotent_iff : IsNilpotent R L M ↔ ∃ k, lowerCentralSeries R L M k = ⊥ :=
  ⟨fun h => h.nilpotent, fun h => ⟨h⟩⟩


theorem _root_.LieSubmodule.isNilpotent_iff_exists_lcs_eq_bot (N : LieSubmodule R L M) :
    LieModule.IsNilpotent R L N ↔ ∃ k, N.lcs k = ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    ⊢ Iff (LieModule.IsNilpotent R L (Subtype fun x => Membership.mem N x)) (Exist …
  -/
  rw [isNilpotent_iff]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    ⊢ Iff (Exists fun k => Eq (LieModule.lowerCentralSeries R L (Subtype fun x =>  …
  -/
  refine exists_congr fun k => ?_
  rw [N.lowerCentralSeries_eq_lcs_comap k, LieSubmodule.comap_incl_eq_bot,
    inf_eq_right.mpr (N.lcs_le_self k)]


instance (priority := 100) trivialIsNilpotent [IsTrivial L M] : IsNilpotent R L M :=
      /-
        R : Type u
        L : Type v
        M : Type w
        inst✝¹¹ : CommRing R
        inst✝¹⁰ : LieRing L
        inst✝⁹ : LieAlgebra R L
        inst✝⁸ : AddCommGroup M
        inst✝⁷ : Module R M
        inst✝⁶ : LieRingModule L M
        k : Nat
        N : LieSubmodule R L M
        M₂ : Type w₁
        inst✝⁵ : AddCommGroup M₂
        inst✝⁴ : Module R M₂
        inst✝³ : LieRingModule L M₂
        inst✝² : LieModule R L M₂
        inst✝¹ : LieModule R L M
        inst✝ : LieModule.IsTrivial L M
        ⊢ Exists fun k => Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
      -/
  ⟨by use 1; change ⁅⊤, ⊤⁆ = ⊥; simp⟩
                                /-
                                  🎉 no goals
                                -/


theorem exists_forall_pow_toEnd_eq_zero [hM : IsNilpotent R L M] :
    ∃ k : ℕ, ∀ x : L, toEnd R L M x ^ k = 0 := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    hM : LieModule.IsNilpotent R L M
    ⊢ Exists fun k => ∀ (x : L), Eq (HPow.hPow ((LieModule.toEnd R L M) x) k) 0
  -/
  obtain ⟨k, hM⟩ := hM
  /-
    case mk.intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    k : Nat
    hM : Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    ⊢ Exists fun k => ∀ (x : L), Eq (HPow.hPow ((LieModule.toEnd R L M) x) k) 0
  -/
  use k
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    k : Nat
    hM : Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    ⊢ ∀ (x : L), Eq (HPow.hPow ((LieModule.toEnd R L M) x) k) 0
  -/
  intro x; ext m
  /-
    case h.h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    k : Nat
    hM : Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    x : L
    m : M
    ⊢ Eq ((HPow.hPow ((LieModule.toEnd R L M) x) k) m) (0 m)
  -/
  rw [LinearMap.pow_apply, LinearMap.zero_apply, ← @LieSubmodule.mem_bot R L M, ← hM]
  /-
    case h.h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    k : Nat
    hM : Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    x : L
    m : M
    ⊢ Membership.mem (LieModule.lowerCentralSeries R L M k) (Nat.iterate (⇑((LieMo …
  -/
  exact iterate_toEnd_mem_lowerCentralSeries R L M x m k
  /-
    🎉 no goals
  -/


theorem isNilpotent_toEnd_of_isNilpotent [IsNilpotent R L M] (x : L) :
    _root_.IsNilpotent (toEnd R L M x) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    x : L
    ⊢ _root_.IsNilpotent ((LieModule.toEnd R L M) x)
  -/
  change ∃ k, toEnd R L M x ^ k = 0
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    x : L
    ⊢ Exists fun k => Eq (HPow.hPow ((LieModule.toEnd R L M) x) k) 0
  -/
  have := exists_forall_pow_toEnd_eq_zero R L M
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    x : L
    this : Exists fun k => ∀ (x : L), Eq (HPow.hPow ((LieModule.toEnd R L M) x) k) 0
    ⊢ Exists fun k => Eq (HPow.hPow ((LieModule.toEnd R L M) x) k) 0
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem isNilpotent_toEnd_of_isNilpotent₂ [IsNilpotent R L M] (x y : L) :
    _root_.IsNilpotent (toEnd R L M x ∘ₗ toEnd R L M y) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    x y : L
    ⊢ _root_.IsNilpotent (LinearMap.comp ((LieModule.toEnd R L M) x) ((LieModule.t …
  -/
  obtain ⟨k, hM⟩ := exists_lowerCentralSeries_eq_bot_of_isNilpotent R L M
  replace hM : lowerCentralSeries R L M (2 * k) = ⊥ := by
    rw [eq_bot_iff, ← hM]; exact antitone_lowerCentralSeries R L M (by omega)
  /-
    case intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    x y : L
    k : Nat
    hM : Eq (LieModule.lowerCentralSeries R L M (HMul.hMul 2 k)) Bot.bot
    ⊢ _root_.IsNilpotent (LinearMap.comp ((LieModule.toEnd R L M) x) ((LieModule.t …
  -/
  use k
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    x y : L
    k : Nat
    hM : Eq (LieModule.lowerCentralSeries R L M (HMul.hMul 2 k)) Bot.bot
    ⊢ Eq (HPow.hPow (LinearMap.comp ((LieModule.toEnd R L M) x) ((LieModule.toEnd  …
  -/
  ext m
  /-
    case h.h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    x y : L
    k : Nat
    hM : Eq (LieModule.lowerCentralSeries R L M (HMul.hMul 2 k)) Bot.bot
    m : M
    ⊢ Eq ((HPow.hPow (LinearMap.comp ((LieModule.toEnd R L M) x) ((LieModule.toEnd …
  -/
  rw [LinearMap.pow_apply, LinearMap.zero_apply, ← LieSubmodule.mem_bot (R := R) (L := L), ← hM]
  /-
    case h.h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    x y : L
    k : Nat
    hM : Eq (LieModule.lowerCentralSeries R L M (HMul.hMul 2 k)) Bot.bot
    m : M
    ⊢ Membership.mem (LieModule.lowerCentralSeries R L M (HMul.hMul 2 k)) (Nat.ite …
  -/
  exact iterate_toEnd_mem_lowerCentralSeries₂ R L M x y m k
  /-
    🎉 no goals
  -/


@[simp] lemma maxGenEigenSpace_toEnd_eq_top [IsNilpotent R L M] (x : L) :
    ((toEnd R L M x).maxGenEigenspace 0) = ⊤ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    x : L
    ⊢ Eq (((LieModule.toEnd R L M) x).maxGenEigenspace 0) Top.top
  -/
  ext m
  simp only [Module.End.mem_maxGenEigenspace, zero_smul, sub_zero, Submodule.mem_top,
    iff_true]
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    x : L
    m : M
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x) k) m) 0
  -/
  obtain ⟨k, hk⟩ := exists_forall_pow_toEnd_eq_zero R L M
  /-
    case h.intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    x : L
    m : M
    k : Nat
    hk : ∀ (x : L), Eq (HPow.hPow ((LieModule.toEnd R L M) x) k) 0
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x) k) m) 0
  -/
  exact ⟨k, by simp [hk x]⟩
  /-
    🎉 no goals
  -/


/-- If the quotient of a Lie module `M` by a Lie submodule on which the Lie algebra acts trivially
is nilpotent then `M` is nilpotent.

This is essentially the Lie module equivalent of the fact that a central
extension of nilpotent Lie algebras is nilpotent. See `LieAlgebra.nilpotent_of_nilpotent_quotient`
below for the corresponding result for Lie algebras. -/
theorem nilpotentOfNilpotentQuotient {N : LieSubmodule R L M} (h₁ : N ≤ maxTrivSubmodule R L M)
    (h₂ : IsNilpotent R L (M ⧸ N)) : IsNilpotent R L M := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    h₁ : LE.le N (LieModule.maxTrivSubmodule R L M)
    h₂ : LieModule.IsNilpotent R L (HasQuotient.Quotient M N)
    ⊢ LieModule.IsNilpotent R L M
  -/
  obtain ⟨k, hk⟩ := h₂
  /-
    case mk.intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    h₁ : LE.le N (LieModule.maxTrivSubmodule R L M)
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R L (HasQuotient.Quotient M N) k) Bot.bot
    ⊢ LieModule.IsNilpotent R L M
  -/
  use k + 1
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    h₁ : LE.le N (LieModule.maxTrivSubmodule R L M)
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R L (HasQuotient.Quotient M N) k) Bot.bot
    ⊢ Eq (LieModule.lowerCentralSeries R L M (HAdd.hAdd k 1)) Bot.bot
  -/
  simp only [lowerCentralSeries_succ]
  suffices lowerCentralSeries R L M k ≤ N by
    replace this := LieSubmodule.mono_lie_right ⊤ (le_trans this h₁)
    rwa [ideal_oper_maxTrivSubmodule_eq_bot, le_bot_iff] at this
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    h₁ : LE.le N (LieModule.maxTrivSubmodule R L M)
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R L (HasQuotient.Quotient M N) k) Bot.bot
    ⊢ LE.le (LieModule.lowerCentralSeries R L M k) N
  -/
  rw [← LieSubmodule.Quotient.map_mk'_eq_bot_le, ← le_bot_iff, ← hk]
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    h₁ : LE.le N (LieModule.maxTrivSubmodule R L M)
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R L (HasQuotient.Quotient M N) k) Bot.bot
    ⊢ LE.le (LieSubmodule.map (LieSubmodule.Quotient.mk' N) (LieModule.lowerCentra …
  -/
  exact map_lowerCentralSeries_le k (LieSubmodule.Quotient.mk' N)
  /-
    🎉 no goals
  -/


theorem isNilpotent_quotient_iff :
    IsNilpotent R L (M ⧸ N) ↔ ∃ k, lowerCentralSeries R L M k ≤ N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieModule R L M
    ⊢ Iff (LieModule.IsNilpotent R L (HasQuotient.Quotient M N)) (Exists fun k =>  …
  -/
  rw [LieModule.isNilpotent_iff]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieModule R L M
    ⊢ Iff (Exists fun k => Eq (LieModule.lowerCentralSeries R L (HasQuotient.Quoti …
  -/
  refine exists_congr fun k ↦ ?_
  rw [← LieSubmodule.Quotient.map_mk'_eq_bot_le, map_lowerCentralSeries_eq k
    (LieSubmodule.Quotient.surjective_mk' N)]


theorem iInf_lcs_le_of_isNilpotent_quot (h : IsNilpotent R L (M ⧸ N)) :
    ⨅ k, lowerCentralSeries R L M k ≤ N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieModule R L M
    h : LieModule.IsNilpotent R L (HasQuotient.Quotient M N)
    ⊢ LE.le (iInf fun k => LieModule.lowerCentralSeries R L M k) N
  -/
  obtain ⟨k, hk⟩ := (isNilpotent_quotient_iff R L M N).mp h
  /-
    case intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieModule R L M
    h : LieModule.IsNilpotent R L (HasQuotient.Quotient M N)
    k : Nat
    hk : LE.le (LieModule.lowerCentralSeries R L M k) N
    ⊢ LE.le (iInf fun k => LieModule.lowerCentralSeries R L M k) N
  -/
  exact iInf_le_of_le k hk
  /-
    🎉 no goals
  -/


/-- Given a nilpotent Lie module `M` with lower central series `M = C₀ ≥ C₁ ≥ ⋯ ≥ Cₖ = ⊥`, this is
the natural number `k` (the number of inclusions).

For a non-nilpotent module, we use the junk value 0. -/
noncomputable def nilpotencyLength : ℕ :=
  sInf {k | lowerCentralSeries R L M k = ⊥}


@[simp]
theorem nilpotencyLength_eq_zero_iff [IsNilpotent R L M] :
    nilpotencyLength R L M = 0 ↔ Subsingleton M := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    ⊢ Iff (Eq (LieModule.nilpotencyLength R L M) 0) (Subsingleton M)
  -/
  let s := {k | lowerCentralSeries R L M k = ⊥}
  have hs : s.Nonempty := by
    obtain ⟨k, hk⟩ := (by infer_instance : IsNilpotent R L M)
    exact ⟨k, hk⟩
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    s : Set Nat := setOf fun k => Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    hs : s.Nonempty
    ⊢ Iff (Eq (LieModule.nilpotencyLength R L M) 0) (Subsingleton M)
  -/
  change sInf s = 0 ↔ _
  rw [← LieSubmodule.subsingleton_iff R L M, ← subsingleton_iff_bot_eq_top, ←
    lowerCentralSeries_zero, @eq_comm (LieSubmodule R L M)]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    s : Set Nat := setOf fun k => Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    hs : s.Nonempty
    ⊢ Iff (Eq (InfSet.sInf s) 0) (Eq (LieModule.lowerCentralSeries R L M 0) Bot.bot)
  -/
  refine ⟨fun h => h ▸ Nat.sInf_mem hs, fun h => ?_⟩
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    s : Set Nat := setOf fun k => Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    hs : s.Nonempty
    h : Eq (LieModule.lowerCentralSeries R L M 0) Bot.bot
    ⊢ Eq (InfSet.sInf s) 0
  -/
  rw [Nat.sInf_eq_zero]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    s : Set Nat := setOf fun k => Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    hs : s.Nonempty
    h : Eq (LieModule.lowerCentralSeries R L M 0) Bot.bot
    ⊢ Or (Membership.mem s 0) (Eq s EmptyCollection.emptyCollection)
  -/
  exact Or.inl h
  /-
    🎉 no goals
  -/


theorem nilpotencyLength_eq_succ_iff (k : ℕ) :
    nilpotencyLength R L M = k + 1 ↔
      lowerCentralSeries R L M (k + 1) = ⊥ ∧ lowerCentralSeries R L M k ≠ ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    k : Nat
    ⊢ Iff (Eq (LieModule.nilpotencyLength R L M) (HAdd.hAdd k 1)) (And (Eq (LieMod …
  -/
  let s := {k | lowerCentralSeries R L M k = ⊥}
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    k : Nat
    s : Set Nat := setOf fun k => Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    ⊢ Iff (Eq (LieModule.nilpotencyLength R L M) (HAdd.hAdd k 1)) (And (Eq (LieMod …
  -/
  change sInf s = k + 1 ↔ k + 1 ∈ s ∧ k ∉ s
  have hs : ∀ k₁ k₂, k₁ ≤ k₂ → k₁ ∈ s → k₂ ∈ s := by
    rintro k₁ k₂ h₁₂ (h₁ : lowerCentralSeries R L M k₁ = ⊥)
    exact eq_bot_iff.mpr (h₁ ▸ antitone_lowerCentralSeries R L M h₁₂)
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    k : Nat
    s : Set Nat := setOf fun k => Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    hs : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
    ⊢ Iff (Eq (InfSet.sInf s) (HAdd.hAdd k 1)) (And (Membership.mem s (HAdd.hAdd k …
  -/
  exact Nat.sInf_upward_closed_eq_succ_iff hs k
  /-
    🎉 no goals
  -/


@[simp]
theorem nilpotencyLength_eq_one_iff [Nontrivial M] :
    nilpotencyLength R L M = 1 ↔ IsTrivial L M := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : Nontrivial M
    ⊢ Iff (Eq (LieModule.nilpotencyLength R L M) 1) (LieModule.IsTrivial L M)
  -/
  rw [nilpotencyLength_eq_succ_iff, ← trivial_iff_lower_central_eq_bot]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : Nontrivial M
    ⊢ Iff (And (LieModule.IsTrivial L M) (Ne (LieModule.lowerCentralSeries R L M 0 …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem isTrivial_of_nilpotencyLength_le_one [IsNilpotent R L M] (h : nilpotencyLength R L M ≤ 1) :
    IsTrivial L M := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    h : LE.le (LieModule.nilpotencyLength R L M) 1
    ⊢ LieModule.IsTrivial L M
  -/
  nontriviality M
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    h : LE.le (LieModule.nilpotencyLength R L M) 1
    a✝ : Nontrivial M
    ⊢ LieModule.IsTrivial L M
  -/
  cases' Nat.le_one_iff_eq_zero_or_eq_one.mp h with h h
    /-
      case inl
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule.IsNilpotent R L M
      h✝ : LE.le (LieModule.nilpotencyLength R L M) 1
      a✝ : Nontrivial M
      h : Eq (LieModule.nilpotencyLength R L M) 0
      ⊢ LieModule.IsTrivial L M
    -/
  · rw [nilpotencyLength_eq_zero_iff] at h; infer_instance
                                            /-
                                              🎉 no goals
                                            -/
    /-
      case inr
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule.IsNilpotent R L M
      h✝ : LE.le (LieModule.nilpotencyLength R L M) 1
      a✝ : Nontrivial M
      h : Eq (LieModule.nilpotencyLength R L M) 1
      ⊢ LieModule.IsTrivial L M
    -/
  · rwa [nilpotencyLength_eq_one_iff] at h
    /-
      🎉 no goals
    -/


/-- Given a non-trivial nilpotent Lie module `M` with lower central series
`M = C₀ ≥ C₁ ≥ ⋯ ≥ Cₖ = ⊥`, this is the `k-1`th term in the lower central series (the last
non-trivial term).

For a trivial or non-nilpotent module, this is the bottom submodule, `⊥`. -/
noncomputable def lowerCentralSeriesLast : LieSubmodule R L M :=
  match nilpotencyLength R L M with
  | 0 => ⊥
  | k + 1 => lowerCentralSeries R L M k


theorem lowerCentralSeriesLast_le_max_triv [LieModule R L M] :
    lowerCentralSeriesLast R L M ≤ maxTrivSubmodule R L M := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    ⊢ LE.le (LieModule.lowerCentralSeriesLast R L M) (LieModule.maxTrivSubmodule R …
  -/
  rw [lowerCentralSeriesLast]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    ⊢ LE.le (LieModule.lowerCentralSeriesLast.match_1 (fun x => LieSubmodule R L M …
  -/
  cases' h : nilpotencyLength R L M with k
    /-
      case zero
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      h : Eq (LieModule.nilpotencyLength R L M) 0
      ⊢ LE.le (LieModule.lowerCentralSeriesLast.match_1 (fun x => LieSubmodule R L M …
    -/
  · exact bot_le
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      k : Nat
      h : Eq (LieModule.nilpotencyLength R L M) (HAdd.hAdd k 1)
      ⊢ LE.le (LieModule.lowerCentralSeriesLast.match_1 (fun x => LieSubmodule R L M …
    -/
  · rw [le_max_triv_iff_bracket_eq_bot]
    /-
      case succ
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      k : Nat
      h : Eq (LieModule.nilpotencyLength R L M) (HAdd.hAdd k 1)
      ⊢ Eq (Bracket.bracket Top.top (LieModule.lowerCentralSeriesLast.match_1 (fun x …
    -/
    rw [nilpotencyLength_eq_succ_iff, lowerCentralSeries_succ] at h
    /-
      case succ
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      k : Nat
      h : And (Eq (Bracket.bracket Top.top (LieModule.lowerCentralSeries R L M k)) B …
      ⊢ Eq (Bracket.bracket Top.top (LieModule.lowerCentralSeriesLast.match_1 (fun x …
    -/
    exact h.1
    /-
      🎉 no goals
    -/


theorem nontrivial_lowerCentralSeriesLast [Nontrivial M] [IsNilpotent R L M] :
    Nontrivial (lowerCentralSeriesLast R L M) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : Nontrivial M
    inst✝ : LieModule.IsNilpotent R L M
    ⊢ Nontrivial (Subtype fun x => Membership.mem (LieModule.lowerCentralSeriesLas …
  -/
  rw [LieSubmodule.nontrivial_iff_ne_bot, lowerCentralSeriesLast]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : Nontrivial M
    inst✝ : LieModule.IsNilpotent R L M
    ⊢ Ne (LieModule.lowerCentralSeriesLast.match_1 (fun x => LieSubmodule R L M) ( …
  -/
  cases h : nilpotencyLength R L M
    /-
      case zero
      R : Type u
      L : Type v
      M : Type w
      inst✝⁷ : CommRing R
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra R L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      inst✝¹ : Nontrivial M
      inst✝ : LieModule.IsNilpotent R L M
      h : Eq (LieModule.nilpotencyLength R L M) 0
      ⊢ Ne (LieModule.lowerCentralSeriesLast.match_1 (fun x => LieSubmodule R L M) 0 …
    -/
  · rw [nilpotencyLength_eq_zero_iff, ← not_nontrivial_iff_subsingleton] at h
    /-
      case zero
      R : Type u
      L : Type v
      M : Type w
      inst✝⁷ : CommRing R
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra R L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      inst✝¹ : Nontrivial M
      inst✝ : LieModule.IsNilpotent R L M
      h : Not (Nontrivial M)
      ⊢ Ne (LieModule.lowerCentralSeriesLast.match_1 (fun x => LieSubmodule R L M) 0 …
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      L : Type v
      M : Type w
      inst✝⁷ : CommRing R
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra R L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      inst✝¹ : Nontrivial M
      inst✝ : LieModule.IsNilpotent R L M
      n✝ : Nat
      h : Eq (LieModule.nilpotencyLength R L M) (HAdd.hAdd n✝ 1)
      ⊢ Ne (LieModule.lowerCentralSeriesLast.match_1 (fun x => LieSubmodule R L M) ( …
    -/
  · rw [nilpotencyLength_eq_succ_iff] at h
    /-
      case succ
      R : Type u
      L : Type v
      M : Type w
      inst✝⁷ : CommRing R
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra R L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      inst✝¹ : Nontrivial M
      inst✝ : LieModule.IsNilpotent R L M
      n✝ : Nat
      h : And (Eq (LieModule.lowerCentralSeries R L M (HAdd.hAdd n✝ 1)) Bot.bot) (Ne …
      ⊢ Ne (LieModule.lowerCentralSeriesLast.match_1 (fun x => LieSubmodule R L M) ( …
    -/
    exact h.2
    /-
      🎉 no goals
    -/


theorem lowerCentralSeriesLast_le_of_not_isTrivial [IsNilpotent R L M] (h : ¬ IsTrivial L M) :
    lowerCentralSeriesLast R L M ≤ lowerCentralSeries R L M 1 := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    h : Not (LieModule.IsTrivial L M)
    ⊢ LE.le (LieModule.lowerCentralSeriesLast R L M) (LieModule.lowerCentralSeries …
  -/
  rw [lowerCentralSeriesLast]
  replace h : 1 < nilpotencyLength R L M := by
    by_contra contra
    have := isTrivial_of_nilpotencyLength_le_one R L M (not_lt.mp contra)
    contradiction
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule.IsNilpotent R L M
    h : LT.lt 1 (LieModule.nilpotencyLength R L M)
    ⊢ LE.le (LieModule.lowerCentralSeriesLast.match_1 (fun x => LieSubmodule R L M …
  -/
  cases' hk : nilpotencyLength R L M with k <;> rw [hk] at h
    /-
      case zero
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule.IsNilpotent R L M
      h : LT.lt 1 0
      hk : Eq (LieModule.nilpotencyLength R L M) 0
      ⊢ LE.le (LieModule.lowerCentralSeriesLast.match_1 (fun x => LieSubmodule R L M …
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule.IsNilpotent R L M
      k : Nat
      h : LT.lt 1 (HAdd.hAdd k 1)
      hk : Eq (LieModule.nilpotencyLength R L M) (HAdd.hAdd k 1)
      ⊢ LE.le (LieModule.lowerCentralSeriesLast.match_1 (fun x => LieSubmodule R L M …
    -/
  · exact antitone_lowerCentralSeries _ _ _ (Nat.lt_succ.mp h)
    /-
      🎉 no goals
    -/


/-- For a nilpotent Lie module `M` of a Lie algebra `L`, the first term in the lower central series
of `M` contains a non-zero element on which `L` acts trivially unless the entire action is trivial.

Taking `M = L`, this provides a useful characterisation of Abelian-ness for nilpotent Lie
algebras. -/
lemma disjoint_lowerCentralSeries_maxTrivSubmodule_iff [IsNilpotent R L M] :
    Disjoint (lowerCentralSeries R L M 1) (maxTrivSubmodule R L M) ↔ IsTrivial L M := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    ⊢ Iff (Disjoint (LieModule.lowerCentralSeries R L M 1) (LieModule.maxTrivSubmo …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ by simp⟩
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    h : Disjoint (LieModule.lowerCentralSeries R L M 1) (LieModule.maxTrivSubmodul …
    ⊢ LieModule.IsTrivial L M
  -/
  nontriviality M
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    h : Disjoint (LieModule.lowerCentralSeries R L M 1) (LieModule.maxTrivSubmodul …
    a✝ : Nontrivial M
    ⊢ LieModule.IsTrivial L M
  -/
  by_contra contra
  have : lowerCentralSeriesLast R L M ≤ lowerCentralSeries R L M 1 ⊓ maxTrivSubmodule R L M :=
    le_inf_iff.mpr ⟨lowerCentralSeriesLast_le_of_not_isTrivial R L M contra,
      lowerCentralSeriesLast_le_max_triv R L M⟩
  suffices ¬ Nontrivial (lowerCentralSeriesLast R L M) by
    exact this (nontrivial_lowerCentralSeriesLast R L M)
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    h : Disjoint (LieModule.lowerCentralSeries R L M 1) (LieModule.maxTrivSubmodul …
    a✝ : Nontrivial M
    contra : Not (LieModule.IsTrivial L M)
    this : LE.le (LieModule.lowerCentralSeriesLast R L M) (Min.min (LieModule.lowe …
    ⊢ Not (Nontrivial (Subtype fun x => Membership.mem (LieModule.lowerCentralSeri …
  -/
  rw [h.eq_bot, le_bot_iff] at this
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsNilpotent R L M
    h : Disjoint (LieModule.lowerCentralSeries R L M 1) (LieModule.maxTrivSubmodul …
    a✝ : Nontrivial M
    contra : Not (LieModule.IsTrivial L M)
    this : Eq (LieModule.lowerCentralSeriesLast R L M) Bot.bot
    ⊢ Not (Nontrivial (Subtype fun x => Membership.mem (LieModule.lowerCentralSeri …
  -/
  exact this ▸ not_nontrivial _
  /-
    🎉 no goals
  -/


theorem nontrivial_max_triv_of_isNilpotent [Nontrivial M] [IsNilpotent R L M] :
    Nontrivial (maxTrivSubmodule R L M) :=
  Set.nontrivial_mono (lowerCentralSeriesLast_le_max_triv R L M)
    (nontrivial_lowerCentralSeriesLast R L M)


@[simp]
theorem coe_lcs_range_toEnd_eq (k : ℕ) :
    (lowerCentralSeries R (toEnd R L M).range M k : Submodule R M) =
      lowerCentralSeries R L M k := by
  induction k with
  | zero => simp
  | succ k ih =>
    simp only [lowerCentralSeries_succ, LieSubmodule.lieIdeal_oper_eq_linear_span', ←
      (lowerCentralSeries R (toEnd R L M).range M k).mem_toSubmodule, ih]
    congr
    ext m
    constructor
    · rintro ⟨⟨-, ⟨y, rfl⟩⟩, -, n, hn, rfl⟩
      exact ⟨y, LieSubmodule.mem_top _, n, hn, rfl⟩
    · rintro ⟨x, -, n, hn, rfl⟩
      exact
        ⟨⟨toEnd R L M x, LieHom.mem_range_self _ x⟩, LieSubmodule.mem_top _, n, hn, rfl⟩


@[simp]
theorem isNilpotent_range_toEnd_iff :
    IsNilpotent R (toEnd R L M).range M ↔ IsNilpotent R L M := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    ⊢ Iff (LieModule.IsNilpotent R (Subtype fun x => Membership.mem (LieModule.toE …
  -/
  constructor <;> rintro ⟨k, hk⟩ <;> use k <;>
      /-
        case h
        R : Type u
        L : Type v
        M : Type w
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        k : Nat
        hk : Eq (LieModule.lowerCentralSeries R (Subtype fun x => Membership.mem (LieM …
        ⊢ Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
      -/
      rw [← LieSubmodule.toSubmodule_inj] at hk ⊢ <;>
    /-
      case h
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      k : Nat
      hk : Eq ↑(LieModule.lowerCentralSeries R (Subtype fun x => Membership.mem (Lie …
      ⊢ Eq ↑(LieModule.lowerCentralSeries R L M k) ↑Bot.bot
    -/
    /-
      🎉 no goals
    -/
    simpa using hk
    /-
      🎉 no goals
    -/


/-- The upper (aka ascending) central series.

See also `LieSubmodule.lcs`. -/
def ucs (k : ℕ) : LieSubmodule R L M → LieSubmodule R L M :=
  normalizer^[k]


@[simp]
theorem ucs_zero : N.ucs 0 = N :=
  rfl


@[simp]
theorem ucs_succ (k : ℕ) : N.ucs (k + 1) = (N.ucs k).normalizer :=
  Function.iterate_succ_apply' normalizer k N


theorem ucs_add (k l : ℕ) : N.ucs (k + l) = (N.ucs l).ucs k :=
  Function.iterate_add_apply normalizer k l N


@[gcongr, mono]
theorem ucs_mono (k : ℕ) (h : N₁ ≤ N₂) : N₁.ucs k ≤ N₂.ucs k := by
  induction k with
  | zero => simpa
  | succ k ih =>
    simp only [ucs_succ]
    gcongr


theorem ucs_eq_self_of_normalizer_eq_self (h : N₁.normalizer = N₁) (k : ℕ) : N₁.ucs k = N₁ := by
  induction k with
  | zero => simp
  | succ k ih => rwa [ucs_succ, ih]


/-- If a Lie module `M` contains a self-normalizing Lie submodule `N`, then all terms of the upper
central series of `M` are contained in `N`.

An important instance of this situation arises from a Cartan subalgebra `H ⊆ L` with the roles of
`L`, `M`, `N` played by `H`, `L`, `H`, respectively. -/
theorem ucs_le_of_normalizer_eq_self (h : N₁.normalizer = N₁) (k : ℕ) :
    (⊥ : LieSubmodule R L M).ucs k ≤ N₁ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N₁ : LieSubmodule R L M
    inst✝ : LieModule R L M
    h : Eq N₁.normalizer N₁
    k : Nat
    ⊢ LE.le (LieSubmodule.ucs k Bot.bot) N₁
  -/
  rw [← ucs_eq_self_of_normalizer_eq_self h k]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N₁ : LieSubmodule R L M
    inst✝ : LieModule R L M
    h : Eq N₁.normalizer N₁
    k : Nat
    ⊢ LE.le (LieSubmodule.ucs k Bot.bot) (LieSubmodule.ucs k N₁)
  -/
  gcongr
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N₁ : LieSubmodule R L M
    inst✝ : LieModule R L M
    h : Eq N₁.normalizer N₁
    k : Nat
    ⊢ LE.le Bot.bot N₁
  -/
  simp
  /-
    🎉 no goals
  -/


theorem lcs_add_le_iff (l k : ℕ) : N₁.lcs (l + k) ≤ N₂ ↔ N₁.lcs l ≤ N₂.ucs k := by
  induction k generalizing l with
  | zero => simp
  | succ k ih =>
    rw [(by abel : l + (k + 1) = l + 1 + k), ih, ucs_succ, lcs_succ, top_lie_le_iff_le_normalizer]


theorem lcs_le_iff (k : ℕ) : N₁.lcs k ≤ N₂ ↔ N₁ ≤ N₂.ucs k := by
  -- Porting note: `convert` needed type annotations
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N₁ N₂ : LieSubmodule R L M
    inst✝ : LieModule R L M
    k : Nat
    ⊢ Iff (LE.le (LieSubmodule.lcs k N₁) N₂) (LE.le N₁ (LieSubmodule.ucs k N₂))
  -/
  convert lcs_add_le_iff (R := R) (L := L) (M := M) 0 k
  /-
    case h.e'_1.h.e'_3.h.e'_10
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N₁ N₂ : LieSubmodule R L M
    inst✝ : LieModule R L M
    k : Nat
    ⊢ Eq k (HAdd.hAdd 0 k)
  -/
  rw [zero_add]
  /-
    🎉 no goals
  -/


theorem gc_lcs_ucs (k : ℕ) :
    GaloisConnection (fun N : LieSubmodule R L M => N.lcs k) fun N : LieSubmodule R L M =>
      N.ucs k :=
  fun _ _ => lcs_le_iff k


theorem ucs_eq_top_iff (k : ℕ) : N.ucs k = ⊤ ↔ LieModule.lowerCentralSeries R L M k ≤ N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieModule R L M
    k : Nat
    ⊢ Iff (Eq (LieSubmodule.ucs k N) Top.top) (LE.le (LieModule.lowerCentralSeries …
  -/
  rw [eq_top_iff, ← lcs_le_iff]; rfl
                                 /-
                                   🎉 no goals
                                 -/


theorem _root_.LieModule.isNilpotent_iff_exists_ucs_eq_top :
    LieModule.IsNilpotent R L M ↔ ∃ k, (⊥ : LieSubmodule R L M).ucs k = ⊤ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    ⊢ Iff (LieModule.IsNilpotent R L M) (Exists fun k => Eq (LieSubmodule.ucs k Bo …
  -/
  rw [LieModule.isNilpotent_iff]; exact exists_congr fun k => by simp [ucs_eq_top_iff]
                                  /-
                                    🎉 no goals
                                  -/


theorem ucs_comap_incl (k : ℕ) :
    ((⊥ : LieSubmodule R L M).ucs k).comap N.incl = (⊥ : LieSubmodule R L N).ucs k := by
  induction k with
  | zero => exact N.ker_incl
  | succ k ih => simp [← ih]


theorem isNilpotent_iff_exists_self_le_ucs :
    LieModule.IsNilpotent R L N ↔ ∃ k, N ≤ (⊥ : LieSubmodule R L M).ucs k := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieModule R L M
    ⊢ Iff (LieModule.IsNilpotent R L (Subtype fun x => Membership.mem N x)) (Exist …
  -/
  simp_rw [LieModule.isNilpotent_iff_exists_ucs_eq_top, ← ucs_comap_incl, comap_incl_eq_top]
  /-
    🎉 no goals
  -/


theorem ucs_bot_one : (⊥ : LieSubmodule R L M).ucs 1 = LieModule.maxTrivSubmodule R L M := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    ⊢ Eq (LieSubmodule.ucs 1 Bot.bot) (LieModule.maxTrivSubmodule R L M)
  -/
  simp [LieSubmodule.normalizer_bot_eq_maxTrivSubmodule]
  /-
    🎉 no goals
  -/


include hf hg hfg in
theorem Function.Surjective.lieModule_lcs_map_eq (k : ℕ) :
    (lowerCentralSeries R L M k : Submodule R M).map g = lowerCentralSeries R L₂ M₂ k := by
  induction k with
  | zero => simpa [LinearMap.range_eq_top]
  | succ k ih =>
    suffices
      g '' {m | ∃ (x : L) (n : _), n ∈ lowerCentralSeries R L M k ∧ ⁅x, n⁆ = m} =
        {m | ∃ (x : L₂) (n : _), n ∈ lowerCentralSeries R L M k ∧ ⁅x, g n⁆ = m} by
      simp only [← LieSubmodule.mem_toSubmodule] at this
      -- Porting note: was
      -- simp [← LieSubmodule.mem_toSubmodule, ← ih, LieSubmodule.lieIdeal_oper_eq_linear_span',
      --   Submodule.map_span, -Submodule.span_image, this,
      --   -LieSubmodule.mem_toSubmodule]
      simp_rw [lowerCentralSeries_succ, LieSubmodule.lieIdeal_oper_eq_linear_span',
        Submodule.map_span, LieSubmodule.mem_top, true_and, ← LieSubmodule.mem_toSubmodule, this,
        ← ih, Submodule.mem_map, exists_exists_and_eq_and]
    ext m₂
    constructor
    · rintro ⟨m, ⟨x, n, hn, rfl⟩, rfl⟩
      exact ⟨f x, n, hn, hfg x n⟩
    · rintro ⟨x, n, hn, rfl⟩
      obtain ⟨y, rfl⟩ := hf x
      exact ⟨⁅y, n⁆, ⟨y, n, hn, rfl⟩, (hfg y n).symm⟩


include hf hg hfg in
theorem Function.Surjective.lieModuleIsNilpotent [IsNilpotent R L M] : IsNilpotent R L₂ M₂ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    L₂ : Type u_1
    M₂ : Type u_2
    inst✝⁶ : LieRing L₂
    inst✝⁵ : LieAlgebra R L₂
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : Module R M₂
    inst✝² : LieRingModule L₂ M₂
    inst✝¹ : LieModule R L₂ M₂
    f : LieHom R L L₂
    g : LinearMap (RingHom.id R) M M₂
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    hfg : ∀ (x : L) (m : M), Eq (Bracket.bracket (f x) (g m)) (g (Bracket.bracket  …
    inst✝ : LieModule.IsNilpotent R L M
    ⊢ LieModule.IsNilpotent R L₂ M₂
  -/
  obtain ⟨k, hk⟩ := id (by infer_instance : IsNilpotent R L M)
  /-
    case mk.intro
    R : Type u
    L : Type v
    M : Type w
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    L₂ : Type u_1
    M₂ : Type u_2
    inst✝⁶ : LieRing L₂
    inst✝⁵ : LieAlgebra R L₂
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : Module R M₂
    inst✝² : LieRingModule L₂ M₂
    inst✝¹ : LieModule R L₂ M₂
    f : LieHom R L L₂
    g : LinearMap (RingHom.id R) M M₂
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    hfg : ∀ (x : L) (m : M), Eq (Bracket.bracket (f x) (g m)) (g (Bracket.bracket  …
    inst✝ : LieModule.IsNilpotent R L M
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    ⊢ LieModule.IsNilpotent R L₂ M₂
  -/
  use k
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    L₂ : Type u_1
    M₂ : Type u_2
    inst✝⁶ : LieRing L₂
    inst✝⁵ : LieAlgebra R L₂
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : Module R M₂
    inst✝² : LieRingModule L₂ M₂
    inst✝¹ : LieModule R L₂ M₂
    f : LieHom R L L₂
    g : LinearMap (RingHom.id R) M M₂
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    hfg : ∀ (x : L) (m : M), Eq (Bracket.bracket (f x) (g m)) (g (Bracket.bracket  …
    inst✝ : LieModule.IsNilpotent R L M
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    ⊢ Eq (LieModule.lowerCentralSeries R L₂ M₂ k) Bot.bot
  -/
  rw [← LieSubmodule.toSubmodule_inj] at hk ⊢
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    L₂ : Type u_1
    M₂ : Type u_2
    inst✝⁶ : LieRing L₂
    inst✝⁵ : LieAlgebra R L₂
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : Module R M₂
    inst✝² : LieRingModule L₂ M₂
    inst✝¹ : LieModule R L₂ M₂
    f : LieHom R L L₂
    g : LinearMap (RingHom.id R) M M₂
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    hfg : ∀ (x : L) (m : M), Eq (Bracket.bracket (f x) (g m)) (g (Bracket.bracket  …
    inst✝ : LieModule.IsNilpotent R L M
    k : Nat
    hk : Eq ↑(LieModule.lowerCentralSeries R L M k) ↑Bot.bot
    ⊢ Eq ↑(LieModule.lowerCentralSeries R L₂ M₂ k) ↑Bot.bot
  -/
  simp [← hf.lieModule_lcs_map_eq hg hfg k, hk]
  /-
    🎉 no goals
  -/


theorem Equiv.lieModule_isNilpotent_iff (f : L ≃ₗ⁅R⁆ L₂) (g : M ≃ₗ[R] M₂)
    (hfg : ∀ x m, ⁅f x, g m⁆ = g ⁅x, m⁆) : IsNilpotent R L M ↔ IsNilpotent R L₂ M₂ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝¹² : CommRing R
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : LieAlgebra R L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    inst✝⁷ : LieRingModule L M
    inst✝⁶ : LieModule R L M
    L₂ : Type u_1
    M₂ : Type u_2
    inst✝⁵ : LieRing L₂
    inst✝⁴ : LieAlgebra R L₂
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L₂ M₂
    inst✝ : LieModule R L₂ M₂
    f : LieEquiv R L L₂
    g : LinearEquiv (RingHom.id R) M M₂
    hfg : ∀ (x : L) (m : M), Eq (Bracket.bracket (f x) (g m)) (g (Bracket.bracket  …
    ⊢ Iff (LieModule.IsNilpotent R L M) (LieModule.IsNilpotent R L₂ M₂)
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u
      L : Type v
      M : Type w
      inst✝¹² : CommRing R
      inst✝¹¹ : LieRing L
      inst✝¹⁰ : LieAlgebra R L
      inst✝⁹ : AddCommGroup M
      inst✝⁸ : Module R M
      inst✝⁷ : LieRingModule L M
      inst✝⁶ : LieModule R L M
      L₂ : Type u_1
      M₂ : Type u_2
      inst✝⁵ : LieRing L₂
      inst✝⁴ : LieAlgebra R L₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L₂ M₂
      inst✝ : LieModule R L₂ M₂
      f : LieEquiv R L L₂
      g : LinearEquiv (RingHom.id R) M M₂
      hfg : ∀ (x : L) (m : M), Eq (Bracket.bracket (f x) (g m)) (g (Bracket.bracket  …
      h : LieModule.IsNilpotent R L M
      ⊢ LieModule.IsNilpotent R L₂ M₂
    -/
  · have hg : Surjective (g : M →ₗ[R] M₂) := g.surjective
    /-
      case mp
      R : Type u
      L : Type v
      M : Type w
      inst✝¹² : CommRing R
      inst✝¹¹ : LieRing L
      inst✝¹⁰ : LieAlgebra R L
      inst✝⁹ : AddCommGroup M
      inst✝⁸ : Module R M
      inst✝⁷ : LieRingModule L M
      inst✝⁶ : LieModule R L M
      L₂ : Type u_1
      M₂ : Type u_2
      inst✝⁵ : LieRing L₂
      inst✝⁴ : LieAlgebra R L₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L₂ M₂
      inst✝ : LieModule R L₂ M₂
      f : LieEquiv R L L₂
      g : LinearEquiv (RingHom.id R) M M₂
      hfg : ∀ (x : L) (m : M), Eq (Bracket.bracket (f x) (g m)) (g (Bracket.bracket  …
      h : LieModule.IsNilpotent R L M
      hg : Function.Surjective ⇑↑g
      ⊢ LieModule.IsNilpotent R L₂ M₂
    -/
    exact f.surjective.lieModuleIsNilpotent hg hfg
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      L : Type v
      M : Type w
      inst✝¹² : CommRing R
      inst✝¹¹ : LieRing L
      inst✝¹⁰ : LieAlgebra R L
      inst✝⁹ : AddCommGroup M
      inst✝⁸ : Module R M
      inst✝⁷ : LieRingModule L M
      inst✝⁶ : LieModule R L M
      L₂ : Type u_1
      M₂ : Type u_2
      inst✝⁵ : LieRing L₂
      inst✝⁴ : LieAlgebra R L₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L₂ M₂
      inst✝ : LieModule R L₂ M₂
      f : LieEquiv R L L₂
      g : LinearEquiv (RingHom.id R) M M₂
      hfg : ∀ (x : L) (m : M), Eq (Bracket.bracket (f x) (g m)) (g (Bracket.bracket  …
      h : LieModule.IsNilpotent R L₂ M₂
      ⊢ LieModule.IsNilpotent R L M
    -/
  · have hg : Surjective (g.symm : M₂ →ₗ[R] M) := g.symm.surjective
    /-
      case mpr
      R : Type u
      L : Type v
      M : Type w
      inst✝¹² : CommRing R
      inst✝¹¹ : LieRing L
      inst✝¹⁰ : LieAlgebra R L
      inst✝⁹ : AddCommGroup M
      inst✝⁸ : Module R M
      inst✝⁷ : LieRingModule L M
      inst✝⁶ : LieModule R L M
      L₂ : Type u_1
      M₂ : Type u_2
      inst✝⁵ : LieRing L₂
      inst✝⁴ : LieAlgebra R L₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L₂ M₂
      inst✝ : LieModule R L₂ M₂
      f : LieEquiv R L L₂
      g : LinearEquiv (RingHom.id R) M M₂
      hfg : ∀ (x : L) (m : M), Eq (Bracket.bracket (f x) (g m)) (g (Bracket.bracket  …
      h : LieModule.IsNilpotent R L₂ M₂
      hg : Function.Surjective ⇑↑g.symm
      ⊢ LieModule.IsNilpotent R L M
    -/
    refine f.symm.surjective.lieModuleIsNilpotent hg fun x m => ?_
    rw [LinearEquiv.coe_coe, LieEquiv.coe_toLieHom, ← g.symm_apply_apply ⁅f.symm x, g.symm m⁆, ←
      hfg, f.apply_symm_apply, g.apply_symm_apply]


@[simp]
theorem LieModule.isNilpotent_of_top_iff :
    IsNilpotent R (⊤ : LieSubalgebra R L) M ↔ IsNilpotent R L M :=
  Equiv.lieModule_isNilpotent_iff LieSubalgebra.topEquiv (1 : M ≃ₗ[R] M) fun _ _ => rfl


@[simp] lemma LieModule.isNilpotent_of_top_iff' :
    IsNilpotent R L {x // x ∈ (⊤ : LieSubmodule R L M)} ↔ IsNilpotent R L M :=
  Equiv.lieModule_isNilpotent_iff 1 (LinearEquiv.ofTop ⊤ rfl) fun _ _ ↦ rfl


instance (priority := 100) LieAlgebra.isSolvable_of_isNilpotent (R : Type u) (L : Type v)
    [CommRing R] [LieRing L] [LieAlgebra R L] [hL : LieModule.IsNilpotent R L L] :
    LieAlgebra.IsSolvable R L := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    hL : LieModule.IsNilpotent R L L
    ⊢ LieAlgebra.IsSolvable R L
  -/
  obtain ⟨k, h⟩ : ∃ k, LieModule.lowerCentralSeries R L L k = ⊥ := hL.nilpotent
  /-
    case intro
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    hL : LieModule.IsNilpotent R L L
    k : Nat
    h : Eq (LieModule.lowerCentralSeries R L L k) Bot.bot
    ⊢ LieAlgebra.IsSolvable R L
  -/
  use k; rw [← le_bot_iff] at h ⊢
  /-
    case h
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    hL : LieModule.IsNilpotent R L L
    k : Nat
    h : LE.le (LieModule.lowerCentralSeries R L L k) Bot.bot
    ⊢ LE.le (LieAlgebra.derivedSeries R L k) Bot.bot
  -/
  exact le_trans (LieModule.derivedSeries_le_lowerCentralSeries R L k) h
  /-
    🎉 no goals
  -/


/-- We say a Lie algebra is nilpotent when it is nilpotent as a Lie module over itself via the
adjoint representation. -/
abbrev LieAlgebra.IsNilpotent (R : Type u) (L : Type v) [CommRing R] [LieRing L] [LieAlgebra R L] :
    Prop :=
  LieModule.IsNilpotent R L L


theorem LieAlgebra.nilpotent_ad_of_nilpotent_algebra [IsNilpotent R L] :
    ∃ k : ℕ, ∀ x : L, ad R L x ^ k = 0 :=
  LieModule.exists_forall_pow_toEnd_eq_zero R L L

-- TODO Generalise the below to Lie modules if / when we define morphisms, equivs of Lie modules
-- covering a Lie algebra morphism of (possibly different) Lie algebras.

/-- Given an ideal `I` of a Lie algebra `L`, the lower central series of `L ⧸ I` is the same
whether we regard `L ⧸ I` as an `L` module or an `L ⧸ I` module.

TODO: This result obviously generalises but the generalisation requires the missing definition of
morphisms between Lie modules over different Lie algebras. -/
-- Porting note: added `LieSubmodule.toSubmodule` in the statement
theorem coe_lowerCentralSeries_ideal_quot_eq {I : LieIdeal R L} (k : ℕ) :
    LieSubmodule.toSubmodule (lowerCentralSeries R L (L ⧸ I) k) =
      LieSubmodule.toSubmodule (lowerCentralSeries R (L ⧸ I) (L ⧸ I) k) := by
  induction k with
  | zero =>
    simp only [LieModule.lowerCentralSeries_zero, LieSubmodule.top_toSubmodule,
      LieIdeal.top_toLieSubalgebra, LieSubalgebra.top_toSubmodule]
  | succ k ih =>
    simp only [LieModule.lowerCentralSeries_succ, LieSubmodule.lieIdeal_oper_eq_linear_span]
    congr
    ext x
    constructor
    · rintro ⟨⟨y, -⟩, ⟨z, hz⟩, rfl : ⁅y, z⁆ = x⟩
      rw [← LieSubmodule.mem_toSubmodule, ih, LieSubmodule.mem_toSubmodule] at hz
      exact ⟨⟨LieSubmodule.Quotient.mk y, LieSubmodule.mem_top _⟩, ⟨z, hz⟩, rfl⟩
    · rintro ⟨⟨⟨y⟩, -⟩, ⟨z, hz⟩, rfl : ⁅y, z⁆ = x⟩
      rw [← LieSubmodule.mem_toSubmodule, ← ih, LieSubmodule.mem_toSubmodule] at hz
      exact ⟨⟨y, LieSubmodule.mem_top _⟩, ⟨z, hz⟩, rfl⟩


/-- Note that the below inequality can be strict. For example the ideal of strictly-upper-triangular
2x2 matrices inside the Lie algebra of upper-triangular 2x2 matrices with `k = 1`. -/
-- Porting note: added `LieSubmodule.toSubmodule` in the statement
theorem LieModule.coe_lowerCentralSeries_ideal_le {I : LieIdeal R L} (k : ℕ) :
    LieSubmodule.toSubmodule (lowerCentralSeries R I I k) ≤ lowerCentralSeries R L I k := by
  induction k with
  | zero => simp
  | succ k ih =>
    simp only [LieModule.lowerCentralSeries_succ, LieSubmodule.lieIdeal_oper_eq_linear_span]
    apply Submodule.span_mono
    rintro x ⟨⟨y, -⟩, ⟨z, hz⟩, rfl : ⁅y, z⁆ = x⟩
    exact ⟨⟨y.val, LieSubmodule.mem_top _⟩, ⟨z, ih hz⟩, rfl⟩


/-- A central extension of nilpotent Lie algebras is nilpotent. -/
theorem LieAlgebra.nilpotent_of_nilpotent_quotient {I : LieIdeal R L} (h₁ : I ≤ center R L)
    (h₂ : IsNilpotent R (L ⧸ I)) : IsNilpotent R L := by
  suffices LieModule.IsNilpotent R L (L ⧸ I) by
    exact LieModule.nilpotentOfNilpotentQuotient R L L h₁ this
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h₁ : LE.le I (LieAlgebra.center R L)
    h₂ : LieAlgebra.IsNilpotent R (HasQuotient.Quotient L I)
    ⊢ LieModule.IsNilpotent R L (HasQuotient.Quotient L I)
  -/
  obtain ⟨k, hk⟩ := h₂
  /-
    case mk.intro
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h₁ : LE.le I (LieAlgebra.center R L)
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R (HasQuotient.Quotient L I) (HasQuotien …
    ⊢ LieModule.IsNilpotent R L (HasQuotient.Quotient L I)
  -/
  use k
  /-
    case h
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h₁ : LE.le I (LieAlgebra.center R L)
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R (HasQuotient.Quotient L I) (HasQuotien …
    ⊢ Eq (LieModule.lowerCentralSeries R L (HasQuotient.Quotient L I) k) Bot.bot
  -/
  simp [← LieSubmodule.toSubmodule_inj, coe_lowerCentralSeries_ideal_quot_eq, hk]
  /-
    🎉 no goals
  -/


theorem LieAlgebra.non_trivial_center_of_isNilpotent [Nontrivial L] [IsNilpotent R L] :
    Nontrivial <| center R L :=
  LieModule.nontrivial_max_triv_of_isNilpotent R L L


theorem LieIdeal.map_lowerCentralSeries_le (k : ℕ) {f : L →ₗ⁅R⁆ L'} :
    LieIdeal.map f (lowerCentralSeries R L L k) ≤ lowerCentralSeries R L' L' k := by
  induction k with
  | zero => simp only [LieModule.lowerCentralSeries_zero, le_top]
  | succ k ih =>
    simp only [LieModule.lowerCentralSeries_succ]
    exact le_trans (LieIdeal.map_bracket_le f) (LieSubmodule.mono_lie le_top ih)


theorem LieIdeal.lowerCentralSeries_map_eq (k : ℕ) {f : L →ₗ⁅R⁆ L'} (h : Function.Surjective f) :
    LieIdeal.map f (lowerCentralSeries R L L k) = lowerCentralSeries R L' L' k := by
  have h' : (⊤ : LieIdeal R L).map f = ⊤ := by
    rw [← f.idealRange_eq_map]
    exact f.idealRange_eq_top_of_surjective h
  induction k with
  | zero => simp only [LieModule.lowerCentralSeries_zero]; exact h'
  | succ k ih => simp only [LieModule.lowerCentralSeries_succ, LieIdeal.map_bracket_eq f h, ih, h']


theorem Function.Injective.lieAlgebra_isNilpotent [h₁ : IsNilpotent R L'] {f : L →ₗ⁅R⁆ L'}
    (h₂ : Function.Injective f) : IsNilpotent R L :=
  { nilpotent := by
      /-
        R : Type u
        L : Type v
        L' : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        h₁ : LieAlgebra.IsNilpotent R L'
        f : LieHom R L L'
        h₂ : Function.Injective ⇑f
        ⊢ Exists fun k => Eq (LieModule.lowerCentralSeries R L L k) Bot.bot
      -/
      obtain ⟨k, hk⟩ := id h₁
      /-
        case mk.intro
        R : Type u
        L : Type v
        L' : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        h₁ : LieAlgebra.IsNilpotent R L'
        f : LieHom R L L'
        h₂ : Function.Injective ⇑f
        k : Nat
        hk : Eq (LieModule.lowerCentralSeries R L' L' k) Bot.bot
        ⊢ Exists fun k => Eq (LieModule.lowerCentralSeries R L L k) Bot.bot
      -/
      use k
      /-
        case h
        R : Type u
        L : Type v
        L' : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        h₁ : LieAlgebra.IsNilpotent R L'
        f : LieHom R L L'
        h₂ : Function.Injective ⇑f
        k : Nat
        hk : Eq (LieModule.lowerCentralSeries R L' L' k) Bot.bot
        ⊢ Eq (LieModule.lowerCentralSeries R L L k) Bot.bot
      -/
      apply LieIdeal.bot_of_map_eq_bot h₂; rw [eq_bot_iff, ← hk]
      /-
        case h
        R : Type u
        L : Type v
        L' : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        h₁ : LieAlgebra.IsNilpotent R L'
        f : LieHom R L L'
        h₂ : Function.Injective ⇑f
        k : Nat
        hk : Eq (LieModule.lowerCentralSeries R L' L' k) Bot.bot
        ⊢ LE.le (LieIdeal.map f (LieModule.lowerCentralSeries R L L k)) (LieModule.low …
      -/
      apply LieIdeal.map_lowerCentralSeries_le }
      /-
        🎉 no goals
      -/


theorem Function.Surjective.lieAlgebra_isNilpotent [h₁ : IsNilpotent R L] {f : L →ₗ⁅R⁆ L'}
    (h₂ : Function.Surjective f) : IsNilpotent R L' :=
  { nilpotent := by
      /-
        R : Type u
        L : Type v
        L' : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        h₁ : LieAlgebra.IsNilpotent R L
        f : LieHom R L L'
        h₂ : Function.Surjective ⇑f
        ⊢ Exists fun k => Eq (LieModule.lowerCentralSeries R L' L' k) Bot.bot
      -/
      obtain ⟨k, hk⟩ := id h₁
      /-
        case mk.intro
        R : Type u
        L : Type v
        L' : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        h₁ : LieAlgebra.IsNilpotent R L
        f : LieHom R L L'
        h₂ : Function.Surjective ⇑f
        k : Nat
        hk : Eq (LieModule.lowerCentralSeries R L L k) Bot.bot
        ⊢ Exists fun k => Eq (LieModule.lowerCentralSeries R L' L' k) Bot.bot
      -/
      use k
      /-
        case h
        R : Type u
        L : Type v
        L' : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        h₁ : LieAlgebra.IsNilpotent R L
        f : LieHom R L L'
        h₂ : Function.Surjective ⇑f
        k : Nat
        hk : Eq (LieModule.lowerCentralSeries R L L k) Bot.bot
        ⊢ Eq (LieModule.lowerCentralSeries R L' L' k) Bot.bot
      -/
      rw [← LieIdeal.lowerCentralSeries_map_eq k h₂, hk]
      /-
        case h
        R : Type u
        L : Type v
        L' : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        h₁ : LieAlgebra.IsNilpotent R L
        f : LieHom R L L'
        h₂ : Function.Surjective ⇑f
        k : Nat
        hk : Eq (LieModule.lowerCentralSeries R L L k) Bot.bot
        ⊢ Eq (LieIdeal.map f Bot.bot) Bot.bot
      -/
      simp only [LieIdeal.map_eq_bot_iff, bot_le] }
      /-
        🎉 no goals
      -/


theorem LieEquiv.nilpotent_iff_equiv_nilpotent (e : L ≃ₗ⁅R⁆ L') :
    IsNilpotent R L ↔ IsNilpotent R L' := by
  /-
    R : Type u
    L : Type v
    L' : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    e : LieEquiv R L L'
    ⊢ Iff (LieAlgebra.IsNilpotent R L) (LieAlgebra.IsNilpotent R L')
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u
      L : Type v
      L' : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      e : LieEquiv R L L'
      h : LieAlgebra.IsNilpotent R L
      ⊢ LieAlgebra.IsNilpotent R L'
    -/
  · exact e.symm.injective.lieAlgebra_isNilpotent
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      L : Type v
      L' : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      e : LieEquiv R L L'
      h : LieAlgebra.IsNilpotent R L'
      ⊢ LieAlgebra.IsNilpotent R L
    -/
  · exact e.injective.lieAlgebra_isNilpotent
    /-
      🎉 no goals
    -/


theorem LieHom.isNilpotent_range [IsNilpotent R L] (f : L →ₗ⁅R⁆ L') : IsNilpotent R f.range :=
  f.surjective_rangeRestrict.lieAlgebra_isNilpotent


/-- Note that this result is not quite a special case of
`LieModule.isNilpotent_range_toEnd_iff` which concerns nilpotency of the
`(ad R L).range`-module `L`, whereas this result concerns nilpotency of the `(ad R L).range`-module
`(ad R L).range`. -/
@[simp]
theorem LieAlgebra.isNilpotent_range_ad_iff : IsNilpotent R (ad R L).range ↔ IsNilpotent R L := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    ⊢ Iff (LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem (LieAlgebra.a …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      h : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem (LieAlgebra.ad R …
      ⊢ LieAlgebra.IsNilpotent R L
    -/
  · have : (ad R L).ker = center R L := by simp
    exact
      LieAlgebra.nilpotent_of_nilpotent_quotient (le_of_eq this)
        ((ad R L).quotKerEquivRange.nilpotent_iff_equiv_nilpotent.mpr h)
    /-
      case refine_2
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      ⊢ LieAlgebra.IsNilpotent R L → LieAlgebra.IsNilpotent R (Subtype fun x => Memb …
    -/
  · intro h
    /-
      case refine_2
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      h : LieAlgebra.IsNilpotent R L
      ⊢ LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem (LieAlgebra.ad R L …
    -/
    exact (ad R L).isNilpotent_range
    /-
      🎉 no goals
    -/


instance [h : LieAlgebra.IsNilpotent R L] : LieAlgebra.IsNilpotent R (⊤ : LieSubalgebra R L) :=
  LieSubalgebra.topEquiv.nilpotent_iff_equiv_nilpotent.mpr h


/-- Given a Lie module `M` over a Lie algebra `L` together with an ideal `I` of `L`, this is the
lower central series of `M` as an `I`-module. The advantage of using this definition instead of
`LieModule.lowerCentralSeries R I M` is that its terms are Lie submodules of `M` as an
`L`-module, rather than just as an `I`-module.

See also `LieIdeal.coe_lcs_eq`. -/
def lcs : LieSubmodule R L M :=
  (fun N => ⁅I, N⁆)^[k] ⊤


@[simp]
theorem lcs_zero : I.lcs M 0 = ⊤ :=
  rfl


@[simp]
theorem lcs_succ : I.lcs M (k + 1) = ⁅I, I.lcs M k⁆ :=
  Function.iterate_succ_apply' (fun N => ⁅I, N⁆) k ⊤


theorem lcs_top : (⊤ : LieIdeal R L).lcs M k = lowerCentralSeries R L M k :=
  rfl

-- Porting note: added `LieSubmodule.toSubmodule` in the statement

theorem coe_lcs_eq [LieModule R L M] :
    LieSubmodule.toSubmodule (I.lcs M k) = lowerCentralSeries R I M k := by
  induction k with
  | zero => simp
  | succ k ih =>
    simp_rw [lowerCentralSeries_succ, lcs_succ, LieSubmodule.lieIdeal_oper_eq_linear_span', ←
      (I.lcs M k).mem_toSubmodule, ih, LieSubmodule.mem_toSubmodule, LieSubmodule.mem_top,
      true_and, (I : LieSubalgebra R L).coe_bracket_of_module]
    congr
    ext m
    constructor
    · rintro ⟨x, hx, m, hm, rfl⟩
      exact ⟨⟨x, hx⟩, m, hm, rfl⟩
    · rintro ⟨⟨x, hx⟩, m, hm, rfl⟩
      exact ⟨x, hx, m, hm, rfl⟩


theorem _root_.LieAlgebra.ad_nilpotent_of_nilpotent {a : A} (h : IsNilpotent a) :
    IsNilpotent (LieAlgebra.ad R A a) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    h : IsNilpotent a
    ⊢ IsNilpotent ((LieAlgebra.ad R A) a)
  -/
  rw [LieAlgebra.ad_eq_lmul_left_sub_lmul_right]
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    h : IsNilpotent a
    ⊢ IsNilpotent (HSub.hSub (LinearMap.mulLeft R) (LinearMap.mulRight R) a)
  -/
  have hl : IsNilpotent (LinearMap.mulLeft R a) := by rwa [LinearMap.isNilpotent_mulLeft_iff]
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    h : IsNilpotent a
    hl : IsNilpotent (LinearMap.mulLeft R a)
    ⊢ IsNilpotent (HSub.hSub (LinearMap.mulLeft R) (LinearMap.mulRight R) a)
  -/
  have hr : IsNilpotent (LinearMap.mulRight R a) := by rwa [LinearMap.isNilpotent_mulRight_iff]
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    h : IsNilpotent a
    hl : IsNilpotent (LinearMap.mulLeft R a)
    hr : IsNilpotent (LinearMap.mulRight R a)
    ⊢ IsNilpotent (HSub.hSub (LinearMap.mulLeft R) (LinearMap.mulRight R) a)
  -/
  have := @LinearMap.commute_mulLeft_right R A _ _ _ _ _ a a
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    h : IsNilpotent a
    hl : IsNilpotent (LinearMap.mulLeft R a)
    hr : IsNilpotent (LinearMap.mulRight R a)
    this : Commute (LinearMap.mulLeft R a) (LinearMap.mulRight R a)
    ⊢ IsNilpotent (HSub.hSub (LinearMap.mulLeft R) (LinearMap.mulRight R) a)
  -/
  exact this.isNilpotent_sub hl hr
  /-
    🎉 no goals
  -/


theorem _root_.LieSubalgebra.isNilpotent_ad_of_isNilpotent_ad {L : Type v} [LieRing L]
    [LieAlgebra R L] (K : LieSubalgebra R L) {x : K} (h : IsNilpotent (LieAlgebra.ad R L ↑x)) :
    IsNilpotent (LieAlgebra.ad R K x) := by
  /-
    R : Type u
    inst✝² : CommRing R
    L : Type v
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    x : Subtype fun x => Membership.mem K x
    h : IsNilpotent ((LieAlgebra.ad R L) ↑x)
    ⊢ IsNilpotent ((LieAlgebra.ad R (Subtype fun x => Membership.mem K x)) x)
  -/
  obtain ⟨n, hn⟩ := h
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    L : Type v
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    x : Subtype fun x => Membership.mem K x
    n : Nat
    hn : Eq (HPow.hPow ((LieAlgebra.ad R L) ↑x) n) 0
    ⊢ IsNilpotent ((LieAlgebra.ad R (Subtype fun x => Membership.mem K x)) x)
  -/
  use n
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    L : Type v
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    x : Subtype fun x => Membership.mem K x
    n : Nat
    hn : Eq (HPow.hPow ((LieAlgebra.ad R L) ↑x) n) 0
    ⊢ Eq (HPow.hPow ((LieAlgebra.ad R (Subtype fun x => Membership.mem K x)) x) n) 0
  -/
  exact LinearMap.submodule_pow_eq_zero_of_pow_eq_zero (K.ad_comp_incl_eq x) hn
  /-
    🎉 no goals
  -/


theorem _root_.LieAlgebra.isNilpotent_ad_of_isNilpotent {L : LieSubalgebra R A} {x : L}
    (h : IsNilpotent (x : A)) : IsNilpotent (LieAlgebra.ad R L x) :=
  L.isNilpotent_ad_of_isNilpotent_ad <| LieAlgebra.ad_nilpotent_of_nilpotent R h


@[simp]
lemma LieSubmodule.lowerCentralSeries_tensor_eq_baseChange (k : ℕ) :
    lowerCentralSeries A (A ⊗[R] L) (A ⊗[R] M) k =
    (lowerCentralSeries R L M k).baseChange A := by
  induction k with
  | zero => simp
  | succ k ih => simp only [lowerCentralSeries_succ, ih, ← baseChange_top, lie_baseChange]


instance LieModule.instIsNilpotentTensor [IsNilpotent R L M] :
    IsNilpotent A (A ⊗[R] L) (A ⊗[R] M) := by
  /-
    R : Type u_1
    A : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : LieModule.IsNilpotent R L M
    ⊢ LieModule.IsNilpotent A (TensorProduct R A L) (TensorProduct R A M)
  -/
  obtain ⟨k, hk⟩ := inferInstanceAs (IsNilpotent R L M)
  /-
    case mk.intro
    R : Type u_1
    A : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : LieModule.IsNilpotent R L M
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R L M k) Bot.bot
    ⊢ LieModule.IsNilpotent A (TensorProduct R A L) (TensorProduct R A M)
  -/
  exact ⟨k, by simp [hk]⟩
  /-
    🎉 no goals
  -/


