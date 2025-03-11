/-- Given a Lie module `M` of a Lie algebra `L`, `LieSubmodule.IsUcsLimit` is the proposition
that a Lie submodule `N ⊆ M` is the limiting value for the upper central series.

This is a characteristic property of Cartan subalgebras with the roles of `L`, `M`, `N` played by
`H`, `L`, `H`, respectively. See `LieSubalgebra.isCartanSubalgebra_iff_isUcsLimit`. -/
def LieSubmodule.IsUcsLimit {M : Type*} [AddCommGroup M] [Module R M] [LieRingModule L M]
    [LieModule R L M] (N : LieSubmodule R L M) : Prop :=
  ∃ k, ∀ l, k ≤ l → (⊥ : LieSubmodule R L M).ucs l = N


/-- A Cartan subalgebra is a nilpotent, self-normalizing subalgebra.

A _splitting_ Cartan subalgebra can be defined by mixing in `LieModule.IsTriangularizable R H L`. -/
class IsCartanSubalgebra : Prop where
  nilpotent : LieAlgebra.IsNilpotent R H
  self_normalizing : H.normalizer = H


instance [H.IsCartanSubalgebra] : LieAlgebra.IsNilpotent R H :=
  IsCartanSubalgebra.nilpotent


@[simp]
theorem normalizer_eq_self_of_isCartanSubalgebra (H : LieSubalgebra R L) [H.IsCartanSubalgebra] :
    H.toLieSubmodule.normalizer = H.toLieSubmodule := by
  rw [← LieSubmodule.toSubmodule_inj, coe_normalizer_eq_normalizer,
    IsCartanSubalgebra.self_normalizing, coe_toLieSubmodule]


@[simp]
theorem ucs_eq_self_of_isCartanSubalgebra (H : LieSubalgebra R L) [H.IsCartanSubalgebra] (k : ℕ) :
    H.toLieSubmodule.ucs k = H.toLieSubmodule := by
  induction k with
  | zero => simp
  | succ k ih => simp [ih]


theorem isCartanSubalgebra_iff_isUcsLimit : H.IsCartanSubalgebra ↔ H.toLieSubmodule.IsUcsLimit := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    ⊢ Iff H.IsCartanSubalgebra H.toLieSubmodule.IsUcsLimit
  -/
  constructor
    /-
      case mp
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      ⊢ H.IsCartanSubalgebra → H.toLieSubmodule.IsUcsLimit
    -/
  · intro h
    /-
      case mp
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      h : H.IsCartanSubalgebra
      ⊢ H.toLieSubmodule.IsUcsLimit
    -/
    have h₁ : LieAlgebra.IsNilpotent R H := by infer_instance
    /-
      case mp
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      h : H.IsCartanSubalgebra
      h₁ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
      ⊢ H.toLieSubmodule.IsUcsLimit
    -/
    obtain ⟨k, hk⟩ := H.toLieSubmodule.isNilpotent_iff_exists_self_le_ucs.mp h₁
    replace hk : H.toLieSubmodule = LieSubmodule.ucs k ⊥ :=
      le_antisymm hk
        (LieSubmodule.ucs_le_of_normalizer_eq_self H.normalizer_eq_self_of_isCartanSubalgebra k)
    /-
      case mp.intro
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      h : H.IsCartanSubalgebra
      h₁ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
      k : Nat
      hk : Eq H.toLieSubmodule (LieSubmodule.ucs k Bot.bot)
      ⊢ H.toLieSubmodule.IsUcsLimit
    -/
    refine ⟨k, fun l hl => ?_⟩
    rw [← Nat.sub_add_cancel hl, LieSubmodule.ucs_add, ← hk,
      LieSubalgebra.ucs_eq_self_of_isCartanSubalgebra]
    /-
      case mpr
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      ⊢ H.toLieSubmodule.IsUcsLimit → H.IsCartanSubalgebra
    -/
  · rintro ⟨k, hk⟩
    exact
      { nilpotent := by
          dsimp only [LieAlgebra.IsNilpotent]
          erw [H.toLieSubmodule.isNilpotent_iff_exists_lcs_eq_bot]
          use k
          rw [_root_.eq_bot_iff, LieSubmodule.lcs_le_iff, hk k (le_refl k)]
        self_normalizing := by
          have hk' := hk (k + 1) k.le_succ
          rw [LieSubmodule.ucs_succ, hk k (le_refl k)] at hk'
          rw [← LieSubalgebra.toSubmodule_inj, ← LieSubalgebra.coe_normalizer_eq_normalizer,
            hk', LieSubalgebra.coe_toLieSubmodule] }


lemma ne_bot_of_isCartanSubalgebra [Nontrivial L] (H : LieSubalgebra R L) [H.IsCartanSubalgebra] :
    H ≠ ⊥ := by
  /-
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Nontrivial L
    H : LieSubalgebra R L
    inst✝ : H.IsCartanSubalgebra
    ⊢ Ne H Bot.bot
  -/
  intro e
  /-
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Nontrivial L
    H : LieSubalgebra R L
    inst✝ : H.IsCartanSubalgebra
    e : Eq H Bot.bot
    ⊢ False
  -/
  obtain ⟨x, hx⟩ := exists_ne (0 : L)
  /-
    case intro
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Nontrivial L
    H : LieSubalgebra R L
    inst✝ : H.IsCartanSubalgebra
    e : Eq H Bot.bot
    x : L
    hx : Ne x 0
    ⊢ False
  -/
  have : x ∈ H.normalizer := by simp [LieSubalgebra.mem_normalizer_iff, e]
  /-
    case intro
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Nontrivial L
    H : LieSubalgebra R L
    inst✝ : H.IsCartanSubalgebra
    e : Eq H Bot.bot
    x : L
    hx : Ne x 0
    this : Membership.mem H.normalizer x
    ⊢ False
  -/
  exact hx (by rwa [LieSubalgebra.IsCartanSubalgebra.self_normalizing, e] at this)
  /-
    🎉 no goals
  -/


instance (priority := 500) [Nontrivial L] (H : LieSubalgebra R L) [H.IsCartanSubalgebra] :
    Nontrivial H := by
  /-
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    H✝ : LieSubalgebra R L
    inst✝¹ : Nontrivial L
    H : LieSubalgebra R L
    inst✝ : H.IsCartanSubalgebra
    ⊢ Nontrivial (Subtype fun x => Membership.mem H x)
  -/
  refine (subsingleton_or_nontrivial H).elim (fun inst ↦ False.elim ?_) id
  /-
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    H✝ : LieSubalgebra R L
    inst✝¹ : Nontrivial L
    H : LieSubalgebra R L
    inst✝ : H.IsCartanSubalgebra
    inst : Subsingleton (Subtype fun x => Membership.mem H x)
    ⊢ False
  -/
  apply ne_bot_of_isCartanSubalgebra H
  /-
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    H✝ : LieSubalgebra R L
    inst✝¹ : Nontrivial L
    H : LieSubalgebra R L
    inst✝ : H.IsCartanSubalgebra
    inst : Subsingleton (Subtype fun x => Membership.mem H x)
    ⊢ Eq H Bot.bot
  -/
  rw [eq_bot_iff]
  /-
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    H✝ : LieSubalgebra R L
    inst✝¹ : Nontrivial L
    H : LieSubalgebra R L
    inst✝ : H.IsCartanSubalgebra
    inst : Subsingleton (Subtype fun x => Membership.mem H x)
    ⊢ ∀ (x : L), Membership.mem H x → Eq x 0
  -/
  exact fun x hx ↦ congr_arg Subtype.val (Subsingleton.elim (⟨x, hx⟩ : H) 0)
  /-
    🎉 no goals
  -/


@[simp]
theorem LieIdeal.normalizer_eq_top {R : Type u} {L : Type v} [CommRing R] [LieRing L]
    [LieAlgebra R L] (I : LieIdeal R L) : (I : LieSubalgebra R L).normalizer = ⊤ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ Eq (LieIdeal.toLieSubalgebra R L I).normalizer Top.top
  -/
  ext x
  simpa only [LieSubalgebra.mem_normalizer_iff, LieSubalgebra.mem_top, iff_true] using
    fun y hy => I.lie_mem hy


/-- A nilpotent Lie algebra is its own Cartan subalgebra. -/
instance LieAlgebra.top_isCartanSubalgebra_of_nilpotent [LieAlgebra.IsNilpotent R L] :
    LieSubalgebra.IsCartanSubalgebra (⊤ : LieSubalgebra R L) where
  nilpotent := inferInstance
                         /-
                           R : Type u
                           L : Type v
                           inst✝³ : CommRing R
                           inst✝² : LieRing L
                           inst✝¹ : LieAlgebra R L
                           H : LieSubalgebra R L
                           inst✝ : LieAlgebra.IsNilpotent R L
                           ⊢ Eq Top.top.normalizer Top.top
                         -/
  self_normalizing := by rw [← top_toLieSubalgebra, normalizer_eq_top, top_toLieSubalgebra]
                         /-
                           🎉 no goals
                         -/

