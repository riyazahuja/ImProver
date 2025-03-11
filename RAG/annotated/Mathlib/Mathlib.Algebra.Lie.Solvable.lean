/-- A generalisation of the derived series of a Lie algebra, whose zeroth term is a specified ideal.

It can be more convenient to work with this generalisation when considering the derived series of
an ideal since it provides a type-theoretic expression of the fact that the terms of the ideal's
derived series are also ideals of the enclosing algebra.

See also `LieIdeal.derivedSeries_eq_derivedSeriesOfIdeal_comap` and
`LieIdeal.derivedSeries_eq_derivedSeriesOfIdeal_map` below. -/
def derivedSeriesOfIdeal (k : ℕ) : LieIdeal R L → LieIdeal R L :=
  (fun I => ⁅I, I⁆)^[k]


@[simp]
theorem derivedSeriesOfIdeal_zero : derivedSeriesOfIdeal R L 0 I = I :=
  rfl


@[simp]
theorem derivedSeriesOfIdeal_succ (k : ℕ) :
    derivedSeriesOfIdeal R L (k + 1) I =
      ⁅derivedSeriesOfIdeal R L k I, derivedSeriesOfIdeal R L k I⁆ :=
  Function.iterate_succ_apply' (fun I => ⁅I, I⁆) k I


/-- The derived series of Lie ideals of a Lie algebra. -/
abbrev derivedSeries (k : ℕ) : LieIdeal R L :=
  derivedSeriesOfIdeal R L k ⊤


theorem derivedSeries_def (k : ℕ) : derivedSeries R L k = derivedSeriesOfIdeal R L k ⊤ :=
  rfl


local notation "D" => derivedSeriesOfIdeal R L


theorem derivedSeriesOfIdeal_add (k l : ℕ) : D (k + l) I = D k (D l I) := by
  induction k with
  | zero => rw [Nat.zero_add, derivedSeriesOfIdeal_zero]
  | succ k ih => rw [Nat.succ_add k l, derivedSeriesOfIdeal_succ, derivedSeriesOfIdeal_succ, ih]


@[gcongr, mono]
theorem derivedSeriesOfIdeal_le {I J : LieIdeal R L} {k l : ℕ} (h₁ : I ≤ J) (h₂ : l ≤ k) :
    D k I ≤ D l J := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    k l : Nat
    h₁ : LE.le I J
    h₂ : LE.le l k
    ⊢ LE.le (LieAlgebra.derivedSeriesOfIdeal R L k I) (LieAlgebra.derivedSeriesOfI …
  -/
  revert l; induction' k with k ih <;> intro l h₂
    /-
      case zero
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I J : LieIdeal R L
      h₁ : LE.le I J
      l : Nat
      h₂ : LE.le l 0
      ⊢ LE.le (LieAlgebra.derivedSeriesOfIdeal R L 0 I) (LieAlgebra.derivedSeriesOfI …
    -/
  · rw [le_zero_iff] at h₂; rw [h₂, derivedSeriesOfIdeal_zero]; exact h₁
                                                                /-
                                                                  🎉 no goals
                                                                -/
    /-
      case succ
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I J : LieIdeal R L
      h₁ : LE.le I J
      k : Nat
      ih : ∀ {l : Nat}, LE.le l k → LE.le (LieAlgebra.derivedSeriesOfIdeal R L k I)  …
      l : Nat
      h₂ : LE.le l (HAdd.hAdd k 1)
      ⊢ LE.le (LieAlgebra.derivedSeriesOfIdeal R L (HAdd.hAdd k 1) I) (LieAlgebra.de …
    -/
  · have h : l = k.succ ∨ l ≤ k := by rwa [le_iff_eq_or_lt, Nat.lt_succ_iff] at h₂
    /-
      case succ
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I J : LieIdeal R L
      h₁ : LE.le I J
      k : Nat
      ih : ∀ {l : Nat}, LE.le l k → LE.le (LieAlgebra.derivedSeriesOfIdeal R L k I)  …
      l : Nat
      h₂ : LE.le l (HAdd.hAdd k 1)
      h : Or (Eq l k.succ) (LE.le l k)
      ⊢ LE.le (LieAlgebra.derivedSeriesOfIdeal R L (HAdd.hAdd k 1) I) (LieAlgebra.de …
    -/
    cases' h with h h
      /-
        case succ.inl
        R : Type u
        L : Type v
        inst✝² : CommRing R
        inst✝¹ : LieRing L
        inst✝ : LieAlgebra R L
        I J : LieIdeal R L
        h₁ : LE.le I J
        k : Nat
        ih : ∀ {l : Nat}, LE.le l k → LE.le (LieAlgebra.derivedSeriesOfIdeal R L k I)  …
        l : Nat
        h₂ : LE.le l (HAdd.hAdd k 1)
        h : Eq l k.succ
        ⊢ LE.le (LieAlgebra.derivedSeriesOfIdeal R L (HAdd.hAdd k 1) I) (LieAlgebra.de …
      -/
    · rw [h, derivedSeriesOfIdeal_succ, derivedSeriesOfIdeal_succ]
      /-
        case succ.inl
        R : Type u
        L : Type v
        inst✝² : CommRing R
        inst✝¹ : LieRing L
        inst✝ : LieAlgebra R L
        I J : LieIdeal R L
        h₁ : LE.le I J
        k : Nat
        ih : ∀ {l : Nat}, LE.le l k → LE.le (LieAlgebra.derivedSeriesOfIdeal R L k I)  …
        l : Nat
        h₂ : LE.le l (HAdd.hAdd k 1)
        h : Eq l k.succ
        ⊢ LE.le (Bracket.bracket (LieAlgebra.derivedSeriesOfIdeal R L k I) (LieAlgebra …
      -/
      exact LieSubmodule.mono_lie (ih (le_refl k)) (ih (le_refl k))
      /-
        🎉 no goals
      -/
      /-
        case succ.inr
        R : Type u
        L : Type v
        inst✝² : CommRing R
        inst✝¹ : LieRing L
        inst✝ : LieAlgebra R L
        I J : LieIdeal R L
        h₁ : LE.le I J
        k : Nat
        ih : ∀ {l : Nat}, LE.le l k → LE.le (LieAlgebra.derivedSeriesOfIdeal R L k I)  …
        l : Nat
        h₂ : LE.le l (HAdd.hAdd k 1)
        h : LE.le l k
        ⊢ LE.le (LieAlgebra.derivedSeriesOfIdeal R L (HAdd.hAdd k 1) I) (LieAlgebra.de …
      -/
    · rw [derivedSeriesOfIdeal_succ]; exact le_trans (LieSubmodule.lie_le_left _ _) (ih h)
                                      /-
                                        🎉 no goals
                                      -/


theorem derivedSeriesOfIdeal_succ_le (k : ℕ) : D (k + 1) I ≤ D k I :=
  derivedSeriesOfIdeal_le (le_refl I) k.le_succ


theorem derivedSeriesOfIdeal_le_self (k : ℕ) : D k I ≤ I :=
  derivedSeriesOfIdeal_le (le_refl I) (zero_le k)


theorem derivedSeriesOfIdeal_mono {I J : LieIdeal R L} (h : I ≤ J) (k : ℕ) : D k I ≤ D k J :=
  derivedSeriesOfIdeal_le h (le_refl k)


theorem derivedSeriesOfIdeal_antitone {k l : ℕ} (h : l ≤ k) : D k I ≤ D l I :=
  derivedSeriesOfIdeal_le (le_refl I) h


theorem derivedSeriesOfIdeal_add_le_add (J : LieIdeal R L) (k l : ℕ) :
    D (k + l) (I + J) ≤ D k I + D l J := by
  let D₁ : LieIdeal R L →o LieIdeal R L :=
    { toFun := fun I => ⁅I, I⁆
      monotone' := fun I J h => LieSubmodule.mono_lie h h }
  have h₁ : ∀ I J : LieIdeal R L, D₁ (I ⊔ J) ≤ D₁ I ⊔ J := by
    simp [D₁, LieSubmodule.lie_le_right, LieSubmodule.lie_le_left, le_sup_of_le_right]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    k l : Nat
    D₁ : OrderHom (LieIdeal R L) (LieIdeal R L) := { toFun := fun I => Bracket.bra …
    h₁ : ∀ (I J : LieIdeal R L), LE.le (D₁ (Max.max I J)) (Max.max (D₁ I) J)
    ⊢ LE.le (LieAlgebra.derivedSeriesOfIdeal R L (HAdd.hAdd k l) (HAdd.hAdd I J))  …
  -/
  rw [← D₁.iterate_sup_le_sup_iff] at h₁
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    k l : Nat
    D₁ : OrderHom (LieIdeal R L) (LieIdeal R L) := { toFun := fun I => Bracket.bra …
    h₁ : ∀ (n₁ n₂ : Nat) (a₁ a₂ : LieIdeal R L), LE.le (Nat.iterate (⇑D₁) (HAdd.hA …
    ⊢ LE.le (LieAlgebra.derivedSeriesOfIdeal R L (HAdd.hAdd k l) (HAdd.hAdd I J))  …
  -/
  exact h₁ k l I J
  /-
    🎉 no goals
  -/


theorem derivedSeries_of_bot_eq_bot (k : ℕ) : derivedSeriesOfIdeal R L k ⊥ = ⊥ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    k : Nat
    ⊢ Eq (LieAlgebra.derivedSeriesOfIdeal R L k Bot.bot) Bot.bot
  -/
  rw [eq_bot_iff]; exact derivedSeriesOfIdeal_le_self ⊥ k
                   /-
                     🎉 no goals
                   -/


theorem abelian_iff_derived_one_eq_bot : IsLieAbelian I ↔ derivedSeriesOfIdeal R L 1 I = ⊥ := by
  rw [derivedSeriesOfIdeal_succ, derivedSeriesOfIdeal_zero,
    LieSubmodule.lie_abelian_iff_lie_self_eq_bot]


theorem abelian_iff_derived_succ_eq_bot (I : LieIdeal R L) (k : ℕ) :
    IsLieAbelian (derivedSeriesOfIdeal R L k I) ↔ derivedSeriesOfIdeal R L (k + 1) I = ⊥ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    k : Nat
    ⊢ Iff (IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.derivedSeries …
  -/
  rw [add_comm, derivedSeriesOfIdeal_add I 1 k, abelian_iff_derived_one_eq_bot]
  /-
    🎉 no goals
  -/


theorem derivedSeries_eq_derivedSeriesOfIdeal_comap (k : ℕ) :
    derivedSeries R I k = (derivedSeriesOfIdeal R L k I).comap I.incl := by
  induction k with
  | zero => simp only [derivedSeries_def, comap_incl_self, derivedSeriesOfIdeal_zero]
  | succ k ih =>
    simp only [derivedSeries_def, derivedSeriesOfIdeal_succ] at ih ⊢; rw [ih]
    exact comap_bracket_incl_of_le I (derivedSeriesOfIdeal_le_self I k)
      (derivedSeriesOfIdeal_le_self I k)


theorem derivedSeries_eq_derivedSeriesOfIdeal_map (k : ℕ) :
    (derivedSeries R I k).map I.incl = derivedSeriesOfIdeal R L k I := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    k : Nat
    ⊢ Eq (LieIdeal.map I.incl (LieAlgebra.derivedSeries R (Subtype fun x => Member …
  -/
  rw [derivedSeries_eq_derivedSeriesOfIdeal_comap, map_comap_incl, inf_eq_right]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    k : Nat
    ⊢ LE.le (LieAlgebra.derivedSeriesOfIdeal R L k I) I
  -/
  apply derivedSeriesOfIdeal_le_self
  /-
    🎉 no goals
  -/


theorem derivedSeries_eq_bot_iff (k : ℕ) :
    derivedSeries R I k = ⊥ ↔ derivedSeriesOfIdeal R L k I = ⊥ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    k : Nat
    ⊢ Iff (Eq (LieAlgebra.derivedSeries R (Subtype fun x => Membership.mem I x) k) …
  -/
  rw [← derivedSeries_eq_derivedSeriesOfIdeal_map, map_eq_bot_iff, ker_incl, eq_bot_iff]
  /-
    🎉 no goals
  -/


theorem derivedSeries_add_eq_bot {k l : ℕ} {I J : LieIdeal R L} (hI : derivedSeries R I k = ⊥)
    (hJ : derivedSeries R J l = ⊥) : derivedSeries R (I + J) (k + l) = ⊥ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    k l : Nat
    I J : LieIdeal R L
    hI : Eq (LieAlgebra.derivedSeries R (Subtype fun x => Membership.mem I x) k) B …
    hJ : Eq (LieAlgebra.derivedSeries R (Subtype fun x => Membership.mem J x) l) B …
    ⊢ Eq (LieAlgebra.derivedSeries R (Subtype fun x => Membership.mem (HAdd.hAdd I …
  -/
  rw [LieIdeal.derivedSeries_eq_bot_iff] at hI hJ ⊢
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    k l : Nat
    I J : LieIdeal R L
    hI : Eq (LieAlgebra.derivedSeriesOfIdeal R L k I) Bot.bot
    hJ : Eq (LieAlgebra.derivedSeriesOfIdeal R L l J) Bot.bot
    ⊢ Eq (LieAlgebra.derivedSeriesOfIdeal R L (HAdd.hAdd k l) (HAdd.hAdd I J)) Bot …
  -/
  rw [← le_bot_iff]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    k l : Nat
    I J : LieIdeal R L
    hI : Eq (LieAlgebra.derivedSeriesOfIdeal R L k I) Bot.bot
    hJ : Eq (LieAlgebra.derivedSeriesOfIdeal R L l J) Bot.bot
    ⊢ LE.le (LieAlgebra.derivedSeriesOfIdeal R L (HAdd.hAdd k l) (HAdd.hAdd I J))  …
  -/
  let D := derivedSeriesOfIdeal R L; change D k I = ⊥ at hI; change D l J = ⊥ at hJ
  calc
    D (k + l) (I + J) ≤ D k I + D l J := derivedSeriesOfIdeal_add_le_add I J k l
    _ ≤ ⊥ := by rw [hI, hJ]; simp


theorem derivedSeries_map_le (k : ℕ) : (derivedSeries R L' k).map f ≤ derivedSeries R L k := by
  induction k with
  | zero => simp only [derivedSeries_def, derivedSeriesOfIdeal_zero, le_top]
  | succ k ih =>
    simp only [derivedSeries_def, derivedSeriesOfIdeal_succ] at ih ⊢
    exact le_trans (map_bracket_le f) (LieSubmodule.mono_lie ih ih)


theorem derivedSeries_map_eq (k : ℕ) (h : Function.Surjective f) :
    (derivedSeries R L' k).map f = derivedSeries R L k := by
  induction k with
  | zero =>
    change (⊤ : LieIdeal R L').map f = ⊤
    rw [← f.idealRange_eq_map]
    exact f.idealRange_eq_top_of_surjective h
  | succ k ih => simp only [derivedSeries_def, map_bracket_eq f h, ih, derivedSeriesOfIdeal_succ]


theorem derivedSeries_succ_eq_top_iff (n : ℕ) :
    derivedSeries R L (n + 1) = ⊤ ↔ derivedSeries R L 1 = ⊤ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    n : Nat
    ⊢ Iff (Eq (LieAlgebra.derivedSeries R L (HAdd.hAdd n 1)) Top.top) (Eq (LieAlge …
  -/
  simp only [derivedSeries_def]
  induction n with
  | zero => simp
  | succ n ih =>
    rw [derivedSeriesOfIdeal_succ]
    refine ⟨fun h ↦ ?_, fun h ↦ by rwa [ih.mpr h]⟩
    rw [← ih, eq_top_iff]
    conv_lhs => rw [← h]
    exact LieSubmodule.lie_le_right _ _


theorem derivedSeries_eq_top (n : ℕ) (h : derivedSeries R L 1 = ⊤) :
    derivedSeries R L n = ⊤ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    n : Nat
    h : Eq (LieAlgebra.derivedSeries R L 1) Top.top
    ⊢ Eq (LieAlgebra.derivedSeries R L n) Top.top
  -/
  cases n
    /-
      case zero
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      h : Eq (LieAlgebra.derivedSeries R L 1) Top.top
      ⊢ Eq (LieAlgebra.derivedSeries R L 0) Top.top
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      h : Eq (LieAlgebra.derivedSeries R L 1) Top.top
      n✝ : Nat
      ⊢ Eq (LieAlgebra.derivedSeries R L (HAdd.hAdd n✝ 1)) Top.top
    -/
  · rwa [derivedSeries_succ_eq_top_iff]
    /-
      🎉 no goals
    -/


/-- A Lie algebra is solvable if its derived series reaches 0 (in a finite number of steps). -/
class IsSolvable : Prop where
  solvable : ∃ k, derivedSeries R L k = ⊥


instance isSolvableBot : IsSolvable R (⊥ : LieIdeal R L) :=
  ⟨⟨0, Subsingleton.elim _ ⊥⟩⟩


instance isSolvableAdd {I J : LieIdeal R L} [hI : IsSolvable R I] [hJ : IsSolvable R J] :
    IsSolvable R (I + J) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    L' : Type w₁
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    I✝ J✝ : LieIdeal R L
    f : LieHom R L' L
    I J : LieIdeal R L
    hI : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
    hJ : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem J x)
    ⊢ LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem (HAdd.hAdd I J) x)
  -/
  obtain ⟨k, hk⟩ := id hI; obtain ⟨l, hl⟩ := id hJ
  /-
    case mk.intro.mk.intro
    R : Type u
    L : Type v
    M : Type w
    L' : Type w₁
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    I✝ J✝ : LieIdeal R L
    f : LieHom R L' L
    I J : LieIdeal R L
    hI : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
    hJ : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem J x)
    k : Nat
    hk : Eq (LieAlgebra.derivedSeries R (Subtype fun x => Membership.mem I x) k) B …
    l : Nat
    hl : Eq (LieAlgebra.derivedSeries R (Subtype fun x => Membership.mem J x) l) B …
    ⊢ LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem (HAdd.hAdd I J) x)
  -/
  exact ⟨⟨k + l, LieIdeal.derivedSeries_add_eq_bot hk hl⟩⟩
  /-
    🎉 no goals
  -/


theorem derivedSeries_lt_top_of_solvable [IsSolvable R L] [Nontrivial L] :
    derivedSeries R L 1 < ⊤ := by
  /-
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieAlgebra.IsSolvable R L
    inst✝ : Nontrivial L
    ⊢ LT.lt (LieAlgebra.derivedSeries R L 1) Top.top
  -/
  obtain ⟨n, hn⟩ := IsSolvable.solvable (R := R) (L := L)
  /-
    case intro
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieAlgebra.IsSolvable R L
    inst✝ : Nontrivial L
    n : Nat
    hn : Eq (LieAlgebra.derivedSeries R L n) Bot.bot
    ⊢ LT.lt (LieAlgebra.derivedSeries R L 1) Top.top
  -/
  rw [lt_top_iff_ne_top]
  /-
    case intro
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieAlgebra.IsSolvable R L
    inst✝ : Nontrivial L
    n : Nat
    hn : Eq (LieAlgebra.derivedSeries R L n) Bot.bot
    ⊢ Ne (LieAlgebra.derivedSeries R L 1) Top.top
  -/
  intro contra
  /-
    case intro
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieAlgebra.IsSolvable R L
    inst✝ : Nontrivial L
    n : Nat
    hn : Eq (LieAlgebra.derivedSeries R L n) Bot.bot
    contra : Eq (LieAlgebra.derivedSeries R L 1) Top.top
    ⊢ False
  -/
  rw [LieIdeal.derivedSeries_eq_top n contra] at hn
  /-
    case intro
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieAlgebra.IsSolvable R L
    inst✝ : Nontrivial L
    n : Nat
    hn : Eq Top.top Bot.bot
    contra : Eq (LieAlgebra.derivedSeries R L 1) Top.top
    ⊢ False
  -/
  exact top_ne_bot hn
  /-
    🎉 no goals
  -/


theorem Injective.lieAlgebra_isSolvable [h₁ : IsSolvable R L] (h₂ : Injective f) :
    IsSolvable R L' := by
  /-
    R : Type u
    L : Type v
    L' : Type w₁
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L' L
    h₁ : LieAlgebra.IsSolvable R L
    h₂ : Function.Injective ⇑f
    ⊢ LieAlgebra.IsSolvable R L'
  -/
  obtain ⟨k, hk⟩ := id h₁
  /-
    case mk.intro
    R : Type u
    L : Type v
    L' : Type w₁
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L' L
    h₁ : LieAlgebra.IsSolvable R L
    h₂ : Function.Injective ⇑f
    k : Nat
    hk : Eq (LieAlgebra.derivedSeries R L k) Bot.bot
    ⊢ LieAlgebra.IsSolvable R L'
  -/
  use k
  /-
    case h
    R : Type u
    L : Type v
    L' : Type w₁
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L' L
    h₁ : LieAlgebra.IsSolvable R L
    h₂ : Function.Injective ⇑f
    k : Nat
    hk : Eq (LieAlgebra.derivedSeries R L k) Bot.bot
    ⊢ Eq (LieAlgebra.derivedSeries R L' k) Bot.bot
  -/
  apply LieIdeal.bot_of_map_eq_bot h₂; rw [eq_bot_iff, ← hk]
  /-
    case h
    R : Type u
    L : Type v
    L' : Type w₁
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L' L
    h₁ : LieAlgebra.IsSolvable R L
    h₂ : Function.Injective ⇑f
    k : Nat
    hk : Eq (LieAlgebra.derivedSeries R L k) Bot.bot
    ⊢ LE.le (LieIdeal.map f (LieAlgebra.derivedSeries R L' k)) (LieAlgebra.derived …
  -/
  apply LieIdeal.derivedSeries_map_le
  /-
    🎉 no goals
  -/


instance (A : LieIdeal R L) [IsSolvable R L] : IsSolvable R A :=
  A.incl_injective.lieAlgebra_isSolvable


theorem Surjective.lieAlgebra_isSolvable [h₁ : IsSolvable R L'] (h₂ : Surjective f) :
    IsSolvable R L := by
  /-
    R : Type u
    L : Type v
    L' : Type w₁
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L' L
    h₁ : LieAlgebra.IsSolvable R L'
    h₂ : Function.Surjective ⇑f
    ⊢ LieAlgebra.IsSolvable R L
  -/
  obtain ⟨k, hk⟩ := id h₁
  /-
    case mk.intro
    R : Type u
    L : Type v
    L' : Type w₁
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L' L
    h₁ : LieAlgebra.IsSolvable R L'
    h₂ : Function.Surjective ⇑f
    k : Nat
    hk : Eq (LieAlgebra.derivedSeries R L' k) Bot.bot
    ⊢ LieAlgebra.IsSolvable R L
  -/
  use k
  /-
    case h
    R : Type u
    L : Type v
    L' : Type w₁
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L' L
    h₁ : LieAlgebra.IsSolvable R L'
    h₂ : Function.Surjective ⇑f
    k : Nat
    hk : Eq (LieAlgebra.derivedSeries R L' k) Bot.bot
    ⊢ Eq (LieAlgebra.derivedSeries R L k) Bot.bot
  -/
  rw [← LieIdeal.derivedSeries_map_eq k h₂, hk]
  /-
    case h
    R : Type u
    L : Type v
    L' : Type w₁
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L' L
    h₁ : LieAlgebra.IsSolvable R L'
    h₂ : Function.Surjective ⇑f
    k : Nat
    hk : Eq (LieAlgebra.derivedSeries R L' k) Bot.bot
    ⊢ Eq (LieIdeal.map f Bot.bot) Bot.bot
  -/
  simp only [LieIdeal.map_eq_bot_iff, bot_le]
  /-
    🎉 no goals
  -/


instance LieHom.isSolvable_range (f : L' →ₗ⁅R⁆ L) [LieAlgebra.IsSolvable R L'] :
    LieAlgebra.IsSolvable R f.range :=
  f.surjective_rangeRestrict.lieAlgebra_isSolvable


theorem solvable_iff_equiv_solvable (e : L' ≃ₗ⁅R⁆ L) : IsSolvable R L' ↔ IsSolvable R L := by
  /-
    R : Type u
    L : Type v
    L' : Type w₁
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    e : LieEquiv R L' L
    ⊢ Iff (LieAlgebra.IsSolvable R L') (LieAlgebra.IsSolvable R L)
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u
      L : Type v
      L' : Type w₁
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      e : LieEquiv R L' L
      h : LieAlgebra.IsSolvable R L'
      ⊢ LieAlgebra.IsSolvable R L
    -/
  · exact e.symm.injective.lieAlgebra_isSolvable
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      L : Type v
      L' : Type w₁
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      e : LieEquiv R L' L
      h : LieAlgebra.IsSolvable R L
      ⊢ LieAlgebra.IsSolvable R L'
    -/
  · exact e.injective.lieAlgebra_isSolvable
    /-
      🎉 no goals
    -/


theorem le_solvable_ideal_solvable {I J : LieIdeal R L} (h₁ : I ≤ J) (_ : IsSolvable R J) :
    IsSolvable R I :=
  (LieIdeal.inclusion_injective h₁).lieAlgebra_isSolvable


instance (priority := 100) ofAbelianIsSolvable [IsLieAbelian L] : IsSolvable R L := by
  /-
    R : Type u
    L : Type v
    M : Type w
    L' : Type w₁
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    I J : LieIdeal R L
    f : LieHom R L' L
    inst✝ : IsLieAbelian L
    ⊢ LieAlgebra.IsSolvable R L
  -/
  use 1
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    L' : Type w₁
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    I J : LieIdeal R L
    f : LieHom R L' L
    inst✝ : IsLieAbelian L
    ⊢ Eq (LieAlgebra.derivedSeries R L 1) Bot.bot
  -/
  rw [← abelian_iff_derived_one_eq_bot, lie_abelian_iff_equiv_lie_abelian LieIdeal.topEquiv]
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    L' : Type w₁
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    I J : LieIdeal R L
    f : LieHom R L' L
    inst✝ : IsLieAbelian L
    ⊢ IsLieAbelian L
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The (solvable) radical of Lie algebra is the `sSup` of all solvable ideals. -/
def radical :=
  sSup { I : LieIdeal R L | IsSolvable R I }


/-- The radical of a Noetherian Lie algebra is solvable. -/
instance radicalIsSolvable [IsNoetherian R L] : IsSolvable R (radical R L) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    L' : Type w₁
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    I J : LieIdeal R L
    f : LieHom R L' L
    inst✝ : IsNoetherian R L
    ⊢ LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem (LieAlgebra.radical …
  -/
  have hwf := LieSubmodule.wellFoundedGT_of_noetherian R L L
  /-
    R : Type u
    L : Type v
    M : Type w
    L' : Type w₁
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    I J : LieIdeal R L
    f : LieHom R L' L
    inst✝ : IsNoetherian R L
    hwf : WellFoundedGT (LieSubmodule R L L)
    ⊢ LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem (LieAlgebra.radical …
  -/
  rw [← CompleteLattice.isSupClosedCompact_iff_wellFoundedGT] at hwf
  /-
    R : Type u
    L : Type v
    M : Type w
    L' : Type w₁
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    I J : LieIdeal R L
    f : LieHom R L' L
    inst✝ : IsNoetherian R L
    hwf : CompleteLattice.IsSupClosedCompact (LieSubmodule R L L)
    ⊢ LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem (LieAlgebra.radical …
  -/
  refine hwf { I : LieIdeal R L | IsSolvable R I } ⟨⊥, ?_⟩ fun I hI J hJ => ?_
    /-
      case refine_1
      R : Type u
      L : Type v
      M : Type w
      L' : Type w₁
      inst✝⁵ : CommRing R
      inst✝⁴ : LieRing L
      inst✝³ : LieAlgebra R L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      I J : LieIdeal R L
      f : LieHom R L' L
      inst✝ : IsNoetherian R L
      hwf : CompleteLattice.IsSupClosedCompact (LieSubmodule R L L)
      ⊢ Membership.mem (setOf fun I => LieAlgebra.IsSolvable R (Subtype fun x => Mem …
    -/
  · exact LieAlgebra.isSolvableBot R L
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      L : Type v
      M : Type w
      L' : Type w₁
      inst✝⁵ : CommRing R
      inst✝⁴ : LieRing L
      inst✝³ : LieAlgebra R L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      I✝ J✝ : LieIdeal R L
      f : LieHom R L' L
      inst✝ : IsNoetherian R L
      hwf : CompleteLattice.IsSupClosedCompact (LieSubmodule R L L)
      I : LieSubmodule R L L
      hI : Membership.mem (setOf fun I => LieAlgebra.IsSolvable R (Subtype fun x =>  …
      J : LieSubmodule R L L
      hJ : Membership.mem (setOf fun I => LieAlgebra.IsSolvable R (Subtype fun x =>  …
      ⊢ Membership.mem (setOf fun I => LieAlgebra.IsSolvable R (Subtype fun x => Mem …
    -/
  · rw [Set.mem_setOf_eq] at hI hJ ⊢
    /-
      case refine_2
      R : Type u
      L : Type v
      M : Type w
      L' : Type w₁
      inst✝⁵ : CommRing R
      inst✝⁴ : LieRing L
      inst✝³ : LieAlgebra R L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      I✝ J✝ : LieIdeal R L
      f : LieHom R L' L
      inst✝ : IsNoetherian R L
      hwf : CompleteLattice.IsSupClosedCompact (LieSubmodule R L L)
      I : LieSubmodule R L L
      hI : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
      J : LieSubmodule R L L
      hJ : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem J x)
      ⊢ LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem (Max.max I J) x)
    -/
    apply LieAlgebra.isSolvableAdd R L
    /-
      🎉 no goals
    -/


/-- The `→` direction of this lemma is actually true without the `IsNoetherian` assumption. -/
theorem LieIdeal.solvable_iff_le_radical [IsNoetherian R L] (I : LieIdeal R L) :
    IsSolvable R I ↔ I ≤ radical R L :=
  ⟨fun h => le_sSup h, fun h => le_solvable_ideal_solvable h inferInstance⟩


theorem center_le_radical : center R L ≤ radical R L :=
  have h : IsSolvable R (center R L) := inferInstance
  le_sSup h


instance [IsSolvable R L] : IsSolvable R (⊤ : LieSubalgebra R L) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    L' : Type w₁
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    I J : LieIdeal R L
    f : LieHom R L' L
    inst✝ : LieAlgebra.IsSolvable R L
    ⊢ LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem Top.top x)
  -/
  rwa [solvable_iff_equiv_solvable LieSubalgebra.topEquiv]
  /-
    🎉 no goals
  -/


@[simp] lemma radical_eq_top_of_isSolvable [IsSolvable R L] :
    radical R L = ⊤ := by
  /-
    R : Type u
    L : Type v
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSolvable R L
    ⊢ Eq (LieAlgebra.radical R L) Top.top
  -/
  rw [eq_top_iff]
  /-
    R : Type u
    L : Type v
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSolvable R L
    ⊢ LE.le Top.top (LieAlgebra.radical R L)
  -/
  have h : IsSolvable R (⊤ : LieSubalgebra R L) := inferInstance
  /-
    R : Type u
    L : Type v
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSolvable R L
    h : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem Top.top x)
    ⊢ LE.le Top.top (LieAlgebra.radical R L)
  -/
  exact le_sSup h
  /-
    🎉 no goals
  -/


/-- Given a solvable Lie ideal `I` with derived series `I = D₀ ≥ D₁ ≥ ⋯ ≥ Dₖ = ⊥`, this is the
natural number `k` (the number of inclusions).

For a non-solvable ideal, the value is 0. -/
noncomputable def derivedLengthOfIdeal (I : LieIdeal R L) : ℕ :=
  sInf { k | derivedSeriesOfIdeal R L k I = ⊥ }


/-- The derived length of a Lie algebra is the derived length of its 'top' Lie ideal.

See also `LieAlgebra.derivedLength_eq_derivedLengthOfIdeal`. -/
noncomputable abbrev derivedLength : ℕ :=
  derivedLengthOfIdeal R L ⊤


theorem derivedSeries_of_derivedLength_succ (I : LieIdeal R L) (k : ℕ) :
    derivedLengthOfIdeal R L I = k + 1 ↔
      IsLieAbelian (derivedSeriesOfIdeal R L k I) ∧ derivedSeriesOfIdeal R L k I ≠ ⊥ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    k : Nat
    ⊢ Iff (Eq (LieAlgebra.derivedLengthOfIdeal R L I) (HAdd.hAdd k 1)) (And (IsLie …
  -/
  rw [abelian_iff_derived_succ_eq_bot]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    k : Nat
    ⊢ Iff (Eq (LieAlgebra.derivedLengthOfIdeal R L I) (HAdd.hAdd k 1)) (And (Eq (L …
  -/
  let s := { k | derivedSeriesOfIdeal R L k I = ⊥ }
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    k : Nat
    s : Set Nat := setOf fun k => Eq (LieAlgebra.derivedSeriesOfIdeal R L k I) Bot …
    ⊢ Iff (Eq (LieAlgebra.derivedLengthOfIdeal R L I) (HAdd.hAdd k 1)) (And (Eq (L …
  -/
  change sInf s = k + 1 ↔ k + 1 ∈ s ∧ k ∉ s
  have hs : ∀ k₁ k₂ : ℕ, k₁ ≤ k₂ → k₁ ∈ s → k₂ ∈ s := by
    intro k₁ k₂ h₁₂ h₁
    suffices derivedSeriesOfIdeal R L k₂ I ≤ ⊥ by exact eq_bot_iff.mpr this
    change derivedSeriesOfIdeal R L k₁ I = ⊥ at h₁; rw [← h₁]
    exact derivedSeriesOfIdeal_antitone I h₁₂
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    k : Nat
    s : Set Nat := setOf fun k => Eq (LieAlgebra.derivedSeriesOfIdeal R L k I) Bot …
    hs : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
    ⊢ Iff (Eq (InfSet.sInf s) (HAdd.hAdd k 1)) (And (Membership.mem s (HAdd.hAdd k …
  -/
  exact Nat.sInf_upward_closed_eq_succ_iff hs k
  /-
    🎉 no goals
  -/


theorem derivedLength_eq_derivedLengthOfIdeal (I : LieIdeal R L) :
    derivedLength R I = derivedLengthOfIdeal R L I := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ Eq (LieAlgebra.derivedLength R (Subtype fun x => Membership.mem I x)) (LieAl …
  -/
  let s₁ := { k | derivedSeries R I k = ⊥ }
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    s₁ : Set Nat := setOf fun k => Eq (LieAlgebra.derivedSeries R (Subtype fun x = …
    ⊢ Eq (LieAlgebra.derivedLength R (Subtype fun x => Membership.mem I x)) (LieAl …
  -/
  let s₂ := { k | derivedSeriesOfIdeal R L k I = ⊥ }
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    s₁ : Set Nat := setOf fun k => Eq (LieAlgebra.derivedSeries R (Subtype fun x = …
    s₂ : Set Nat := setOf fun k => Eq (LieAlgebra.derivedSeriesOfIdeal R L k I) Bo …
    ⊢ Eq (LieAlgebra.derivedLength R (Subtype fun x => Membership.mem I x)) (LieAl …
  -/
  change sInf s₁ = sInf s₂
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    s₁ : Set Nat := setOf fun k => Eq (LieAlgebra.derivedSeries R (Subtype fun x = …
    s₂ : Set Nat := setOf fun k => Eq (LieAlgebra.derivedSeriesOfIdeal R L k I) Bo …
    ⊢ Eq (InfSet.sInf s₁) (InfSet.sInf s₂)
  -/
  congr; ext k; exact I.derivedSeries_eq_bot_iff k
                /-
                  🎉 no goals
                -/


/-- Given a solvable Lie ideal `I` with derived series `I = D₀ ≥ D₁ ≥ ⋯ ≥ Dₖ = ⊥`, this is the
`k-1`th term in the derived series (and is therefore an Abelian ideal contained in `I`).

For a non-solvable ideal, this is the zero ideal, `⊥`. -/
noncomputable def derivedAbelianOfIdeal (I : LieIdeal R L) : LieIdeal R L :=
  match derivedLengthOfIdeal R L I with
  | 0 => ⊥
  | k + 1 => derivedSeriesOfIdeal R L k I


instance : Unique {x // x ∈ (⊥ : LieIdeal R L)} :=
  inferInstanceAs <| Unique {x // x ∈ (⊥ : Submodule R L)}


theorem abelian_derivedAbelianOfIdeal (I : LieIdeal R L) :
    IsLieAbelian (derivedAbelianOfIdeal I) := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.derivedAbelianOfId …
  -/
  dsimp only [derivedAbelianOfIdeal]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.derivedAbelianOfId …
  -/
  cases' h : derivedLengthOfIdeal R L I with k
    /-
      case zero
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I : LieIdeal R L
      h : Eq (LieAlgebra.derivedLengthOfIdeal R L I) 0
      ⊢ IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.derivedAbelianOfId …
    -/
  · dsimp; infer_instance
           /-
             🎉 no goals
           -/
    /-
      case succ
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I : LieIdeal R L
      k : Nat
      h : Eq (LieAlgebra.derivedLengthOfIdeal R L I) (HAdd.hAdd k 1)
      ⊢ IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.derivedAbelianOfId …
    -/
  · rw [derivedSeries_of_derivedLength_succ] at h; exact h.1
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem derivedLength_zero (I : LieIdeal R L) [hI : IsSolvable R I] :
    derivedLengthOfIdeal R L I = 0 ↔ I = ⊥ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    hI : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
    ⊢ Iff (Eq (LieAlgebra.derivedLengthOfIdeal R L I) 0) (Eq I Bot.bot)
  -/
  let s := { k | derivedSeriesOfIdeal R L k I = ⊥ }
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    hI : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
    s : Set Nat := setOf fun k => Eq (LieAlgebra.derivedSeriesOfIdeal R L k I) Bot …
    ⊢ Iff (Eq (LieAlgebra.derivedLengthOfIdeal R L I) 0) (Eq I Bot.bot)
  -/
  change sInf s = 0 ↔ _
  have hne : s ≠ ∅ := by
    obtain ⟨k, hk⟩ := id hI
    refine Set.Nonempty.ne_empty ⟨k, ?_⟩
    rw [derivedSeries_def, LieIdeal.derivedSeries_eq_bot_iff] at hk; exact hk
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    hI : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
    s : Set Nat := setOf fun k => Eq (LieAlgebra.derivedSeriesOfIdeal R L k I) Bot …
    hne : Ne s EmptyCollection.emptyCollection
    ⊢ Iff (Eq (InfSet.sInf s) 0) (Eq I Bot.bot)
  -/
  simp [s, hne]
  /-
    🎉 no goals
  -/


theorem abelian_of_solvable_ideal_eq_bot_iff (I : LieIdeal R L) [h : IsSolvable R I] :
    derivedAbelianOfIdeal I = ⊥ ↔ I = ⊥ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
    ⊢ Iff (Eq (LieAlgebra.derivedAbelianOfIdeal I) Bot.bot) (Eq I Bot.bot)
  -/
  dsimp only [derivedAbelianOfIdeal]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
    ⊢ Iff (Eq (LieAlgebra.derivedAbelianOfIdeal.match_1 (fun x => LieIdeal R L) (L …
  -/
  split -- Porting note: Original tactic was `cases' h : derivedAbelianOfIdeal R L I with k`
    /-
      case h_1
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I : LieIdeal R L
      h : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
      x✝ : Nat
      heq✝ : Eq (LieAlgebra.derivedLengthOfIdeal R L I) 0
      ⊢ Iff (Eq Bot.bot Bot.bot) (Eq I Bot.bot)
    -/
  · rename_i h
    /-
      case h_1
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I : LieIdeal R L
      h✝ : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
      x✝ : Nat
      h : Eq (LieAlgebra.derivedLengthOfIdeal R L I) 0
      ⊢ Iff (Eq Bot.bot Bot.bot) (Eq I Bot.bot)
    -/
    rw [derivedLength_zero] at h
    /-
      case h_1
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I : LieIdeal R L
      h✝ : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
      x✝ : Nat
      h : Eq I Bot.bot
      ⊢ Iff (Eq Bot.bot Bot.bot) (Eq I Bot.bot)
    -/
    rw [h]
    /-
      🎉 no goals
    -/
    /-
      case h_2
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I : LieIdeal R L
      h : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
      x✝ k✝ : Nat
      heq✝ : Eq (LieAlgebra.derivedLengthOfIdeal R L I) k✝.succ
      ⊢ Iff (Eq (LieAlgebra.derivedSeriesOfIdeal R L k✝ I) Bot.bot) (Eq I Bot.bot)
    -/
  · rename_i k h
    /-
      case h_2
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I : LieIdeal R L
      h✝ : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
      x✝ k : Nat
      h : Eq (LieAlgebra.derivedLengthOfIdeal R L I) k.succ
      ⊢ Iff (Eq (LieAlgebra.derivedSeriesOfIdeal R L k I) Bot.bot) (Eq I Bot.bot)
    -/
    obtain ⟨_, h₂⟩ := (derivedSeries_of_derivedLength_succ R L I k).mp h
    /-
      case h_2.intro
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I : LieIdeal R L
      h✝ : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
      x✝ k : Nat
      h : Eq (LieAlgebra.derivedLengthOfIdeal R L I) k.succ
      left✝ : IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.derivedSerie …
      h₂ : Ne (LieAlgebra.derivedSeriesOfIdeal R L k I) Bot.bot
      ⊢ Iff (Eq (LieAlgebra.derivedSeriesOfIdeal R L k I) Bot.bot) (Eq I Bot.bot)
    -/
    have h₃ : I ≠ ⊥ := by intro contra; apply h₂; rw [contra]; apply derivedSeries_of_bot_eq_bot
    /-
      case h_2.intro
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I : LieIdeal R L
      h✝ : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
      x✝ k : Nat
      h : Eq (LieAlgebra.derivedLengthOfIdeal R L I) k.succ
      left✝ : IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.derivedSerie …
      h₂ : Ne (LieAlgebra.derivedSeriesOfIdeal R L k I) Bot.bot
      h₃ : Ne I Bot.bot
      ⊢ Iff (Eq (LieAlgebra.derivedSeriesOfIdeal R L k I) Bot.bot) (Eq I Bot.bot)
    -/
    simp only [h₂, h₃]
    /-
      🎉 no goals
    -/


