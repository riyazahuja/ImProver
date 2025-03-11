/-- The lex order on multivariate power series.  -/
noncomputable def lexOrder (φ : MvPowerSeries σ R) : (WithTop (Lex (σ →₀ ℕ))) := by
  classical
  exact if h : φ = 0 then ⊤ else by
    have ne : Set.Nonempty (toLex '' φ.support) := by
      simp only [Set.image_nonempty, Function.support_nonempty_iff, ne_eq, h, not_false_eq_true]
    apply WithTop.some
    apply WellFounded.min _ (toLex '' φ.support) ne
    · exact Finsupp.instLTLex.lt
    · exact wellFounded_lt


theorem lexOrder_def_of_ne_zero {φ : MvPowerSeries σ R} (hφ : φ ≠ 0) :
    ∃ (ne : Set.Nonempty (toLex '' φ.support)),
      lexOrder φ = WithTop.some ((@wellFounded_lt (Lex (σ →₀ ℕ))
        (instLTLex) (Lex.wellFoundedLT)).min (toLex '' φ.support) ne) := by
  suffices ne : Set.Nonempty (toLex '' φ.support) by
    use ne
    unfold lexOrder
    simp only [dif_neg hφ]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    hφ : Ne φ 0
    ⊢ (Set.image (⇑toLex) (Function.support φ)).Nonempty
  -/
  simp only [Set.image_nonempty, Function.support_nonempty_iff, ne_eq, hφ, not_false_eq_true]
  /-
    🎉 no goals
  -/


@[simp]
theorem lexOrder_eq_top_iff_eq_zero (φ : MvPowerSeries σ R) :
    lexOrder φ = ⊤ ↔ φ = 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    ⊢ Iff (Eq φ.lexOrder Top.top) (Eq φ 0)
  -/
  unfold lexOrder
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    ⊢ Iff (Eq (dite (Eq φ 0) (fun h => Top.top) fun h => letFun ⋯ fun ne => ↑(⋯.mi …
  -/
  split_ifs with h
    /-
      case pos
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      h : Eq φ 0
      ⊢ Iff (Eq Top.top Top.top) (Eq φ 0)
    -/
  · simp only [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      h : Not (Eq φ 0)
      ⊢ Iff (Eq (letFun ⋯ fun ne => ↑(⋯.min (Set.image (⇑toLex) (Function.support φ) …
    -/
  · simp only [h, WithTop.coe_ne_top]
    /-
      🎉 no goals
    -/


theorem lexOrder_zero : lexOrder (0 : MvPowerSeries σ R) = ⊤ := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    ⊢ Eq (MvPowerSeries.lexOrder 0) Top.top
  -/
  unfold lexOrder
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    ⊢ Eq (dite (Eq 0 0) (fun h => Top.top) fun h => letFun ⋯ fun ne => ↑(⋯.min (Se …
  -/
  rw [dif_pos rfl]
  /-
    🎉 no goals
  -/


theorem exists_finsupp_eq_lexOrder_of_ne_zero {φ : MvPowerSeries σ R} (hφ : φ ≠ 0) :
    ∃ (d : σ →₀ ℕ), lexOrder φ = toLex d := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    hφ : Ne φ 0
    ⊢ Exists fun d => Eq φ.lexOrder ↑(toLex d)
  -/
  simp only [ne_eq, ← lexOrder_eq_top_iff_eq_zero, WithTop.ne_top_iff_exists] at hφ
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    hφ : Exists fun a => Eq (↑a) φ.lexOrder
    ⊢ Exists fun d => Eq φ.lexOrder ↑(toLex d)
  -/
  obtain ⟨p, hp⟩ := hφ
  /-
    case intro
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    p : Lex (Finsupp σ Nat)
    hp : Eq (↑p) φ.lexOrder
    ⊢ Exists fun d => Eq φ.lexOrder ↑(toLex d)
  -/
  exact ⟨ofLex p, by simp only [toLex_ofLex, hp]⟩
  /-
    🎉 no goals
  -/


theorem coeff_ne_zero_of_lexOrder {φ : MvPowerSeries σ R} {d : σ →₀ ℕ}
    (h : toLex d = lexOrder φ) : coeff R d φ ≠ 0 := by
  have hφ : φ ≠ 0 := by
    simp only [ne_eq, ← lexOrder_eq_top_iff_eq_zero, ← h, WithTop.coe_ne_top, not_false_eq_true]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : Eq (↑(toLex d)) φ.lexOrder
    hφ : Ne φ 0
    ⊢ Ne ((MvPowerSeries.coeff R d) φ) 0
  -/
  have hφ' := lexOrder_def_of_ne_zero hφ
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : Eq (↑(toLex d)) φ.lexOrder
    hφ : Ne φ 0
    hφ' : Exists fun ne => Eq φ.lexOrder ↑(⋯.min (Set.image (⇑toLex) (Function.sup …
    ⊢ Ne ((MvPowerSeries.coeff R d) φ) 0
  -/
  rcases hφ' with ⟨ne, hφ'⟩
  /-
    case intro
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : Eq (↑(toLex d)) φ.lexOrder
    hφ : Ne φ 0
    ne : (Set.image (⇑toLex) (Function.support φ)).Nonempty
    hφ' : Eq φ.lexOrder ↑(⋯.min (Set.image (⇑toLex) (Function.support φ)) ne)
    ⊢ Ne ((MvPowerSeries.coeff R d) φ) 0
  -/
  simp only [← h, WithTop.coe_eq_coe] at hφ'
  suffices toLex d ∈ toLex '' φ.support by
    simp only [Set.mem_image_equiv, toLex_symm_eq, ofLex_toLex, Function.mem_support, ne_eq] at this
    apply this
  /-
    case intro
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : Eq (↑(toLex d)) φ.lexOrder
    hφ : Ne φ 0
    ne : (Set.image (⇑toLex) (Function.support φ)).Nonempty
    hφ' : Eq (toLex d) (⋯.min (Set.image (⇑toLex) (Function.support φ)) ne)
    ⊢ Membership.mem (Set.image (⇑toLex) (Function.support φ)) (toLex d)
  -/
  rw [hφ']
  /-
    case intro
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : Eq (↑(toLex d)) φ.lexOrder
    hφ : Ne φ 0
    ne : (Set.image (⇑toLex) (Function.support φ)).Nonempty
    hφ' : Eq (toLex d) (⋯.min (Set.image (⇑toLex) (Function.support φ)) ne)
    ⊢ Membership.mem (Set.image (⇑toLex) (Function.support φ)) (⋯.min (Set.image ( …
  -/
  apply WellFounded.min_mem
  /-
    🎉 no goals
  -/


theorem coeff_eq_zero_of_lt_lexOrder {φ : MvPowerSeries σ R} {d : σ →₀ ℕ}
    (h : toLex d < lexOrder φ) : coeff R d φ = 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑(toLex d)) φ.lexOrder
    ⊢ Eq ((MvPowerSeries.coeff R d) φ) 0
  -/
  by_cases hφ : φ = 0
    /-
      case pos
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      d : Finsupp σ Nat
      h : LT.lt (↑(toLex d)) φ.lexOrder
      hφ : Eq φ 0
      ⊢ Eq ((MvPowerSeries.coeff R d) φ) 0
    -/
  · simp only [hφ, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      d : Finsupp σ Nat
      h : LT.lt (↑(toLex d)) φ.lexOrder
      hφ : Not (Eq φ 0)
      ⊢ Eq ((MvPowerSeries.coeff R d) φ) 0
    -/
  · rcases lexOrder_def_of_ne_zero hφ with ⟨ne, hφ'⟩
    /-
      case neg.intro
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      d : Finsupp σ Nat
      h : LT.lt (↑(toLex d)) φ.lexOrder
      hφ : Not (Eq φ 0)
      ne : (Set.image (⇑toLex) (Function.support φ)).Nonempty
      hφ' : Eq φ.lexOrder ↑(⋯.min (Set.image (⇑toLex) (Function.support φ)) ne)
      ⊢ Eq ((MvPowerSeries.coeff R d) φ) 0
    -/
    rw [hφ', WithTop.coe_lt_coe] at h
    /-
      case neg.intro
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      d : Finsupp σ Nat
      hφ : Not (Eq φ 0)
      ne : (Set.image (⇑toLex) (Function.support φ)).Nonempty
      h : LT.lt (toLex d) (⋯.min (Set.image (⇑toLex) (Function.support φ)) ne)
      hφ' : Eq φ.lexOrder ↑(⋯.min (Set.image (⇑toLex) (Function.support φ)) ne)
      ⊢ Eq ((MvPowerSeries.coeff R d) φ) 0
    -/
    by_contra h'
    /-
      case neg.intro
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      d : Finsupp σ Nat
      hφ : Not (Eq φ 0)
      ne : (Set.image (⇑toLex) (Function.support φ)).Nonempty
      h : LT.lt (toLex d) (⋯.min (Set.image (⇑toLex) (Function.support φ)) ne)
      hφ' : Eq φ.lexOrder ↑(⋯.min (Set.image (⇑toLex) (Function.support φ)) ne)
      h' : Not (Eq ((MvPowerSeries.coeff R d) φ) 0)
      ⊢ False
    -/
    exact WellFounded.not_lt_min _ (toLex '' φ.support) ne (Set.mem_image_equiv.mpr h') h
    /-
      🎉 no goals
    -/


theorem lexOrder_le_of_coeff_ne_zero {φ : MvPowerSeries σ R} {d : σ →₀ ℕ}
    (h : coeff R d φ ≠ 0) : lexOrder φ ≤ toLex d := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : Ne ((MvPowerSeries.coeff R d) φ) 0
    ⊢ LE.le φ.lexOrder ↑(toLex d)
  -/
  rw [← not_lt]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : Ne ((MvPowerSeries.coeff R d) φ) 0
    ⊢ Not (LT.lt (↑(toLex d)) φ.lexOrder)
  -/
  intro h'
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : Ne ((MvPowerSeries.coeff R d) φ) 0
    h' : LT.lt (↑(toLex d)) φ.lexOrder
    ⊢ False
  -/
  exact h (coeff_eq_zero_of_lt_lexOrder h')
  /-
    🎉 no goals
  -/


theorem le_lexOrder_iff {φ : MvPowerSeries σ R} {w : WithTop (Lex (σ →₀ ℕ))} :
    w ≤ lexOrder φ ↔ (∀ (d : σ →₀ ℕ) (_ : toLex d < w), coeff R d φ = 0) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ : MvPowerSeries σ R
    w : WithTop (Lex (Finsupp σ Nat))
    ⊢ Iff (LE.le w φ.lexOrder) (∀ (d : Finsupp σ Nat), LT.lt (↑(toLex d)) w → Eq ( …
  -/
  constructor
    /-
      case mp
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      w : WithTop (Lex (Finsupp σ Nat))
      ⊢ LE.le w φ.lexOrder → ∀ (d : Finsupp σ Nat), LT.lt (↑(toLex d)) w → Eq ((MvPo …
    -/
  · intro h d hd
    /-
      case mp
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      w : WithTop (Lex (Finsupp σ Nat))
      h : LE.le w φ.lexOrder
      d : Finsupp σ Nat
      hd : LT.lt (↑(toLex d)) w
      ⊢ Eq ((MvPowerSeries.coeff R d) φ) 0
    -/
    apply coeff_eq_zero_of_lt_lexOrder
    /-
      case mp.h
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      w : WithTop (Lex (Finsupp σ Nat))
      h : LE.le w φ.lexOrder
      d : Finsupp σ Nat
      hd : LT.lt (↑(toLex d)) w
      ⊢ LT.lt (↑(toLex d)) φ.lexOrder
    -/
    exact lt_of_lt_of_le hd h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      w : WithTop (Lex (Finsupp σ Nat))
      ⊢ (∀ (d : Finsupp σ Nat), LT.lt (↑(toLex d)) w → Eq ((MvPowerSeries.coeff R d) …
    -/
  · intro h
    /-
      case mpr
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      w : WithTop (Lex (Finsupp σ Nat))
      h : ∀ (d : Finsupp σ Nat), LT.lt (↑(toLex d)) w → Eq ((MvPowerSeries.coeff R d …
      ⊢ LE.le w φ.lexOrder
    -/
    rw [← not_lt]
    /-
      case mpr
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      w : WithTop (Lex (Finsupp σ Nat))
      h : ∀ (d : Finsupp σ Nat), LT.lt (↑(toLex d)) w → Eq ((MvPowerSeries.coeff R d …
      ⊢ Not (LT.lt φ.lexOrder w)
    -/
    intro h'
    have hφ : φ ≠ 0 := by
      rw [ne_eq, ← lexOrder_eq_top_iff_eq_zero]
      exact ne_top_of_lt h'
    /-
      case mpr
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      w : WithTop (Lex (Finsupp σ Nat))
      h : ∀ (d : Finsupp σ Nat), LT.lt (↑(toLex d)) w → Eq ((MvPowerSeries.coeff R d …
      h' : LT.lt φ.lexOrder w
      hφ : Ne φ 0
      ⊢ False
    -/
    obtain ⟨d, hd⟩ := exists_finsupp_eq_lexOrder_of_ne_zero hφ
    /-
      case mpr.intro
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      w : WithTop (Lex (Finsupp σ Nat))
      h : ∀ (d : Finsupp σ Nat), LT.lt (↑(toLex d)) w → Eq ((MvPowerSeries.coeff R d …
      h' : LT.lt φ.lexOrder w
      hφ : Ne φ 0
      d : Finsupp σ Nat
      hd : Eq φ.lexOrder ↑(toLex d)
      ⊢ False
    -/
    refine coeff_ne_zero_of_lexOrder hd.symm (h d ?_)
    /-
      case mpr.intro
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ : MvPowerSeries σ R
      w : WithTop (Lex (Finsupp σ Nat))
      h : ∀ (d : Finsupp σ Nat), LT.lt (↑(toLex d)) w → Eq ((MvPowerSeries.coeff R d …
      h' : LT.lt φ.lexOrder w
      hφ : Ne φ 0
      d : Finsupp σ Nat
      hd : Eq φ.lexOrder ↑(toLex d)
      ⊢ LT.lt (↑(toLex d)) w
    -/
    rwa [← hd]
    /-
      🎉 no goals
    -/


theorem min_lexOrder_le {φ ψ : MvPowerSeries σ R} :
    min (lexOrder φ) (lexOrder ψ) ≤ lexOrder (φ + ψ)  := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    ⊢ LE.le (Min.min φ.lexOrder ψ.lexOrder) (HAdd.hAdd φ ψ).lexOrder
  -/
  rw [le_lexOrder_iff]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    ⊢ ∀ (d : Finsupp σ Nat), LT.lt (↑(toLex d)) (Min.min φ.lexOrder ψ.lexOrder) →  …
  -/
  intro d hd
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : LT.lt (↑(toLex d)) (Min.min φ.lexOrder ψ.lexOrder)
    ⊢ Eq ((MvPowerSeries.coeff R d) (HAdd.hAdd φ ψ)) 0
  -/
  simp only [lt_min_iff] at hd
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : And (LT.lt (↑(toLex d)) φ.lexOrder) (LT.lt (↑(toLex d)) ψ.lexOrder)
    ⊢ Eq ((MvPowerSeries.coeff R d) (HAdd.hAdd φ ψ)) 0
  -/
  rw [map_add, coeff_eq_zero_of_lt_lexOrder hd.1, coeff_eq_zero_of_lt_lexOrder hd.2, add_zero]
  /-
    🎉 no goals
  -/


theorem coeff_mul_of_add_lexOrder {φ ψ : MvPowerSeries σ R}
    {p q : σ →₀ ℕ} (hp : lexOrder φ = toLex p) (hq : lexOrder ψ = toLex q) :
    coeff R (p + q) (φ * ψ) = coeff R p φ * coeff R q ψ := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    p q : Finsupp σ Nat
    hp : Eq φ.lexOrder ↑(toLex p)
    hq : Eq ψ.lexOrder ↑(toLex q)
    ⊢ Eq ((MvPowerSeries.coeff R (HAdd.hAdd p q)) (HMul.hMul φ ψ)) (HMul.hMul ((Mv …
  -/
  rw [coeff_mul]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    p q : Finsupp σ Nat
    hp : Eq φ.lexOrder ↑(toLex p)
    hq : Eq ψ.lexOrder ↑(toLex q)
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p q)).sum fun p => HMul. …
  -/
  apply Finset.sum_eq_single (⟨p, q⟩ : (σ →₀ ℕ) × (σ →₀ ℕ))
    /-
      case h₀
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ ψ : MvPowerSeries σ R
      p q : Finsupp σ Nat
      hp : Eq φ.lexOrder ↑(toLex p)
      hq : Eq ψ.lexOrder ↑(toLex q)
      ⊢ ∀ (b : Prod (Finsupp σ Nat) (Finsupp σ Nat)), Membership.mem (Finset.HasAnti …
    -/
  · rintro ⟨u, v⟩ h h'
    /-
      case h₀.mk
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ ψ : MvPowerSeries σ R
      p q : Finsupp σ Nat
      hp : Eq φ.lexOrder ↑(toLex p)
      hq : Eq ψ.lexOrder ↑(toLex q)
      u v : Finsupp σ Nat
      h : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p q)) { fst …
      h' : Ne { fst := u, snd := v } { fst := p, snd := q }
      ⊢ Eq (HMul.hMul ((MvPowerSeries.coeff R { fst := u, snd := v }.1) φ) ((MvPower …
    -/
    simp only [Finset.mem_antidiagonal] at h
    /-
      case h₀.mk
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ ψ : MvPowerSeries σ R
      p q : Finsupp σ Nat
      hp : Eq φ.lexOrder ↑(toLex p)
      hq : Eq ψ.lexOrder ↑(toLex q)
      u v : Finsupp σ Nat
      h' : Ne { fst := u, snd := v } { fst := p, snd := q }
      h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
      ⊢ Eq (HMul.hMul ((MvPowerSeries.coeff R { fst := u, snd := v }.1) φ) ((MvPower …
    -/
    simp only
    /-
      case h₀.mk
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ ψ : MvPowerSeries σ R
      p q : Finsupp σ Nat
      hp : Eq φ.lexOrder ↑(toLex p)
      hq : Eq ψ.lexOrder ↑(toLex q)
      u v : Finsupp σ Nat
      h' : Ne { fst := u, snd := v } { fst := p, snd := q }
      h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
      ⊢ Eq (HMul.hMul ((MvPowerSeries.coeff R u) φ) ((MvPowerSeries.coeff R v) ψ)) 0
    -/
    by_cases hu : toLex u < toLex p
      /-
        case pos
        σ : Type u_1
        R : Type u_2
        inst✝² : Semiring R
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        φ ψ : MvPowerSeries σ R
        p q : Finsupp σ Nat
        hp : Eq φ.lexOrder ↑(toLex p)
        hq : Eq ψ.lexOrder ↑(toLex q)
        u v : Finsupp σ Nat
        h' : Ne { fst := u, snd := v } { fst := p, snd := q }
        h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
        hu : LT.lt (toLex u) (toLex p)
        ⊢ Eq (HMul.hMul ((MvPowerSeries.coeff R u) φ) ((MvPowerSeries.coeff R v) ψ)) 0
      -/
    · rw [coeff_eq_zero_of_lt_lexOrder (R := R) (d := u), zero_mul]
      /-
        case pos
        σ : Type u_1
        R : Type u_2
        inst✝² : Semiring R
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        φ ψ : MvPowerSeries σ R
        p q : Finsupp σ Nat
        hp : Eq φ.lexOrder ↑(toLex p)
        hq : Eq ψ.lexOrder ↑(toLex q)
        u v : Finsupp σ Nat
        h' : Ne { fst := u, snd := v } { fst := p, snd := q }
        h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
        hu : LT.lt (toLex u) (toLex p)
        ⊢ LT.lt (↑(toLex u)) φ.lexOrder
      -/
      simp only [hp, WithTop.coe_lt_coe, hu]
      /-
        🎉 no goals
      -/
      /-
        case neg
        σ : Type u_1
        R : Type u_2
        inst✝² : Semiring R
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        φ ψ : MvPowerSeries σ R
        p q : Finsupp σ Nat
        hp : Eq φ.lexOrder ↑(toLex p)
        hq : Eq ψ.lexOrder ↑(toLex q)
        u v : Finsupp σ Nat
        h' : Ne { fst := u, snd := v } { fst := p, snd := q }
        h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
        hu : Not (LT.lt (toLex u) (toLex p))
        ⊢ Eq (HMul.hMul ((MvPowerSeries.coeff R u) φ) ((MvPowerSeries.coeff R v) ψ)) 0
      -/
    · rw [coeff_eq_zero_of_lt_lexOrder (d := v), mul_zero]
      /-
        case neg
        σ : Type u_1
        R : Type u_2
        inst✝² : Semiring R
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        φ ψ : MvPowerSeries σ R
        p q : Finsupp σ Nat
        hp : Eq φ.lexOrder ↑(toLex p)
        hq : Eq ψ.lexOrder ↑(toLex q)
        u v : Finsupp σ Nat
        h' : Ne { fst := u, snd := v } { fst := p, snd := q }
        h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
        hu : Not (LT.lt (toLex u) (toLex p))
        ⊢ LT.lt (↑(toLex v)) ψ.lexOrder
      -/
      simp only [hq, WithTop.coe_lt_coe, ← not_le]
      /-
        case neg
        σ : Type u_1
        R : Type u_2
        inst✝² : Semiring R
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        φ ψ : MvPowerSeries σ R
        p q : Finsupp σ Nat
        hp : Eq φ.lexOrder ↑(toLex p)
        hq : Eq ψ.lexOrder ↑(toLex q)
        u v : Finsupp σ Nat
        h' : Ne { fst := u, snd := v } { fst := p, snd := q }
        h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
        hu : Not (LT.lt (toLex u) (toLex p))
        ⊢ Not (LE.le ↑(toLex q) ↑(toLex v))
      -/
      simp only [not_lt] at hu
      /-
        case neg
        σ : Type u_1
        R : Type u_2
        inst✝² : Semiring R
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        φ ψ : MvPowerSeries σ R
        p q : Finsupp σ Nat
        hp : Eq φ.lexOrder ↑(toLex p)
        hq : Eq ψ.lexOrder ↑(toLex q)
        u v : Finsupp σ Nat
        h' : Ne { fst := u, snd := v } { fst := p, snd := q }
        h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
        hu : LE.le (toLex p) (toLex u)
        ⊢ Not (LE.le ↑(toLex q) ↑(toLex v))
      -/
      intro hv
      /-
        case neg
        σ : Type u_1
        R : Type u_2
        inst✝² : Semiring R
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        φ ψ : MvPowerSeries σ R
        p q : Finsupp σ Nat
        hp : Eq φ.lexOrder ↑(toLex p)
        hq : Eq ψ.lexOrder ↑(toLex q)
        u v : Finsupp σ Nat
        h' : Ne { fst := u, snd := v } { fst := p, snd := q }
        h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
        hu : LE.le (toLex p) (toLex u)
        hv : LE.le ↑(toLex q) ↑(toLex v)
        ⊢ False
      -/
      simp only [WithTop.coe_le_coe] at hv
      /-
        case neg
        σ : Type u_1
        R : Type u_2
        inst✝² : Semiring R
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        φ ψ : MvPowerSeries σ R
        p q : Finsupp σ Nat
        hp : Eq φ.lexOrder ↑(toLex p)
        hq : Eq ψ.lexOrder ↑(toLex q)
        u v : Finsupp σ Nat
        h' : Ne { fst := u, snd := v } { fst := p, snd := q }
        h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
        hu : LE.le (toLex p) (toLex u)
        hv : LE.le (toLex q) (toLex v)
        ⊢ False
      -/
      apply h'
      /-
        case neg
        σ : Type u_1
        R : Type u_2
        inst✝² : Semiring R
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        φ ψ : MvPowerSeries σ R
        p q : Finsupp σ Nat
        hp : Eq φ.lexOrder ↑(toLex p)
        hq : Eq ψ.lexOrder ↑(toLex q)
        u v : Finsupp σ Nat
        h' : Ne { fst := u, snd := v } { fst := p, snd := q }
        h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
        hu : LE.le (toLex p) (toLex u)
        hv : LE.le (toLex q) (toLex v)
        ⊢ Eq { fst := u, snd := v } { fst := p, snd := q }
      -/
      simp only [Prod.mk.injEq]
      /-
        case neg
        σ : Type u_1
        R : Type u_2
        inst✝² : Semiring R
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        φ ψ : MvPowerSeries σ R
        p q : Finsupp σ Nat
        hp : Eq φ.lexOrder ↑(toLex p)
        hq : Eq ψ.lexOrder ↑(toLex q)
        u v : Finsupp σ Nat
        h' : Ne { fst := u, snd := v } { fst := p, snd := q }
        h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
        hu : LE.le (toLex p) (toLex u)
        hv : LE.le (toLex q) (toLex v)
        ⊢ And (Eq u p) (Eq v q)
      -/
      constructor
        /-
          case neg.left
          σ : Type u_1
          R : Type u_2
          inst✝² : Semiring R
          inst✝¹ : LinearOrder σ
          inst✝ : WellFoundedGT σ
          φ ψ : MvPowerSeries σ R
          p q : Finsupp σ Nat
          hp : Eq φ.lexOrder ↑(toLex p)
          hq : Eq ψ.lexOrder ↑(toLex q)
          u v : Finsupp σ Nat
          h' : Ne { fst := u, snd := v } { fst := p, snd := q }
          h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
          hu : LE.le (toLex p) (toLex u)
          hv : LE.le (toLex q) (toLex v)
          ⊢ Eq u p
        -/
      · apply toLex.injective
        /-
          case neg.left.a
          σ : Type u_1
          R : Type u_2
          inst✝² : Semiring R
          inst✝¹ : LinearOrder σ
          inst✝ : WellFoundedGT σ
          φ ψ : MvPowerSeries σ R
          p q : Finsupp σ Nat
          hp : Eq φ.lexOrder ↑(toLex p)
          hq : Eq ψ.lexOrder ↑(toLex q)
          u v : Finsupp σ Nat
          h' : Ne { fst := u, snd := v } { fst := p, snd := q }
          h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
          hu : LE.le (toLex p) (toLex u)
          hv : LE.le (toLex q) (toLex v)
          ⊢ Eq (toLex u) (toLex p)
        -/
        apply Or.resolve_right (eq_or_gt_of_le hu)
        /-
          case neg.left.a
          σ : Type u_1
          R : Type u_2
          inst✝² : Semiring R
          inst✝¹ : LinearOrder σ
          inst✝ : WellFoundedGT σ
          φ ψ : MvPowerSeries σ R
          p q : Finsupp σ Nat
          hp : Eq φ.lexOrder ↑(toLex p)
          hq : Eq ψ.lexOrder ↑(toLex q)
          u v : Finsupp σ Nat
          h' : Ne { fst := u, snd := v } { fst := p, snd := q }
          h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
          hu : LE.le (toLex p) (toLex u)
          hv : LE.le (toLex q) (toLex v)
          ⊢ Not (LT.lt (toLex p) (toLex u))
        -/
        intro hu'
        /-
          case neg.left.a
          σ : Type u_1
          R : Type u_2
          inst✝² : Semiring R
          inst✝¹ : LinearOrder σ
          inst✝ : WellFoundedGT σ
          φ ψ : MvPowerSeries σ R
          p q : Finsupp σ Nat
          hp : Eq φ.lexOrder ↑(toLex p)
          hq : Eq ψ.lexOrder ↑(toLex q)
          u v : Finsupp σ Nat
          h' : Ne { fst := u, snd := v } { fst := p, snd := q }
          h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
          hu : LE.le (toLex p) (toLex u)
          hv : LE.le (toLex q) (toLex v)
          hu' : LT.lt (toLex p) (toLex u)
          ⊢ False
        -/
        exact not_le.mpr (add_lt_add_of_lt_of_le hu' hv) (le_of_eq h)
        /-
          🎉 no goals
        -/
        /-
          case neg.right
          σ : Type u_1
          R : Type u_2
          inst✝² : Semiring R
          inst✝¹ : LinearOrder σ
          inst✝ : WellFoundedGT σ
          φ ψ : MvPowerSeries σ R
          p q : Finsupp σ Nat
          hp : Eq φ.lexOrder ↑(toLex p)
          hq : Eq ψ.lexOrder ↑(toLex q)
          u v : Finsupp σ Nat
          h' : Ne { fst := u, snd := v } { fst := p, snd := q }
          h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
          hu : LE.le (toLex p) (toLex u)
          hv : LE.le (toLex q) (toLex v)
          ⊢ Eq v q
        -/
      · apply toLex.injective
        /-
          case neg.right.a
          σ : Type u_1
          R : Type u_2
          inst✝² : Semiring R
          inst✝¹ : LinearOrder σ
          inst✝ : WellFoundedGT σ
          φ ψ : MvPowerSeries σ R
          p q : Finsupp σ Nat
          hp : Eq φ.lexOrder ↑(toLex p)
          hq : Eq ψ.lexOrder ↑(toLex q)
          u v : Finsupp σ Nat
          h' : Ne { fst := u, snd := v } { fst := p, snd := q }
          h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
          hu : LE.le (toLex p) (toLex u)
          hv : LE.le (toLex q) (toLex v)
          ⊢ Eq (toLex v) (toLex q)
        -/
        apply Or.resolve_right (eq_or_gt_of_le hv)
        /-
          case neg.right.a
          σ : Type u_1
          R : Type u_2
          inst✝² : Semiring R
          inst✝¹ : LinearOrder σ
          inst✝ : WellFoundedGT σ
          φ ψ : MvPowerSeries σ R
          p q : Finsupp σ Nat
          hp : Eq φ.lexOrder ↑(toLex p)
          hq : Eq ψ.lexOrder ↑(toLex q)
          u v : Finsupp σ Nat
          h' : Ne { fst := u, snd := v } { fst := p, snd := q }
          h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
          hu : LE.le (toLex p) (toLex u)
          hv : LE.le (toLex q) (toLex v)
          ⊢ Not (LT.lt (toLex q) (toLex v))
        -/
        intro hv'
        /-
          case neg.right.a
          σ : Type u_1
          R : Type u_2
          inst✝² : Semiring R
          inst✝¹ : LinearOrder σ
          inst✝ : WellFoundedGT σ
          φ ψ : MvPowerSeries σ R
          p q : Finsupp σ Nat
          hp : Eq φ.lexOrder ↑(toLex p)
          hq : Eq ψ.lexOrder ↑(toLex q)
          u v : Finsupp σ Nat
          h' : Ne { fst := u, snd := v } { fst := p, snd := q }
          h : Eq (HAdd.hAdd u v) (HAdd.hAdd p q)
          hu : LE.le (toLex p) (toLex u)
          hv : LE.le (toLex q) (toLex v)
          hv' : LT.lt (toLex q) (toLex v)
          ⊢ False
        -/
        exact not_le.mpr (add_lt_add_of_le_of_lt hu hv') (le_of_eq h)
        /-
          🎉 no goals
        -/
    /-
      case h₁
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ ψ : MvPowerSeries σ R
      p q : Finsupp σ Nat
      hp : Eq φ.lexOrder ↑(toLex p)
      hq : Eq ψ.lexOrder ↑(toLex q)
      ⊢ Not (Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p q)) {  …
    -/
  · intro h
    /-
      case h₁
      σ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      φ ψ : MvPowerSeries σ R
      p q : Finsupp σ Nat
      hp : Eq φ.lexOrder ↑(toLex p)
      hq : Eq ψ.lexOrder ↑(toLex q)
      h : Not (Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p q))  …
      ⊢ Eq (HMul.hMul ((MvPowerSeries.coeff R { fst := p, snd := q }.1) φ) ((MvPower …
    -/
    simp only [Finset.mem_antidiagonal, not_true_eq_false] at h
    /-
      🎉 no goals
    -/


theorem le_lexOrder_mul (φ ψ : MvPowerSeries σ R) :
    lexOrder φ + lexOrder ψ ≤ lexOrder (φ * ψ) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    ⊢ LE.le (HAdd.hAdd φ.lexOrder ψ.lexOrder) (HMul.hMul φ ψ).lexOrder
  -/
  rw [le_lexOrder_iff]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    ⊢ ∀ (d : Finsupp σ Nat), LT.lt (↑(toLex d)) (HAdd.hAdd φ.lexOrder ψ.lexOrder)  …
  -/
  intro d hd
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : LT.lt (↑(toLex d)) (HAdd.hAdd φ.lexOrder ψ.lexOrder)
    ⊢ Eq ((MvPowerSeries.coeff R d) (HMul.hMul φ ψ)) 0
  -/
  rw [coeff_mul]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : LT.lt (↑(toLex d)) (HAdd.hAdd φ.lexOrder ψ.lexOrder)
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal d).sum fun p => HMul.hMul ((MvPower …
  -/
  apply Finset.sum_eq_zero
  /-
    case h
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : LT.lt (↑(toLex d)) (HAdd.hAdd φ.lexOrder ψ.lexOrder)
    ⊢ ∀ (x : Prod (Finsupp σ Nat) (Finsupp σ Nat)), Membership.mem (Finset.HasAnti …
  -/
  rintro ⟨u, v⟩ h
  /-
    case h.mk
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : LT.lt (↑(toLex d)) (HAdd.hAdd φ.lexOrder ψ.lexOrder)
    u v : Finsupp σ Nat
    h : Membership.mem (Finset.HasAntidiagonal.antidiagonal d) { fst := u, snd :=  …
    ⊢ Eq (HMul.hMul ((MvPowerSeries.coeff R { fst := u, snd := v }.1) φ) ((MvPower …
  -/
  simp only [Finset.mem_antidiagonal] at h
  /-
    case h.mk
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : LT.lt (↑(toLex d)) (HAdd.hAdd φ.lexOrder ψ.lexOrder)
    u v : Finsupp σ Nat
    h : Eq (HAdd.hAdd u v) d
    ⊢ Eq (HMul.hMul ((MvPowerSeries.coeff R { fst := u, snd := v }.1) φ) ((MvPower …
  -/
  simp only
  suffices toLex u < lexOrder φ ∨ toLex v < lexOrder ψ by
    rcases this with (hu | hv)
    · rw [coeff_eq_zero_of_lt_lexOrder hu, zero_mul]
    · rw [coeff_eq_zero_of_lt_lexOrder hv, mul_zero]
  /-
    case h.mk
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : LT.lt (↑(toLex d)) (HAdd.hAdd φ.lexOrder ψ.lexOrder)
    u v : Finsupp σ Nat
    h : Eq (HAdd.hAdd u v) d
    ⊢ Or (LT.lt (↑(toLex u)) φ.lexOrder) (LT.lt (↑(toLex v)) ψ.lexOrder)
  -/
  rw [or_iff_not_imp_left, not_lt, ← not_le]
  /-
    case h.mk
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : LT.lt (↑(toLex d)) (HAdd.hAdd φ.lexOrder ψ.lexOrder)
    u v : Finsupp σ Nat
    h : Eq (HAdd.hAdd u v) d
    ⊢ LE.le φ.lexOrder ↑(toLex u) → Not (LE.le ψ.lexOrder ↑(toLex v))
  -/
  intro hu hv
  /-
    case h.mk
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : LT.lt (↑(toLex d)) (HAdd.hAdd φ.lexOrder ψ.lexOrder)
    u v : Finsupp σ Nat
    h : Eq (HAdd.hAdd u v) d
    hu : LE.le φ.lexOrder ↑(toLex u)
    hv : LE.le ψ.lexOrder ↑(toLex v)
    ⊢ False
  -/
  rw [← not_le] at hd
  /-
    case h.mk
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : Not (LE.le (HAdd.hAdd φ.lexOrder ψ.lexOrder) ↑(toLex d))
    u v : Finsupp σ Nat
    h : Eq (HAdd.hAdd u v) d
    hu : LE.le φ.lexOrder ↑(toLex u)
    hv : LE.le ψ.lexOrder ↑(toLex v)
    ⊢ False
  -/
  apply hd
  /-
    case h.mk
    σ : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    φ ψ : MvPowerSeries σ R
    d : Finsupp σ Nat
    hd : Not (LE.le (HAdd.hAdd φ.lexOrder ψ.lexOrder) ↑(toLex d))
    u v : Finsupp σ Nat
    h : Eq (HAdd.hAdd u v) d
    hu : LE.le φ.lexOrder ↑(toLex u)
    hv : LE.le ψ.lexOrder ↑(toLex v)
    ⊢ LE.le (HAdd.hAdd φ.lexOrder ψ.lexOrder) ↑(toLex d)
  -/
  simp only [← h, toLex_add, WithTop.coe_add, add_le_add hu hv]
  /-
    🎉 no goals
  -/


alias lexOrder_mul_ge := le_lexOrder_mul


theorem lexOrder_mul [NoZeroDivisors R] (φ ψ : MvPowerSeries σ R) :
    lexOrder (φ * ψ) = lexOrder φ + lexOrder ψ := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : LinearOrder σ
    inst✝¹ : WellFoundedGT σ
    inst✝ : NoZeroDivisors R
    φ ψ : MvPowerSeries σ R
    ⊢ Eq (HMul.hMul φ ψ).lexOrder (HAdd.hAdd φ.lexOrder ψ.lexOrder)
  -/
  by_cases hφ : φ = 0
    /-
      case pos
      σ : Type u_1
      R : Type u_2
      inst✝³ : Semiring R
      inst✝² : LinearOrder σ
      inst✝¹ : WellFoundedGT σ
      inst✝ : NoZeroDivisors R
      φ ψ : MvPowerSeries σ R
      hφ : Eq φ 0
      ⊢ Eq (HMul.hMul φ ψ).lexOrder (HAdd.hAdd φ.lexOrder ψ.lexOrder)
    -/
  · simp only [hφ, zero_mul, lexOrder_zero, top_add]
    /-
      🎉 no goals
    -/
  /-
    case neg
    σ : Type u_1
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : LinearOrder σ
    inst✝¹ : WellFoundedGT σ
    inst✝ : NoZeroDivisors R
    φ ψ : MvPowerSeries σ R
    hφ : Not (Eq φ 0)
    ⊢ Eq (HMul.hMul φ ψ).lexOrder (HAdd.hAdd φ.lexOrder ψ.lexOrder)
  -/
  by_cases hψ : ψ = 0
    /-
      case pos
      σ : Type u_1
      R : Type u_2
      inst✝³ : Semiring R
      inst✝² : LinearOrder σ
      inst✝¹ : WellFoundedGT σ
      inst✝ : NoZeroDivisors R
      φ ψ : MvPowerSeries σ R
      hφ : Not (Eq φ 0)
      hψ : Eq ψ 0
      ⊢ Eq (HMul.hMul φ ψ).lexOrder (HAdd.hAdd φ.lexOrder ψ.lexOrder)
    -/
  · simp only [hψ, mul_zero, lexOrder_zero, add_top]
    /-
      🎉 no goals
    -/
  /-
    case neg
    σ : Type u_1
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : LinearOrder σ
    inst✝¹ : WellFoundedGT σ
    inst✝ : NoZeroDivisors R
    φ ψ : MvPowerSeries σ R
    hφ : Not (Eq φ 0)
    hψ : Not (Eq ψ 0)
    ⊢ Eq (HMul.hMul φ ψ).lexOrder (HAdd.hAdd φ.lexOrder ψ.lexOrder)
  -/
  rcases exists_finsupp_eq_lexOrder_of_ne_zero hφ with ⟨p, hp⟩
  /-
    case neg.intro
    σ : Type u_1
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : LinearOrder σ
    inst✝¹ : WellFoundedGT σ
    inst✝ : NoZeroDivisors R
    φ ψ : MvPowerSeries σ R
    hφ : Not (Eq φ 0)
    hψ : Not (Eq ψ 0)
    p : Finsupp σ Nat
    hp : Eq φ.lexOrder ↑(toLex p)
    ⊢ Eq (HMul.hMul φ ψ).lexOrder (HAdd.hAdd φ.lexOrder ψ.lexOrder)
  -/
  rcases exists_finsupp_eq_lexOrder_of_ne_zero hψ with ⟨q, hq⟩
  /-
    case neg.intro.intro
    σ : Type u_1
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : LinearOrder σ
    inst✝¹ : WellFoundedGT σ
    inst✝ : NoZeroDivisors R
    φ ψ : MvPowerSeries σ R
    hφ : Not (Eq φ 0)
    hψ : Not (Eq ψ 0)
    p : Finsupp σ Nat
    hp : Eq φ.lexOrder ↑(toLex p)
    q : Finsupp σ Nat
    hq : Eq ψ.lexOrder ↑(toLex q)
    ⊢ Eq (HMul.hMul φ ψ).lexOrder (HAdd.hAdd φ.lexOrder ψ.lexOrder)
  -/
  apply le_antisymm _ (lexOrder_mul_ge φ ψ)
  /-
    σ : Type u_1
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : LinearOrder σ
    inst✝¹ : WellFoundedGT σ
    inst✝ : NoZeroDivisors R
    φ ψ : MvPowerSeries σ R
    hφ : Not (Eq φ 0)
    hψ : Not (Eq ψ 0)
    p : Finsupp σ Nat
    hp : Eq φ.lexOrder ↑(toLex p)
    q : Finsupp σ Nat
    hq : Eq ψ.lexOrder ↑(toLex q)
    ⊢ LE.le (HMul.hMul φ ψ).lexOrder (HAdd.hAdd φ.lexOrder ψ.lexOrder)
  -/
  rw [hp, hq]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : LinearOrder σ
    inst✝¹ : WellFoundedGT σ
    inst✝ : NoZeroDivisors R
    φ ψ : MvPowerSeries σ R
    hφ : Not (Eq φ 0)
    hψ : Not (Eq ψ 0)
    p : Finsupp σ Nat
    hp : Eq φ.lexOrder ↑(toLex p)
    q : Finsupp σ Nat
    hq : Eq ψ.lexOrder ↑(toLex q)
    ⊢ LE.le (HMul.hMul φ ψ).lexOrder (HAdd.hAdd ↑(toLex p) ↑(toLex q))
  -/
  apply lexOrder_le_of_coeff_ne_zero (d := p + q)
  /-
    σ : Type u_1
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : LinearOrder σ
    inst✝¹ : WellFoundedGT σ
    inst✝ : NoZeroDivisors R
    φ ψ : MvPowerSeries σ R
    hφ : Not (Eq φ 0)
    hψ : Not (Eq ψ 0)
    p : Finsupp σ Nat
    hp : Eq φ.lexOrder ↑(toLex p)
    q : Finsupp σ Nat
    hq : Eq ψ.lexOrder ↑(toLex q)
    ⊢ Ne ((MvPowerSeries.coeff R (HAdd.hAdd p q)) (HMul.hMul φ ψ)) 0
  -/
  rw [coeff_mul_of_add_lexOrder hp hq, mul_ne_zero_iff]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : LinearOrder σ
    inst✝¹ : WellFoundedGT σ
    inst✝ : NoZeroDivisors R
    φ ψ : MvPowerSeries σ R
    hφ : Not (Eq φ 0)
    hψ : Not (Eq ψ 0)
    p : Finsupp σ Nat
    hp : Eq φ.lexOrder ↑(toLex p)
    q : Finsupp σ Nat
    hq : Eq ψ.lexOrder ↑(toLex q)
    ⊢ And (Ne ((MvPowerSeries.coeff R p) φ) 0) (Ne ((MvPowerSeries.coeff R q) ψ) 0)
  -/
  exact ⟨coeff_ne_zero_of_lexOrder hp.symm, coeff_ne_zero_of_lexOrder hq.symm⟩
  /-
    🎉 no goals
  -/


