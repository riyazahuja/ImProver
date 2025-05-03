/-- The house of an algebraic number as the norm of its image by the canonical embedding. -/
def house (α : K) : ℝ := ‖canonicalEmbedding K α‖


/-- The house is the largest of the modulus of the conjugates of an algebraic number. -/
theorem house_eq_sup' (α : K) :
    house α = univ.sup' univ_nonempty (fun φ : K →+* ℂ ↦ ‖φ α‖₊) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : K
    ⊢ Eq (NumberField.house α) ↑(Finset.univ.sup' ⋯ fun φ => NNNorm.nnnorm (φ α))
  -/
  rw [house, ← coe_nnnorm, nnnorm_eq, ← sup'_eq_sup univ_nonempty]
  /-
    🎉 no goals
  -/


theorem house_sum_le_sum_house {ι : Type*} (s : Finset ι) (α : ι → K) :
    house (∑ i ∈ s, α i) ≤ ∑ i ∈ s, house (α i) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ι : Type u_2
    s : Finset ι
    α : ι → K
    ⊢ LE.le (NumberField.house (s.sum fun i => α i)) (s.sum fun i => NumberField.h …
  -/
  simp only [house, map_sum]; apply norm_sum_le_of_le; intros; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem house_nonneg (α : K) : 0 ≤ house α := norm_nonneg _


theorem house_mul_le (α β : K) : house (α * β) ≤ house α * house β := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α β : K
    ⊢ LE.le (NumberField.house (HMul.hMul α β)) (HMul.hMul (NumberField.house α) ( …
  -/
  simp only [house, _root_.map_mul]; apply norm_mul_le
                                     /-
                                       🎉 no goals
                                     -/


@[simp] theorem house_intCast (x : ℤ) : house (x : K) = |x| := by
  simp only [house, map_intCast, Pi.intCast_def, pi_norm_const, Complex.norm_eq_abs,
    Complex.abs_intCast, Int.cast_abs]


/-- `c` is defined as the product of the maximum absolute
  value of the entries of the inverse of the matrix `basisMatrix` and  `finrank ℚ K`. -/
private def c := (finrank ℚ K) * ‖((basisMatrix K).transpose)⁻¹‖


private theorem c_nonneg : 0 ≤ c K := by
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : DecidableEq (RingHom K Complex)
    ⊢ LE.le 0 (NumberField.house.c K)
  -/
  rw [c, mul_nonneg_iff]; left
  /-
    case h
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : DecidableEq (RingHom K Complex)
    ⊢ And (LE.le 0 ↑(Module.finrank Rat K)) (LE.le 0 (Norm.norm (Inv.inv (NumberFi …
  -/
  exact ⟨by simp only [Nat.cast_nonneg], norm_nonneg ((basisMatrix K).transpose)⁻¹⟩
  /-
    🎉 no goals
  -/


theorem basis_repr_abs_le_const_mul_house (α : 𝓞 K) (i : K →+* ℂ) :
    Complex.abs (((integralBasis K).reindex (equivReindex K).symm).repr α i) ≤
      (c K) * house (algebraMap (𝓞 K) K α) := by
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : DecidableEq (RingHom K Complex)
    α : NumberField.RingOfIntegers K
    i : RingHom K Complex
    ⊢ LE.le (Complex.abs ↑((((NumberField.integralBasis K).reindex (NumberField.eq …
  -/
  let σ := canonicalEmbedding K
  calc
    _ ≤ ∑ j, ‖((basisMatrix K).transpose)⁻¹‖ * Complex.abs (σ (algebraMap (𝓞 K) K α) j) := ?_
    _ ≤ ∑ _ : K →+* ℂ, ‖fun i j => ((basisMatrix K).transpose)⁻¹ i j‖
        * house (algebraMap (𝓞 K) K α) := ?_
    _ = ↑(finrank ℚ K) * ‖((basisMatrix K).transpose)⁻¹‖ * house (algebraMap (𝓞 K) K α) := ?_

    /-
      case calc_1
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : NumberField K
      inst✝ : DecidableEq (RingHom K Complex)
      α : NumberField.RingOfIntegers K
      i : RingHom K Complex
      σ : RingHom K (RingHom K Complex → Complex) := NumberField.canonicalEmbedding K
      ⊢ LE.le (Complex.abs ↑((((NumberField.integralBasis K).reindex (NumberField.eq …
    -/
  · rw [← inverse_basisMatrix_mulVec_eq_repr]
    /-
      case calc_1
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : NumberField K
      inst✝ : DecidableEq (RingHom K Complex)
      α : NumberField.RingOfIntegers K
      i : RingHom K Complex
      σ : RingHom K (RingHom K Complex → Complex) := NumberField.canonicalEmbedding K
      ⊢ LE.le (Complex.abs ((Inv.inv (NumberField.basisMatrix K).transpose).mulVec ( …
    -/
    apply le_trans
      /-
        case calc_1.a
        K : Type u_1
        inst✝² : Field K
        inst✝¹ : NumberField K
        inst✝ : DecidableEq (RingHom K Complex)
        α : NumberField.RingOfIntegers K
        i : RingHom K Complex
        σ : RingHom K (RingHom K Complex → Complex) := NumberField.canonicalEmbedding K
        ⊢ LE.le (Complex.abs ((Inv.inv (NumberField.basisMatrix K).transpose).mulVec ( …
      -/
    · apply le_trans (AbsoluteValue.sum_le Complex.abs _ _)
        /-
          case calc_1.a
          K : Type u_1
          inst✝² : Field K
          inst✝¹ : NumberField K
          inst✝ : DecidableEq (RingHom K Complex)
          α : NumberField.RingOfIntegers K
          i : RingHom K Complex
          σ : RingHom K (RingHom K Complex → Complex) := NumberField.canonicalEmbedding K
          ⊢ LE.le (Finset.univ.sum fun i_1 => Complex.abs (HMul.hMul ((fun j => Inv.inv  …
        -/
      · exact sum_le_sum (fun _ _ => (AbsoluteValue.map_mul Complex.abs _ _).le)
        /-
          🎉 no goals
        -/
    · apply sum_le_sum (fun _ _ => mul_le_mul_of_nonneg_right ?_
        (AbsoluteValue.nonneg Complex.abs _))
        /-
          K : Type u_1
          inst✝² : Field K
          inst✝¹ : NumberField K
          inst✝ : DecidableEq (RingHom K Complex)
          α : NumberField.RingOfIntegers K
          i : RingHom K Complex
          σ : RingHom K (RingHom K Complex → Complex) := NumberField.canonicalEmbedding K
          x✝¹ : RingHom K Complex
          x✝ : Membership.mem Finset.univ x✝¹
          ⊢ LE.le (Complex.abs ((fun j => Inv.inv (NumberField.basisMatrix K).transpose  …
        -/
      · exact norm_entry_le_entrywise_sup_norm ((basisMatrix K).transpose)⁻¹
        /-
          🎉 no goals
        -/
    /-
      case calc_2
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : NumberField K
      inst✝ : DecidableEq (RingHom K Complex)
      α : NumberField.RingOfIntegers K
      i : RingHom K Complex
      σ : RingHom K (RingHom K Complex → Complex) := NumberField.canonicalEmbedding K
      ⊢ LE.le (Finset.univ.sum fun j => HMul.hMul (Norm.norm (Inv.inv (NumberField.b …
    -/
  · apply sum_le_sum; intros j _
    /-
      case calc_2.h
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : NumberField K
      inst✝ : DecidableEq (RingHom K Complex)
      α : NumberField.RingOfIntegers K
      i : RingHom K Complex
      σ : RingHom K (RingHom K Complex → Complex) := NumberField.canonicalEmbedding K
      j : RingHom K Complex
      a✝ : Membership.mem Finset.univ j
      ⊢ LE.le (HMul.hMul (Norm.norm (Inv.inv (NumberField.basisMatrix K).transpose)) …
    -/
    apply mul_le_mul_of_nonneg_left _ (norm_nonneg fun i j ↦ ((basisMatrix K).transpose)⁻¹ i j)
      /-
        K : Type u_1
        inst✝² : Field K
        inst✝¹ : NumberField K
        inst✝ : DecidableEq (RingHom K Complex)
        α : NumberField.RingOfIntegers K
        i : RingHom K Complex
        σ : RingHom K (RingHom K Complex → Complex) := NumberField.canonicalEmbedding K
        j : RingHom K Complex
        a✝ : Membership.mem Finset.univ j
        ⊢ LE.le (Complex.abs (σ ((algebraMap (NumberField.RingOfIntegers K) K) α) j))  …
      -/
    · exact norm_le_pi_norm (σ ((algebraMap (𝓞 K) K) α)) j
      /-
        🎉 no goals
      -/
    /-
      case calc_3
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : NumberField K
      inst✝ : DecidableEq (RingHom K Complex)
      α : NumberField.RingOfIntegers K
      i : RingHom K Complex
      σ : RingHom K (RingHom K Complex → Complex) := NumberField.canonicalEmbedding K
      ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (Norm.norm fun i j => Inv.inv (Number …
    -/
  · rw [sum_const, card_univ, nsmul_eq_mul, Embeddings.card, mul_assoc]
    /-
      🎉 no goals
    -/


/-- `newBasis K` defines a reindexed basis of the ring of integers of `K`,
  adjusted by the inverse of the equivalence `equivReindex`. -/
private def newBasis := (RingOfIntegers.basis K).reindex (equivReindex K).symm


/-- `supOfBasis K` calculates the supremum of the absolute values of
  the elements in `newBasis K`. -/
private def supOfBasis : ℝ := univ.sup' univ_nonempty
  fun r ↦ house (algebraMap (𝓞 K) K (newBasis K r))


private theorem supOfBasis_nonneg : 0 ≤ supOfBasis K := by
  simp only [supOfBasis, le_sup'_iff, mem_univ, and_self,
    exists_const, house_nonneg]


/-- `a' K a` returns the integer coefficients of the basis vector in the
  expansion of the product of an algebraic integer and a basis vectors. -/
private def a' : α → β → (K →+* ℂ) → (K →+* ℂ) → ℤ := fun k l r =>
  (newBasis K).repr (a k l * (newBasis K) r)


/--`asiegel K a` the integer matrix of the coefficients of the
  product of matrix elements and basis vectors -/
private def asiegel : Matrix (α × (K →+* ℂ)) (β × (K →+* ℂ)) ℤ := fun k l => a' K a k.1 l.1 l.2 k.2


include ha in
private theorem asiegel_ne_0 : asiegel K a ≠ 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    ha : Ne a 0
    ⊢ Ne (NumberField.house.asiegel K a) 0
  -/
  simp (config := { unfoldPartialApp := true }) only [asiegel, a']
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    ha : Ne a 0
    ⊢ Ne (fun k l => ((NumberField.house.newBasis K).repr (HMul.hMul (a k.1 l.1) ( …
  -/
  simp only [ne_eq]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    ha : Ne a 0
    ⊢ Not (Eq (fun k l => ((NumberField.house.newBasis K).repr (HMul.hMul (a k.1 l …
  -/
  rw [funext_iff]; intros hs
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    ha : Ne a 0
    hs : ∀ (x : Prod α (RingHom K Complex)), Eq (fun l => ((NumberField.house.newB …
    ⊢ False
  -/
  simp only [Prod.forall] at hs;
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    ha : Ne a 0
    hs : ∀ (a_1 : α) (b : RingHom K Complex), Eq (fun l => ((NumberField.house.new …
    ⊢ False
  -/
  apply ha
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    ha : Ne a 0
    hs : ∀ (a_1 : α) (b : RingHom K Complex), Eq (fun l => ((NumberField.house.new …
    ⊢ Eq a 0
  -/
  rw [← Matrix.ext_iff]; intros k' l
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    ha : Ne a 0
    hs : ∀ (a_1 : α) (b : RingHom K Complex), Eq (fun l => ((NumberField.house.new …
    k' : α
    l : β
    ⊢ Eq (a k' l) (0 k' l)
  -/
  specialize hs k'
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    ha : Ne a 0
    k' : α
    l : β
    hs : ∀ (b : RingHom K Complex), Eq (fun l => ((NumberField.house.newBasis K).r …
    ⊢ Eq (a k' l) (0 k' l)
  -/
  let ⟨b⟩ := Fintype.card_pos_iff.1 (Fintype.card_pos (α := (K →+* ℂ)))
  have := ((newBasis K).repr.map_eq_zero_iff (x := (a k' l * (newBasis K) b))).1 <| by
    ext b'
    specialize hs b'
    rw [funext_iff] at hs
    simp only [Prod.forall] at hs
    apply hs
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    ha : Ne a 0
    k' : α
    l : β
    hs : ∀ (b : RingHom K Complex), Eq (fun l => ((NumberField.house.newBasis K).r …
    b : RingHom K Complex
    this : Eq (HMul.hMul (a k' l) ((NumberField.house.newBasis K) b)) 0
    ⊢ Eq (a k' l) (0 k' l)
  -/
  simp only [mul_eq_zero] at this
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    ha : Ne a 0
    k' : α
    l : β
    hs : ∀ (b : RingHom K Complex), Eq (fun l => ((NumberField.house.newBasis K).r …
    b : RingHom K Complex
    this : Or (Eq (a k' l) 0) (Eq ((NumberField.house.newBasis K) b) 0)
    ⊢ Eq (a k' l) (0 k' l)
  -/
  exact this.resolve_right (Basis.ne_zero (newBasis K) b)
  /-
    🎉 no goals
  -/


/-- `ξ` is the product of `x (l, r)` and the `r`-th basis element of the newBasis of `K`. -/
private def ξ : β → 𝓞 K := fun l => ∑ r : K →+* ℂ, x (l, r) * (newBasis K r)


include hxl in
private theorem ξ_ne_0 : ξ K x ≠ 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    β : Type u_3
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    ⊢ Ne (NumberField.house.ξ K x) 0
  -/
  intro H
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    β : Type u_3
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    H : Eq (NumberField.house.ξ K x) 0
    ⊢ False
  -/
  apply hxl
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    β : Type u_3
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    H : Eq (NumberField.house.ξ K x) 0
    ⊢ Eq x 0
  -/
  ext ⟨l, r⟩
  /-
    case h.mk
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    β : Type u_3
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    H : Eq (NumberField.house.ξ K x) 0
    l : β
    r : RingHom K Complex
    ⊢ Eq (x { fst := l, snd := r }) (0 { fst := l, snd := r })
  -/
  rw [funext_iff] at H
  /-
    case h.mk
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    β : Type u_3
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    H : ∀ (x_1 : β), Eq (NumberField.house.ξ K x x_1) (0 x_1)
    l : β
    r : RingHom K Complex
    ⊢ Eq (x { fst := l, snd := r }) (0 { fst := l, snd := r })
  -/
  have hblin := Basis.linearIndependent (newBasis K)
  /-
    case h.mk
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    β : Type u_3
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    H : ∀ (x_1 : β), Eq (NumberField.house.ξ K x x_1) (0 x_1)
    l : β
    r : RingHom K Complex
    hblin : LinearIndependent Int ⇑(NumberField.house.newBasis K)
    ⊢ Eq (x { fst := l, snd := r }) (0 { fst := l, snd := r })
  -/
  simp only [zsmul_eq_mul, Fintype.linearIndependent_iff] at hblin
  /-
    case h.mk
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    β : Type u_3
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    H : ∀ (x_1 : β), Eq (NumberField.house.ξ K x x_1) (0 x_1)
    l : β
    r : RingHom K Complex
    hblin : ∀ (g : RingHom K Complex → Int), Eq (Finset.univ.sum fun x => HMul.hMu …
    ⊢ Eq (x { fst := l, snd := r }) (0 { fst := l, snd := r })
  -/
  exact hblin (fun r ↦ x (l,r)) (H _) r
  /-
    🎉 no goals
  -/


private theorem lin_1 (l k r) : a k l * (newBasis K) r =
    ∑ u, (a' K a k l r u) * (newBasis K) u := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    l : β
    k : α
    r : RingHom K Complex
    ⊢ Eq (HMul.hMul (a k l) ((NumberField.house.newBasis K) r)) (Finset.univ.sum f …
  -/
  simp only [Basis.sum_repr (newBasis K) (a k l * (newBasis K) r), a', ← zsmul_eq_mul]
  /-
    🎉 no goals
  -/


include hxl hmulvec0 in
private theorem ξ_mulVec_eq_0 : a *ᵥ ξ K x = 0 := by
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    inst✝ : Fintype β
    hmulvec0 : Eq ((NumberField.house.asiegel K a).mulVec x) 0
    ⊢ Eq (a.mulVec (NumberField.house.ξ K x)) 0
  -/
  funext k; simp only [Pi.zero_apply]; rw [eq_comm]

  have lin_0 : ∀ u, ∑ r, ∑ l, (a' K a k l r u * x (l, r) : 𝓞 K) = 0 := by
    intros u
    have hξ := ξ_ne_0 K x hxl
    rw [Ne, funext_iff, not_forall] at hξ
    rcases hξ with ⟨l, hξ⟩
    rw [funext_iff] at hmulvec0
    specialize hmulvec0 ⟨k, u⟩
    simp only [Fintype.sum_prod_type, mulVec, dotProduct, asiegel] at hmulvec0
    rw [sum_comm] at hmulvec0
    exact mod_cast hmulvec0

  have : 0 = ∑ u, (∑ r, ∑ l, a' K a k l r u * x (l, r) : 𝓞 K) * (newBasis K) u := by
    simp only [lin_0, zero_mul, sum_const_zero]

  have : 0 = ∑ r, ∑ l, x (l, r) * ∑ u, a' K a k l r u * (newBasis K) u := by
    conv at this => enter [2, 2, u]; rw [sum_mul]
    rw [sum_comm] at this
    rw [this]; congr 1; ext1 r
    conv => enter [1, 2, l]; rw [sum_mul]
    rw [sum_comm]; congr 1; ext1 r
    rw [mul_sum]; congr 1; ext1 r
    ring
  /-
    case h
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    inst✝ : Fintype β
    hmulvec0 : Eq ((NumberField.house.asiegel K a).mulVec x) 0
    k : α
    lin_0 : ∀ (u : RingHom K Complex), Eq (Finset.univ.sum fun r => Finset.univ.su …
    this✝ : Eq 0 (Finset.univ.sum fun u => HMul.hMul (Finset.univ.sum fun r => Fin …
    this : Eq 0 (Finset.univ.sum fun r => Finset.univ.sum fun l => HMul.hMul (↑(x  …
    ⊢ Eq 0 (a.mulVec (NumberField.house.ξ K x) k)
  -/
  rw [sum_comm] at this
  /-
    case h
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    inst✝ : Fintype β
    hmulvec0 : Eq ((NumberField.house.asiegel K a).mulVec x) 0
    k : α
    lin_0 : ∀ (u : RingHom K Complex), Eq (Finset.univ.sum fun r => Finset.univ.su …
    this✝ : Eq 0 (Finset.univ.sum fun u => HMul.hMul (Finset.univ.sum fun r => Fin …
    this : Eq 0 (Finset.univ.sum fun y => Finset.univ.sum fun x_1 => HMul.hMul (↑( …
    ⊢ Eq 0 (a.mulVec (NumberField.house.ξ K x) k)
  -/
  rw [this]; congr 1; ext1 l
  /-
    case h.e_f.h
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    inst✝ : Fintype β
    hmulvec0 : Eq ((NumberField.house.asiegel K a).mulVec x) 0
    k : α
    lin_0 : ∀ (u : RingHom K Complex), Eq (Finset.univ.sum fun r => Finset.univ.su …
    this✝ : Eq 0 (Finset.univ.sum fun u => HMul.hMul (Finset.univ.sum fun r => Fin …
    this : Eq 0 (Finset.univ.sum fun y => Finset.univ.sum fun x_1 => HMul.hMul (↑( …
    l : β
    ⊢ Eq (Finset.univ.sum fun x_1 => HMul.hMul (↑(x { fst := l, snd := x_1 })) (Fi …
  -/
  rw [ξ, mul_sum]; congr 1; ext1 l
  /-
    case h.e_f.h.e_f.h
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    x : Prod β (RingHom K Complex) → Int
    hxl : Ne x 0
    inst✝ : Fintype β
    hmulvec0 : Eq ((NumberField.house.asiegel K a).mulVec x) 0
    k : α
    lin_0 : ∀ (u : RingHom K Complex), Eq (Finset.univ.sum fun r => Finset.univ.su …
    this✝ : Eq 0 (Finset.univ.sum fun u => HMul.hMul (Finset.univ.sum fun r => Fin …
    this : Eq 0 (Finset.univ.sum fun y => Finset.univ.sum fun x_1 => HMul.hMul (↑( …
    l✝ : β
    l : RingHom K Complex
    ⊢ Eq (HMul.hMul (↑(x { fst := l✝, snd := l })) (Finset.univ.sum fun u => HMul. …
  -/
  rw [← lin_1]; ring
                /-
                  🎉 no goals
                -/


/-- `c₂` is the product of the maximum of `1` and `c`, and `supOfBasis`. -/
private abbrev c₂ := max 1 (c K) * (supOfBasis K)


private theorem c₂_nonneg : 0 ≤ c₂ K :=
  mul_nonneg (le_trans zero_le_one (le_max_left ..)) (supOfBasis_nonneg _)


include habs Apos in
private theorem asiegel_remark : ‖asiegel K a‖ ≤ c₂ K * A := by
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    inst✝² : Fintype β
    A : Real
    habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
    inst✝¹ : DecidableEq (RingHom K Complex)
    inst✝ : Fintype α
    Apos : LE.le 0 A
    ⊢ LE.le (Norm.norm (NumberField.house.asiegel K a)) (HMul.hMul (NumberField.ho …
  -/
  rw [Matrix.norm_le_iff]
    /-
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      ⊢ ∀ (i : Prod α (RingHom K Complex)) (j : Prod β (RingHom K Complex)), LE.le ( …
    -/
  · intro kr lu
    calc
      ‖asiegel K a kr lu‖ = |asiegel K a kr lu| := ?_
      _ ≤ (c K) *
        house ((algebraMap (𝓞 K) K) (a kr.1 lu.1 * ((newBasis K) lu.2))) := ?_
      _ ≤ (c K) * house ((algebraMap (𝓞 K) K) (a kr.1 lu.1)) *
        house ((algebraMap (𝓞 K) K) ((newBasis K) lu.2)) := ?_
      _ ≤ (c K) * A * house ((algebraMap (𝓞 K) K) ((newBasis K) lu.2)) := ?_
      _ ≤ (c K) * A * (supOfBasis K) := ?_
      _ ≤ (c₂ K) * A := ?_
      /-
        case calc_1
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : NumberField K
        α : Type u_2
        β : Type u_3
        a : Matrix α β (NumberField.RingOfIntegers K)
        inst✝² : Fintype β
        A : Real
        habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
        inst✝¹ : DecidableEq (RingHom K Complex)
        inst✝ : Fintype α
        Apos : LE.le 0 A
        kr : Prod α (RingHom K Complex)
        lu : Prod β (RingHom K Complex)
        ⊢ Eq (Norm.norm (NumberField.house.asiegel K a kr lu)) ↑(abs (NumberField.hous …
      -/
    · simp only [Int.cast_abs, ← Real.norm_eq_abs (asiegel K a kr lu)]; rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
      /-
        case calc_2
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : NumberField K
        α : Type u_2
        β : Type u_3
        a : Matrix α β (NumberField.RingOfIntegers K)
        inst✝² : Fintype β
        A : Real
        habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
        inst✝¹ : DecidableEq (RingHom K Complex)
        inst✝ : Fintype α
        Apos : LE.le 0 A
        kr : Prod α (RingHom K Complex)
        lu : Prod β (RingHom K Complex)
        ⊢ LE.le (↑(abs (NumberField.house.asiegel K a kr lu))) (HMul.hMul (NumberField …
      -/
    · have remark := basis_repr_abs_le_const_mul_house K
      simp only [Basis.repr_reindex, Finsupp.mapDomain_equiv_apply,
        integralBasis_repr_apply, eq_intCast, Rat.cast_intCast,
          Complex.abs_intCast] at remark
      /-
        case calc_2
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : NumberField K
        α : Type u_2
        β : Type u_3
        a : Matrix α β (NumberField.RingOfIntegers K)
        inst✝² : Fintype β
        A : Real
        habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
        inst✝¹ : DecidableEq (RingHom K Complex)
        inst✝ : Fintype α
        Apos : LE.le 0 A
        kr : Prod α (RingHom K Complex)
        lu : Prod β (RingHom K Complex)
        remark : ∀ (α : NumberField.RingOfIntegers K) (i : RingHom K Complex), LE.le ( …
        ⊢ LE.le (↑(abs (NumberField.house.asiegel K a kr lu))) (HMul.hMul (NumberField …
      -/
      exact mod_cast remark ((a kr.1 lu.1 * ((newBasis K) lu.2))) kr.2
      /-
        🎉 no goals
      -/
      /-
        case calc_3
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : NumberField K
        α : Type u_2
        β : Type u_3
        a : Matrix α β (NumberField.RingOfIntegers K)
        inst✝² : Fintype β
        A : Real
        habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
        inst✝¹ : DecidableEq (RingHom K Complex)
        inst✝ : Fintype α
        Apos : LE.le 0 A
        kr : Prod α (RingHom K Complex)
        lu : Prod β (RingHom K Complex)
        ⊢ LE.le (HMul.hMul (NumberField.house.c K) (NumberField.house ((algebraMap (Nu …
      -/
    · simp only [house, _root_.map_mul, mul_assoc]
      /-
        case calc_3
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : NumberField K
        α : Type u_2
        β : Type u_3
        a : Matrix α β (NumberField.RingOfIntegers K)
        inst✝² : Fintype β
        A : Real
        habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
        inst✝¹ : DecidableEq (RingHom K Complex)
        inst✝ : Fintype α
        Apos : LE.le 0 A
        kr : Prod α (RingHom K Complex)
        lu : Prod β (RingHom K Complex)
        ⊢ LE.le (HMul.hMul (NumberField.house.c K) (Norm.norm (HMul.hMul ((NumberField …
      -/
      exact mul_le_mul_of_nonneg_left (norm_mul_le _ _) (c_nonneg K)
      /-
        🎉 no goals
      -/
      /-
        case calc_4
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : NumberField K
        α : Type u_2
        β : Type u_3
        a : Matrix α β (NumberField.RingOfIntegers K)
        inst✝² : Fintype β
        A : Real
        habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
        inst✝¹ : DecidableEq (RingHom K Complex)
        inst✝ : Fintype α
        Apos : LE.le 0 A
        kr : Prod α (RingHom K Complex)
        lu : Prod β (RingHom K Complex)
        ⊢ LE.le (HMul.hMul (HMul.hMul (NumberField.house.c K) (NumberField.house ((alg …
      -/
    · rw [mul_assoc, mul_assoc]
      /-
        case calc_4
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : NumberField K
        α : Type u_2
        β : Type u_3
        a : Matrix α β (NumberField.RingOfIntegers K)
        inst✝² : Fintype β
        A : Real
        habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
        inst✝¹ : DecidableEq (RingHom K Complex)
        inst✝ : Fintype α
        Apos : LE.le 0 A
        kr : Prod α (RingHom K Complex)
        lu : Prod β (RingHom K Complex)
        ⊢ LE.le (HMul.hMul (NumberField.house.c K) (HMul.hMul (NumberField.house ((alg …
      -/
      apply mul_le_mul_of_nonneg_left ?_ (c_nonneg K)
        /-
          K : Type u_1
          inst✝⁴ : Field K
          inst✝³ : NumberField K
          α : Type u_2
          β : Type u_3
          a : Matrix α β (NumberField.RingOfIntegers K)
          inst✝² : Fintype β
          A : Real
          habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
          inst✝¹ : DecidableEq (RingHom K Complex)
          inst✝ : Fintype α
          Apos : LE.le 0 A
          kr : Prod α (RingHom K Complex)
          lu : Prod β (RingHom K Complex)
          ⊢ LE.le (HMul.hMul (NumberField.house ((algebraMap (NumberField.RingOfIntegers …
        -/
      · apply mul_le_mul_of_nonneg_right (habs kr.1 lu.1) ?_
        · exact norm_nonneg ((canonicalEmbedding K) ((algebraMap (𝓞 K) K)
            ((newBasis K) lu.2)))
       /-
         case calc_5
         K : Type u_1
         inst✝⁴ : Field K
         inst✝³ : NumberField K
         α : Type u_2
         β : Type u_3
         a : Matrix α β (NumberField.RingOfIntegers K)
         inst✝² : Fintype β
         A : Real
         habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
         inst✝¹ : DecidableEq (RingHom K Complex)
         inst✝ : Fintype α
         Apos : LE.le 0 A
         kr : Prod α (RingHom K Complex)
         lu : Prod β (RingHom K Complex)
         ⊢ LE.le (HMul.hMul (HMul.hMul (NumberField.house.c K) A) (NumberField.house (( …
       -/
    ·  apply mul_le_mul_of_nonneg_left ?_ (mul_nonneg (c_nonneg K) Apos)
         /-
           K : Type u_1
           inst✝⁴ : Field K
           inst✝³ : NumberField K
           α : Type u_2
           β : Type u_3
           a : Matrix α β (NumberField.RingOfIntegers K)
           inst✝² : Fintype β
           A : Real
           habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
           inst✝¹ : DecidableEq (RingHom K Complex)
           inst✝ : Fintype α
           Apos : LE.le 0 A
           kr : Prod α (RingHom K Complex)
           lu : Prod β (RingHom K Complex)
           ⊢ LE.le (NumberField.house ((algebraMap (NumberField.RingOfIntegers K) K) ((Nu …
         -/
       · simp only [supOfBasis, le_sup'_iff, mem_univ]; use lu.2
                                                        /-
                                                          🎉 no goals
                                                        -/
      /-
        case calc_6
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : NumberField K
        α : Type u_2
        β : Type u_3
        a : Matrix α β (NumberField.RingOfIntegers K)
        inst✝² : Fintype β
        A : Real
        habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
        inst✝¹ : DecidableEq (RingHom K Complex)
        inst✝ : Fintype α
        Apos : LE.le 0 A
        kr : Prod α (RingHom K Complex)
        lu : Prod β (RingHom K Complex)
        ⊢ LE.le (HMul.hMul (HMul.hMul (NumberField.house.c K) A) (NumberField.house.su …
      -/
    · rw [mul_right_comm]
      exact mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_right (le_max_right ..) (supOfBasis_nonneg K)) Apos
    /-
      case hr
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      ⊢ LE.le 0 (HMul.hMul (NumberField.house.c₂ K) A)
    -/
  · rw [mul_nonneg_iff]; left; exact ⟨c₂_nonneg K, Apos⟩
                               /-
                                 🎉 no goals
                               -/


/-- `c₁ K` is the product of `finrank ℚ K` and  `c₂ K` and depends on `K`. -/
private def c₁ := finrank ℚ K * c₂ K


include habs Apos hxbound hpq in
private theorem house_le_bound : ∀ l, house (ξ K x l).1 ≤ (c₁ K) *
    ((c₁ K * q * A)^((p : ℝ) / (q - p))) := by
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    p q : Nat
    hpq : LT.lt p q
    x : Prod β (RingHom K Complex) → Int
    inst✝² : Fintype β
    A : Real
    habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
    inst✝¹ : DecidableEq (RingHom K Complex)
    inst✝ : Fintype α
    Apos : LE.le 0 A
    hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
    ⊢ ∀ (l : β), LE.le (NumberField.house ↑(NumberField.house.ξ K x l)) (HMul.hMul …
  -/
  let h := finrank ℚ K
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    α : Type u_2
    β : Type u_3
    a : Matrix α β (NumberField.RingOfIntegers K)
    p q : Nat
    hpq : LT.lt p q
    x : Prod β (RingHom K Complex) → Int
    inst✝² : Fintype β
    A : Real
    habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
    inst✝¹ : DecidableEq (RingHom K Complex)
    inst✝ : Fintype α
    Apos : LE.le 0 A
    hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
    h : Nat := Module.finrank Rat K
    ⊢ ∀ (l : β), LE.le (NumberField.house ↑(NumberField.house.ξ K x l)) (HMul.hMul …
  -/
  intros l
  calc _ = house (algebraMap (𝓞 K) K (∑ r, (x (l, r)) * ((newBasis K) r))) := rfl
       _ ≤ ∑ r, house (((algebraMap (𝓞 K) K) (x (l, r))) *
        ((algebraMap (𝓞 K) K) ((newBasis K) r))) := ?_
       _ ≤ ∑ r, ‖x (l,r)‖ * house ((algebraMap (𝓞 K) K) ((newBasis K) r)) := ?_
       _ ≤ ∑ r, ‖x (l, r)‖ * (supOfBasis K) := ?_
       _ ≤ ∑ _r : K →+* ℂ, ((↑q * h * ‖asiegel K a‖) ^ ((p : ℝ) / (q - p))) * supOfBasis K := ?_
       _ ≤ h * (c₂ K) * ((q * c₁ K * A) ^ ((p : ℝ) / (q - p))) := ?_
       _ ≤ c₁ K * ((c₁ K * ↑q * A) ^ ((p : ℝ) / (q - p))) := ?_
    /-
      case calc_1
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      ⊢ LE.le (NumberField.house ((algebraMap (NumberField.RingOfIntegers K) K) (Fin …
    -/
  · simp_rw [← _root_.map_mul, map_sum]; apply house_sum_le_sum_house
                                         /-
                                           🎉 no goals
                                         -/
    /-
      case calc_2
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      ⊢ LE.le (Finset.univ.sum fun r => NumberField.house (HMul.hMul ((algebraMap (N …
    -/
  · apply sum_le_sum; intros r _; convert house_mul_le ..
    /-
      case h.e'_4.h.e'_5
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      r : RingHom K Complex
      a✝ : Membership.mem Finset.univ r
      ⊢ Eq (Norm.norm (x { fst := l, snd := r })) (NumberField.house ((algebraMap (N …
    -/
    simp only [map_intCast, house_intCast, Int.cast_abs, Int.norm_eq_abs]
    /-
      🎉 no goals
    -/
    /-
      case calc_3
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      ⊢ LE.le (Finset.univ.sum fun r => HMul.hMul (Norm.norm (x { fst := l, snd := r …
    -/
  · apply sum_le_sum; intros r _; unfold supOfBasis
    /-
      case calc_3.h
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      r : RingHom K Complex
      a✝ : Membership.mem Finset.univ r
      ⊢ LE.le (HMul.hMul (Norm.norm (x { fst := l, snd := r })) (NumberField.house ( …
    -/
    apply mul_le_mul_of_nonneg_left ?_ (norm_nonneg (x (l,r)))
      /-
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : NumberField K
        α : Type u_2
        β : Type u_3
        a : Matrix α β (NumberField.RingOfIntegers K)
        p q : Nat
        hpq : LT.lt p q
        x : Prod β (RingHom K Complex) → Int
        inst✝² : Fintype β
        A : Real
        habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
        inst✝¹ : DecidableEq (RingHom K Complex)
        inst✝ : Fintype α
        Apos : LE.le 0 A
        hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
        h : Nat := Module.finrank Rat K
        l : β
        r : RingHom K Complex
        a✝ : Membership.mem Finset.univ r
        ⊢ LE.le (NumberField.house ((algebraMap (NumberField.RingOfIntegers K) K) ((Nu …
      -/
    · simp only [le_sup'_iff, mem_univ, true_and]; use r
                                                   /-
                                                     🎉 no goals
                                                   -/
    /-
      case calc_4
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      ⊢ LE.le (Finset.univ.sum fun r => HMul.hMul (Norm.norm (x { fst := l, snd := r …
    -/
  · apply sum_le_sum; intros r _
    /-
      case calc_4.h
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      r : RingHom K Complex
      a✝ : Membership.mem Finset.univ r
      ⊢ LE.le (HMul.hMul (Norm.norm (x { fst := l, snd := r })) (NumberField.house.s …
    -/
    apply mul_le_mul_of_nonneg_right ?_ (supOfBasis_nonneg K)
    /-
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      r : RingHom K Complex
      a✝ : Membership.mem Finset.univ r
      ⊢ LE.le (Norm.norm (x { fst := l, snd := r })) (HPow.hPow (HMul.hMul (HMul.hMu …
    -/
    exact le_trans (norm_le_pi_norm x ⟨l, r⟩) hxbound
    /-
      🎉 no goals
    -/
    /-
      case calc_5
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      ⊢ LE.le (Finset.univ.sum fun _r => HMul.hMul (HPow.hPow (HMul.hMul (HMul.hMul  …
    -/
  · simp only [Nat.cast_mul, sum_const, card_univ, nsmul_eq_mul]
    /-
      case calc_5
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      ⊢ LE.le (HMul.hMul (↑(Fintype.card (RingHom K Complex))) (HMul.hMul (HPow.hPow …
    -/
    rw [Embeddings.card, mul_comm _ (supOfBasis K), c₂, c₁, ← mul_assoc]
    /-
      case calc_5
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      ⊢ LE.le (HMul.hMul (HMul.hMul (↑(Module.finrank Rat K)) (NumberField.house.sup …
    -/
    apply mul_le_mul
      /-
        case calc_5.h₁
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : NumberField K
        α : Type u_2
        β : Type u_3
        a : Matrix α β (NumberField.RingOfIntegers K)
        p q : Nat
        hpq : LT.lt p q
        x : Prod β (RingHom K Complex) → Int
        inst✝² : Fintype β
        A : Real
        habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
        inst✝¹ : DecidableEq (RingHom K Complex)
        inst✝ : Fintype α
        Apos : LE.le 0 A
        hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
        h : Nat := Module.finrank Rat K
        l : β
        ⊢ LE.le (HMul.hMul (↑(Module.finrank Rat K)) (NumberField.house.supOfBasis K)) …
      -/
    · apply mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg' _)
        /-
          K : Type u_1
          inst✝⁴ : Field K
          inst✝³ : NumberField K
          α : Type u_2
          β : Type u_3
          a : Matrix α β (NumberField.RingOfIntegers K)
          p q : Nat
          hpq : LT.lt p q
          x : Prod β (RingHom K Complex) → Int
          inst✝² : Fintype β
          A : Real
          habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
          inst✝¹ : DecidableEq (RingHom K Complex)
          inst✝ : Fintype α
          Apos : LE.le 0 A
          hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
          h : Nat := Module.finrank Rat K
          l : β
          ⊢ LE.le (NumberField.house.supOfBasis K) (HMul.hMul (Max.max 1 (NumberField.ho …
        -/
      · exact le_mul_of_one_le_left (supOfBasis_nonneg K) (le_max_left ..)
        /-
          🎉 no goals
        -/
    · apply Real.rpow_le_rpow (mul_nonneg (mul_nonneg (Nat.cast_nonneg' _) (Nat.cast_nonneg' _))
        (norm_nonneg _))
        /-
          case calc_5.h₂.h₁
          K : Type u_1
          inst✝⁴ : Field K
          inst✝³ : NumberField K
          α : Type u_2
          β : Type u_3
          a : Matrix α β (NumberField.RingOfIntegers K)
          p q : Nat
          hpq : LT.lt p q
          x : Prod β (RingHom K Complex) → Int
          inst✝² : Fintype β
          A : Real
          habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
          inst✝¹ : DecidableEq (RingHom K Complex)
          inst✝ : Fintype α
          Apos : LE.le 0 A
          hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
          h : Nat := Module.finrank Rat K
          l : β
          ⊢ LE.le (HMul.hMul (HMul.hMul ↑q ↑h) (Norm.norm (NumberField.house.asiegel K a …
        -/
      · rw [← mul_assoc, mul_assoc (_*_)]
        apply mul_le_mul_of_nonneg_left (asiegel_remark K a habs Apos)
          (mul_nonneg (Nat.cast_nonneg' _) (Nat.cast_nonneg _))
        /-
          case calc_5.h₂.h₂
          K : Type u_1
          inst✝⁴ : Field K
          inst✝³ : NumberField K
          α : Type u_2
          β : Type u_3
          a : Matrix α β (NumberField.RingOfIntegers K)
          p q : Nat
          hpq : LT.lt p q
          x : Prod β (RingHom K Complex) → Int
          inst✝² : Fintype β
          A : Real
          habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
          inst✝¹ : DecidableEq (RingHom K Complex)
          inst✝ : Fintype α
          Apos : LE.le 0 A
          hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
          h : Nat := Module.finrank Rat K
          l : β
          ⊢ LE.le 0 (HDiv.hDiv (↑p) (HSub.hSub ↑q ↑p))
        -/
      · exact div_nonneg (Nat.cast_nonneg' _) (sub_nonneg.2 (mod_cast hpq.le))
        /-
          🎉 no goals
        -/
      /-
        case calc_5.c0
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : NumberField K
        α : Type u_2
        β : Type u_3
        a : Matrix α β (NumberField.RingOfIntegers K)
        p q : Nat
        hpq : LT.lt p q
        x : Prod β (RingHom K Complex) → Int
        inst✝² : Fintype β
        A : Real
        habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
        inst✝¹ : DecidableEq (RingHom K Complex)
        inst✝ : Fintype α
        Apos : LE.le 0 A
        hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
        h : Nat := Module.finrank Rat K
        l : β
        ⊢ LE.le 0 (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑h) (Norm.norm (NumberField.hous …
      -/
    · apply Real.rpow_nonneg
      exact mul_nonneg (mul_nonneg (Nat.cast_nonneg' _) (Nat.cast_nonneg' _))
        (norm_nonneg _)
    · exact mul_nonneg (Nat.cast_nonneg' _) (mul_nonneg (le_trans zero_le_one (le_max_left ..))
        (supOfBasis_nonneg _))
    /-
      case calc_6
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      α : Type u_2
      β : Type u_3
      a : Matrix α β (NumberField.RingOfIntegers K)
      p q : Nat
      hpq : LT.lt p q
      x : Prod β (RingHom K Complex) → Int
      inst✝² : Fintype β
      A : Real
      habs : ∀ (k : α) (l : β), LE.le (NumberField.house ((algebraMap (NumberField.R …
      inst✝¹ : DecidableEq (RingHom K Complex)
      inst✝ : Fintype α
      Apos : LE.le 0 A
      hxbound : LE.le (Norm.norm x) (HPow.hPow (HMul.hMul (HMul.hMul ↑q ↑(Module.fin …
      h : Nat := Module.finrank Rat K
      l : β
      ⊢ LE.le (HMul.hMul (HMul.hMul (↑h) (NumberField.house.c₂ K)) (HPow.hPow (HMul. …
    -/
  · rw [mul_comm (q : ℝ) (c₁ K)]; rfl
                                  /-
                                    🎉 no goals
                                  -/


include hpq h0p cardα cardβ ha habs in
/-- There exists a "small" non-zero algebraic integral solution of an
 non-trivial underdetermined system of linear equations with algebraic integer coefficients.-/
theorem exists_ne_zero_int_vec_house_le :
    ∃ (ξ : β → 𝓞 K), ξ ≠ 0 ∧ a *ᵥ ξ = 0 ∧
    ∀ l, house (ξ l).1 ≤ c₁ K * ((c₁ K * q * A) ^ ((p : ℝ) / (q - p))) := by
  classical
  let h := finrank ℚ K
  have hphqh : p * h < q * h := mul_lt_mul_of_pos_right hpq finrank_pos
  have h0ph : 0 < p * h := by rw [mul_pos_iff]; constructor; exact ⟨h0p, finrank_pos⟩
  have hfinp : Fintype.card (α × (K →+* ℂ)) = p * h := by
    rw [Fintype.card_prod, cardα, Embeddings.card]
  have hfinq : Fintype.card (β × (K →+* ℂ)) = q * h := by
    rw [Fintype.card_prod, cardβ, Embeddings.card]
  have ⟨x, hxl, hmulvec0, hxbound⟩ :=
    Int.Matrix.exists_ne_zero_int_vec_norm_le' (asiegel K a)
      (by rwa [hfinp, hfinq]) (by rwa [hfinp]) (asiegel_ne_0 K a ha)
  simp only [hfinp, hfinq, Nat.cast_mul] at hmulvec0 hxbound
  rw [← sub_mul, mul_div_mul_right _ _ (mod_cast finrank_pos.ne')] at hxbound
  have Apos : 0 ≤ A := by
    have ⟨k⟩ := Fintype.card_pos_iff.1 (cardα ▸ h0p)
    have ⟨l⟩ := Fintype.card_pos_iff.1 (cardβ ▸ h0p.trans hpq)
    exact le_trans (house_nonneg _) (habs k l)
  use ξ K x, ξ_ne_0 K x hxl, ξ_mulVec_eq_0 K a x hxl hmulvec0,
    house_le_bound K a hpq x habs Apos hxbound


