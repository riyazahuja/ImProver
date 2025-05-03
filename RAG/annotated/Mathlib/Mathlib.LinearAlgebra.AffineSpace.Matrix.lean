/-- Given an affine basis `p`, and a family of points `q : ι' → P`, this is the matrix whose
rows are the barycentric coordinates of `q` with respect to `p`.

It is an affine equivalent of `Basis.toMatrix`. -/
noncomputable def toMatrix {ι' : Type*} (q : ι' → P) : Matrix ι' ι k :=
  fun i j => b.coord j (q i)


@[simp]
theorem toMatrix_apply {ι' : Type*} (q : ι' → P) (i : ι') (j : ι) :
    b.toMatrix q i j = b.coord j (q i) := rfl


@[simp]
theorem toMatrix_self [DecidableEq ι] : b.toMatrix b = (1 : Matrix ι ι k) := by
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : DecidableEq ι
    ⊢ Eq (b.toMatrix ⇑b) 1
  -/
  ext i j
  /-
    case a
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : DecidableEq ι
    i j : ι
    ⊢ Eq (b.toMatrix (⇑b) i j) (1 i j)
  -/
  rw [toMatrix_apply, coord_apply, Matrix.one_eq_pi_single, Pi.single_apply]
  /-
    🎉 no goals
  -/


theorem toMatrix_row_sum_one [Fintype ι] (q : ι' → P) (i : ι') : ∑ j, b.toMatrix q i j = 1 := by
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    ι' : Type u_1
    inst✝ : Fintype ι
    q : ι' → P
    i : ι'
    ⊢ Eq (Finset.univ.sum fun j => b.toMatrix q i j) 1
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a family of points `p : ι' → P` and an affine basis `b`, if the matrix whose rows are the
coordinates of `p` with respect `b` has a right inverse, then `p` is affine independent. -/
theorem affineIndependent_of_toMatrix_right_inv [Fintype ι] [Finite ι'] [DecidableEq ι']
    (p : ι' → P) {A : Matrix ι ι' k} (hA : b.toMatrix p * A = 1) : AffineIndependent k p := by
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁶ : AddCommGroup V
    inst✝⁵ : AddTorsor V P
    inst✝⁴ : Ring k
    inst✝³ : Module k V
    b : AffineBasis ι k P
    ι' : Type u_1
    inst✝² : Fintype ι
    inst✝¹ : Finite ι'
    inst✝ : DecidableEq ι'
    p : ι' → P
    A : Matrix ι ι' k
    hA : Eq (HMul.hMul (b.toMatrix p) A) 1
    ⊢ AffineIndependent k p
  -/
  cases nonempty_fintype ι'
  /-
    case intro
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁶ : AddCommGroup V
    inst✝⁵ : AddTorsor V P
    inst✝⁴ : Ring k
    inst✝³ : Module k V
    b : AffineBasis ι k P
    ι' : Type u_1
    inst✝² : Fintype ι
    inst✝¹ : Finite ι'
    inst✝ : DecidableEq ι'
    p : ι' → P
    A : Matrix ι ι' k
    hA : Eq (HMul.hMul (b.toMatrix p) A) 1
    val✝ : Fintype ι'
    ⊢ AffineIndependent k p
  -/
  rw [affineIndependent_iff_eq_of_fintype_affineCombination_eq]
  /-
    case intro
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁶ : AddCommGroup V
    inst✝⁵ : AddTorsor V P
    inst✝⁴ : Ring k
    inst✝³ : Module k V
    b : AffineBasis ι k P
    ι' : Type u_1
    inst✝² : Fintype ι
    inst✝¹ : Finite ι'
    inst✝ : DecidableEq ι'
    p : ι' → P
    A : Matrix ι ι' k
    hA : Eq (HMul.hMul (b.toMatrix p) A) 1
    val✝ : Fintype ι'
    ⊢ ∀ (w1 w2 : ι' → k), Eq (Finset.univ.sum fun i => w1 i) 1 → Eq (Finset.univ.s …
  -/
  intro w₁ w₂ hw₁ hw₂ hweq
  have hweq' : w₁ ᵥ* b.toMatrix p = w₂ ᵥ* b.toMatrix p := by
    ext j
    change (∑ i, w₁ i • b.coord j (p i)) = ∑ i, w₂ i • b.coord j (p i)
    -- Porting note: Added `u` because `∘` was causing trouble
    have u : (fun i => b.coord j (p i)) = b.coord j ∘ p := by simp only [Function.comp_def]
    rw [← Finset.univ.affineCombination_eq_linear_combination _ _ hw₁,
      ← Finset.univ.affineCombination_eq_linear_combination _ _ hw₂, u,
      ← Finset.univ.map_affineCombination p w₁ hw₁, ← Finset.univ.map_affineCombination p w₂ hw₂,
      hweq]
  /-
    case intro
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁶ : AddCommGroup V
    inst✝⁵ : AddTorsor V P
    inst✝⁴ : Ring k
    inst✝³ : Module k V
    b : AffineBasis ι k P
    ι' : Type u_1
    inst✝² : Fintype ι
    inst✝¹ : Finite ι'
    inst✝ : DecidableEq ι'
    p : ι' → P
    A : Matrix ι ι' k
    hA : Eq (HMul.hMul (b.toMatrix p) A) 1
    val✝ : Fintype ι'
    w₁ w₂ : ι' → k
    hw₁ : Eq (Finset.univ.sum fun i => w₁ i) 1
    hw₂ : Eq (Finset.univ.sum fun i => w₂ i) 1
    hweq : Eq ((Finset.affineCombination k Finset.univ p) w₁) ((Finset.affineCombi …
    hweq' : Eq (Matrix.vecMul w₁ (b.toMatrix p)) (Matrix.vecMul w₂ (b.toMatrix p))
    ⊢ Eq w₁ w₂
  -/
  replace hweq' := congr_arg (fun w => w ᵥ* A) hweq'
  /-
    case intro
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁶ : AddCommGroup V
    inst✝⁵ : AddTorsor V P
    inst✝⁴ : Ring k
    inst✝³ : Module k V
    b : AffineBasis ι k P
    ι' : Type u_1
    inst✝² : Fintype ι
    inst✝¹ : Finite ι'
    inst✝ : DecidableEq ι'
    p : ι' → P
    A : Matrix ι ι' k
    hA : Eq (HMul.hMul (b.toMatrix p) A) 1
    val✝ : Fintype ι'
    w₁ w₂ : ι' → k
    hw₁ : Eq (Finset.univ.sum fun i => w₁ i) 1
    hw₂ : Eq (Finset.univ.sum fun i => w₂ i) 1
    hweq : Eq ((Finset.affineCombination k Finset.univ p) w₁) ((Finset.affineCombi …
    hweq' : Eq ((fun w => Matrix.vecMul w A) (Matrix.vecMul w₁ (b.toMatrix p))) (( …
    ⊢ Eq w₁ w₂
  -/
  simpa only [Matrix.vecMul_vecMul, hA, Matrix.vecMul_one] using hweq'
  /-
    🎉 no goals
  -/


/-- Given a family of points `p : ι' → P` and an affine basis `b`, if the matrix whose rows are the
coordinates of `p` with respect `b` has a left inverse, then `p` spans the entire space. -/
theorem affineSpan_eq_top_of_toMatrix_left_inv [Finite ι] [Fintype ι'] [DecidableEq ι]
    [Nontrivial k] (p : ι' → P) {A : Matrix ι ι' k} (hA : A * b.toMatrix p = 1) :
    affineSpan k (range p) = ⊤ := by
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : AddTorsor V P
    inst✝⁵ : Ring k
    inst✝⁴ : Module k V
    b : AffineBasis ι k P
    ι' : Type u_1
    inst✝³ : Finite ι
    inst✝² : Fintype ι'
    inst✝¹ : DecidableEq ι
    inst✝ : Nontrivial k
    p : ι' → P
    A : Matrix ι ι' k
    hA : Eq (HMul.hMul A (b.toMatrix p)) 1
    ⊢ Eq (affineSpan k (Set.range p)) Top.top
  -/
  cases nonempty_fintype ι
  suffices ∀ i, b i ∈ affineSpan k (range p) by
    rw [eq_top_iff, ← b.tot, affineSpan_le]
    rintro q ⟨i, rfl⟩
    exact this i
  /-
    case intro
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : AddTorsor V P
    inst✝⁵ : Ring k
    inst✝⁴ : Module k V
    b : AffineBasis ι k P
    ι' : Type u_1
    inst✝³ : Finite ι
    inst✝² : Fintype ι'
    inst✝¹ : DecidableEq ι
    inst✝ : Nontrivial k
    p : ι' → P
    A : Matrix ι ι' k
    hA : Eq (HMul.hMul A (b.toMatrix p)) 1
    val✝ : Fintype ι
    ⊢ ∀ (i : ι), Membership.mem (affineSpan k (Set.range p)) (b i)
  -/
  intro i
  have hAi : ∑ j, A i j = 1 := by
    calc
      ∑ j, A i j = ∑ j, A i j * ∑ l, b.toMatrix p j l := by simp
      _ = ∑ j, ∑ l, A i j * b.toMatrix p j l := by simp_rw [Finset.mul_sum]
      _ = ∑ l, ∑ j, A i j * b.toMatrix p j l := by rw [Finset.sum_comm]
      _ = ∑ l, (A * b.toMatrix p) i l := rfl
      _ = 1 := by simp [hA, Matrix.one_apply, Finset.filter_eq]
  have hbi : b i = Finset.univ.affineCombination k p (A i) := by
    apply b.ext_elem
    intro j
    rw [b.coord_apply, Finset.univ.map_affineCombination _ _ hAi,
      Finset.univ.affineCombination_eq_linear_combination _ _ hAi]
    change _ = (A * b.toMatrix p) i j
    simp_rw [hA, Matrix.one_apply, @eq_comm _ i j]
  /-
    case intro
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : AddTorsor V P
    inst✝⁵ : Ring k
    inst✝⁴ : Module k V
    b : AffineBasis ι k P
    ι' : Type u_1
    inst✝³ : Finite ι
    inst✝² : Fintype ι'
    inst✝¹ : DecidableEq ι
    inst✝ : Nontrivial k
    p : ι' → P
    A : Matrix ι ι' k
    hA : Eq (HMul.hMul A (b.toMatrix p)) 1
    val✝ : Fintype ι
    i : ι
    hAi : Eq (Finset.univ.sum fun j => A i j) 1
    hbi : Eq (b i) ((Finset.affineCombination k Finset.univ p) (A i))
    ⊢ Membership.mem (affineSpan k (Set.range p)) (b i)
  -/
  rw [hbi]
  /-
    case intro
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : AddTorsor V P
    inst✝⁵ : Ring k
    inst✝⁴ : Module k V
    b : AffineBasis ι k P
    ι' : Type u_1
    inst✝³ : Finite ι
    inst✝² : Fintype ι'
    inst✝¹ : DecidableEq ι
    inst✝ : Nontrivial k
    p : ι' → P
    A : Matrix ι ι' k
    hA : Eq (HMul.hMul A (b.toMatrix p)) 1
    val✝ : Fintype ι
    i : ι
    hAi : Eq (Finset.univ.sum fun j => A i j) 1
    hbi : Eq (b i) ((Finset.affineCombination k Finset.univ p) (A i))
    ⊢ Membership.mem (affineSpan k (Set.range p)) ((Finset.affineCombination k Fin …
  -/
  exact affineCombination_mem_affineSpan hAi p
  /-
    🎉 no goals
  -/


/-- A change of basis formula for barycentric coordinates.

See also `AffineBasis.toMatrix_inv_vecMul_toMatrix`. -/
@[simp]
theorem toMatrix_vecMul_coords (x : P) : b₂.coords x ᵥ* b.toMatrix b₂ = b.coords x := by
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    b₂ : AffineBasis ι k P
    x : P
    ⊢ Eq (Matrix.vecMul (b₂.coords x) (b.toMatrix ⇑b₂)) (b.coords x)
  -/
  ext j
  /-
    case h
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    b₂ : AffineBasis ι k P
    x : P
    j : ι
    ⊢ Eq (Matrix.vecMul (b₂.coords x) (b.toMatrix ⇑b₂) j) (b.coords x j)
  -/
  change _ = b.coord j x
  /-
    case h
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    b₂ : AffineBasis ι k P
    x : P
    j : ι
    ⊢ Eq (Matrix.vecMul (b₂.coords x) (b.toMatrix ⇑b₂) j) ((b.coord j) x)
  -/
  conv_rhs => rw [← b₂.affineCombination_coord_eq_self x]
  /-
    case h
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    b₂ : AffineBasis ι k P
    x : P
    j : ι
    ⊢ Eq (Matrix.vecMul (b₂.coords x) (b.toMatrix ⇑b₂) j) ((b.coord j) ((Finset.af …
  -/
  rw [Finset.map_affineCombination _ _ _ (b₂.sum_coord_apply_eq_one x)]
  /-
    case h
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    b₂ : AffineBasis ι k P
    x : P
    j : ι
    ⊢ Eq (Matrix.vecMul (b₂.coords x) (b.toMatrix ⇑b₂) j) ((Finset.affineCombinati …
  -/
  simp [Matrix.vecMul, dotProduct, toMatrix_apply, coords]
  /-
    🎉 no goals
  -/


theorem toMatrix_mul_toMatrix : b.toMatrix b₂ * b₂.toMatrix b = 1 := by
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : Ring k
    inst✝² : Module k V
    b : AffineBasis ι k P
    inst✝¹ : Fintype ι
    b₂ : AffineBasis ι k P
    inst✝ : DecidableEq ι
    ⊢ Eq (HMul.hMul (b.toMatrix ⇑b₂) (b₂.toMatrix ⇑b)) 1
  -/
  ext l m
  /-
    case a
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : Ring k
    inst✝² : Module k V
    b : AffineBasis ι k P
    inst✝¹ : Fintype ι
    b₂ : AffineBasis ι k P
    inst✝ : DecidableEq ι
    l m : ι
    ⊢ Eq (HMul.hMul (b.toMatrix ⇑b₂) (b₂.toMatrix ⇑b) l m) (1 l m)
  -/
  change (b.coords (b₂ l) ᵥ* b₂.toMatrix b) m = _
  /-
    case a
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : Ring k
    inst✝² : Module k V
    b : AffineBasis ι k P
    inst✝¹ : Fintype ι
    b₂ : AffineBasis ι k P
    inst✝ : DecidableEq ι
    l m : ι
    ⊢ Eq (Matrix.vecMul (b.coords (b₂ l)) (b₂.toMatrix ⇑b) m) (1 l m)
  -/
  rw [toMatrix_vecMul_coords, coords_apply, ← toMatrix_apply, toMatrix_self]
  /-
    🎉 no goals
  -/


theorem isUnit_toMatrix : IsUnit (b.toMatrix b₂) :=
  ⟨{  val := b.toMatrix b₂
      inv := b₂.toMatrix b
      val_inv := b.toMatrix_mul_toMatrix b₂
      inv_val := b₂.toMatrix_mul_toMatrix b }, rfl⟩


theorem isUnit_toMatrix_iff [Nontrivial k] (p : ι → P) :
    IsUnit (b.toMatrix p) ↔ AffineIndependent k p ∧ affineSpan k (range p) = ⊤ := by
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁶ : AddCommGroup V
    inst✝⁵ : AddTorsor V P
    inst✝⁴ : Ring k
    inst✝³ : Module k V
    b : AffineBasis ι k P
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : Nontrivial k
    p : ι → P
    ⊢ Iff (IsUnit (b.toMatrix p)) (And (AffineIndependent k p) (Eq (affineSpan k ( …
  -/
  constructor
    /-
      case mp
      ι : Type u₁
      k : Type u₂
      V : Type u₃
      P : Type u₄
      inst✝⁶ : AddCommGroup V
      inst✝⁵ : AddTorsor V P
      inst✝⁴ : Ring k
      inst✝³ : Module k V
      b : AffineBasis ι k P
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nontrivial k
      p : ι → P
      ⊢ IsUnit (b.toMatrix p) → And (AffineIndependent k p) (Eq (affineSpan k (Set.r …
    -/
  · rintro ⟨⟨B, A, hA, hA'⟩, rfl : B = b.toMatrix p⟩
    exact ⟨b.affineIndependent_of_toMatrix_right_inv p hA,
      b.affineSpan_eq_top_of_toMatrix_left_inv p hA'⟩
    /-
      case mpr
      ι : Type u₁
      k : Type u₂
      V : Type u₃
      P : Type u₄
      inst✝⁶ : AddCommGroup V
      inst✝⁵ : AddTorsor V P
      inst✝⁴ : Ring k
      inst✝³ : Module k V
      b : AffineBasis ι k P
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nontrivial k
      p : ι → P
      ⊢ And (AffineIndependent k p) (Eq (affineSpan k (Set.range p)) Top.top) → IsUn …
    -/
  · rintro ⟨h_tot, h_ind⟩
    /-
      case mpr.intro
      ι : Type u₁
      k : Type u₂
      V : Type u₃
      P : Type u₄
      inst✝⁶ : AddCommGroup V
      inst✝⁵ : AddTorsor V P
      inst✝⁴ : Ring k
      inst✝³ : Module k V
      b : AffineBasis ι k P
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nontrivial k
      p : ι → P
      h_tot : AffineIndependent k p
      h_ind : Eq (affineSpan k (Set.range p)) Top.top
      ⊢ IsUnit (b.toMatrix p)
    -/
    let b' : AffineBasis ι k P := ⟨p, h_tot, h_ind⟩
    /-
      case mpr.intro
      ι : Type u₁
      k : Type u₂
      V : Type u₃
      P : Type u₄
      inst✝⁶ : AddCommGroup V
      inst✝⁵ : AddTorsor V P
      inst✝⁴ : Ring k
      inst✝³ : Module k V
      b : AffineBasis ι k P
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nontrivial k
      p : ι → P
      h_tot : AffineIndependent k p
      h_ind : Eq (affineSpan k (Set.range p)) Top.top
      b' : AffineBasis ι k P := { toFun := p, ind' := h_tot, tot' := h_ind }
      ⊢ IsUnit (b.toMatrix p)
    -/
    change IsUnit (b.toMatrix b')
    /-
      case mpr.intro
      ι : Type u₁
      k : Type u₂
      V : Type u₃
      P : Type u₄
      inst✝⁶ : AddCommGroup V
      inst✝⁵ : AddTorsor V P
      inst✝⁴ : Ring k
      inst✝³ : Module k V
      b : AffineBasis ι k P
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nontrivial k
      p : ι → P
      h_tot : AffineIndependent k p
      h_ind : Eq (affineSpan k (Set.range p)) Top.top
      b' : AffineBasis ι k P := { toFun := p, ind' := h_tot, tot' := h_ind }
      ⊢ IsUnit (b.toMatrix ⇑b')
    -/
    exact b.isUnit_toMatrix b'
    /-
      🎉 no goals
    -/


/-- A change of basis formula for barycentric coordinates.

See also `AffineBasis.toMatrix_vecMul_coords`. -/
@[simp]
theorem toMatrix_inv_vecMul_toMatrix (x : P) :
    b.coords x ᵥ* (b.toMatrix b₂)⁻¹ = b₂.coords x := by
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : CommRing k
    inst✝² : Module k V
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b b₂ : AffineBasis ι k P
    x : P
    ⊢ Eq (Matrix.vecMul (b.coords x) (Inv.inv (b.toMatrix ⇑b₂))) (b₂.coords x)
  -/
  have hu := b.isUnit_toMatrix b₂
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : CommRing k
    inst✝² : Module k V
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b b₂ : AffineBasis ι k P
    x : P
    hu : IsUnit (b.toMatrix ⇑b₂)
    ⊢ Eq (Matrix.vecMul (b.coords x) (Inv.inv (b.toMatrix ⇑b₂))) (b₂.coords x)
  -/
  rw [Matrix.isUnit_iff_isUnit_det] at hu
  rw [← b.toMatrix_vecMul_coords b₂, Matrix.vecMul_vecMul, Matrix.mul_nonsing_inv _ hu,
    Matrix.vecMul_one]


/-- If we fix a background affine basis `b`, then for any other basis `b₂`, we can characterise
the barycentric coordinates provided by `b₂` in terms of determinants relative to `b`. -/
theorem det_smul_coords_eq_cramer_coords (x : P) :
    (b.toMatrix b₂).det • b₂.coords x = (b.toMatrix b₂)ᵀ.cramer (b.coords x) := by
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : CommRing k
    inst✝² : Module k V
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b b₂ : AffineBasis ι k P
    x : P
    ⊢ Eq (HSMul.hSMul (b.toMatrix ⇑b₂).det (b₂.coords x)) ((b.toMatrix ⇑b₂).transp …
  -/
  have hu := b.isUnit_toMatrix b₂
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : CommRing k
    inst✝² : Module k V
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b b₂ : AffineBasis ι k P
    x : P
    hu : IsUnit (b.toMatrix ⇑b₂)
    ⊢ Eq (HSMul.hSMul (b.toMatrix ⇑b₂).det (b₂.coords x)) ((b.toMatrix ⇑b₂).transp …
  -/
  rw [Matrix.isUnit_iff_isUnit_det] at hu
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : CommRing k
    inst✝² : Module k V
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b b₂ : AffineBasis ι k P
    x : P
    hu : IsUnit (b.toMatrix ⇑b₂).det
    ⊢ Eq (HSMul.hSMul (b.toMatrix ⇑b₂).det (b₂.coords x)) ((b.toMatrix ⇑b₂).transp …
  -/
  rw [← b.toMatrix_inv_vecMul_toMatrix, Matrix.det_smul_inv_vecMul_eq_cramer_transpose _ _ hu]
  /-
    🎉 no goals
  -/


