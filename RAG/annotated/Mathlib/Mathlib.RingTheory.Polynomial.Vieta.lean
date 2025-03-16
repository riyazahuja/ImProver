/-- A sum version of **Vieta's formula** for `Multiset`: the product of the linear terms `X + λ`
where `λ` runs through a multiset `s` is equal to a linear combination of the symmetric functions
`esymm s` of the `λ`'s . -/
theorem prod_X_add_C_eq_sum_esymm (s : Multiset R) :
    (s.map fun r => X + C r).prod =
      ∑ j ∈ Finset.range (Multiset.card s + 1), (C (s.esymm j) * X ^ (Multiset.card s - j)) := by
  classical
    rw [prod_map_add, antidiagonal_eq_map_powerset, map_map, ← bind_powerset_len,
      map_bind, sum_bind, Finset.sum_eq_multiset_sum, Finset.range_val, map_congr (Eq.refl _)]
    intro _ _
    rw [esymm, ← sum_hom', ← sum_map_mul_right, map_congr (Eq.refl _)]
    intro s ht
    rw [mem_powersetCard] at ht
    dsimp
    rw [prod_hom' s (Polynomial.C : R →+* R[X])]
    simp [ht, map_const, prod_replicate, prod_hom', map_id', card_sub]


/-- Vieta's formula for the coefficients of the product of linear terms `X + λ` where `λ` runs
through a multiset `s` : the `k`th coefficient is the symmetric function `esymm (card s - k) s`. -/
theorem prod_X_add_C_coeff (s : Multiset R) {k : ℕ} (h : k ≤ Multiset.card s) :
    (s.map fun r => X + C r).prod.coeff k = s.esymm (Multiset.card s - k) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    s : Multiset R
    k : Nat
    h : LE.le k s.card
    ⊢ Eq ((Multiset.map (fun r => HAdd.hAdd Polynomial.X (Polynomial.C r)) s).prod …
  -/
  convert Polynomial.ext_iff.mp (prod_X_add_C_eq_sum_esymm s) k using 1
  /-
    case h.e'_3
    R : Type u_1
    inst✝ : CommSemiring R
    s : Multiset R
    k : Nat
    h : LE.le k s.card
    ⊢ Eq (s.esymm (HSub.hSub s.card k)) (((Finset.range (HAdd.hAdd s.card 1)).sum  …
  -/
  simp_rw [finset_sum_coeff, coeff_C_mul_X_pow]
  /-
    case h.e'_3
    R : Type u_1
    inst✝ : CommSemiring R
    s : Multiset R
    k : Nat
    h : LE.le k s.card
    ⊢ Eq (s.esymm (HSub.hSub s.card k)) ((Finset.range (HAdd.hAdd s.card 1)).sum f …
  -/
  rw [Finset.sum_eq_single_of_mem (Multiset.card s - k) _]
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : CommSemiring R
      s : Multiset R
      k : Nat
      h : LE.le k s.card
      ⊢ Eq (s.esymm (HSub.hSub s.card k)) (ite (Eq k (HSub.hSub s.card (HSub.hSub s. …
    -/
  · rw [if_pos (Nat.sub_sub_self h).symm]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : CommSemiring R
      s : Multiset R
      k : Nat
      h : LE.le k s.card
      ⊢ ∀ (b : Nat), Membership.mem (Finset.range (HAdd.hAdd s.card 1)) b → Ne b (HS …
    -/
  · intro j hj1 hj2
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : CommSemiring R
      s : Multiset R
      k : Nat
      h : LE.le k s.card
      j : Nat
      hj1 : Membership.mem (Finset.range (HAdd.hAdd s.card 1)) j
      hj2 : Ne j (HSub.hSub s.card k)
      ⊢ Eq (ite (Eq k (HSub.hSub s.card j)) (s.esymm j) 0) 0
    -/
    suffices k ≠ card s - j by rw [if_neg this]
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : CommSemiring R
      s : Multiset R
      k : Nat
      h : LE.le k s.card
      j : Nat
      hj1 : Membership.mem (Finset.range (HAdd.hAdd s.card 1)) j
      hj2 : Ne j (HSub.hSub s.card k)
      ⊢ Ne k (HSub.hSub s.card j)
    -/
    intro hn
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : CommSemiring R
      s : Multiset R
      k : Nat
      h : LE.le k s.card
      j : Nat
      hj1 : Membership.mem (Finset.range (HAdd.hAdd s.card 1)) j
      hj2 : Ne j (HSub.hSub s.card k)
      hn : Eq k (HSub.hSub s.card j)
      ⊢ False
    -/
    rw [hn, Nat.sub_sub_self (Nat.lt_succ_iff.mp (Finset.mem_range.mp hj1))] at hj2
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : CommSemiring R
      s : Multiset R
      k : Nat
      h : LE.le k s.card
      j : Nat
      hj1 : Membership.mem (Finset.range (HAdd.hAdd s.card 1)) j
      hj2 : Ne j j
      hn : Eq k (HSub.hSub s.card j)
      ⊢ False
    -/
    exact Ne.irrefl hj2
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      s : Multiset R
      k : Nat
      h : LE.le k s.card
      ⊢ Membership.mem (Finset.range (HAdd.hAdd s.card 1)) (HSub.hSub s.card k)
    -/
  · rw [Finset.mem_range]
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      s : Multiset R
      k : Nat
      h : LE.le k s.card
      ⊢ LT.lt (HSub.hSub s.card k) (HAdd.hAdd s.card 1)
    -/
    exact Nat.lt_succ_of_le (Nat.sub_le (Multiset.card s) k)
    /-
      🎉 no goals
    -/


theorem prod_X_add_C_coeff' {σ} (s : Multiset σ) (r : σ → R) {k : ℕ} (h : k ≤ Multiset.card s) :
    (s.map fun i => X + C (r i)).prod.coeff k = (s.map r).esymm (Multiset.card s - k) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    σ : Type u_2
    s : Multiset σ
    r : σ → R
    k : Nat
    h : LE.le k s.card
    ⊢ Eq ((Multiset.map (fun i => HAdd.hAdd Polynomial.X (Polynomial.C (r i))) s). …
  -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  erw [← map_map (fun r => X + C r) r, prod_X_add_C_coeff] <;> rw [s.card_map r]; assumption
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem _root_.Finset.prod_X_add_C_coeff {σ} (s : Finset σ) (r : σ → R) {k : ℕ} (h : k ≤ #s) :
    (∏ i ∈ s, (X + C (r i))).coeff k = ∑ t ∈ s.powersetCard (#s - k), ∏ i ∈ t, r i := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    σ : Type u_2
    s : Finset σ
    r : σ → R
    k : Nat
    h : LE.le k s.card
    ⊢ Eq ((s.prod fun i => HAdd.hAdd Polynomial.X (Polynomial.C (r i))).coeff k) ( …
  -/
  rw [Finset.prod, prod_X_add_C_coeff' _ r h, Finset.esymm_map_val]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    σ : Type u_2
    s : Finset σ
    r : σ → R
    k : Nat
    h : LE.le k s.card
    ⊢ Eq ((Finset.powersetCard (HSub.hSub s.val.card k) s).sum fun t => t.prod r)  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem esymm_neg (s : Multiset R) (k : ℕ) : (map Neg.neg s).esymm k = (-1) ^ k * esymm s k := by
  rw [esymm, esymm, ← Multiset.sum_map_mul_left, Multiset.powersetCard_map, Multiset.map_map,
    map_congr rfl]
  /-
    R : Type u_1
    inst✝ : CommRing R
    s : Multiset R
    k : Nat
    ⊢ ∀ (x : Multiset R), Membership.mem (Multiset.powersetCard k s) x → Eq (Funct …
  -/
  intro x hx
  /-
    R : Type u_1
    inst✝ : CommRing R
    s : Multiset R
    k : Nat
    x : Multiset R
    hx : Membership.mem (Multiset.powersetCard k s) x
    ⊢ Eq (Function.comp Multiset.prod (Multiset.map Neg.neg) x) (HMul.hMul (HPow.h …
  -/
  rw [(mem_powersetCard.mp hx).right.symm, ← prod_replicate, ← Multiset.map_const]
  /-
    R : Type u_1
    inst✝ : CommRing R
    s : Multiset R
    k : Nat
    x : Multiset R
    hx : Membership.mem (Multiset.powersetCard k s) x
    ⊢ Eq (Function.comp Multiset.prod (Multiset.map Neg.neg) x) (HMul.hMul (Multis …
  -/
  nth_rw 3 [← map_id' x]
  /-
    R : Type u_1
    inst✝ : CommRing R
    s : Multiset R
    k : Nat
    x : Multiset R
    hx : Membership.mem (Multiset.powersetCard k s) x
    ⊢ Eq (Function.comp Multiset.prod (Multiset.map Neg.neg) x) (HMul.hMul (Multis …
  -/
  rw [← prod_map_mul, map_congr rfl, Function.comp_apply]
  /-
    R : Type u_1
    inst✝ : CommRing R
    s : Multiset R
    k : Nat
    x : Multiset R
    hx : Membership.mem (Multiset.powersetCard k s) x
    ⊢ ∀ (x_1 : R), Membership.mem x x_1 → Eq (HMul.hMul (Function.const R (-1) x_1 …
  -/
  exact fun z _ => neg_one_mul z
  /-
    🎉 no goals
  -/


theorem prod_X_sub_X_eq_sum_esymm (s : Multiset R) :
    (s.map fun t => X - C t).prod =
      ∑ j ∈ Finset.range (Multiset.card s + 1),
        (-1) ^ j * (C (s.esymm j) * X ^ (Multiset.card s - j)) := by
  conv_lhs =>
    congr
    congr
    ext x
    rw [sub_eq_add_neg]
    rw [← map_neg C x]
  /-
    R : Type u_1
    inst✝ : CommRing R
    s : Multiset R
    ⊢ Eq (Multiset.map (fun x => HAdd.hAdd Polynomial.X (Polynomial.C (Neg.neg x)) …
  -/
  convert prod_X_add_C_eq_sum_esymm (map (fun t => -t) s) using 1
    /-
      case h.e'_2
      R : Type u_1
      inst✝ : CommRing R
      s : Multiset R
      ⊢ Eq (Multiset.map (fun x => HAdd.hAdd Polynomial.X (Polynomial.C (Neg.neg x)) …
    -/
  · rw [map_map]; rfl
                  /-
                    🎉 no goals
                  -/
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : CommRing R
      s : Multiset R
      ⊢ Eq ((Finset.range (HAdd.hAdd s.card 1)).sum fun j => HMul.hMul (HPow.hPow (- …
    -/
  · simp only [esymm_neg, card_map, mul_assoc, map_mul, map_pow, map_neg, map_one]
    /-
      🎉 no goals
    -/


theorem prod_X_sub_C_coeff (s : Multiset R) {k : ℕ} (h : k ≤ Multiset.card s) :
    (s.map fun t => X - C t).prod.coeff k =
    (-1) ^ (Multiset.card s - k) * s.esymm (Multiset.card s - k) := by
  conv_lhs =>
    congr
    congr
    congr
    ext x
    rw [sub_eq_add_neg]
    rw [← map_neg C x]
  /-
    R : Type u_1
    inst✝ : CommRing R
    s : Multiset R
    k : Nat
    h : LE.le k s.card
    ⊢ Eq ((Multiset.map (fun x => HAdd.hAdd Polynomial.X (Polynomial.C (Neg.neg x) …
  -/
  convert prod_X_add_C_coeff (map (fun t => -t) s) _ using 1
    /-
      case h.e'_2
      R : Type u_1
      inst✝ : CommRing R
      s : Multiset R
      k : Nat
      h : LE.le k s.card
      ⊢ Eq ((Multiset.map (fun x => HAdd.hAdd Polynomial.X (Polynomial.C (Neg.neg x) …
    -/
  · rw [map_map]; rfl
                  /-
                    🎉 no goals
                  -/
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : CommRing R
      s : Multiset R
      k : Nat
      h : LE.le k s.card
      ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HSub.hSub s.card k)) (s.esymm (HSub.hSub s.ca …
    -/
  · rw [esymm_neg, card_map]
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      R : Type u_1
      inst✝ : CommRing R
      s : Multiset R
      k : Nat
      h : LE.le k s.card
      ⊢ LE.le k (Multiset.map (fun t => Neg.neg t) s).card
    -/
  · rwa [card_map]
    /-
      🎉 no goals
    -/


/-- Vieta's formula for the coefficients and the roots of a polynomial over an integral domain
  with as many roots as its degree. -/
theorem _root_.Polynomial.coeff_eq_esymm_roots_of_card [IsDomain R] {p : R[X]}
    (hroots : Multiset.card p.roots = p.natDegree) {k : ℕ} (h : k ≤ p.natDegree) :
    p.coeff k = p.leadingCoeff * (-1) ^ (p.natDegree - k) * p.roots.esymm (p.natDegree - k) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hroots : Eq p.roots.card p.natDegree
    k : Nat
    h : LE.le k p.natDegree
    ⊢ Eq (p.coeff k) (HMul.hMul (HMul.hMul p.leadingCoeff (HPow.hPow (-1) (HSub.hS …
  -/
  conv_lhs => rw [← C_leadingCoeff_mul_prod_multiset_X_sub_C hroots]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hroots : Eq p.roots.card p.natDegree
    k : Nat
    h : LE.le k p.natDegree
    ⊢ Eq ((HMul.hMul (Polynomial.C p.leadingCoeff) (Multiset.map (fun a => HSub.hS …
  -/
  rw [coeff_C_mul, mul_assoc]; congr
  /-
    case e_a
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hroots : Eq p.roots.card p.natDegree
    k : Nat
    h : LE.le k p.natDegree
    ⊢ Eq ((Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) p.roots …
  -/
  have : k ≤ card (roots p) := by rw [hroots]; exact h
  /-
    case e_a
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hroots : Eq p.roots.card p.natDegree
    k : Nat
    h : LE.le k p.natDegree
    this : LE.le k p.roots.card
    ⊢ Eq ((Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) p.roots …
  -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  convert p.roots.prod_X_sub_C_coeff this using 3 <;> rw [hroots]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- Vieta's formula for split polynomials over a field. -/
theorem _root_.Polynomial.coeff_eq_esymm_roots_of_splits {F} [Field F] {p : F[X]}
    (hsplit : p.Splits (RingHom.id F)) {k : ℕ} (h : k ≤ p.natDegree) :
    p.coeff k = p.leadingCoeff * (-1) ^ (p.natDegree - k) * p.roots.esymm (p.natDegree - k) :=
  Polynomial.coeff_eq_esymm_roots_of_card (splits_iff_card_roots.1 hsplit) h


/-- A sum version of Vieta's formula for `MvPolynomial`: viewing `X i` as variables,
the product of linear terms `λ + X i` is equal to a linear combination of
the symmetric polynomials `esymm σ R j`. -/
theorem MvPolynomial.prod_C_add_X_eq_sum_esymm :
    (∏ i : σ, (Polynomial.X + Polynomial.C (MvPolynomial.X i))) =
      ∑ j ∈ range (card σ + 1), Polynomial.C
        (MvPolynomial.esymm σ R j) * Polynomial.X ^ (card σ - j) := by
  /-
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    ⊢ Eq (Finset.univ.prod fun i => HAdd.hAdd Polynomial.X (Polynomial.C (MvPolyno …
  -/
  let s := Finset.univ.val.map fun i : σ => (MvPolynomial.X i : MvPolynomial σ R)
  have : Fintype.card σ = Multiset.card s := by
    rw [Multiset.card_map, ← Finset.card_univ, Finset.card_def]
  /-
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    s : Multiset (MvPolynomial σ R) := Multiset.map (fun i => MvPolynomial.X i) Fi …
    this : Eq (Fintype.card σ) s.card
    ⊢ Eq (Finset.univ.prod fun i => HAdd.hAdd Polynomial.X (Polynomial.C (MvPolyno …
  -/
  simp_rw [this, MvPolynomial.esymm_eq_multiset_esymm σ R, Finset.prod_eq_multiset_prod]
  /-
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    s : Multiset (MvPolynomial σ R) := Multiset.map (fun i => MvPolynomial.X i) Fi …
    this : Eq (Fintype.card σ) s.card
    ⊢ Eq (Multiset.map (fun i => HAdd.hAdd Polynomial.X (Polynomial.C (MvPolynomia …
  -/
  convert Multiset.prod_X_add_C_eq_sum_esymm s
  /-
    case h.e'_2.h.e'_3
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    s : Multiset (MvPolynomial σ R) := Multiset.map (fun i => MvPolynomial.X i) Fi …
    this : Eq (Fintype.card σ) s.card
    ⊢ Eq (Multiset.map (fun i => HAdd.hAdd Polynomial.X (Polynomial.C (MvPolynomia …
  -/
  simp_rw [s, Multiset.map_map, Function.comp_apply]
  /-
    🎉 no goals
  -/


theorem MvPolynomial.prod_X_add_C_coeff (k : ℕ) (h : k ≤ card σ) :
    (∏ i : σ, (Polynomial.X + Polynomial.C (MvPolynomial.X i)) : Polynomial _).coeff k =
    MvPolynomial.esymm σ R (card σ - k) := by
  /-
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    k : Nat
    h : LE.le k (Fintype.card σ)
    ⊢ Eq ((Finset.univ.prod fun i => HAdd.hAdd Polynomial.X (Polynomial.C (MvPolyn …
  -/
  let s := Finset.univ.val.map fun i => (MvPolynomial.X i : MvPolynomial σ R)
  have : Fintype.card σ = Multiset.card s := by
    rw [Multiset.card_map, ← Finset.card_univ, Finset.card_def]
  /-
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    k : Nat
    h : LE.le k (Fintype.card σ)
    s : Multiset (MvPolynomial σ R) := Multiset.map (fun i => MvPolynomial.X i) Fi …
    this : Eq (Fintype.card σ) s.card
    ⊢ Eq ((Finset.univ.prod fun i => HAdd.hAdd Polynomial.X (Polynomial.C (MvPolyn …
  -/
  rw [this] at h ⊢
  /-
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    k : Nat
    s : Multiset (MvPolynomial σ R) := Multiset.map (fun i => MvPolynomial.X i) Fi …
    h : LE.le k s.card
    this : Eq (Fintype.card σ) s.card
    ⊢ Eq ((Finset.univ.prod fun i => HAdd.hAdd Polynomial.X (Polynomial.C (MvPolyn …
  -/
  rw [MvPolynomial.esymm_eq_multiset_esymm σ R, Finset.prod_eq_multiset_prod]
  /-
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    k : Nat
    s : Multiset (MvPolynomial σ R) := Multiset.map (fun i => MvPolynomial.X i) Fi …
    h : LE.le k s.card
    this : Eq (Fintype.card σ) s.card
    ⊢ Eq ((Multiset.map (fun i => HAdd.hAdd Polynomial.X (Polynomial.C (MvPolynomi …
  -/
  convert Multiset.prod_X_add_C_coeff s h
  /-
    case h.e'_2.h.e'_3.h.e'_3
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    k : Nat
    s : Multiset (MvPolynomial σ R) := Multiset.map (fun i => MvPolynomial.X i) Fi …
    h : LE.le k s.card
    this : Eq (Fintype.card σ) s.card
    ⊢ Eq (Multiset.map (fun i => HAdd.hAdd Polynomial.X (Polynomial.C (MvPolynomia …
  -/
  dsimp
  /-
    case h.e'_2.h.e'_3.h.e'_3
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    k : Nat
    s : Multiset (MvPolynomial σ R) := Multiset.map (fun i => MvPolynomial.X i) Fi …
    h : LE.le k s.card
    this : Eq (Fintype.card σ) s.card
    ⊢ Eq (Multiset.map (fun i => HAdd.hAdd Polynomial.X (Polynomial.C (MvPolynomia …
  -/
  simp_rw [s, Multiset.map_map, Function.comp_apply]
  /-
    🎉 no goals
  -/


