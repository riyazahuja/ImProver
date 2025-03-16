local macro "C_simp" : tactic =>
  `(tactic| simp only [map_ofNat, C_0, C_1, C_neg, C_add, C_sub, C_mul, C_pow])


local macro "eval_simp" : tactic =>
  `(tactic| simp only [eval_C, eval_X, eval_neg, eval_add, eval_sub, eval_mul, eval_pow])


/-- The coordinate ring $R[W] := R[X, Y] / \langle W(X, Y) \rangle$ of `W`. -/
abbrev CoordinateRing : Type u :=
  AdjoinRoot W.polynomial


/-- The function field $R(W) := \mathrm{Frac}(R[W])$ of `W`. -/
abbrev FunctionField : Type u :=
  FractionRing W.CoordinateRing


noncomputable instance : Algebra R W.CoordinateRing :=
  Quotient.algebra R


noncomputable instance : Algebra R[X] W.CoordinateRing :=
  Quotient.algebra R[X]


instance : IsScalarTower R R[X] W.CoordinateRing :=
  Quotient.isScalarTower R R[X] _


instance [Subsingleton R] : Subsingleton W.CoordinateRing :=
  Module.subsingleton R[X] _

-- Porting note: added the abbreviation `mk` for `AdjoinRoot.mk W.polynomial`

/-- The natural ring homomorphism mapping an element of `R[X][Y]` to an element of `R[W]`. -/
noncomputable abbrev mk : R[X][Y] →+* W.CoordinateRing :=
  AdjoinRoot.mk W.polynomial

-- Porting note: added `classical` explicitly

/-- The basis $\{1, Y\}$ for the coordinate ring $R[W]$ over the polynomial ring $R[X]$. -/
protected noncomputable def basis : Basis (Fin 2) R[X] W.CoordinateRing := by
  classical exact (subsingleton_or_nontrivial R).by_cases (fun _ => default) fun _ =>
    (AdjoinRoot.powerBasis' W.monic_polynomial).basis.reindex <| finCongr W.natDegree_polynomial


lemma basis_apply (n : Fin 2) :
    CoordinateRing.basis W n = (AdjoinRoot.powerBasis' W.monic_polynomial).gen ^ (n : ℕ) := by
  classical
  nontriviality R
  rw [CoordinateRing.basis, Or.by_cases, dif_neg <| not_subsingleton R, Basis.reindex_apply,
    PowerBasis.basis_eq_pow]
  rfl

-- Porting note: added `@[simp]` in lieu of `coe_basis`

@[simp]
lemma basis_zero : CoordinateRing.basis W 0 = 1 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq ((WeierstrassCurve.Affine.CoordinateRing.basis W) 0) 1
  -/
  simpa only [basis_apply] using pow_zero _
  /-
    🎉 no goals
  -/

-- Porting note: added `@[simp]` in lieu of `coe_basis`

@[simp]
lemma basis_one : CoordinateRing.basis W 1 = mk W Y := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq ((WeierstrassCurve.Affine.CoordinateRing.basis W) 1) ((WeierstrassCurve.A …
  -/
  simpa only [basis_apply] using pow_one _
  /-
    🎉 no goals
  -/

-- Porting note: removed `@[simp]` in lieu of `basis_zero` and `basis_one`

lemma coe_basis : (CoordinateRing.basis W : Fin 2 → W.CoordinateRing) = ![1, mk W Y] := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq (⇑(WeierstrassCurve.Affine.CoordinateRing.basis W)) (Matrix.vecCons 1 (Ma …
  -/
  ext n
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    n : Fin 2
    ⊢ Eq ((WeierstrassCurve.Affine.CoordinateRing.basis W) n) (Matrix.vecCons 1 (M …
  -/
  fin_cases n
  /-
    case h.«0»
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq ((WeierstrassCurve.Affine.CoordinateRing.basis W) ((fun i => i) ⟨0, ⋯⟩))  …
  -/
  exacts [basis_zero W, basis_one W]
  /-
    🎉 no goals
  -/


variable {W} in
lemma smul (x : R[X]) (y : W.CoordinateRing) : x • y = mk W (C x) * y :=
  (algebraMap_smul W.CoordinateRing x y).symm


variable {W} in
lemma smul_basis_eq_zero {p q : R[X]} (hpq : p • (1 : W.CoordinateRing) + q • mk W Y = 0) :
    p = 0 ∧ q = 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    p q : Polynomial R
    hpq : Eq (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul q ((WeierstrassCurve.Affine …
    ⊢ And (Eq p 0) (Eq q 0)
  -/
  have h := Fintype.linearIndependent_iff.mp (CoordinateRing.basis W).linearIndependent ![p, q]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    p q : Polynomial R
    hpq : Eq (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul q ((WeierstrassCurve.Affine …
    h : Eq (Finset.univ.sum fun i => HSMul.hSMul (Matrix.vecCons p (Matrix.vecCons …
    ⊢ And (Eq p 0) (Eq q 0)
  -/
  erw [Fin.sum_univ_succ, basis_zero, Fin.sum_univ_one, basis_one] at h
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    p q : Polynomial R
    hpq : Eq (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul q ((WeierstrassCurve.Affine …
    h : Eq (HAdd.hAdd (HSMul.hSMul (Matrix.vecCons p (Matrix.vecCons q Matrix.vecE …
    ⊢ And (Eq p 0) (Eq q 0)
  -/
  exact ⟨h hpq 0, h hpq 1⟩
  /-
    🎉 no goals
  -/


variable {W} in
lemma exists_smul_basis_eq (x : W.CoordinateRing) :
    ∃ p q : R[X], p • (1 : W.CoordinateRing) + q • mk W Y = x := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x : W.CoordinateRing
    ⊢ Exists fun p => Exists fun q => Eq (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul …
  -/
  have h := (CoordinateRing.basis W).sum_equivFun x
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x : W.CoordinateRing
    h : Eq (Finset.univ.sum fun i => HSMul.hSMul ((WeierstrassCurve.Affine.Coordin …
    ⊢ Exists fun p => Exists fun q => Eq (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul …
  -/
  erw [Fin.sum_univ_succ, Fin.sum_univ_one, basis_zero, basis_one] at h
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x : W.CoordinateRing
    h : Eq (HAdd.hAdd (HSMul.hSMul ((WeierstrassCurve.Affine.CoordinateRing.basis  …
    ⊢ Exists fun p => Exists fun q => Eq (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul …
  -/
  exact ⟨_, _, h⟩
  /-
    🎉 no goals
  -/


lemma smul_basis_mul_C (y : R[X]) (p q : R[X]) :
    (p • (1 : W.CoordinateRing) + q • mk W Y) * mk W (C y) =
      (p * y) • (1 : W.CoordinateRing) + (q * y) • mk W Y := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    y p q : Polynomial R
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul q ((WeierstrassCurve …
  -/
  simp only [smul, _root_.map_mul]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    y p q : Polynomial R
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul ((WeierstrassCurve.Affine.CoordinateRing …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma smul_basis_mul_Y (p q : R[X]) : (p • (1 : W.CoordinateRing) + q • mk W Y) * mk W Y =
    (q * (X ^ 3 + C W.a₂ * X ^ 2 + C W.a₄ * X + C W.a₆)) • (1 : W.CoordinateRing) +
      (p - q * (C W.a₁ * X + C W.a₃)) • mk W Y := by
  have Y_sq : mk W Y ^ 2 =
      mk W (C (X ^ 3 + C W.a₂ * X ^ 2 + C W.a₄ * X + C W.a₆) - C (C W.a₁ * X + C W.a₃) * Y) := by
    exact AdjoinRoot.mk_eq_mk.mpr ⟨1, by rw [polynomial]; ring1⟩
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    p q : Polynomial R
    Y_sq : Eq (HPow.hPow ((WeierstrassCurve.Affine.CoordinateRing.mk W) Polynomial …
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul q ((WeierstrassCurve …
  -/
  simp only [smul, add_mul, mul_assoc, ← sq, Y_sq, C_sub, map_sub, C_mul, _root_.map_mul]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    p q : Polynomial R
    Y_sq : Eq (HPow.hPow ((WeierstrassCurve.Affine.CoordinateRing.mk W) Polynomial …
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((WeierstrassCurve.Affine.CoordinateRing.mk W) (Pol …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- The ring homomorphism `R[W] →+* S[W.map f]` induced by a ring homomorphism `f : R →+* S`. -/
noncomputable def map : W.CoordinateRing →+* (W.map f).toAffine.CoordinateRing :=
  AdjoinRoot.lift ((AdjoinRoot.of _).comp <| mapRingHom f)
    ((AdjoinRoot.root (WeierstrassCurve.map W f).toAffine.polynomial)) <| by
      /-
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        W : WeierstrassCurve.Affine R
        f : RingHom R S
        ⊢ Eq (Polynomial.eval₂ ((AdjoinRoot.of (WeierstrassCurve.map W f).toAffine.pol …
      -/
      rw [← eval₂_map, ← map_polynomial, AdjoinRoot.eval₂_root]
      /-
        🎉 no goals
      -/


lemma map_mk (x : R[X][Y]) : map W f (mk W x) = mk (W.map f) (x.map <| mapRingHom f) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    W : WeierstrassCurve.Affine R
    f : RingHom R S
    x : Polynomial (Polynomial R)
    ⊢ Eq ((WeierstrassCurve.Affine.CoordinateRing.map W f) ((WeierstrassCurve.Affi …
  -/
  rw [map, AdjoinRoot.lift_mk, ← eval₂_map]
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    W : WeierstrassCurve.Affine R
    f : RingHom R S
    x : Polynomial (Polynomial R)
    ⊢ Eq (Polynomial.eval₂ (AdjoinRoot.of (WeierstrassCurve.map W f).toAffine.poly …
  -/
  exact AdjoinRoot.aeval_eq <| x.map <| mapRingHom f
  /-
    🎉 no goals
  -/


variable {W} in
protected lemma map_smul (x : R[X]) (y : W.CoordinateRing) :
    map W f (x • y) = x.map f • map W f y := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    W : WeierstrassCurve.Affine R
    f : RingHom R S
    x : Polynomial R
    y : W.CoordinateRing
    ⊢ Eq ((WeierstrassCurve.Affine.CoordinateRing.map W f) (HSMul.hSMul x y)) (HSM …
  -/
  rw [smul, _root_.map_mul, map_mk, map_C, smul]
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    W : WeierstrassCurve.Affine R
    f : RingHom R S
    x : Polynomial R
    y : W.CoordinateRing
    ⊢ Eq (HMul.hMul ((WeierstrassCurve.Affine.CoordinateRing.mk (WeierstrassCurve. …
  -/
  rfl
  /-
    🎉 no goals
  -/


variable {f} in
lemma map_injective (hf : Function.Injective f) : Function.Injective <| map W f :=
  (injective_iff_map_eq_zero _).mpr fun y hy => by
    /-
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      W : WeierstrassCurve.Affine R
      f : RingHom R S
      hf : Function.Injective ⇑f
      y : W.CoordinateRing
      hy : Eq ((WeierstrassCurve.Affine.CoordinateRing.map W f) y) 0
      ⊢ Eq y 0
    -/
    obtain ⟨p, q, rfl⟩ := exists_smul_basis_eq y
    /-
      case intro.intro
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      W : WeierstrassCurve.Affine R
      f : RingHom R S
      hf : Function.Injective ⇑f
      p q : Polynomial R
      hy : Eq ((WeierstrassCurve.Affine.CoordinateRing.map W f) (HAdd.hAdd (HSMul.hS …
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul q ((WeierstrassCurve.Affine.Coo …
    -/
    simp_rw [map_add, CoordinateRing.map_smul, map_one, map_mk, map_X] at hy
    /-
      case intro.intro
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      W : WeierstrassCurve.Affine R
      f : RingHom R S
      hf : Function.Injective ⇑f
      p q : Polynomial R
      hy : Eq (HAdd.hAdd (HSMul.hSMul (Polynomial.map f p) 1) (HSMul.hSMul (Polynomi …
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul q ((WeierstrassCurve.Affine.Coo …
    -/
    obtain ⟨hp, hq⟩ := smul_basis_eq_zero hy
    /-
      case intro.intro.intro
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      W : WeierstrassCurve.Affine R
      f : RingHom R S
      hf : Function.Injective ⇑f
      p q : Polynomial R
      hy : Eq (HAdd.hAdd (HSMul.hSMul (Polynomial.map f p) 1) (HSMul.hSMul (Polynomi …
      hp : Eq (Polynomial.map f p) 0
      hq : Eq (Polynomial.map f q) 0
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul q ((WeierstrassCurve.Affine.Coo …
    -/
    rw [Polynomial.map_eq_zero_iff hf] at hp hq
    /-
      case intro.intro.intro
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      W : WeierstrassCurve.Affine R
      f : RingHom R S
      hf : Function.Injective ⇑f
      p q : Polynomial R
      hy : Eq (HAdd.hAdd (HSMul.hSMul (Polynomial.map f p) 1) (HSMul.hSMul (Polynomi …
      hp : Eq p 0
      hq : Eq q 0
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul q ((WeierstrassCurve.Affine.Coo …
    -/
    simp_rw [hp, hq, zero_smul, add_zero]
    /-
      🎉 no goals
    -/


instance [IsDomain R] : IsDomain W.CoordinateRing :=
  have : IsDomain (W.map <| algebraMap R <| FractionRing R).toAffine.CoordinateRing :=
    AdjoinRoot.isDomain_of_prime (irreducible_polynomial _).prime
  (map_injective W <| IsFractionRing.injective R <| FractionRing R).isDomain


/-- The class of the element $X - x$ in $R[W]$ for some $x \in R$. -/
noncomputable def XClass (x : R) : W.CoordinateRing :=
  mk W <| C <| X - C x


lemma XClass_ne_zero [Nontrivial R] (x : R) : XClass W x ≠ 0 :=
  AdjoinRoot.mk_ne_zero_of_natDegree_lt W.monic_polynomial (C_ne_zero.mpr <| X_sub_C_ne_zero x) <|
       /-
         R : Type u
         inst✝¹ : CommRing R
         W : WeierstrassCurve.Affine R
         inst✝ : Nontrivial R
         x : R
         ⊢ LT.lt (Polynomial.C (HSub.hSub Polynomial.X (Polynomial.C x))).natDegree W.p …
       -/
    by rw [natDegree_polynomial, natDegree_C]; norm_num1
                                               /-
                                                 🎉 no goals
                                               -/


/-- The class of the element $Y - y(X)$ in $R[W]$ for some $y(X) \in R[X]$. -/
noncomputable def YClass (y : R[X]) : W.CoordinateRing :=
  mk W <| Y - C y


lemma YClass_ne_zero [Nontrivial R] (y : R[X]) : YClass W y ≠ 0 :=
  AdjoinRoot.mk_ne_zero_of_natDegree_lt W.monic_polynomial (X_sub_C_ne_zero y) <|
       /-
         R : Type u
         inst✝¹ : CommRing R
         W : WeierstrassCurve.Affine R
         inst✝ : Nontrivial R
         y : Polynomial R
         ⊢ LT.lt (HSub.hSub Polynomial.X (Polynomial.C y)).natDegree W.polynomial.natDe …
       -/
    by rw [natDegree_polynomial, natDegree_X_sub_C]; norm_num1
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma C_addPolynomial (x y L : R) : mk W (C <| W.addPolynomial x y L) =
    mk W ((Y - C (linePolynomial x y L)) * (W.negPolynomial - C (linePolynomial x y L))) :=
                                 /-
                                   R : Type u
                                   inst✝ : CommRing R
                                   W : WeierstrassCurve.Affine R
                                   x y L : R
                                   ⊢ Eq (HSub.hSub (Polynomial.C (W.addPolynomial x y L)) (HMul.hMul (HSub.hSub P …
                                 -/
  AdjoinRoot.mk_eq_mk.mpr ⟨1, by rw [W.C_addPolynomial, add_sub_cancel_left, mul_one]⟩
                                 /-
                                   🎉 no goals
                                 -/


/-- The ideal $\langle X - x \rangle$ of $R[W]$ for some $x \in R$. -/
noncomputable def XIdeal (x : R) : Ideal W.CoordinateRing :=
  span {XClass W x}


/-- The ideal $\langle Y - y(X) \rangle$ of $R[W]$ for some $y(X) \in R[X]$. -/
noncomputable def YIdeal (y : R[X]) : Ideal W.CoordinateRing :=
  span {YClass W y}


/-- The ideal $\langle X - x, Y - y(X) \rangle$ of $R[W]$ for some $x \in R$ and $y(X) \in R[X]$. -/
noncomputable def XYIdeal (x : R) (y : R[X]) : Ideal W.CoordinateRing :=
  span {XClass W x, YClass W y}


lemma XYIdeal_eq₁ (x y L : R) : XYIdeal W x (C y) = XYIdeal W x (linePolynomial x y L) := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (WeierstrassCurve.Affine.CoordinateRing.XYIdeal W x (Polynomial.C y)) (We …
  -/
  simp only [XYIdeal, XClass, YClass, linePolynomial]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (Ideal.span (Insert.insert ((WeierstrassCurve.Affine.CoordinateRing.mk W) …
  -/
  rw [← span_pair_add_mul_right <| mk W <| C <| C <| -L, ← _root_.map_mul, ← map_add]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (Ideal.span (Insert.insert ((WeierstrassCurve.Affine.CoordinateRing.mk W) …
  -/
  apply congr_arg (_ ∘ _ ∘ _ ∘ _)
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (HAdd.hAdd (HSub.hSub Polynomial.X (Polynomial.C (Polynomial.C y))) (HMul …
  -/
  C_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (HAdd.hAdd (HSub.hSub Polynomial.X (Polynomial.C (Polynomial.C y))) (HMul …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma XYIdeal_add_eq (x₁ x₂ y₁ L : R) : XYIdeal W (W.addX x₁ x₂ L) (C <| W.addY x₁ x₂ y₁ L) =
    span {mk W <| W.negPolynomial - C (linePolynomial x₁ y₁ L)} ⊔ XIdeal W (W.addX x₁ x₂ L) := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    ⊢ Eq (WeierstrassCurve.Affine.CoordinateRing.XYIdeal W (W.addX x₁ x₂ L) (Polyn …
  -/
  simp only [XYIdeal, XIdeal, XClass, YClass, addY, negAddY, negY, negPolynomial, linePolynomial]
  rw [sub_sub <| -(Y : R[X][Y]), neg_sub_left (Y : R[X][Y]), map_neg, span_singleton_neg, sup_comm,
    ← span_insert, ← span_pair_add_mul_right <| mk W <| C <| C <| W.a₁ + L, ← _root_.map_mul,
    ← map_add]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    ⊢ Eq (Ideal.span (Insert.insert ((WeierstrassCurve.Affine.CoordinateRing.mk W) …
  -/
  apply congr_arg (_ ∘ _ ∘ _ ∘ _)
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    ⊢ Eq (HAdd.hAdd (HSub.hSub Polynomial.X (Polynomial.C (Polynomial.C (HSub.hSub …
  -/
  C_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    ⊢ Eq (HAdd.hAdd (HSub.hSub Polynomial.X (HSub.hSub (HSub.hSub (Neg.neg (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- The $R$-algebra isomorphism from $R[W] / \langle X - x, Y - y(X) \rangle$ to $R$ obtained by
evaluation at $y(X)$ and at $x$ provided that $W(x, y(x)) = 0$. -/
noncomputable def quotientXYIdealEquiv {x : R} {y : R[X]} (h : (W.polynomial.eval y).eval x = 0) :
    (W.CoordinateRing ⧸ XYIdeal W x y) ≃ₐ[R] R :=
  ((quotientEquivAlgOfEq R <| by
      /-
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        W : WeierstrassCurve.Affine R
        f : RingHom R S
        x : R
        y : Polynomial R
        h : Eq (Polynomial.eval x (Polynomial.eval y W.polynomial)) 0
        ⊢ Eq (WeierstrassCurve.Affine.CoordinateRing.XYIdeal W x y) (Ideal.map (Ideal. …
      -/
      simp only [XYIdeal, XClass, YClass, ← Set.image_pair, ← map_span]; rfl).trans <|
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
        DoubleQuot.quotQuotEquivQuotOfLEₐ R <| (span_singleton_le_iff_mem _).mpr <|
          mem_span_C_X_sub_C_X_sub_C_iff_eval_eval_eq_zero.mpr h).trans
    quotientSpanCXSubCXSubCAlgEquiv


lemma C_addPolynomial_slope {x₁ x₂ y₁ y₂ : F} (h₁ : W.Equation x₁ y₁) (h₂ : W.Equation x₂ y₂)
    (hxy : x₁ = x₂ → y₁ ≠ W.negY x₂ y₂) : mk W (C <| W.addPolynomial x₁ y₁ <| W.slope x₁ x₂ y₁ y₂) =
      -(XClass W x₁ * XClass W x₂ * XClass W (W.addX x₁ x₂ <| W.slope x₁ x₂ y₁ y₂)) := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq ((WeierstrassCurve.Affine.CoordinateRing.mk W) (Polynomial.C (W.addPolyno …
  -/
  simp only [addPolynomial_slope h₁ h₂ hxy, C_neg, mk, map_neg, neg_inj, _root_.map_mul]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (HMul.hMul (HMul.hMul ((AdjoinRoot.mk W.polynomial) (Polynomial.C (HSub.h …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma XYIdeal_eq₂ {x₁ x₂ y₁ y₂ : F} (h₁ : W.Equation x₁ y₁)
    (h₂ : W.Equation x₂ y₂) (hxy : x₁ = x₂ → y₁ ≠ W.negY x₂ y₂) :
    XYIdeal W x₂ (C y₂) = XYIdeal W x₂ (linePolynomial x₁ y₁ <| W.slope x₁ x₂ y₁ y₂) := by
  have hy₂ : y₂ = (linePolynomial x₁ y₁ <| W.slope x₁ x₂ y₁ y₂).eval x₂ := by
    by_cases hx : x₁ = x₂
    · rcases hx, Y_eq_of_Y_ne h₁ h₂ hx <| hxy hx with ⟨rfl, rfl⟩
      field_simp [linePolynomial, sub_ne_zero_of_ne <| hxy rfl]
    · field_simp [linePolynomial, slope_of_X_ne hx, sub_ne_zero_of_ne hx]
      ring1
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    hy₂ : Eq y₂ (Polynomial.eval x₂ (WeierstrassCurve.Affine.linePolynomial x₁ y₁  …
    ⊢ Eq (WeierstrassCurve.Affine.CoordinateRing.XYIdeal W x₂ (Polynomial.C y₂)) ( …
  -/
  nth_rw 1 [hy₂]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    hy₂ : Eq y₂ (Polynomial.eval x₂ (WeierstrassCurve.Affine.linePolynomial x₁ y₁  …
    ⊢ Eq (WeierstrassCurve.Affine.CoordinateRing.XYIdeal W x₂ (Polynomial.C (Polyn …
  -/
  simp only [XYIdeal, XClass, YClass, linePolynomial]
  rw [← span_pair_add_mul_right <| mk W <| C <| C <| -W.slope x₁ x₂ y₁ y₂, ← _root_.map_mul,
    ← map_add]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    hy₂ : Eq y₂ (Polynomial.eval x₂ (WeierstrassCurve.Affine.linePolynomial x₁ y₁  …
    ⊢ Eq (Ideal.span (Insert.insert ((WeierstrassCurve.Affine.CoordinateRing.mk W) …
  -/
  apply congr_arg (_ ∘ _ ∘ _ ∘ _)
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    hy₂ : Eq y₂ (Polynomial.eval x₂ (WeierstrassCurve.Affine.linePolynomial x₁ y₁  …
    ⊢ Eq (HAdd.hAdd (HSub.hSub Polynomial.X (Polynomial.C (Polynomial.C (Polynomia …
  -/
  eval_simp
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    hy₂ : Eq y₂ (Polynomial.eval x₂ (WeierstrassCurve.Affine.linePolynomial x₁ y₁  …
    ⊢ Eq (HAdd.hAdd (HSub.hSub Polynomial.X (Polynomial.C (Polynomial.C (HAdd.hAdd …
  -/
  C_simp
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    hy₂ : Eq y₂ (Polynomial.eval x₂ (WeierstrassCurve.Affine.linePolynomial x₁ y₁  …
    ⊢ Eq (HAdd.hAdd (HSub.hSub Polynomial.X (HAdd.hAdd (HMul.hMul (Polynomial.C (P …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma XYIdeal_neg_mul {x y : F} (h : W.Nonsingular x y) :
    XYIdeal W x (C <| W.negY x y) * XYIdeal W x (C y) = XIdeal W x := by
  have Y_rw : (Y - C (C y)) * (Y - C (C <| W.negY x y)) -
      C (X - C x) * (C (X ^ 2 + C (x + W.a₂) * X + C (x ^ 2 + W.a₂ * x + W.a₄)) - C (C W.a₁) * Y) =
        W.polynomial * 1 := by
    linear_combination (norm := (rw [negY, polynomial]; C_simp; ring1))
      congr_arg C (congr_arg C ((equation_iff ..).mp h.left).symm)
  simp_rw [XYIdeal, XClass, YClass, span_pair_mul_span_pair, mul_comm, ← _root_.map_mul,
    AdjoinRoot.mk_eq_mk.mpr ⟨1, Y_rw⟩, _root_.map_mul, span_insert,
    ← span_singleton_mul_span_singleton, ← Ideal.mul_sup, ← span_insert]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x y : F
    h : W.Nonsingular x y
    Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
    ⊢ Eq (HMul.hMul (Ideal.span (Singleton.singleton ((WeierstrassCurve.Affine.Coo …
  -/
  convert mul_top (_ : Ideal W.CoordinateRing) using 2
  /-
    case h.e'_2.h.e'_6
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x y : F
    h : W.Nonsingular x y
    Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
    ⊢ Eq (Ideal.span (Insert.insert ((WeierstrassCurve.Affine.CoordinateRing.mk W) …
  -/
  simp_rw [← Set.image_singleton (f := mk W), ← Set.image_insert_eq, ← map_span]
  /-
    case h.e'_2.h.e'_6
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x y : F
    h : W.Nonsingular x y
    Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
    ⊢ Eq (Ideal.map (WeierstrassCurve.Affine.CoordinateRing.mk W) (Ideal.span (Ins …
  -/
  convert map_top (R := F[X][Y]) (mk W) using 1
  /-
    case h.e'_2
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x y : F
    h : W.Nonsingular x y
    Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
    ⊢ Eq (Ideal.map (WeierstrassCurve.Affine.CoordinateRing.mk W) (Ideal.span (Ins …
  -/
  apply congr_arg
  /-
    case h.e'_2.h
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x y : F
    h : W.Nonsingular x y
    Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
    ⊢ Eq (Ideal.span (Insert.insert (Polynomial.C (HSub.hSub Polynomial.X (Polynom …
  -/
  simp_rw [eq_top_iff_one, mem_span_insert', mem_span_singleton']
  /-
    case h.e'_2.h
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x y : F
    h : W.Nonsingular x y
    Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
    ⊢ Exists fun a => Exists fun a_1 => Exists fun a_2 => Exists fun a_3 => Eq (HM …
  -/
  rcases ((nonsingular_iff' ..).mp h).right with hx | hy
    /-
      case h.e'_2.h.inl
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x y : F
      h : W.Nonsingular x y
      Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
      hx : Ne (HSub.hSub (HMul.hMul W.a₁ y) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow …
      ⊢ Exists fun a => Exists fun a_1 => Exists fun a_2 => Exists fun a_3 => Eq (HM …
    -/
  · let W_X := W.a₁ * y - (3 * x ^ 2 + 2 * W.a₂ * x + W.a₄)
    refine
      ⟨C <| C W_X⁻¹ * -(X + C (2 * x + W.a₂)), C <| C <| W_X⁻¹ * W.a₁, 0, C <| C <| W_X⁻¹ * -1, ?_⟩
    /-
      case h.e'_2.h.inl
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x y : F
      h : W.Nonsingular x y
      Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
      hx : Ne (HSub.hSub (HMul.hMul W.a₁ y) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow …
      W_X : F := HSub.hSub (HMul.hMul W.a₁ y) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HP …
      ⊢ Eq (HMul.hMul (Polynomial.C (Polynomial.C (HMul.hMul (Inv.inv W_X) (-1)))) ( …
    -/
    rw [← mul_right_inj' <| C_ne_zero.mpr <| C_ne_zero.mpr hx]
    /-
      case h.e'_2.h.inl
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x y : F
      h : W.Nonsingular x y
      Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
      hx : Ne (HSub.hSub (HMul.hMul W.a₁ y) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow …
      W_X : F := HSub.hSub (HMul.hMul W.a₁ y) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HP …
      ⊢ Eq (HMul.hMul (Polynomial.C (Polynomial.C (HSub.hSub (HMul.hMul W.a₁ y) (HAd …
    -/
    simp only [W_X, mul_add, ← mul_assoc, ← C_mul, mul_inv_cancel₀ hx]
    /-
      case h.e'_2.h.inl
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x y : F
      h : W.Nonsingular x y
      Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
      hx : Ne (HSub.hSub (HMul.hMul W.a₁ y) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow …
      W_X : F := HSub.hSub (HMul.hMul W.a₁ y) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HP …
      ⊢ Eq (HMul.hMul (Polynomial.C (Polynomial.C (HMul.hMul 1 (-1)))) (HSub.hSub (P …
    -/
    C_simp
    /-
      case h.e'_2.h.inl
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x y : F
      h : W.Nonsingular x y
      Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
      hx : Ne (HSub.hSub (HMul.hMul W.a₁ y) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow …
      W_X : F := HSub.hSub (HMul.hMul W.a₁ y) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HP …
      ⊢ Eq (HMul.hMul (HMul.hMul 1 (-1)) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow …
    -/
    ring1
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.inr
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x y : F
      h : W.Nonsingular x y
      Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
      hy : Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃) 0
      ⊢ Exists fun a => Exists fun a_1 => Exists fun a_2 => Exists fun a_3 => Eq (HM …
    -/
  · let W_Y := 2 * y + W.a₁ * x + W.a₃
    /-
      case h.e'_2.h.inr
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x y : F
      h : W.Nonsingular x y
      Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
      hy : Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃) 0
      W_Y : F := HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃
      ⊢ Exists fun a => Exists fun a_1 => Exists fun a_2 => Exists fun a_3 => Eq (HM …
    -/
    refine ⟨0, C <| C W_Y⁻¹, C <| C <| W_Y⁻¹ * -1, 0, ?_⟩
    /-
      case h.e'_2.h.inr
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x y : F
      h : W.Nonsingular x y
      Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
      hy : Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃) 0
      W_Y : F := HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃
      ⊢ Eq (HMul.hMul 0 (HSub.hSub (Polynomial.C (HAdd.hAdd (HAdd.hAdd (HPow.hPow Po …
    -/
    rw [negY, ← mul_right_inj' <| C_ne_zero.mpr <| C_ne_zero.mpr hy]
    /-
      case h.e'_2.h.inr
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x y : F
      h : W.Nonsingular x y
      Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
      hy : Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃) 0
      W_Y : F := HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃
      ⊢ Eq (HMul.hMul (Polynomial.C (Polynomial.C (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 …
    -/
    simp only [W_Y, mul_add, ← mul_assoc, ← C_mul, mul_inv_cancel₀ hy]
    /-
      case h.e'_2.h.inr
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x y : F
      h : W.Nonsingular x y
      Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
      hy : Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃) 0
      W_Y : F := HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃
      ⊢ Eq (HMul.hMul (HMul.hMul (Polynomial.C (Polynomial.C (HAdd.hAdd (HAdd.hAdd ( …
    -/
    C_simp
    /-
      case h.e'_2.h.inr
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x y : F
      h : W.Nonsingular x y
      Y_rw : Eq (HSub.hSub (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (Polynom …
      hy : Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃) 0
      W_Y : F := HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃
      ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 (Polynomial.C (P …
    -/
    ring1
    /-
      🎉 no goals
    -/


private lemma XYIdeal'_mul_inv {x y : F} (h : W.Nonsingular x y) :
    XYIdeal W x (C y) * (XYIdeal W x (C <| W.negY x y) *
        (XIdeal W x : FractionalIdeal W.CoordinateRing⁰ W.FunctionField)⁻¹) = 1 := by
  rw [← mul_assoc, ← FractionalIdeal.coeIdeal_mul, mul_comm <| XYIdeal W .., XYIdeal_neg_mul h,
    XIdeal, FractionalIdeal.coe_ideal_span_singleton_mul_inv W.FunctionField <| XClass_ne_zero W x]


lemma XYIdeal_mul_XYIdeal {x₁ x₂ y₁ y₂ : F} (h₁ : W.Equation x₁ y₁) (h₂ : W.Equation x₂ y₂)
    (hxy : x₁ = x₂ → y₁ ≠ W.negY x₂ y₂) :
    XIdeal W (W.addX x₁ x₂ <| W.slope x₁ x₂ y₁ y₂) * (XYIdeal W x₁ (C y₁) * XYIdeal W x₂ (C y₂)) =
      YIdeal W (linePolynomial x₁ y₁ <| W.slope x₁ x₂ y₁ y₂) *
        XYIdeal W (W.addX x₁ x₂ <| W.slope x₁ x₂ y₁ y₂)
          (C <| W.addY x₁ x₂ y₁ <| W.slope x₁ x₂ y₁ y₂) := by
  have sup_rw : ∀ a b c d : Ideal W.CoordinateRing, a ⊔ (b ⊔ (c ⊔ d)) = a ⊔ d ⊔ b ⊔ c :=
    fun _ _ c _ => by rw [← sup_assoc, sup_comm c, sup_sup_sup_comm, ← sup_assoc]
  rw [XYIdeal_add_eq, XIdeal, mul_comm, XYIdeal_eq₁ W x₁ y₁ <| W.slope x₁ x₂ y₁ y₂, XYIdeal,
    XYIdeal_eq₂ h₁ h₂ hxy, XYIdeal, span_pair_mul_span_pair]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
    ⊢ Eq (HMul.hMul (Ideal.span (Insert.insert (HMul.hMul (WeierstrassCurve.Affine …
  -/
  simp_rw [span_insert, sup_rw, Ideal.sup_mul, span_singleton_mul_span_singleton]
  rw [← neg_eq_iff_eq_neg.mpr <| C_addPolynomial_slope h₁ h₂ hxy, span_singleton_neg,
    C_addPolynomial, _root_.map_mul, YClass]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
    ⊢ Eq (Max.max (Max.max (Max.max (Ideal.span (Singleton.singleton (HMul.hMul (( …
  -/
  simp_rw [mul_comm <| XClass W x₁, mul_assoc, ← span_singleton_mul_span_singleton, ← Ideal.mul_sup]
  rw [span_singleton_mul_span_singleton, ← span_insert,
    ← span_pair_add_mul_right <| -(XClass W <| W.addX x₁ x₂ <| W.slope x₁ x₂ y₁ y₂), mul_neg,
    ← sub_eq_add_neg, ← sub_mul, ← map_sub <| mk W, sub_sub_sub_cancel_right, span_insert,
    ← span_singleton_mul_span_singleton, ← sup_rw, ← Ideal.sup_mul, ← Ideal.sup_mul]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
    ⊢ Eq (HMul.hMul (Ideal.span (Singleton.singleton ((WeierstrassCurve.Affine.Coo …
  -/
  apply congr_arg (_ ∘ _)
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
    ⊢ Eq (HMul.hMul (Max.max (Ideal.span (Singleton.singleton (WeierstrassCurve.Af …
  -/
  convert top_mul (_ : Ideal W.CoordinateRing)
  simp_rw [XClass, ← Set.image_singleton (f := mk W), ← map_span, ← Ideal.map_sup, eq_top_iff_one,
    mem_map_iff_of_surjective _ AdjoinRoot.mk_surjective, ← span_insert, mem_span_insert',
    mem_span_singleton']
  /-
    case h.e'_2.h.e'_5
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
    ⊢ Exists fun x => And (Exists fun a => Exists fun a_1 => Exists fun a_2 => Eq  …
  -/
  by_cases hx : x₁ = x₂
    /-
      case pos
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : W.Equation x₁ y₁
      h₂ : W.Equation x₂ y₂
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
      hx : Eq x₁ x₂
      ⊢ Exists fun x => And (Exists fun a => Exists fun a_1 => Exists fun a_2 => Eq  …
    -/
  · rcases hx, Y_eq_of_Y_ne h₁ h₂ hx (hxy hx) with ⟨rfl, rfl⟩
    /-
      case pos
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ y₁ : F
      h₁ : W.Equation x₁ y₁
      sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
      h₂ : W.Equation x₁ y₁
      hxy : Eq x₁ x₁ → Ne y₁ (W.negY x₁ y₁)
      ⊢ Exists fun x => And (Exists fun a => Exists fun a_1 => Exists fun a_2 => Eq  …
    -/
    let y := (y₁ - W.negY x₁ y₁) ^ 2
    /-
      case pos
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ y₁ : F
      h₁ : W.Equation x₁ y₁
      sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
      h₂ : W.Equation x₁ y₁
      hxy : Eq x₁ x₁ → Ne y₁ (W.negY x₁ y₁)
      y : F := HPow.hPow (HSub.hSub y₁ (W.negY x₁ y₁)) 2
      ⊢ Exists fun x => And (Exists fun a => Exists fun a_1 => Exists fun a_2 => Eq  …
    -/
    replace hxy := pow_ne_zero 2 <| sub_ne_zero_of_ne <| hxy rfl
    refine ⟨1 + C (C <| y⁻¹ * 4) * W.polynomial,
      ⟨C <| C y⁻¹ * (C 4 * X ^ 2 + C (4 * x₁ + W.b₂) * X + C (4 * x₁ ^ 2 + W.b₂ * x₁ + 2 * W.b₄)),
        0, C (C y⁻¹) * (Y - W.negPolynomial), ?_⟩, by
      rw [map_add, map_one, _root_.map_mul <| mk W, AdjoinRoot.mk_self, mul_zero, add_zero]⟩
    /-
      case pos
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ y₁ : F
      h₁ : W.Equation x₁ y₁
      sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
      h₂ : W.Equation x₁ y₁
      y : F := HPow.hPow (HSub.hSub y₁ (W.negY x₁ y₁)) 2
      hxy : Ne (HPow.hPow (HSub.hSub y₁ (W.negY x₁ y₁)) 2) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (Polynomial.C (Polynomial.C (Inv.inv y))) (HSub.hSu …
    -/
    rw [polynomial, negPolynomial, ← mul_right_inj' <| C_ne_zero.mpr <| C_ne_zero.mpr hxy]
    /-
      case pos
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ y₁ : F
      h₁ : W.Equation x₁ y₁
      sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
      h₂ : W.Equation x₁ y₁
      y : F := HPow.hPow (HSub.hSub y₁ (W.negY x₁ y₁)) 2
      hxy : Ne (HPow.hPow (HSub.hSub y₁ (W.negY x₁ y₁)) 2) 0
      ⊢ Eq (HMul.hMul (Polynomial.C (Polynomial.C (HPow.hPow (HSub.hSub y₁ (W.negY x …
    -/
    simp only [y, mul_add, ← mul_assoc, ← C_mul, mul_inv_cancel₀ hxy]
    linear_combination (norm := (rw [b₂, b₄, negY]; C_simp; ring1))
      -4 * congr_arg C (congr_arg C <| (equation_iff ..).mp h₁)
    /-
      case neg
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : W.Equation x₁ y₁
      h₂ : W.Equation x₂ y₂
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
      hx : Not (Eq x₁ x₂)
      ⊢ Exists fun x => And (Exists fun a => Exists fun a_1 => Exists fun a_2 => Eq  …
    -/
  · replace hx := sub_ne_zero_of_ne hx
    /-
      case neg
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : W.Equation x₁ y₁
      h₂ : W.Equation x₂ y₂
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
      hx : Ne (HSub.hSub x₁ x₂) 0
      ⊢ Exists fun x => And (Exists fun a => Exists fun a_1 => Exists fun a_2 => Eq  …
    -/
    refine ⟨_, ⟨⟨C <| C (x₁ - x₂)⁻¹, C <| C <| (x₁ - x₂)⁻¹ * -1, 0, ?_⟩, map_one _⟩⟩
    /-
      case neg
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : W.Equation x₁ y₁
      h₂ : W.Equation x₂ y₂
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
      hx : Ne (HSub.hSub x₁ x₂) 0
      ⊢ Eq (HMul.hMul 0 (HSub.hSub Polynomial.X W.negPolynomial)) (HAdd.hAdd (HAdd.h …
    -/
    rw [← mul_right_inj' <| C_ne_zero.mpr <| C_ne_zero.mpr hx]
    /-
      case neg
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : W.Equation x₁ y₁
      h₂ : W.Equation x₂ y₂
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
      hx : Ne (HSub.hSub x₁ x₂) 0
      ⊢ Eq (HMul.hMul (Polynomial.C (Polynomial.C (HSub.hSub x₁ x₂))) (HMul.hMul 0 ( …
    -/
    simp only [← mul_assoc, mul_add, ← C_mul, mul_inv_cancel₀ hx]
    /-
      case neg
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : W.Equation x₁ y₁
      h₂ : W.Equation x₂ y₂
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
      hx : Ne (HSub.hSub x₁ x₂) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (Polynomial.C (Polynomial.C (HSub.hSub x₁ x₂))) 0)  …
    -/
    C_simp
    /-
      case neg
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : W.Equation x₁ y₁
      h₂ : W.Equation x₂ y₂
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      sup_rw : ∀ (a b c d : Ideal W.CoordinateRing), Eq (Max.max a (Max.max b (Max.m …
      hx : Ne (HSub.hSub x₁ x₂) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub (Polynomial.C (Polynomial.C x₁)) (Polyno …
    -/
    ring1
    /-
      🎉 no goals
    -/


/-- The non-zero fractional ideal $\langle X - x, Y - y \rangle$ of $F(W)$ for some $x, y \in F$. -/
noncomputable def XYIdeal' {x y : F} (h : W.Nonsingular x y) :
    (FractionalIdeal W.CoordinateRing⁰ W.FunctionField)ˣ :=
  Units.mkOfMulEqOne _ _ <| XYIdeal'_mul_inv h


lemma XYIdeal'_eq {x y : F} (h : W.Nonsingular x y) :
    (XYIdeal' h : FractionalIdeal W.CoordinateRing⁰ W.FunctionField) = XYIdeal W x (C y) :=
  rfl


lemma mk_XYIdeal'_mul_mk_XYIdeal'_of_Yeq {x y : F} (h : W.Nonsingular x y) :
    ClassGroup.mk (XYIdeal' <| nonsingular_neg h) * ClassGroup.mk (XYIdeal' h) = 1 := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x y : F
    h : W.Nonsingular x y
    ⊢ Eq (HMul.hMul (ClassGroup.mk (WeierstrassCurve.Affine.CoordinateRing.XYIdeal …
  -/
  rw [← _root_.map_mul]
  exact
    (ClassGroup.mk_eq_one_of_coe_ideal <| by exact (FractionalIdeal.coeIdeal_mul ..).symm.trans <|
      FractionalIdeal.coeIdeal_inj.mpr <| XYIdeal_neg_mul h).mpr ⟨_, XClass_ne_zero W _, rfl⟩


lemma mk_XYIdeal'_mul_mk_XYIdeal' {x₁ x₂ y₁ y₂ : F} (h₁ : W.Nonsingular x₁ y₁)
    (h₂ : W.Nonsingular x₂ y₂) (hxy : x₁ = x₂ → y₁ ≠ W.negY x₂ y₂) :
    ClassGroup.mk (XYIdeal' h₁) * ClassGroup.mk (XYIdeal' h₂) =
      ClassGroup.mk (XYIdeal' <| nonsingular_add h₁ h₂ hxy) := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Nonsingular x₁ y₁
    h₂ : W.Nonsingular x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (HMul.hMul (ClassGroup.mk (WeierstrassCurve.Affine.CoordinateRing.XYIdeal …
  -/
  rw [← _root_.map_mul]
  exact (ClassGroup.mk_eq_mk_of_coe_ideal (by exact (FractionalIdeal.coeIdeal_mul ..).symm) <|
      XYIdeal'_eq _).mpr
    ⟨_, _, XClass_ne_zero W _, YClass_ne_zero W _, XYIdeal_mul_XYIdeal h₁.left h₂.left hxy⟩


lemma norm_smul_basis (p q : R[X]) :
    Algebra.norm R[X] (p • (1 : W.CoordinateRing) + q • mk W Y) =
      p ^ 2 - p * q * (C W.a₁ * X + C W.a₃) -
        q ^ 2 * (X ^ 3 + C W.a₂ * X ^ 2 + C W.a₄ * X + C W.a₆) := by
  simp_rw [Algebra.norm_eq_matrix_det <| CoordinateRing.basis W, Matrix.det_fin_two,
    Algebra.leftMulMatrix_eq_repr_mul, basis_zero, mul_one, basis_one, smul_basis_mul_Y, map_add,
    Finsupp.add_apply, map_smul, Finsupp.smul_apply, ← basis_zero, ← basis_one,
    Basis.repr_self_apply, if_pos, one_ne_zero, if_false, smul_eq_mul]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    p q : Polynomial R
    ⊢ Eq (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul p 1) (HMul.hMul q 0)) (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma coe_norm_smul_basis (p q : R[X]) :
    Algebra.norm R[X] (p • (1 : W.CoordinateRing) + q • mk W Y) =
      mk W ((C p + C q * X) * (C p + C q * (-(Y : R[X][Y]) - C (C W.a₁ * X + C W.a₃)))) :=
  AdjoinRoot.mk_eq_mk.mpr
                 /-
                   R : Type u
                   inst✝ : CommRing R
                   W : WeierstrassCurve.Affine R
                   p q : Polynomial R
                   ⊢ Eq (HSub.hSub (Polynomial.C ((Algebra.norm (Polynomial R)) (HAdd.hAdd (HSMul …
                 -/
    ⟨C q ^ 2, by simp only [norm_smul_basis, polynomial]; C_simp; ring1⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma degree_norm_smul_basis [IsDomain R] (p q : R[X]) :
    (Algebra.norm R[X] <| p • (1 : W.CoordinateRing) + q • mk W Y).degree =
      max (2 • p.degree) (2 • q.degree + 3) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    p q : Polynomial R
    ⊢ Eq ((Algebra.norm (Polynomial R)) (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul  …
  -/
  have hdp : (p ^ 2).degree = 2 • p.degree := degree_pow p 2
  have hdpq : (p * q * (C W.a₁ * X + C W.a₃)).degree ≤ p.degree + q.degree + 1 := by
    simpa only [degree_mul] using add_le_add_left degree_linear_le (p.degree + q.degree)
  have hdq :
      (q ^ 2 * (X ^ 3 + C W.a₂ * X ^ 2 + C W.a₄ * X + C W.a₆)).degree = 2 • q.degree + 3 := by
    rw [degree_mul, degree_pow, ← one_mul <| X ^ 3, ← C_1, degree_cubic <| one_ne_zero' R]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    p q : Polynomial R
    hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 p.degree)
    hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
    hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
    ⊢ Eq ((Algebra.norm (Polynomial R)) (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul  …
  -/
  rw [norm_smul_basis]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    p q : Polynomial R
    hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 p.degree)
    hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
    hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
    ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow p 2) (HMul.hMul (HMul.hMul p q) (HAdd.hA …
  -/
  by_cases hp : p = 0
  · simpa only [hp, hdq, neg_zero, zero_sub, zero_mul, zero_pow two_ne_zero, degree_neg] using
      (max_bot_left _).symm
    /-
      case neg
      R : Type u
      inst✝¹ : CommRing R
      W : WeierstrassCurve.Affine R
      inst✝ : IsDomain R
      p q : Polynomial R
      hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 p.degree)
      hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
      hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
      hp : Not (Eq p 0)
      ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow p 2) (HMul.hMul (HMul.hMul p q) (HAdd.hA …
    -/
  · by_cases hq : q = 0
    · simpa only [hq, hdp, sub_zero, zero_mul, mul_zero, zero_pow two_ne_zero] using
        (max_bot_right _).symm
      /-
        case neg
        R : Type u
        inst✝¹ : CommRing R
        W : WeierstrassCurve.Affine R
        inst✝ : IsDomain R
        p q : Polynomial R
        hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 p.degree)
        hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
        hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
        hp : Not (Eq p 0)
        hq : Not (Eq q 0)
        ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow p 2) (HMul.hMul (HMul.hMul p q) (HAdd.hA …
      -/
    · rw [← not_congr degree_eq_bot] at hp hq
      -- Porting note: BUG `cases` tactic does not modify assumptions in `hp'` and `hq'`
      /-
        case neg
        R : Type u
        inst✝¹ : CommRing R
        W : WeierstrassCurve.Affine R
        inst✝ : IsDomain R
        p q : Polynomial R
        hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 p.degree)
        hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
        hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
        hp : Not (Eq p.degree Bot.bot)
        hq : Not (Eq q.degree Bot.bot)
        ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow p 2) (HMul.hMul (HMul.hMul p q) (HAdd.hA …
      -/
      rcases hp' : p.degree with _ | dp -- `hp' : ` should be redundant
        /-
          case neg.none
          R : Type u
          inst✝¹ : CommRing R
          W : WeierstrassCurve.Affine R
          inst✝ : IsDomain R
          p q : Polynomial R
          hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 p.degree)
          hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
          hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
          hp : Not (Eq p.degree Bot.bot)
          hq : Not (Eq q.degree Bot.bot)
          hp' : Eq p.degree Option.none
          ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow p 2) (HMul.hMul (HMul.hMul p q) (HAdd.hA …
        -/
      · exact (hp hp').elim -- `hp'` should be `rfl`
        /-
          🎉 no goals
        -/
        /-
          case neg.some
          R : Type u
          inst✝¹ : CommRing R
          W : WeierstrassCurve.Affine R
          inst✝ : IsDomain R
          p q : Polynomial R
          hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 p.degree)
          hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
          hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
          hp : Not (Eq p.degree Bot.bot)
          hq : Not (Eq q.degree Bot.bot)
          dp : Nat
          hp' : Eq p.degree (Option.some dp)
          ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow p 2) (HMul.hMul (HMul.hMul p q) (HAdd.hA …
        -/
      · rw [hp'] at hdp hdpq -- line should be redundant
        /-
          case neg.some
          R : Type u
          inst✝¹ : CommRing R
          W : WeierstrassCurve.Affine R
          inst✝ : IsDomain R
          p q : Polynomial R
          hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
          hp : Not (Eq p.degree Bot.bot)
          hq : Not (Eq q.degree Bot.bot)
          dp : Nat
          hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
          hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 (Option.some dp))
          hp' : Eq p.degree (Option.some dp)
          ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow p 2) (HMul.hMul (HMul.hMul p q) (HAdd.hA …
        -/
        rcases hq' : q.degree with _ | dq -- `hq' : ` should be redundant
          /-
            case neg.some.none
            R : Type u
            inst✝¹ : CommRing R
            W : WeierstrassCurve.Affine R
            inst✝ : IsDomain R
            p q : Polynomial R
            hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
            hp : Not (Eq p.degree Bot.bot)
            hq : Not (Eq q.degree Bot.bot)
            dp : Nat
            hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
            hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 (Option.some dp))
            hp' : Eq p.degree (Option.some dp)
            hq' : Eq q.degree Option.none
            ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow p 2) (HMul.hMul (HMul.hMul p q) (HAdd.hA …
          -/
        · exact (hq hq').elim -- `hq'` should be `rfl`
          /-
            🎉 no goals
          -/
          /-
            case neg.some.some
            R : Type u
            inst✝¹ : CommRing R
            W : WeierstrassCurve.Affine R
            inst✝ : IsDomain R
            p q : Polynomial R
            hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
            hp : Not (Eq p.degree Bot.bot)
            hq : Not (Eq q.degree Bot.bot)
            dp : Nat
            hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
            hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 (Option.some dp))
            hp' : Eq p.degree (Option.some dp)
            dq : Nat
            hq' : Eq q.degree (Option.some dq)
            ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow p 2) (HMul.hMul (HMul.hMul p q) (HAdd.hA …
          -/
        · rw [hq'] at hdpq hdq -- line should be redundant
          /-
            case neg.some.some
            R : Type u
            inst✝¹ : CommRing R
            W : WeierstrassCurve.Affine R
            inst✝ : IsDomain R
            p q : Polynomial R
            hp : Not (Eq p.degree Bot.bot)
            hq : Not (Eq q.degree Bot.bot)
            dp : Nat
            hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 (Option.some dp))
            hp' : Eq p.degree (Option.some dp)
            dq : Nat
            hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
            hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
            hq' : Eq q.degree (Option.some dq)
            ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow p 2) (HMul.hMul (HMul.hMul p q) (HAdd.hA …
          -/
          rcases le_or_lt dp (dq + 1) with hpq | hpq
          · convert (degree_sub_eq_right_of_degree_lt <| (degree_sub_le _ _).trans_lt <|
                      max_lt_iff.mpr ⟨hdp.trans_lt _, hdpq.trans_lt _⟩).trans
                                              /-
                                                case h.e'_3.h.e'_4
                                                R : Type u
                                                inst✝¹ : CommRing R
                                                W : WeierstrassCurve.Affine R
                                                inst✝ : IsDomain R
                                                p q : Polynomial R
                                                hp : Not (Eq p.degree Bot.bot)
                                                hq : Not (Eq q.degree Bot.bot)
                                                dp : Nat
                                                hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 (Option.some dp))
                                                hp' : Eq p.degree (Option.some dp)
                                                dq : Nat
                                                hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
                                                hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
                                                hq' : Eq q.degree (Option.some dq)
                                                hpq : LE.le dp (HAdd.hAdd dq 1)
                                                ⊢ Eq (HAdd.hAdd (HSMul.hSMul 2 (Option.some dq)) 3) (HMul.hMul (HPow.hPow q 2) …
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
              (max_eq_right_of_lt _).symm <;> rw [hdq] <;>
                /-
                  case neg.some.some.inl.convert_2
                  R : Type u
                  inst✝¹ : CommRing R
                  W : WeierstrassCurve.Affine R
                  inst✝ : IsDomain R
                  p q : Polynomial R
                  hp : Not (Eq p.degree Bot.bot)
                  hq : Not (Eq q.degree Bot.bot)
                  dp : Nat
                  hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 (Option.some dp))
                  hp' : Eq p.degree (Option.some dp)
                  dq : Nat
                  hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
                  hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
                  hq' : Eq q.degree (Option.some dq)
                  hpq : LE.le dp (HAdd.hAdd dq 1)
                  ⊢ LT.lt (HSMul.hSMul 2 (Option.some dp)) (HAdd.hAdd (HSMul.hSMul 2 (Option.som …
                -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
                exact WithBot.coe_lt_coe.mpr <| by dsimp; linarith only [hpq]
                /-
                  🎉 no goals
                -/
            /-
              case neg.some.some.inr
              R : Type u
              inst✝¹ : CommRing R
              W : WeierstrassCurve.Affine R
              inst✝ : IsDomain R
              p q : Polynomial R
              hp : Not (Eq p.degree Bot.bot)
              hq : Not (Eq q.degree Bot.bot)
              dp : Nat
              hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 (Option.some dp))
              hp' : Eq p.degree (Option.some dp)
              dq : Nat
              hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
              hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
              hq' : Eq q.degree (Option.some dq)
              hpq : LT.lt (HAdd.hAdd dq 1) dp
              ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow p 2) (HMul.hMul (HMul.hMul p q) (HAdd.hA …
            -/
          · rw [sub_sub]
            convert (degree_sub_eq_left_of_degree_lt <| (degree_add_le _ _).trans_lt <|
                      max_lt_iff.mpr ⟨hdpq.trans_lt _, hdq.trans_lt _⟩).trans
                                             /-
                                               case h.e'_3.h.e'_3
                                               R : Type u
                                               inst✝¹ : CommRing R
                                               W : WeierstrassCurve.Affine R
                                               inst✝ : IsDomain R
                                               p q : Polynomial R
                                               hp : Not (Eq p.degree Bot.bot)
                                               hq : Not (Eq q.degree Bot.bot)
                                               dp : Nat
                                               hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 (Option.some dp))
                                               hp' : Eq p.degree (Option.some dp)
                                               dq : Nat
                                               hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
                                               hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
                                               hq' : Eq q.degree (Option.some dq)
                                               hpq : LT.lt (HAdd.hAdd dq 1) dp
                                               ⊢ Eq (HSMul.hSMul 2 (Option.some dp)) (HPow.hPow p 2).degree
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
              (max_eq_left_of_lt _).symm <;> rw [hdp] <;>
                /-
                  case neg.some.some.inr.convert_2
                  R : Type u
                  inst✝¹ : CommRing R
                  W : WeierstrassCurve.Affine R
                  inst✝ : IsDomain R
                  p q : Polynomial R
                  hp : Not (Eq p.degree Bot.bot)
                  hq : Not (Eq q.degree Bot.bot)
                  dp : Nat
                  hdp : Eq (HPow.hPow p 2).degree (HSMul.hSMul 2 (Option.some dp))
                  hp' : Eq p.degree (Option.some dp)
                  dq : Nat
                  hdq : Eq (HMul.hMul (HPow.hPow q 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
                  hdpq : LE.le (HMul.hMul (HMul.hMul p q) (HAdd.hAdd (HMul.hMul (Polynomial.C W. …
                  hq' : Eq q.degree (Option.some dq)
                  hpq : LT.lt (HAdd.hAdd dq 1) dp
                  ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd (Option.some dp) (Option.some dq)) 1) (HSMul.hSM …
                -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
                exact WithBot.coe_lt_coe.mpr <| by dsimp; linarith only [hpq]
                /-
                  🎉 no goals
                -/


variable {W} in
lemma degree_norm_ne_one [IsDomain R] (x : W.CoordinateRing) :
    (Algebra.norm R[X] x).degree ≠ 1 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    x : W.CoordinateRing
    ⊢ Ne ((Algebra.norm (Polynomial R)) x).degree 1
  -/
  rcases exists_smul_basis_eq x with ⟨p, q, rfl⟩
  /-
    case intro.intro
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    p q : Polynomial R
    ⊢ Ne ((Algebra.norm (Polynomial R)) (HAdd.hAdd (HSMul.hSMul p 1) (HSMul.hSMul  …
  -/
  rw [degree_norm_smul_basis]
  /-
    case intro.intro
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    p q : Polynomial R
    ⊢ Ne (Max.max (HSMul.hSMul 2 p.degree) (HAdd.hAdd (HSMul.hSMul 2 q.degree) 3)) 1
  -/
  rcases p.degree with (_ | _ | _ | _) <;> cases q.degree
  /-
    case intro.intro.none.bot
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    p q : Polynomial R
    ⊢ Ne (Max.max (HSMul.hSMul 2 Option.none) (HAdd.hAdd (HSMul.hSMul 2 Bot.bot) 3 …
  -/
  any_goals rintro (_ | _)
  -- Porting note: replaced `dec_trivial` with `by exact (cmp_eq_lt_iff ..).mp rfl`
  /-
    case intro.intro.some.succ.succ.coe
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    p q : Polynomial R
    n✝ a✝ : Nat
    ⊢ Ne (Max.max (HSMul.hSMul 2 (Option.some (HAdd.hAdd (HAdd.hAdd n✝ 1) 1))) (HA …
  -/
  exact (lt_max_of_lt_right <| by exact (cmp_eq_lt_iff ..).mp rfl).ne'
  /-
    🎉 no goals
  -/


variable {W} in
lemma natDegree_norm_ne_one [IsDomain R] (x : W.CoordinateRing) :
    (Algebra.norm R[X] x).natDegree ≠ 1 :=
  degree_norm_ne_one x ∘ (degree_eq_iff_natDegree_eq_of_pos zero_lt_one).mpr


/-- The set function mapping an affine point $(x, y)$ of `W` to the class of the non-zero fractional
ideal $\langle X - x, Y - y \rangle$ of $F(W)$ in the class group of $F[W]$. -/
@[simp]
noncomputable def toClassFun : W.Point → Additive (ClassGroup W.CoordinateRing)
  | 0 => 0
  | some h => Additive.ofMul <| ClassGroup.mk <| CoordinateRing.XYIdeal' h


/-- The group homomorphism mapping an affine point $(x, y)$ of `W` to the class of the non-zero
fractional ideal $\langle X - x, Y - y \rangle$ of $F(W)$ in the class group of $F[W]$. -/
@[simps]
noncomputable def toClass : W.Point →+ Additive (ClassGroup W.CoordinateRing) where
  toFun := toClassFun
  map_zero' := rfl
  map_add' := by
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      W✝ : WeierstrassCurve.Affine R
      f : RingHom R S
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      ⊢ ∀ (x y : W.Point), Eq ({ toFun := WeierstrassCurve.Affine.Point.toClassFun,  …
    -/
    rintro (_ | @⟨x₁, y₁, h₁⟩) (_ | @⟨x₂, y₂, h₂⟩)
    /-
      case zero.zero
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      W✝ : WeierstrassCurve.Affine R
      f : RingHom R S
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      ⊢ Eq ({ toFun := WeierstrassCurve.Affine.Point.toClassFun, map_zero' := ⋯ }.to …
    -/
    any_goals simp only [zero_def, toClassFun, zero_add, add_zero]
    /-
      case some.some
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      W✝ : WeierstrassCurve.Affine R
      f : RingHom R S
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ y₁ : F
      h₁ : W.Nonsingular x₁ y₁
      x₂ y₂ : F
      h₂ : W.Nonsingular x₂ y₂
      ⊢ Eq (WeierstrassCurve.Affine.Point.toClassFun.match_1 (fun x => Additive (Cla …
    -/
    obtain ⟨rfl, rfl⟩ | h := em (x₁ = x₂ ∧ y₁ = W.negY x₂ y₂)
      /-
        case some.some.inl.intro
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        W✝ : WeierstrassCurve.Affine R
        f : RingHom R S
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₂ : F
        h₂ : W.Nonsingular x₁ y₂
        h₁ : W.Nonsingular x₁ (W.negY x₁ y₂)
        ⊢ Eq (WeierstrassCurve.Affine.Point.toClassFun.match_1 (fun x => Additive (Cla …
      -/
    · rw [add_of_Y_eq rfl rfl]
      /-
        case some.some.inl.intro
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        W✝ : WeierstrassCurve.Affine R
        f : RingHom R S
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₂ : F
        h₂ : W.Nonsingular x₁ y₂
        h₁ : W.Nonsingular x₁ (W.negY x₁ y₂)
        ⊢ Eq (WeierstrassCurve.Affine.Point.toClassFun.match_1 (fun x => Additive (Cla …
      -/
      exact (CoordinateRing.mk_XYIdeal'_mul_mk_XYIdeal'_of_Yeq h₂).symm
      /-
        🎉 no goals
      -/
      /-
        case some.some.inr
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        W✝ : WeierstrassCurve.Affine R
        f : RingHom R S
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₁ : F
        h₁ : W.Nonsingular x₁ y₁
        x₂ y₂ : F
        h₂ : W.Nonsingular x₂ y₂
        h : Not (And (Eq x₁ x₂) (Eq y₁ (W.negY x₂ y₂)))
        ⊢ Eq (WeierstrassCurve.Affine.Point.toClassFun.match_1 (fun x => Additive (Cla …
      -/
    · have h hx hy := h ⟨hx, hy⟩
      /-
        case some.some.inr
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        W✝ : WeierstrassCurve.Affine R
        f : RingHom R S
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₁ : F
        h₁ : W.Nonsingular x₁ y₁
        x₂ y₂ : F
        h₂ : W.Nonsingular x₂ y₂
        h✝ : Not (And (Eq x₁ x₂) (Eq y₁ (W.negY x₂ y₂)))
        h : Eq x₁ x₂ → Eq y₁ (W.negY x₂ y₂) → False
        ⊢ Eq (WeierstrassCurve.Affine.Point.toClassFun.match_1 (fun x => Additive (Cla …
      -/
      rw [add_of_imp h]
      /-
        case some.some.inr
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        W✝ : WeierstrassCurve.Affine R
        f : RingHom R S
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₁ : F
        h₁ : W.Nonsingular x₁ y₁
        x₂ y₂ : F
        h₂ : W.Nonsingular x₂ y₂
        h✝ : Not (And (Eq x₁ x₂) (Eq y₁ (W.negY x₂ y₂)))
        h : Eq x₁ x₂ → Eq y₁ (W.negY x₂ y₂) → False
        ⊢ Eq (WeierstrassCurve.Affine.Point.toClassFun.match_1 (fun x => Additive (Cla …
      -/
      exact (CoordinateRing.mk_XYIdeal'_mul_mk_XYIdeal' h₁ h₂ h).symm
      /-
        🎉 no goals
      -/


lemma toClass_zero : toClass (0 : W.Point) = 0 :=
  rfl


lemma toClass_some {x y : F} (h : W.Nonsingular x y) :
    toClass (some h) = ClassGroup.mk (CoordinateRing.XYIdeal' h) :=
  rfl


private lemma add_eq_zero (P Q : W.Point) : P + Q = 0 ↔ P = -Q := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    P Q : W.Point
    ⊢ Iff (Eq (HAdd.hAdd P Q) 0) (Eq P (Neg.neg Q))
  -/
  rcases P, Q with ⟨_ | @⟨x₁, y₁, _⟩, _ | @⟨x₂, y₂, _⟩⟩
  /-
    case zero.zero
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    ⊢ Iff (Eq (HAdd.hAdd WeierstrassCurve.Affine.Point.zero WeierstrassCurve.Affin …
  -/
  any_goals rfl
    /-
      case zero.some
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₂ y₂ : F
      h✝ : W.Nonsingular x₂ y₂
      ⊢ Iff (Eq (HAdd.hAdd WeierstrassCurve.Affine.Point.zero (WeierstrassCurve.Affi …
    -/
  · rw [zero_def, zero_add, ← neg_eq_iff_eq_neg, neg_zero, eq_comm]
    /-
      🎉 no goals
    -/
    /-
      case some.some
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ y₁ : F
      h✝¹ : W.Nonsingular x₁ y₁
      x₂ y₂ : F
      h✝ : W.Nonsingular x₂ y₂
      ⊢ Iff (Eq (HAdd.hAdd (WeierstrassCurve.Affine.Point.some h✝¹) (WeierstrassCurv …
    -/
  · rw [neg_some, some.injEq]
    /-
      case some.some
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ y₁ : F
      h✝¹ : W.Nonsingular x₁ y₁
      x₂ y₂ : F
      h✝ : W.Nonsingular x₂ y₂
      ⊢ Iff (Eq (HAdd.hAdd (WeierstrassCurve.Affine.Point.some h✝¹) (WeierstrassCurv …
    -/
    constructor
      /-
        case some.some.mp
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₁ : F
        h✝¹ : W.Nonsingular x₁ y₁
        x₂ y₂ : F
        h✝ : W.Nonsingular x₂ y₂
        ⊢ Eq (HAdd.hAdd (WeierstrassCurve.Affine.Point.some h✝¹) (WeierstrassCurve.Aff …
      -/
    · contrapose!; intro h; rw [add_of_imp h]; exact some_ne_zero _
                                               /-
                                                 🎉 no goals
                                               -/
      /-
        case some.some.mpr
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₁ : F
        h✝¹ : W.Nonsingular x₁ y₁
        x₂ y₂ : F
        h✝ : W.Nonsingular x₂ y₂
        ⊢ And (Eq x₁ x₂) (Eq y₁ (W.negY x₂ y₂)) → Eq (HAdd.hAdd (WeierstrassCurve.Affi …
      -/
    · exact fun ⟨hx, hy⟩ ↦ add_of_Y_eq hx hy
      /-
        🎉 no goals
      -/


lemma toClass_eq_zero (P : W.Point) : toClass P = 0 ↔ P = 0 := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    P : W.Point
    ⊢ Iff (Eq (WeierstrassCurve.Affine.Point.toClass P) 0) (Eq P 0)
  -/
  constructor
    /-
      case mp
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      P : W.Point
      ⊢ Eq (WeierstrassCurve.Affine.Point.toClass P) 0 → Eq P 0
    -/
  · intro hP
    /-
      case mp
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      P : W.Point
      hP : Eq (WeierstrassCurve.Affine.Point.toClass P) 0
      ⊢ Eq P 0
    -/
    rcases P with (_ | ⟨h, _⟩)
      /-
        case mp.zero
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        hP : Eq (WeierstrassCurve.Affine.Point.toClass WeierstrassCurve.Affine.Point.z …
        ⊢ Eq WeierstrassCurve.Affine.Point.zero 0
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case mp.some.intro
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x✝ y✝ : F
        h : W.Equation x✝ y✝
        right✝ : Or (Ne (Polynomial.evalEval x✝ y✝ W.polynomialX) 0) (Ne (Polynomial.e …
        hP : Eq (WeierstrassCurve.Affine.Point.toClass (WeierstrassCurve.Affine.Point. …
        ⊢ Eq (WeierstrassCurve.Affine.Point.some ⋯) 0
      -/
    · rcases (ClassGroup.mk_eq_one_of_coe_ideal <| by rfl).mp hP with ⟨p, h0, hp⟩
      /-
        case mp.some.intro.intro.intro
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x✝ y✝ : F
        h : W.Equation x✝ y✝
        right✝ : Or (Ne (Polynomial.evalEval x✝ y✝ W.polynomialX) 0) (Ne (Polynomial.e …
        hP : Eq (WeierstrassCurve.Affine.Point.toClass (WeierstrassCurve.Affine.Point. …
        p : W.CoordinateRing
        h0 : Ne p 0
        hp : Eq (WeierstrassCurve.Affine.CoordinateRing.XYIdeal W x✝ (Polynomial.C y✝) …
        ⊢ Eq (WeierstrassCurve.Affine.Point.some ⋯) 0
      -/
      apply (p.natDegree_norm_ne_one _).elim
      rw [← finrank_quotient_span_eq_natDegree_norm (CoordinateRing.basis W) h0,
        ← (quotientEquivAlgOfEq F hp).toLinearEquiv.finrank_eq,
        (CoordinateRing.quotientXYIdealEquiv W h).toLinearEquiv.finrank_eq,
        Module.finrank_self]
    /-
      case mpr
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      P : W.Point
      ⊢ Eq P 0 → Eq (WeierstrassCurve.Affine.Point.toClass P) 0
    -/
  · exact congr_arg toClass
    /-
      🎉 no goals
    -/


lemma toClass_injective : Function.Injective <| @toClass _ _ W := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    ⊢ Function.Injective ⇑WeierstrassCurve.Affine.Point.toClass
  -/
  rintro (_ | h) _ hP
  /-
    case zero
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    a₂✝ : W.Point
    hP : Eq (WeierstrassCurve.Affine.Point.toClass WeierstrassCurve.Affine.Point.z …
    ⊢ Eq WeierstrassCurve.Affine.Point.zero a₂✝
  -/
  all_goals rw [← neg_inj, ← add_eq_zero, ← toClass_eq_zero, map_add, ← hP]
    /-
      case zero
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      a₂✝ : W.Point
      hP : Eq (WeierstrassCurve.Affine.Point.toClass WeierstrassCurve.Affine.Point.z …
      ⊢ Eq (HAdd.hAdd (WeierstrassCurve.Affine.Point.toClass (Neg.neg WeierstrassCur …
    -/
  · exact zero_add 0
    /-
      🎉 no goals
    -/
    /-
      case some
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x✝ y✝ : F
      h : W.Nonsingular x✝ y✝
      a₂✝ : W.Point
      hP : Eq (WeierstrassCurve.Affine.Point.toClass (WeierstrassCurve.Affine.Point. …
      ⊢ Eq (HAdd.hAdd (WeierstrassCurve.Affine.Point.toClass (Neg.neg (WeierstrassCu …
    -/
  · exact CoordinateRing.mk_XYIdeal'_mul_mk_XYIdeal'_of_Yeq h
    /-
      🎉 no goals
    -/


noncomputable instance : AddCommGroup W.Point where
  nsmul := nsmulRec
  zsmul := zsmulRec
  zero_add := zero_add
  add_zero := add_zero
                         /-
                           R : Type u
                           S : Type v
                           inst✝² : CommRing R
                           inst✝¹ : CommRing S
                           W✝ : WeierstrassCurve.Affine R
                           f : RingHom R S
                           F : Type u
                           inst✝ : Field F
                           W : WeierstrassCurve.Affine F
                           x✝ : W.Point
                           ⊢ Eq (HAdd.hAdd (Neg.neg x✝) x✝) 0
                         -/
  neg_add_cancel _ := by rw [add_eq_zero]
                                             /-
                                               R : Type u
                                               S : Type v
                                               inst✝² : CommRing R
                                               inst✝¹ : CommRing S
                                               W✝ : WeierstrassCurve.Affine R
                                               f : RingHom R S
                                               F : Type u
                                               inst✝ : Field F
                                               W : WeierstrassCurve.Affine F
                                               x✝² x✝¹ x✝ : W.Point
                                               ⊢ Eq (WeierstrassCurve.Affine.Point.toClass (HAdd.hAdd (HAdd.hAdd x✝² x✝¹) x✝) …
                                             -/
                         /-
                           🎉 no goals
                         -/
                                             /-
                                               🎉 no goals
                                             -/
                                          /-
                                            R : Type u
                                            S : Type v
                                            inst✝² : CommRing R
                                            inst✝¹ : CommRing S
                                            W✝ : WeierstrassCurve.Affine R
                                            f : RingHom R S
                                            F : Type u
                                            inst✝ : Field F
                                            W : WeierstrassCurve.Affine F
                                            x✝¹ x✝ : W.Point
                                            ⊢ Eq (WeierstrassCurve.Affine.Point.toClass (HAdd.hAdd x✝¹ x✝)) (WeierstrassCu …
                                          -/
  add_comm _ _ := toClass_injective <| by simp only [map_add, add_comm]
                                          /-
                                            🎉 no goals
                                          -/
  add_assoc _ _ _ := toClass_injective <| by simp only [map_add, add_assoc]


noncomputable instance : AddCommGroup W.Point where
  nsmul := nsmulRec
  zsmul := zsmulRec
  zero_add _ := (toAffineAddEquiv W).injective <| by
    /-
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      x✝ : W.Point
      ⊢ Eq ((WeierstrassCurve.Projective.Point.toAffineAddEquiv W) (HAdd.hAdd 0 x✝)) …
    -/
    simp only [map_add, toAffineAddEquiv_apply, toAffineLift_zero, zero_add]
    /-
      🎉 no goals
    -/
  add_zero _ := (toAffineAddEquiv W).injective <| by
    /-
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      x✝ : W.Point
      ⊢ Eq ((WeierstrassCurve.Projective.Point.toAffineAddEquiv W) (HAdd.hAdd x✝ 0)) …
    -/
    simp only [map_add, toAffineAddEquiv_apply, toAffineLift_zero, add_zero]
                                                          /-
                                                            F : Type u
                                                            inst✝ : Field F
                                                            W : WeierstrassCurve.Projective F
                                                            x✝² x✝¹ x✝ : W.Point
                                                            ⊢ Eq ((WeierstrassCurve.Projective.Point.toAffineAddEquiv W) (HAdd.hAdd (HAdd. …
                                                          -/
    /-
      🎉 no goals
    -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  neg_add_cancel P := (toAffineAddEquiv W).injective <| by
    /-
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P : W.Point
      ⊢ Eq ((WeierstrassCurve.Projective.Point.toAffineAddEquiv W) (HAdd.hAdd (Neg.n …
    -/
    simp only [map_add, toAffineAddEquiv_apply, toAffineLift_neg, neg_add_cancel, toAffineLift_zero]
    /-
      🎉 no goals
    -/
                                                       /-
                                                         F : Type u
                                                         inst✝ : Field F
                                                         W : WeierstrassCurve.Projective F
                                                         x✝¹ x✝ : W.Point
                                                         ⊢ Eq ((WeierstrassCurve.Projective.Point.toAffineAddEquiv W) (HAdd.hAdd x✝¹ x✝ …
                                                       -/
  add_comm _ _ := (toAffineAddEquiv W).injective <| by simp only [map_add, add_comm]
                                                       /-
                                                         🎉 no goals
                                                       -/
  add_assoc _ _ _ := (toAffineAddEquiv W).injective <| by simp only [map_add, add_assoc]


noncomputable instance : AddCommGroup W.Point where
  nsmul := nsmulRec
  zsmul := zsmulRec
  zero_add _ := (toAffineAddEquiv W).injective <| by
    /-
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      x✝ : W.Point
      ⊢ Eq ((WeierstrassCurve.Jacobian.Point.toAffineAddEquiv W) (HAdd.hAdd 0 x✝)) ( …
    -/
    simp only [map_add, toAffineAddEquiv_apply, toAffineLift_zero, zero_add]
    /-
      🎉 no goals
    -/
  add_zero _ := (toAffineAddEquiv W).injective <| by
    /-
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      x✝ : W.Point
      ⊢ Eq ((WeierstrassCurve.Jacobian.Point.toAffineAddEquiv W) (HAdd.hAdd x✝ 0)) ( …
    -/
    simp only [map_add, toAffineAddEquiv_apply, toAffineLift_zero, add_zero]
                                                          /-
                                                            F : Type u
                                                            inst✝ : Field F
                                                            W : WeierstrassCurve.Jacobian F
                                                            x✝² x✝¹ x✝ : W.Point
                                                            ⊢ Eq ((WeierstrassCurve.Jacobian.Point.toAffineAddEquiv W) (HAdd.hAdd (HAdd.hA …
                                                          -/
    /-
      🎉 no goals
    -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  neg_add_cancel P := (toAffineAddEquiv W).injective <| by
    /-
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P : W.Point
      ⊢ Eq ((WeierstrassCurve.Jacobian.Point.toAffineAddEquiv W) (HAdd.hAdd (Neg.neg …
    -/
    simp only [map_add, toAffineAddEquiv_apply, toAffineLift_neg, neg_add_cancel, toAffineLift_zero]
    /-
      🎉 no goals
    -/
                                                       /-
                                                         F : Type u
                                                         inst✝ : Field F
                                                         W : WeierstrassCurve.Jacobian F
                                                         x✝¹ x✝ : W.Point
                                                         ⊢ Eq ((WeierstrassCurve.Jacobian.Point.toAffineAddEquiv W) (HAdd.hAdd x✝¹ x✝)) …
                                                       -/
  add_comm _ _ := (toAffineAddEquiv W).injective <| by simp only [map_add, add_comm]
                                                       /-
                                                         🎉 no goals
                                                       -/
  add_assoc _ _ _ := (toAffineAddEquiv W).injective <| by simp only [map_add, add_assoc]


/-- An affine point on an elliptic curve `E` over `R`. -/
def mk {x y : R} (h : E.toAffine.Equation x y) : E.toAffine.Point :=
  WeierstrassCurve.Affine.Point.some <| nonsingular E h


