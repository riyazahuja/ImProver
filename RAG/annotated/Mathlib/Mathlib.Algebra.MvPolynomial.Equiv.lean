/-- The ring isomorphism between multivariable polynomials in a single variable and
polynomials over the ground ring.
-/
@[simps]
def pUnitAlgEquiv : MvPolynomial PUnit R ≃ₐ[R] R[X] where
  toFun := eval₂ Polynomial.C fun _ => Polynomial.X
  invFun := Polynomial.eval₂ MvPolynomial.C (X PUnit.unit)
  left_inv := by
    /-
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      σ : Type u_1
      a a' a₁ a₂ : R
      e : Nat
      s : Finsupp σ Nat
      inst✝ : CommSemiring R
      ⊢ Function.LeftInverse (Polynomial.eval₂ MvPolynomial.C (MvPolynomial.X PUnit. …
    -/
    let f : R[X] →+* MvPolynomial PUnit R := Polynomial.eval₂RingHom MvPolynomial.C (X PUnit.unit)
    /-
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      σ : Type u_1
      a a' a₁ a₂ : R
      e : Nat
      s : Finsupp σ Nat
      inst✝ : CommSemiring R
      f : RingHom (Polynomial R) (MvPolynomial PUnit.{?u.1720 + 1} R) := Polynomial. …
      ⊢ Function.LeftInverse (Polynomial.eval₂ MvPolynomial.C (MvPolynomial.X PUnit. …
    -/
    let g : MvPolynomial PUnit R →+* R[X] := eval₂Hom Polynomial.C fun _ => Polynomial.X
    /-
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      σ : Type u_1
      a a' a₁ a₂ : R
      e : Nat
      s : Finsupp σ Nat
      inst✝ : CommSemiring R
      f : RingHom (Polynomial R) (MvPolynomial PUnit.{?u.1720 + 1} R) := Polynomial. …
      g : RingHom (MvPolynomial PUnit.{?u.2124 + 1} R) (Polynomial R) := MvPolynomia …
      ⊢ Function.LeftInverse (Polynomial.eval₂ MvPolynomial.C (MvPolynomial.X PUnit. …
    -/
    show ∀ p, f.comp g p = p
    /-
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      σ : Type u_1
      a a' a₁ a₂ : R
      e : Nat
      s : Finsupp σ Nat
      inst✝ : CommSemiring R
      f : RingHom (Polynomial R) (MvPolynomial PUnit.{?u.2124 + 1} R) := Polynomial. …
      g : RingHom (MvPolynomial PUnit.{?u.2124 + 1} R) (Polynomial R) := MvPolynomia …
      ⊢ ∀ (p : MvPolynomial PUnit.{?u.2124 + 1} R), Eq ((f.comp g) p) p
    -/
    apply is_id
      /-
        case hC
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        f : RingHom (Polynomial R) (MvPolynomial PUnit.{?u.2124 + 1} R) := Polynomial. …
        g : RingHom (MvPolynomial PUnit.{?u.2124 + 1} R) (Polynomial R) := MvPolynomia …
        ⊢ Eq ((f.comp g).comp MvPolynomial.C) MvPolynomial.C
      -/
    · ext a
      /-
        case hC.a.a
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a✝ a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        f : RingHom (Polynomial R) (MvPolynomial PUnit.{?u.2124 + 1} R) := Polynomial. …
        g : RingHom (MvPolynomial PUnit.{?u.2124 + 1} R) (Polynomial R) := MvPolynomia …
        a : R
        m✝ : Finsupp PUnit.{?u.2124 + 1} Nat
        ⊢ Eq (MvPolynomial.coeff m✝ (((f.comp g).comp MvPolynomial.C) a)) (MvPolynomia …
      -/
      dsimp [f, g]
      /-
        case hC.a.a
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a✝ a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        f : RingHom (Polynomial R) (MvPolynomial PUnit.{?u.2124 + 1} R) := Polynomial. …
        g : RingHom (MvPolynomial PUnit.{?u.2124 + 1} R) (Polynomial R) := MvPolynomia …
        a : R
        m✝ : Finsupp PUnit.{?u.2124 + 1} Nat
        ⊢ Eq (MvPolynomial.coeff m✝ (Polynomial.eval₂ MvPolynomial.C (MvPolynomial.X P …
      -/
      rw [eval₂_C, Polynomial.eval₂_C]
      /-
        🎉 no goals
      -/
      /-
        case hX
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        f : RingHom (Polynomial R) (MvPolynomial PUnit.{?u.2124 + 1} R) := Polynomial. …
        g : RingHom (MvPolynomial PUnit.{?u.2124 + 1} R) (Polynomial R) := MvPolynomia …
        ⊢ ∀ (n : PUnit.{?u.2124 + 1}), Eq ((f.comp g) (MvPolynomial.X n)) (MvPolynomia …
      -/
    · rintro ⟨⟩
      /-
        case hX.unit
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        f : RingHom (Polynomial R) (MvPolynomial PUnit.{?u.2124 + 1} R) := Polynomial. …
        g : RingHom (MvPolynomial PUnit.{?u.2124 + 1} R) (Polynomial R) := MvPolynomia …
        ⊢ Eq ((f.comp g) (MvPolynomial.X PUnit.unit)) (MvPolynomial.X PUnit.unit)
      -/
      dsimp [f, g]
      /-
        case hX.unit
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        f : RingHom (Polynomial R) (MvPolynomial PUnit.{?u.2124 + 1} R) := Polynomial. …
        g : RingHom (MvPolynomial PUnit.{?u.2124 + 1} R) (Polynomial R) := MvPolynomia …
        ⊢ Eq (Polynomial.eval₂ MvPolynomial.C (MvPolynomial.X PUnit.unit) (MvPolynomia …
      -/
      rw [eval₂_X, Polynomial.eval₂_X]
      /-
        🎉 no goals
      -/
  right_inv p :=
                                           /-
                                             R : Type u
                                             S₁ : Type v
                                             S₂ : Type w
                                             S₃ : Type x
                                             σ : Type u_1
                                             a✝ a' a₁ a₂ : R
                                             e : Nat
                                             s : Finsupp σ Nat
                                             inst✝ : CommSemiring R
                                             p : Polynomial R
                                             a : R
                                             ⊢ Eq (MvPolynomial.eval₂ Polynomial.C (fun x => Polynomial.X) (Polynomial.eval …
                                           -/
    Polynomial.induction_on p (fun a => by rw [Polynomial.eval₂_C, MvPolynomial.eval₂_C])
                                           /-
                                             🎉 no goals
                                           -/
                         /-
                           R : Type u
                           S₁ : Type v
                           S₂ : Type w
                           S₃ : Type x
                           σ : Type u_1
                           a a' a₁ a₂ : R
                           e : Nat
                           s : Finsupp σ Nat
                           inst✝ : CommSemiring R
                           p✝ p q : Polynomial R
                           hp : Eq (MvPolynomial.eval₂ Polynomial.C (fun x => Polynomial.X) (Polynomial.e …
                           hq : Eq (MvPolynomial.eval₂ Polynomial.C (fun x => Polynomial.X) (Polynomial.e …
                           ⊢ Eq (MvPolynomial.eval₂ Polynomial.C (fun x => Polynomial.X) (Polynomial.eval …
                         -/
    (fun p q hp hq => by rw [Polynomial.eval₂_add, MvPolynomial.eval₂_add, hp, hq]) fun p n _ => by
                         /-
                           🎉 no goals
                         -/
      rw [Polynomial.eval₂_mul, Polynomial.eval₂_pow, Polynomial.eval₂_X, Polynomial.eval₂_C,
        eval₂_mul, eval₂_C, eval₂_pow, eval₂_X]
  map_mul' _ _ := eval₂_mul _ _
  map_add' _ _ := eval₂_add _ _
  commutes' _ := eval₂_C _ _ _


/-- If `e : A ≃+* B` is an isomorphism of rings, then so is `map e`. -/
@[simps apply]
def mapEquiv [CommSemiring S₁] [CommSemiring S₂] (e : S₁ ≃+* S₂) :
    MvPolynomial σ S₁ ≃+* MvPolynomial σ S₂ :=
  { map (e : S₁ →+* S₂) with
    toFun := map (e : S₁ →+* S₂)
    invFun := map (e.symm : S₂ →+* S₁)
    left_inv := map_leftInverse e.left_inv
    right_inv := map_rightInverse e.right_inv }


@[simp]
theorem mapEquiv_refl : mapEquiv σ (RingEquiv.refl R) = RingEquiv.refl _ :=
  RingEquiv.ext map_id


@[simp]
theorem mapEquiv_symm [CommSemiring S₁] [CommSemiring S₂] (e : S₁ ≃+* S₂) :
    (mapEquiv σ e).symm = mapEquiv σ e.symm :=
  rfl


@[simp]
theorem mapEquiv_trans [CommSemiring S₁] [CommSemiring S₂] [CommSemiring S₃] (e : S₁ ≃+* S₂)
    (f : S₂ ≃+* S₃) : (mapEquiv σ e).trans (mapEquiv σ f) = mapEquiv σ (e.trans f) :=
  RingEquiv.ext fun p => by
    simp only [RingEquiv.coe_trans, comp_apply, mapEquiv_apply, RingEquiv.coe_ringHom_trans,
      map_map]


/-- If `e : A ≃ₐ[R] B` is an isomorphism of `R`-algebras, then so is `map e`. -/
@[simps apply]
def mapAlgEquiv (e : A₁ ≃ₐ[R] A₂) : MvPolynomial σ A₁ ≃ₐ[R] MvPolynomial σ A₂ :=
  { mapAlgHom (e : A₁ →ₐ[R] A₂), mapEquiv σ (e : A₁ ≃+* A₂) with toFun := map (e : A₁ →+* A₂) }


@[simp]
theorem mapAlgEquiv_refl : mapAlgEquiv σ (AlgEquiv.refl : A₁ ≃ₐ[R] A₁) = AlgEquiv.refl :=
  AlgEquiv.ext map_id


@[simp]
theorem mapAlgEquiv_symm (e : A₁ ≃ₐ[R] A₂) : (mapAlgEquiv σ e).symm = mapAlgEquiv σ e.symm :=
  rfl


@[simp]
theorem mapAlgEquiv_trans (e : A₁ ≃ₐ[R] A₂) (f : A₂ ≃ₐ[R] A₃) :
    (mapAlgEquiv σ e).trans (mapAlgEquiv σ f) = mapAlgEquiv σ (e.trans f) := by
  /-
    R : Type u
    σ : Type u_1
    inst✝⁶ : CommSemiring R
    A₁ : Type u_2
    A₂ : Type u_3
    A₃ : Type u_4
    inst✝⁵ : CommSemiring A₁
    inst✝⁴ : CommSemiring A₂
    inst✝³ : CommSemiring A₃
    inst✝² : Algebra R A₁
    inst✝¹ : Algebra R A₂
    inst✝ : Algebra R A₃
    e : AlgEquiv R A₁ A₂
    f : AlgEquiv R A₂ A₃
    ⊢ Eq ((MvPolynomial.mapAlgEquiv σ e).trans (MvPolynomial.mapAlgEquiv σ f)) (Mv …
  -/
  ext
  /-
    case h.a
    R : Type u
    σ : Type u_1
    inst✝⁶ : CommSemiring R
    A₁ : Type u_2
    A₂ : Type u_3
    A₃ : Type u_4
    inst✝⁵ : CommSemiring A₁
    inst✝⁴ : CommSemiring A₂
    inst✝³ : CommSemiring A₃
    inst✝² : Algebra R A₁
    inst✝¹ : Algebra R A₂
    inst✝ : Algebra R A₃
    e : AlgEquiv R A₁ A₂
    f : AlgEquiv R A₂ A₃
    a✝ : MvPolynomial σ A₁
    m✝ : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff m✝ (((MvPolynomial.mapAlgEquiv σ e).trans (MvPolynomi …
  -/
  simp only [AlgEquiv.trans_apply, mapAlgEquiv_apply, map_map]
  /-
    case h.a
    R : Type u
    σ : Type u_1
    inst✝⁶ : CommSemiring R
    A₁ : Type u_2
    A₂ : Type u_3
    A₃ : Type u_4
    inst✝⁵ : CommSemiring A₁
    inst✝⁴ : CommSemiring A₂
    inst✝³ : CommSemiring A₃
    inst✝² : Algebra R A₁
    inst✝¹ : Algebra R A₂
    inst✝ : Algebra R A₃
    e : AlgEquiv R A₁ A₂
    f : AlgEquiv R A₂ A₃
    a✝ : MvPolynomial σ A₁
    m✝ : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff m✝ ((MvPolynomial.map ((↑f).comp ↑e)) a✝)) (MvPolynom …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The function from multivariable polynomials in a sum of two types,
to multivariable polynomials in one of the types,
with coefficients in multivariable polynomials in the other type.

See `sumRingEquiv` for the ring isomorphism.
-/
def sumToIter : MvPolynomial (S₁ ⊕ S₂) R →+* MvPolynomial S₁ (MvPolynomial S₂ R) :=
  eval₂Hom (C.comp C) fun bc => Sum.recOn bc X (C ∘ X)


@[simp]
theorem sumToIter_C (a : R) : sumToIter R S₁ S₂ (C a) = C (C a) :=
  eval₂_C _ _ a


@[simp]
theorem sumToIter_Xl (b : S₁) : sumToIter R S₁ S₂ (X (Sum.inl b)) = X b :=
  eval₂_X _ _ (Sum.inl b)


@[simp]
theorem sumToIter_Xr (c : S₂) : sumToIter R S₁ S₂ (X (Sum.inr c)) = C (X c) :=
  eval₂_X _ _ (Sum.inr c)


/-- The function from multivariable polynomials in one type,
with coefficients in multivariable polynomials in another type,
to multivariable polynomials in the sum of the two types.

See `sumRingEquiv` for the ring isomorphism.
-/
def iterToSum : MvPolynomial S₁ (MvPolynomial S₂ R) →+* MvPolynomial (S₁ ⊕ S₂) R :=
  eval₂Hom (eval₂Hom C (X ∘ Sum.inr)) (X ∘ Sum.inl)


@[simp]
theorem iterToSum_C_C (a : R) : iterToSum R S₁ S₂ (C (C a)) = C a :=
  Eq.trans (eval₂_C _ _ (C a)) (eval₂_C _ _ _)


@[simp]
theorem iterToSum_X (b : S₁) : iterToSum R S₁ S₂ (X b) = X (Sum.inl b) :=
  eval₂_X _ _ _


@[simp]
theorem iterToSum_C_X (c : S₂) : iterToSum R S₁ S₂ (C (X c)) = X (Sum.inr c) :=
  Eq.trans (eval₂_C _ _ (X c)) (eval₂_X _ _ _)


/-- The algebra isomorphism between multivariable polynomials in no variables
and the ground ring. -/
@[simps!]
def isEmptyAlgEquiv [he : IsEmpty σ] : MvPolynomial σ R ≃ₐ[R] R :=
  AlgEquiv.ofAlgHom (aeval (IsEmpty.elim he)) (Algebra.ofId _ _)
        /-
          R : Type u
          S₁ : Type v
          S₂ : Type w
          S₃ : Type x
          σ : Type u_1
          a a' a₁ a₂ : R
          e : Nat
          s : Finsupp σ Nat
          inst✝ : CommSemiring R
          he : IsEmpty σ
          ⊢ Eq ((MvPolynomial.aeval fun a => he.elim a).comp (Algebra.ofId R (MvPolynomi …
        -/
    (by ext)
        /-
          🎉 no goals
        -/
    (by
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        he : IsEmpty σ
        ⊢ Eq ((Algebra.ofId R (MvPolynomial σ R)).comp (MvPolynomial.aeval fun a => he …
      -/
      ext i m
      /-
        case hf.a
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        he : IsEmpty σ
        i : σ
        m : Finsupp σ Nat
        ⊢ Eq (MvPolynomial.coeff m (((Algebra.ofId R (MvPolynomial σ R)).comp (MvPolyn …
      -/
      exact IsEmpty.elim' he i)
      /-
        🎉 no goals
      -/


variable {R S₁ σ} in
@[simp]
lemma aeval_injective_iff_of_isEmpty [IsEmpty σ] [CommSemiring S₁] [Algebra R S₁] {f : σ → S₁} :
    Function.Injective (aeval f : MvPolynomial σ R →ₐ[R] S₁) ↔
      Function.Injective (algebraMap R S₁) := by
  have : aeval f = (Algebra.ofId R S₁).comp (@isEmptyAlgEquiv R σ _ _).toAlgHom := by
    ext i
    exact IsEmpty.elim' ‹IsEmpty σ› i
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝³ : CommSemiring R
    inst✝² : IsEmpty σ
    inst✝¹ : CommSemiring S₁
    inst✝ : Algebra R S₁
    f : σ → S₁
    this : Eq (MvPolynomial.aeval f) ((Algebra.ofId R S₁).comp ↑(MvPolynomial.isEm …
    ⊢ Iff (Function.Injective ⇑(MvPolynomial.aeval f)) (Function.Injective ⇑(algeb …
  -/
  rw [this, ← Injective.of_comp_iff' _ (@isEmptyAlgEquiv R σ _ _).bijective]
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝³ : CommSemiring R
    inst✝² : IsEmpty σ
    inst✝¹ : CommSemiring S₁
    inst✝ : Algebra R S₁
    f : σ → S₁
    this : Eq (MvPolynomial.aeval f) ((Algebra.ofId R S₁).comp ↑(MvPolynomial.isEm …
    ⊢ Iff (Function.Injective ⇑((Algebra.ofId R S₁).comp ↑(MvPolynomial.isEmptyAlg …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The ring isomorphism between multivariable polynomials in no variables
and the ground ring. -/
@[simps!]
def isEmptyRingEquiv [IsEmpty σ] : MvPolynomial σ R ≃+* R :=
  (isEmptyAlgEquiv R σ).toRingEquiv


/-- A helper function for `sumRingEquiv`. -/
@[simps]
def mvPolynomialEquivMvPolynomial [CommSemiring S₃] (f : MvPolynomial S₁ R →+* MvPolynomial S₂ S₃)
    (g : MvPolynomial S₂ S₃ →+* MvPolynomial S₁ R) (hfgC : (f.comp g).comp C = C)
    (hfgX : ∀ n, f (g (X n)) = X n) (hgfC : (g.comp f).comp C = C) (hgfX : ∀ n, g (f (X n)) = X n) :
    MvPolynomial S₁ R ≃+* MvPolynomial S₂ S₃ where
  toFun := f
  invFun := g
  left_inv := is_id (RingHom.comp _ _) hgfC hgfX
  right_inv := is_id (RingHom.comp _ _) hfgC hfgX
  map_mul' := f.map_mul
  map_add' := f.map_add


/-- The ring isomorphism between multivariable polynomials in a sum of two types,
and multivariable polynomials in one of the types,
with coefficients in multivariable polynomials in the other type.
-/
def sumRingEquiv : MvPolynomial (S₁ ⊕ S₂) R ≃+* MvPolynomial S₁ (MvPolynomial S₂ R) := by
  /-
    R : Type u
    S₁ : Type v
    S₂ : Type w
    S₃ : Type x
    σ : Type u_1
    a a' a₁ a₂ : R
    e : Nat
    s : Finsupp σ Nat
    inst✝ : CommSemiring R
    ⊢ RingEquiv (MvPolynomial (Sum S₁ S₂) R) (MvPolynomial S₁ (MvPolynomial S₂ R))
  -/
  apply mvPolynomialEquivMvPolynomial R (S₁ ⊕ S₂) _ _ (sumToIter R S₁ S₂) (iterToSum R S₁ S₂)
    /-
      case hfgC
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      σ : Type u_1
      a a' a₁ a₂ : R
      e : Nat
      s : Finsupp σ Nat
      inst✝ : CommSemiring R
      ⊢ Eq (((MvPolynomial.sumToIter R S₁ S₂).comp (MvPolynomial.iterToSum R S₁ S₂)) …
    -/
  · refine RingHom.ext (hom_eq_hom _ _ ?hC ?hX)
    /-
      case hC
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      σ : Type u_1
      a a' a₁ a₂ : R
      e : Nat
      s : Finsupp σ Nat
      inst✝ : CommSemiring R
      ⊢ Eq ((((MvPolynomial.sumToIter R S₁ S₂).comp (MvPolynomial.iterToSum R S₁ S₂) …
    -/
    case hC => ext1; simp only [RingHom.comp_apply, iterToSum_C_C, sumToIter_C]
    /-
      case hX
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      σ : Type u_1
      a a' a₁ a₂ : R
      e : Nat
      s : Finsupp σ Nat
      inst✝ : CommSemiring R
      ⊢ ∀ (n : S₂), Eq ((((MvPolynomial.sumToIter R S₁ S₂).comp (MvPolynomial.iterTo …
    -/
    case hX => intro; simp only [RingHom.comp_apply, iterToSum_C_X, sumToIter_Xr]
    /-
      🎉 no goals
    -/
    /-
      case hfgX
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      σ : Type u_1
      a a' a₁ a₂ : R
      e : Nat
      s : Finsupp σ Nat
      inst✝ : CommSemiring R
      ⊢ ∀ (n : S₁), Eq ((MvPolynomial.sumToIter R S₁ S₂) ((MvPolynomial.iterToSum R  …
    -/
  · simp [iterToSum_X, sumToIter_Xl]
    /-
      🎉 no goals
    -/
    /-
      case hgfC
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      σ : Type u_1
      a a' a₁ a₂ : R
      e : Nat
      s : Finsupp σ Nat
      inst✝ : CommSemiring R
      ⊢ Eq (((MvPolynomial.iterToSum R S₁ S₂).comp (MvPolynomial.sumToIter R S₁ S₂)) …
    -/
  · ext1; simp only [RingHom.comp_apply, sumToIter_C, iterToSum_C_C]
          /-
            🎉 no goals
          -/
    /-
      case hgfX
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      σ : Type u_1
      a a' a₁ a₂ : R
      e : Nat
      s : Finsupp σ Nat
      inst✝ : CommSemiring R
      ⊢ ∀ (n : Sum S₁ S₂), Eq ((MvPolynomial.iterToSum R S₁ S₂) ((MvPolynomial.sumTo …
    -/
                  /-
                    🎉 no goals
                  -/
  · rintro ⟨⟩ <;> simp only [sumToIter_Xl, iterToSum_X, sumToIter_Xr, iterToSum_C_X]
                  /-
                    🎉 no goals
                  -/


/-- The algebra isomorphism between multivariable polynomials in a sum of two types,
and multivariable polynomials in one of the types,
with coefficients in multivariable polynomials in the other type.
-/
@[simps!]
def sumAlgEquiv : MvPolynomial (S₁ ⊕ S₂) R ≃ₐ[R] MvPolynomial S₁ (MvPolynomial S₂ R) :=
  { sumRingEquiv R S₁ S₂ with
    commutes' := by
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        ⊢ ∀ (r : R), Eq (__src✝.toFun ((algebraMap R (MvPolynomial (Sum S₁ S₂) R)) r)) …
      -/
      intro r
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        r : R
        ⊢ Eq (__src✝.toFun ((algebraMap R (MvPolynomial (Sum S₁ S₂) R)) r)) ((algebraM …
      -/
      have A : algebraMap R (MvPolynomial S₁ (MvPolynomial S₂ R)) r = (C (C r) : _) := rfl
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        r : R
        A : Eq ((algebraMap R (MvPolynomial S₁ (MvPolynomial S₂ R))) r) (MvPolynomial. …
        ⊢ Eq (__src✝.toFun ((algebraMap R (MvPolynomial (Sum S₁ S₂) R)) r)) ((algebraM …
      -/
      have B : algebraMap R (MvPolynomial (S₁ ⊕ S₂) R) r = C r := rfl
      simp only [sumRingEquiv, mvPolynomialEquivMvPolynomial, Equiv.toFun_as_coe,
        Equiv.coe_fn_mk, B, sumToIter_C, A] }


lemma sumAlgEquiv_comp_rename_inr :
    (sumAlgEquiv R S₁ S₂).toAlgHom.comp (rename Sum.inr) = IsScalarTower.toAlgHom R
        (MvPolynomial S₂ R) (MvPolynomial S₁ (MvPolynomial S₂ R)) := by
  /-
    R : Type u
    S₁ : Type v
    S₂ : Type w
    inst✝ : CommSemiring R
    ⊢ Eq ((↑(MvPolynomial.sumAlgEquiv R S₁ S₂)).comp (MvPolynomial.rename Sum.inr) …
  -/
  ext i
  /-
    case hf.a.a
    R : Type u
    S₁ : Type v
    S₂ : Type w
    inst✝ : CommSemiring R
    i : S₂
    m✝¹ : Finsupp S₁ Nat
    m✝ : Finsupp S₂ Nat
    ⊢ Eq (MvPolynomial.coeff m✝ (MvPolynomial.coeff m✝¹ (((↑(MvPolynomial.sumAlgEq …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma sumAlgEquiv_comp_rename_inl :
    (sumAlgEquiv R S₁ S₂).toAlgHom.comp (rename Sum.inl) =
      MvPolynomial.mapAlgHom (Algebra.ofId _ _) := by
  /-
    R : Type u
    S₁ : Type v
    S₂ : Type w
    inst✝ : CommSemiring R
    ⊢ Eq ((↑(MvPolynomial.sumAlgEquiv R S₁ S₂)).comp (MvPolynomial.rename Sum.inl) …
  -/
  ext i
  /-
    case hf.a.a
    R : Type u
    S₁ : Type v
    S₂ : Type w
    inst✝ : CommSemiring R
    i : S₁
    m✝¹ : Finsupp S₁ Nat
    m✝ : Finsupp S₂ Nat
    ⊢ Eq (MvPolynomial.coeff m✝ (MvPolynomial.coeff m✝¹ (((↑(MvPolynomial.sumAlgEq …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The algebra isomorphism between multivariable polynomials in `Option S₁` and
polynomials with coefficients in `MvPolynomial S₁ R`.
-/
@[simps!]
def optionEquivLeft : MvPolynomial (Option S₁) R ≃ₐ[R] Polynomial (MvPolynomial S₁ R) :=
  AlgEquiv.ofAlgHom (MvPolynomial.aeval fun o => o.elim Polynomial.X fun s => Polynomial.C (X s))
    (Polynomial.aevalTower (MvPolynomial.rename some) (X none))
        /-
          R : Type u
          S₁ : Type v
          S₂ : Type w
          S₃ : Type x
          σ : Type u_1
          a a' a₁ a₂ : R
          e : Nat
          s : Finsupp σ Nat
          inst✝ : CommSemiring R
          ⊢ Eq ((MvPolynomial.aeval fun o => o.elim Polynomial.X fun s => Polynomial.C ( …
        -/
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
                                                     /-
                                                       🎉 no goals
                                                     -/
    (by ext : 2 <;> simp) (by ext i : 2; cases i <;> simp)
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma optionEquivLeft_X_some (x : S₁) : optionEquivLeft R S₁ (X (some x)) = Polynomial.C (X x) := by
  /-
    R : Type u
    S₁ : Type v
    inst✝ : CommSemiring R
    x : S₁
    ⊢ Eq ((MvPolynomial.optionEquivLeft R S₁) (MvPolynomial.X (Option.some x))) (P …
  -/
  simp [optionEquivLeft_apply, aeval_X]
  /-
    🎉 no goals
  -/


lemma optionEquivLeft_X_none : optionEquivLeft R S₁ (X none) = Polynomial.X := by
  /-
    R : Type u
    S₁ : Type v
    inst✝ : CommSemiring R
    ⊢ Eq ((MvPolynomial.optionEquivLeft R S₁) (MvPolynomial.X Option.none)) Polyno …
  -/
  simp [optionEquivLeft_apply, aeval_X]
  /-
    🎉 no goals
  -/


lemma optionEquivLeft_C (r : R) : optionEquivLeft R S₁ (C r) = Polynomial.C (C r) := by
  /-
    R : Type u
    S₁ : Type v
    inst✝ : CommSemiring R
    r : R
    ⊢ Eq ((MvPolynomial.optionEquivLeft R S₁) (MvPolynomial.C r)) (Polynomial.C (M …
  -/
  simp only [optionEquivLeft_apply, aeval_C, Polynomial.algebraMap_apply, algebraMap_eq]
  /-
    🎉 no goals
  -/


/-- The algebra isomorphism between multivariable polynomials in `Option S₁` and
multivariable polynomials with coefficients in polynomials.
-/
@[simps!]
def optionEquivRight : MvPolynomial (Option S₁) R ≃ₐ[R] MvPolynomial S₁ R[X] :=
  AlgEquiv.ofAlgHom (MvPolynomial.aeval fun o => o.elim (C Polynomial.X) X)
    (MvPolynomial.aevalTower (Polynomial.aeval (X none)) fun i => X (Option.some i))
    (by
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        ⊢ Eq ((MvPolynomial.aeval fun o => o.elim (MvPolynomial.C Polynomial.X) MvPoly …
      -/
      ext : 2 <;>
        simp only [MvPolynomial.algebraMap_eq, Option.elim, AlgHom.coe_comp, AlgHom.id_comp,
          IsScalarTower.coe_toAlgHom', comp_apply, aevalTower_C, Polynomial.aeval_X, aeval_X,
          Option.elim', aevalTower_X, AlgHom.coe_id, id, eq_self_iff_true, imp_true_iff])
    (by
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        s : Finsupp σ Nat
        inst✝ : CommSemiring R
        ⊢ Eq ((MvPolynomial.aevalTower (Polynomial.aeval (MvPolynomial.X Option.none)) …
      -/
      ext ⟨i⟩ : 2 <;>
        simp only [Option.elim, AlgHom.coe_comp, comp_apply, aeval_X, aevalTower_C,
          Polynomial.aeval_X, AlgHom.coe_id, id, aevalTower_X])


lemma optionEquivRight_X_some (x : S₁) : optionEquivRight R S₁ (X (some x)) = X x := by
  /-
    R : Type u
    S₁ : Type v
    inst✝ : CommSemiring R
    x : S₁
    ⊢ Eq ((MvPolynomial.optionEquivRight R S₁) (MvPolynomial.X (Option.some x))) ( …
  -/
  simp [optionEquivRight_apply, aeval_X]
  /-
    🎉 no goals
  -/


lemma optionEquivRight_X_none : optionEquivRight R S₁ (X none) = C Polynomial.X := by
  /-
    R : Type u
    S₁ : Type v
    inst✝ : CommSemiring R
    ⊢ Eq ((MvPolynomial.optionEquivRight R S₁) (MvPolynomial.X Option.none)) (MvPo …
  -/
  simp [optionEquivRight_apply, aeval_X]
  /-
    🎉 no goals
  -/


lemma optionEquivRight_C (r : R) : optionEquivRight R S₁ (C r) = C (Polynomial.C r) := by
  /-
    R : Type u
    S₁ : Type v
    inst✝ : CommSemiring R
    r : R
    ⊢ Eq ((MvPolynomial.optionEquivRight R S₁) (MvPolynomial.C r)) (MvPolynomial.C …
  -/
  simp only [optionEquivRight_apply, aeval_C, algebraMap_apply, Polynomial.algebraMap_eq]
  /-
    🎉 no goals
  -/


/-- The algebra isomorphism between multivariable polynomials in `Fin (n + 1)` and
polynomials over multivariable polynomials in `Fin n`.
-/
def finSuccEquiv : MvPolynomial (Fin (n + 1)) R ≃ₐ[R] Polynomial (MvPolynomial (Fin n) R) :=
  (renameEquiv R (_root_.finSuccEquiv n)).trans (optionEquivLeft R (Fin n))


theorem finSuccEquiv_eq :
    (finSuccEquiv R n : MvPolynomial (Fin (n + 1)) R →+* Polynomial (MvPolynomial (Fin n) R)) =
      eval₂Hom (Polynomial.C.comp (C : R →+* MvPolynomial (Fin n) R)) fun i : Fin (n + 1) =>
        Fin.cases Polynomial.X (fun k => Polynomial.C (X k)) i := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    ⊢ Eq (↑(MvPolynomial.finSuccEquiv R n)) (MvPolynomial.eval₂Hom (Polynomial.C.c …
  -/
  ext i : 2
  · simp only [finSuccEquiv, optionEquivLeft_apply, aeval_C, AlgEquiv.coe_trans, RingHom.coe_coe,
      coe_eval₂Hom, comp_apply, renameEquiv_apply, eval₂_C, RingHom.coe_comp, rename_C]
    /-
      case hC.a
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      i : R
      ⊢ Eq ((algebraMap R (Polynomial (MvPolynomial (Fin n) R))) i) (Polynomial.C (M …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case hX.a
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      n✝ : Nat
      ⊢ Eq ((↑(MvPolynomial.finSuccEquiv R n) (MvPolynomial.X i)).coeff n✝) (((MvPol …
    -/
                                 /-
                                   🎉 no goals
                                 -/
  · refine Fin.cases ?_ ?_ i <;> simp [finSuccEquiv]
                                 /-
                                   🎉 no goals
                                 -/


theorem finSuccEquiv_apply (p : MvPolynomial (Fin (n + 1)) R) :
    finSuccEquiv R n p =
      eval₂Hom (Polynomial.C.comp (C : R →+* MvPolynomial (Fin n) R))
        (fun i : Fin (n + 1) => Fin.cases Polynomial.X (fun k => Polynomial.C (X k)) i) p := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    ⊢ Eq ((MvPolynomial.finSuccEquiv R n) p) ((MvPolynomial.eval₂Hom (Polynomial.C …
  -/
  rw [← finSuccEquiv_eq, RingHom.coe_coe]
  /-
    🎉 no goals
  -/


theorem finSuccEquiv_comp_C_eq_C {R : Type u} [CommSemiring R] (n : ℕ) :
    (↑(MvPolynomial.finSuccEquiv R n).symm : Polynomial (MvPolynomial (Fin n) R) →+* _).comp
        (Polynomial.C.comp MvPolynomial.C) =
      (MvPolynomial.C : R →+* MvPolynomial (Fin n.succ) R) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    ⊢ Eq ((↑(MvPolynomial.finSuccEquiv R n).symm).comp (Polynomial.C.comp MvPolyno …
  -/
  refine RingHom.ext fun x => ?_
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    x : R
    ⊢ Eq (((↑(MvPolynomial.finSuccEquiv R n).symm).comp (Polynomial.C.comp MvPolyn …
  -/
  rw [RingHom.comp_apply]
  refine
    (MvPolynomial.finSuccEquiv R n).injective
      (Trans.trans ((MvPolynomial.finSuccEquiv R n).apply_symm_apply _) ?_)
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    x : R
    ⊢ Eq ((Polynomial.C.comp MvPolynomial.C) x) ((MvPolynomial.finSuccEquiv R n) ( …
  -/
  simp only [MvPolynomial.finSuccEquiv_apply, MvPolynomial.eval₂Hom_C]
  /-
    🎉 no goals
  -/


                                                                          /-
                                                                            R : Type u
                                                                            inst✝ : CommSemiring R
                                                                            n : Nat
                                                                            ⊢ Eq ((MvPolynomial.finSuccEquiv R n) (MvPolynomial.X 0)) Polynomial.X
                                                                          -/
theorem finSuccEquiv_X_zero : finSuccEquiv R n (X 0) = Polynomial.X := by simp [finSuccEquiv_apply]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem finSuccEquiv_X_succ {j : Fin n} : finSuccEquiv R n (X j.succ) = Polynomial.C (X j) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    j : Fin n
    ⊢ Eq ((MvPolynomial.finSuccEquiv R n) (MvPolynomial.X j.succ)) (Polynomial.C ( …
  -/
  simp [finSuccEquiv_apply]
  /-
    🎉 no goals
  -/


/-- The coefficient of `m` in the `i`-th coefficient of `finSuccEquiv R n f` equals the
    coefficient of `Finsupp.cons i m` in `f`. -/
theorem finSuccEquiv_coeff_coeff (m : Fin n →₀ ℕ) (f : MvPolynomial (Fin (n + 1)) R) (i : ℕ) :
    coeff m (Polynomial.coeff (finSuccEquiv R n f) i) = coeff (m.cons i) f := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    m : Finsupp (Fin n) Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    i : Nat
    ⊢ Eq (MvPolynomial.coeff m (((MvPolynomial.finSuccEquiv R n) f).coeff i)) (MvP …
  -/
  induction' f using MvPolynomial.induction_on' with j r p q hp hq generalizing i m
  /-
    case h1
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    j : Finsupp (Fin (HAdd.hAdd n 1)) Nat
    r : R
    m : Finsupp (Fin n) Nat
    i : Nat
    ⊢ Eq (MvPolynomial.coeff m (((MvPolynomial.finSuccEquiv R n) ((MvPolynomial.mo …
  -/
  swap
    /-
      case h2
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      p q : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      hp : ∀ (m : Finsupp (Fin n) Nat) (i : Nat), Eq (MvPolynomial.coeff m (((MvPoly …
      hq : ∀ (m : Finsupp (Fin n) Nat) (i : Nat), Eq (MvPolynomial.coeff m (((MvPoly …
      m : Finsupp (Fin n) Nat
      i : Nat
      ⊢ Eq (MvPolynomial.coeff m (((MvPolynomial.finSuccEquiv R n) (HAdd.hAdd p q)). …
    -/
  · simp only [map_add, Polynomial.coeff_add, coeff_add, hp, hq]
    /-
      🎉 no goals
    -/
  simp only [finSuccEquiv_apply, coe_eval₂Hom, eval₂_monomial, RingHom.coe_comp, prod_pow,
    Polynomial.coeff_C_mul, coeff_C_mul, coeff_monomial, Fin.prod_univ_succ, Fin.cases_zero,
    Fin.cases_succ, ← map_prod, ← RingHom.map_pow, Function.comp_apply]
  /-
    case h1
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    j : Finsupp (Fin (HAdd.hAdd n 1)) Nat
    r : R
    m : Finsupp (Fin n) Nat
    i : Nat
    ⊢ Eq (HMul.hMul r (MvPolynomial.coeff m ((HMul.hMul (HPow.hPow Polynomial.X (j …
  -/
  rw [← mul_boole, mul_comm (Polynomial.X ^ j 0), Polynomial.coeff_C_mul_X_pow]; congr 1
  /-
    case h1.e_a
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    j : Finsupp (Fin (HAdd.hAdd n 1)) Nat
    r : R
    m : Finsupp (Fin n) Nat
    i : Nat
    ⊢ Eq (MvPolynomial.coeff m (ite (Eq i (j 0)) (Finset.univ.prod fun x => HPow.h …
  -/
  obtain rfl | hjmi := eq_or_ne j (m.cons i)
  · simpa only [cons_zero, cons_succ, if_pos rfl, monomial_eq, C_1, one_mul, prod_pow] using
      coeff_monomial m m (1 : R)
    /-
      case h1.e_a.inr
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      j : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      r : R
      m : Finsupp (Fin n) Nat
      i : Nat
      hjmi : Ne j (Finsupp.cons i m)
      ⊢ Eq (MvPolynomial.coeff m (ite (Eq i (j 0)) (Finset.univ.prod fun x => HPow.h …
    -/
  · simp only [hjmi, if_false]
    /-
      case h1.e_a.inr
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      j : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      r : R
      m : Finsupp (Fin n) Nat
      i : Nat
      hjmi : Ne j (Finsupp.cons i m)
      ⊢ Eq (MvPolynomial.coeff m (ite (Eq i (j 0)) (Finset.univ.prod fun x => HPow.h …
    -/
    obtain hij | rfl := ne_or_eq i (j 0)
      /-
        case h1.e_a.inr.inl
        R : Type u
        inst✝ : CommSemiring R
        n : Nat
        j : Finsupp (Fin (HAdd.hAdd n 1)) Nat
        r : R
        m : Finsupp (Fin n) Nat
        i : Nat
        hjmi : Ne j (Finsupp.cons i m)
        hij : Ne i (j 0)
        ⊢ Eq (MvPolynomial.coeff m (ite (Eq i (j 0)) (Finset.univ.prod fun x => HPow.h …
      -/
    · simp only [hij, if_false, coeff_zero]
      /-
        🎉 no goals
      -/
    /-
      case h1.e_a.inr.inr
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      j : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      r : R
      m : Finsupp (Fin n) Nat
      hjmi : Ne j (Finsupp.cons (j 0) m)
      ⊢ Eq (MvPolynomial.coeff m (ite (Eq (j 0) (j 0)) (Finset.univ.prod fun x => HP …
    -/
    simp only [eq_self_iff_true, if_true]
    have hmj : m ≠ j.tail := by
      rintro rfl
      rw [cons_tail] at hjmi
      contradiction
    simpa only [monomial_eq, C_1, one_mul, prod_pow, Finsupp.tail_apply, if_neg hmj.symm] using
      coeff_monomial m j.tail (1 : R)


theorem eval_eq_eval_mv_eval' (s : Fin n → R) (y : R) (f : MvPolynomial (Fin (n + 1)) R) :
    eval (Fin.cons y s : Fin (n + 1) → R) f =
      Polynomial.eval y (Polynomial.map (eval s) (finSuccEquiv R n f)) := by
  -- turn this into a def `Polynomial.mapAlgHom`
  let φ : (MvPolynomial (Fin n) R)[X] →ₐ[R] R[X] :=
    { Polynomial.mapRingHom (eval s) with
      commutes' := fun r => by
        convert Polynomial.map_C (eval s)
        exact (eval_C _).symm }
  show
    aeval (Fin.cons y s : Fin (n + 1) → R) f =
      (Polynomial.aeval y).comp (φ.comp (finSuccEquiv R n).toAlgHom) f
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    s : Fin n → R
    y : R
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    φ : AlgHom R (Polynomial (MvPolynomial (Fin n) R)) (Polynomial R) :=
      let __src := Polynomial.mapRingHom (MvPolynomial.eval s);
      { toRingHom := __src, commutes' := ⋯ }
    ⊢ Eq ((MvPolynomial.aeval (Fin.cons y s)) f) (((Polynomial.aeval y).comp (φ.co …
  -/
  congr 2
  /-
    case e_a
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    s : Fin n → R
    y : R
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    φ : AlgHom R (Polynomial (MvPolynomial (Fin n) R)) (Polynomial R) :=
      let __src := Polynomial.mapRingHom (MvPolynomial.eval s);
      { toRingHom := __src, commutes' := ⋯ }
    ⊢ Eq (MvPolynomial.aeval (Fin.cons y s)) ((Polynomial.aeval y).comp (φ.comp ↑( …
  -/
  apply MvPolynomial.algHom_ext
  /-
    case e_a.hf
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    s : Fin n → R
    y : R
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    φ : AlgHom R (Polynomial (MvPolynomial (Fin n) R)) (Polynomial R) :=
      let __src := Polynomial.mapRingHom (MvPolynomial.eval s);
      { toRingHom := __src, commutes' := ⋯ }
    ⊢ ∀ (i : Fin (HAdd.hAdd n 1)), Eq ((MvPolynomial.aeval (Fin.cons y s)) (MvPoly …
  -/
  rw [Fin.forall_iff_succ]
  simp only [φ, aeval_X, Fin.cons_zero, AlgEquiv.toAlgHom_eq_coe, AlgHom.coe_comp,
    Polynomial.coe_aeval_eq_eval, Polynomial.map_C, AlgHom.coe_mk, RingHom.toFun_eq_coe,
    Polynomial.coe_mapRingHom, comp_apply, finSuccEquiv_apply, eval₂Hom_X',
    Fin.cases_zero, Polynomial.map_X, Polynomial.eval_X, Fin.cons_succ,
    Fin.cases_succ, eval_X, Polynomial.eval_C,
    RingHom.coe_mk, MonoidHom.coe_coe, AlgHom.coe_coe, implies_true, and_self,
    RingHom.toMonoidHom_eq_coe]


theorem coeff_eval_eq_eval_coeff (s' : Fin n → R) (f : Polynomial (MvPolynomial (Fin n) R))
    (i : ℕ) : Polynomial.coeff (Polynomial.map (eval s') f) i = eval s' (Polynomial.coeff f i) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    s' : Fin n → R
    f : Polynomial (MvPolynomial (Fin n) R)
    i : Nat
    ⊢ Eq ((Polynomial.map (MvPolynomial.eval s') f).coeff i) ((MvPolynomial.eval s …
  -/
  simp only [Polynomial.coeff_map]
  /-
    🎉 no goals
  -/


theorem support_coeff_finSuccEquiv {f : MvPolynomial (Fin (n + 1)) R} {i : ℕ} {m : Fin n →₀ ℕ} :
    m ∈ ((finSuccEquiv R n f).coeff i).support ↔ m.cons i ∈ f.support := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    i : Nat
    m : Finsupp (Fin n) Nat
    ⊢ Iff (Membership.mem (((MvPolynomial.finSuccEquiv R n) f).coeff i).support m) …
  -/
  apply Iff.intro
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin n) Nat
      ⊢ Membership.mem (((MvPolynomial.finSuccEquiv R n) f).coeff i).support m → Mem …
    -/
  · intro h
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin n) Nat
      h : Membership.mem (((MvPolynomial.finSuccEquiv R n) f).coeff i).support m
      ⊢ Membership.mem f.support (Finsupp.cons i m)
    -/
    simpa [← finSuccEquiv_coeff_coeff] using h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin n) Nat
      ⊢ Membership.mem f.support (Finsupp.cons i m) → Membership.mem (((MvPolynomial …
    -/
  · intro h
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin n) Nat
      h : Membership.mem f.support (Finsupp.cons i m)
      ⊢ Membership.mem (((MvPolynomial.finSuccEquiv R n) f).coeff i).support m
    -/
    simpa [mem_support_iff, ← finSuccEquiv_coeff_coeff m f i] using h
    /-
      🎉 no goals
    -/


/--
The `totalDegree` of a multivariable polynomial `p` is at least `i` more than the `totalDegree` of
the `i`th coefficient of `finSuccEquiv` applied to `p`, if this is nonzero.
-/
lemma totalDegree_coeff_finSuccEquiv_add_le (f : MvPolynomial (Fin (n + 1)) R) (i : ℕ)
    (hi : (finSuccEquiv R n f).coeff i ≠ 0) :
    totalDegree ((finSuccEquiv R n f).coeff i) + i ≤ totalDegree f := by
  have hf'_sup : ((finSuccEquiv R n f).coeff i).support.Nonempty := by
    rw [Finset.nonempty_iff_ne_empty, ne_eq, support_eq_empty]
    exact hi
  -- Let σ be a monomial index of ((finSuccEquiv R n p).coeff i) of maximal total degree
  have ⟨σ, hσ1, hσ2⟩ := Finset.exists_mem_eq_sup (support _) hf'_sup
                          (fun s => Finsupp.sum s fun _ e => e)
  -- Then cons i σ is a monomial index of p with total degree equal to the desired bound
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    i : Nat
    hi : Ne (((MvPolynomial.finSuccEquiv R n) f).coeff i) 0
    hf'_sup : (((MvPolynomial.finSuccEquiv R n) f).coeff i).support.Nonempty
    σ : Finsupp (Fin n) Nat
    hσ1 : Membership.mem (((MvPolynomial.finSuccEquiv R n) f).coeff i).support σ
    hσ2 : Eq ((((MvPolynomial.finSuccEquiv R n) f).coeff i).support.sup fun s => s …
    ⊢ LE.le (HAdd.hAdd (((MvPolynomial.finSuccEquiv R n) f).coeff i).totalDegree i …
  -/
  let σ' : Fin (n+1) →₀ ℕ := cons i σ
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    i : Nat
    hi : Ne (((MvPolynomial.finSuccEquiv R n) f).coeff i) 0
    hf'_sup : (((MvPolynomial.finSuccEquiv R n) f).coeff i).support.Nonempty
    σ : Finsupp (Fin n) Nat
    hσ1 : Membership.mem (((MvPolynomial.finSuccEquiv R n) f).coeff i).support σ
    hσ2 : Eq ((((MvPolynomial.finSuccEquiv R n) f).coeff i).support.sup fun s => s …
    σ' : Finsupp (Fin (HAdd.hAdd n 1)) Nat := Finsupp.cons i σ
    ⊢ LE.le (HAdd.hAdd (((MvPolynomial.finSuccEquiv R n) f).coeff i).totalDegree i …
  -/
  convert le_totalDegree (s := σ') _
    /-
      case h.e'_3
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      hi : Ne (((MvPolynomial.finSuccEquiv R n) f).coeff i) 0
      hf'_sup : (((MvPolynomial.finSuccEquiv R n) f).coeff i).support.Nonempty
      σ : Finsupp (Fin n) Nat
      hσ1 : Membership.mem (((MvPolynomial.finSuccEquiv R n) f).coeff i).support σ
      hσ2 : Eq ((((MvPolynomial.finSuccEquiv R n) f).coeff i).support.sup fun s => s …
      σ' : Finsupp (Fin (HAdd.hAdd n 1)) Nat := Finsupp.cons i σ
      ⊢ Eq (HAdd.hAdd (((MvPolynomial.finSuccEquiv R n) f).coeff i).totalDegree i) ( …
    -/
  · rw [totalDegree, hσ2, sum_cons, add_comm]
    /-
      🎉 no goals
    -/
    /-
      case convert_4
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      hi : Ne (((MvPolynomial.finSuccEquiv R n) f).coeff i) 0
      hf'_sup : (((MvPolynomial.finSuccEquiv R n) f).coeff i).support.Nonempty
      σ : Finsupp (Fin n) Nat
      hσ1 : Membership.mem (((MvPolynomial.finSuccEquiv R n) f).coeff i).support σ
      hσ2 : Eq ((((MvPolynomial.finSuccEquiv R n) f).coeff i).support.sup fun s => s …
      σ' : Finsupp (Fin (HAdd.hAdd n 1)) Nat := Finsupp.cons i σ
      ⊢ Membership.mem f.support σ'
    -/
  · rw [← support_coeff_finSuccEquiv]
    /-
      case convert_4
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      hi : Ne (((MvPolynomial.finSuccEquiv R n) f).coeff i) 0
      hf'_sup : (((MvPolynomial.finSuccEquiv R n) f).coeff i).support.Nonempty
      σ : Finsupp (Fin n) Nat
      hσ1 : Membership.mem (((MvPolynomial.finSuccEquiv R n) f).coeff i).support σ
      hσ2 : Eq ((((MvPolynomial.finSuccEquiv R n) f).coeff i).support.sup fun s => s …
      σ' : Finsupp (Fin (HAdd.hAdd n 1)) Nat := Finsupp.cons i σ
      ⊢ Membership.mem (((MvPolynomial.finSuccEquiv R n) f).coeff i).support σ
    -/
    exact hσ1
    /-
      🎉 no goals
    -/


theorem support_finSuccEquiv (f : MvPolynomial (Fin (n + 1)) R) :
    (finSuccEquiv R n f).support = Finset.image (fun m : Fin (n + 1) →₀ ℕ => m 0) f.support := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    ⊢ Eq ((MvPolynomial.finSuccEquiv R n) f).support (Finset.image (fun m => m 0)  …
  -/
  ext i
  /-
    case h
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    i : Nat
    ⊢ Iff (Membership.mem ((MvPolynomial.finSuccEquiv R n) f).support i) (Membersh …
  -/
  rw [Polynomial.mem_support_iff, Finset.mem_image, Finsupp.ne_iff]
  /-
    case h
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    i : Nat
    ⊢ Iff (Exists fun a => Ne ((((MvPolynomial.finSuccEquiv R n) f).coeff i) a) (0 …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      ⊢ (Exists fun a => Ne ((((MvPolynomial.finSuccEquiv R n) f).coeff i) a) (0 a)) …
    -/
  · rintro ⟨m, hm⟩
    /-
      case h.mp.intro
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin n) Nat
      hm : Ne ((((MvPolynomial.finSuccEquiv R n) f).coeff i) m) (0 m)
      ⊢ Exists fun a => And (Membership.mem f.support a) (Eq (a 0) i)
    -/
    refine ⟨cons i m, ?_, cons_zero _ _⟩
    /-
      case h.mp.intro
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin n) Nat
      hm : Ne ((((MvPolynomial.finSuccEquiv R n) f).coeff i) m) (0 m)
      ⊢ Membership.mem f.support (Finsupp.cons i m)
    -/
    rw [← support_coeff_finSuccEquiv]
    /-
      case h.mp.intro
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin n) Nat
      hm : Ne ((((MvPolynomial.finSuccEquiv R n) f).coeff i) m) (0 m)
      ⊢ Membership.mem (((MvPolynomial.finSuccEquiv R n) f).coeff i).support m
    -/
    simpa using hm
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      ⊢ (Exists fun a => And (Membership.mem f.support a) (Eq (a 0) i)) → Exists fun …
    -/
  · rintro ⟨m, h, rfl⟩
    /-
      case h.mpr.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      m : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      h : Membership.mem f.support m
      ⊢ Exists fun a => Ne ((((MvPolynomial.finSuccEquiv R n) f).coeff (m 0)) a) (0 a)
    -/
    refine ⟨tail m, ?_⟩
    /-
      case h.mpr.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      m : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      h : Membership.mem f.support m
      ⊢ Ne ((((MvPolynomial.finSuccEquiv R n) f).coeff (m 0)) m.tail) (0 m.tail)
    -/
    rwa [← coeff, zero_apply, ← mem_support_iff, support_coeff_finSuccEquiv, cons_tail]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-05")] alias finSuccEquiv_support := support_finSuccEquiv


theorem mem_support_finSuccEquiv {f : MvPolynomial (Fin (n + 1)) R} {x} :
    x ∈ (finSuccEquiv R n f).support ↔ x ∈ (fun m : Fin (n + 1) →₀ _ ↦ m 0) '' f.support := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    x : Nat
    ⊢ Iff (Membership.mem ((MvPolynomial.finSuccEquiv R n) f).support x) (Membersh …
  -/
  simpa using congr(x ∈ $(support_finSuccEquiv f))
  /-
    🎉 no goals
  -/


theorem image_support_finSuccEquiv {f : MvPolynomial (Fin (n + 1)) R} {i : ℕ} :
    ((finSuccEquiv R n f).coeff i).support.image (Finsupp.cons i) = {m ∈ f.support | m 0 = i} := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    i : Nat
    ⊢ Eq (Finset.image (Finsupp.cons i) (((MvPolynomial.finSuccEquiv R n) f).coeff …
  -/
  ext m
  /-
    case h
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    i : Nat
    m : Finsupp (Fin (HAdd.hAdd n 1)) Nat
    ⊢ Iff (Membership.mem (Finset.image (Finsupp.cons i) (((MvPolynomial.finSuccEq …
  -/
  rw [Finset.mem_filter, Finset.mem_image, mem_support_iff]
  conv_lhs =>
    congr
    ext
    rw [mem_support_iff, finSuccEquiv_coeff_coeff, Ne]
  /-
    case h
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    i : Nat
    m : Finsupp (Fin (HAdd.hAdd n 1)) Nat
    ⊢ Iff (Exists fun x => And (Not (Eq (MvPolynomial.coeff (Finsupp.cons i x) f)  …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      ⊢ (Exists fun x => And (Not (Eq (MvPolynomial.coeff (Finsupp.cons i x) f) 0))  …
    -/
  · rintro ⟨m', ⟨h, hm'⟩⟩
    /-
      case h.mp.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      m' : Finsupp (Fin n) Nat
      h : Not (Eq (MvPolynomial.coeff (Finsupp.cons i m') f) 0)
      hm' : Eq (Finsupp.cons i m') m
      ⊢ And (Ne (MvPolynomial.coeff m f) 0) (Eq (m 0) i)
    -/
    simp only [← hm']
    /-
      case h.mp.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      m' : Finsupp (Fin n) Nat
      h : Not (Eq (MvPolynomial.coeff (Finsupp.cons i m') f) 0)
      hm' : Eq (Finsupp.cons i m') m
      ⊢ And (Ne (MvPolynomial.coeff (Finsupp.cons i m') f) 0) (Eq ((Finsupp.cons i m …
    -/
    exact ⟨h, by rw [cons_zero]⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      ⊢ And (Ne (MvPolynomial.coeff m f) 0) (Eq (m 0) i) → Exists fun x => And (Not  …
    -/
  · intro h
    /-
      case h.mpr
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      h : And (Ne (MvPolynomial.coeff m f) 0) (Eq (m 0) i)
      ⊢ Exists fun x => And (Not (Eq (MvPolynomial.coeff (Finsupp.cons i x) f) 0)) ( …
    -/
    use tail m
    /-
      case h
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      h : And (Ne (MvPolynomial.coeff m f) 0) (Eq (m 0) i)
      ⊢ And (Not (Eq (MvPolynomial.coeff (Finsupp.cons i m.tail) f) 0)) (Eq (Finsupp …
    -/
    rw [← h.2, cons_tail]
    /-
      case h
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      i : Nat
      m : Finsupp (Fin (HAdd.hAdd n 1)) Nat
      h : And (Ne (MvPolynomial.coeff m f) 0) (Eq (m 0) i)
      ⊢ And (Not (Eq (MvPolynomial.coeff m f) 0)) (Eq m m)
    -/
    simp [h.1]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-05")] alias finSuccEquiv_support' := image_support_finSuccEquiv


lemma mem_image_support_coeff_finSuccEquiv {f : MvPolynomial (Fin (n + 1)) R} {i : ℕ} {x} :
    x ∈ Finsupp.cons i '' ((finSuccEquiv R n f).coeff i).support ↔
      x ∈ f.support ∧ x 0 = i := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    i : Nat
    x : Finsupp (Fin (HAdd.hAdd n 1)) Nat
    ⊢ Iff (Membership.mem (Set.image (Finsupp.cons i) ↑(((MvPolynomial.finSuccEqui …
  -/
  simpa using congr(x ∈ $image_support_finSuccEquiv)
  /-
    🎉 no goals
  -/


lemma mem_support_coeff_finSuccEquiv {f : MvPolynomial (Fin (n + 1)) R} {i : ℕ} {x} :
    x ∈ ((finSuccEquiv R n f).coeff i).support ↔ x.cons i ∈ f.support := by
  rw [← (Finsupp.cons_right_injective i).mem_finset_image (a := x),
    image_support_finSuccEquiv]
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    i : Nat
    x : Finsupp (Fin n) Nat
    ⊢ Iff (Membership.mem (Finset.filter (fun m => Eq (m 0) i) f.support) (Finsupp …
  -/
  simp only [Finset.mem_filter, mem_support_iff, ne_eq, cons_zero, and_true]
  /-
    🎉 no goals
  -/

-- TODO: generalize `finSuccEquiv R n` to an arbitrary ZeroHom

theorem support_finSuccEquiv_nonempty {f : MvPolynomial (Fin (n + 1)) R} (h : f ≠ 0) :
    (finSuccEquiv R n f).support.Nonempty := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    h : Ne f 0
    ⊢ ((MvPolynomial.finSuccEquiv R n) f).support.Nonempty
  -/
  rwa [Polynomial.support_nonempty, EmbeddingLike.map_ne_zero_iff]
  /-
    🎉 no goals
  -/


theorem degree_finSuccEquiv {f : MvPolynomial (Fin (n + 1)) R} (h : f ≠ 0) :
    (finSuccEquiv R n f).degree = degreeOf 0 f := by
  -- TODO: these should be lemmas
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    h : Ne f 0
    ⊢ Eq ((MvPolynomial.finSuccEquiv R n) f).degree ↑(MvPolynomial.degreeOf 0 f)
  -/
  have h₀ : ∀ {α β : Type _} (f : α → β), (fun x => x) ∘ f = f := fun f => rfl
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    h : Ne f 0
    h₀ : ∀ {α : Type ?u.158943} {β : Type ?u.158946} (f : α → β), Eq (Function.com …
    ⊢ Eq ((MvPolynomial.finSuccEquiv R n) f).degree ↑(MvPolynomial.degreeOf 0 f)
  -/
  have h₁ : ∀ {α β : Type _} (f : α → β), f ∘ (fun x => x) = f := fun f => rfl
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    h : Ne f 0
    h₀ : ∀ {α : Type ?u.158943} {β : Type ?u.158946} (f : α → β), Eq (Function.com …
    h₁ : ∀ {α : Type ?u.159011} {β : Type ?u.159014} (f : α → β), Eq (Function.com …
    ⊢ Eq ((MvPolynomial.finSuccEquiv R n) f).degree ↑(MvPolynomial.degreeOf 0 f)
  -/
  have h₂ : WithBot.some = Nat.cast := rfl

  have h' : ((finSuccEquiv R n f).support.sup fun x => x) = degreeOf 0 f := by
    rw [degreeOf_eq_sup, support_finSuccEquiv, Finset.sup_image, h₀]
  rw [Polynomial.degree, ← h', ← h₂, Finset.coe_sup_of_nonempty (support_finSuccEquiv_nonempty h),
    Finset.max_eq_sup_coe, h₁]


theorem natDegree_finSuccEquiv (f : MvPolynomial (Fin (n + 1)) R) :
    (finSuccEquiv R n f).natDegree = degreeOf 0 f := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    ⊢ Eq ((MvPolynomial.finSuccEquiv R n) f).natDegree (MvPolynomial.degreeOf 0 f)
  -/
  by_cases c : f = 0
    /-
      case pos
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      c : Eq f 0
      ⊢ Eq ((MvPolynomial.finSuccEquiv R n) f).natDegree (MvPolynomial.degreeOf 0 f)
    -/
  · rw [c, map_zero, Polynomial.natDegree_zero, degreeOf_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      c : Not (Eq f 0)
      ⊢ Eq ((MvPolynomial.finSuccEquiv R n) f).natDegree (MvPolynomial.degreeOf 0 f)
    -/
  · rw [Polynomial.natDegree, degree_finSuccEquiv (by simpa only [Ne] )]
    /-
      case neg
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      c : Not (Eq f 0)
      ⊢ Eq (WithBot.unbot' 0 ↑(MvPolynomial.degreeOf 0 f)) (MvPolynomial.degreeOf 0 f)
    -/
    erw [WithBot.unbot'_coe]
    /-
      case neg
      R : Type u
      inst✝ : CommSemiring R
      n : Nat
      f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      c : Not (Eq f 0)
      ⊢ Eq (↑(MvPolynomial.degreeOf 0 f)) (MvPolynomial.degreeOf 0 f)
    -/
    simp
    /-
      🎉 no goals
    -/


theorem degreeOf_coeff_finSuccEquiv (p : MvPolynomial (Fin (n + 1)) R) (j : Fin n) (i : ℕ) :
    degreeOf j (Polynomial.coeff (finSuccEquiv R n p) i) ≤ degreeOf j.succ p := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    j : Fin n
    i : Nat
    ⊢ LE.le (MvPolynomial.degreeOf j (((MvPolynomial.finSuccEquiv R n) p).coeff i) …
  -/
  rw [degreeOf_eq_sup, degreeOf_eq_sup, Finset.sup_le_iff]
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    j : Fin n
    i : Nat
    ⊢ ∀ (b : Finsupp (Fin n) Nat), Membership.mem (((MvPolynomial.finSuccEquiv R n …
  -/
  intro m hm
  /-
    R : Type u
    inst✝ : CommSemiring R
    n : Nat
    p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    j : Fin n
    i : Nat
    m : Finsupp (Fin n) Nat
    hm : Membership.mem (((MvPolynomial.finSuccEquiv R n) p).coeff i).support m
    ⊢ LE.le (m j) (p.support.sup fun m => m j.succ)
  -/
  rw [← Finsupp.cons_succ j i m]
  exact Finset.le_sup
    (f := fun (g : Fin (Nat.succ n) →₀ ℕ) => g (Fin.succ j))
    (support_coeff_finSuccEquiv.1 hm)


/-- Consider a multivariate polynomial `φ` whose variables are indexed by `Option σ`,
and suppose that `σ ≃ Fin n`.
Then one may view `φ` as a polynomial over `MvPolynomial (Fin n) R`, by

1. renaming the variables via `Option σ ≃ Fin (n+1)`, and then singling out the `0`-th variable
    via `MvPolynomial.finSuccEquiv`;
2. first viewing it as polynomial over `MvPolynomial σ R` via `MvPolynomial.optionEquivLeft`,
    and then renaming the variables.

This lemma shows that both constructions are the same. -/
lemma finSuccEquiv_rename_finSuccEquiv (e : σ ≃ Fin n) (φ : MvPolynomial (Option σ) R) :
    ((finSuccEquiv R n) ((rename ((Equiv.optionCongr e).trans (_root_.finSuccEquiv n).symm)) φ)) =
      Polynomial.map (rename e).toRingHom (optionEquivLeft R σ φ) := by
  suffices (finSuccEquiv R n).toRingEquiv.toRingHom.comp (rename ((Equiv.optionCongr e).trans
        (_root_.finSuccEquiv n).symm)).toRingHom =
      (Polynomial.mapRingHom (rename e).toRingHom).comp (optionEquivLeft R σ) by
    exact DFunLike.congr_fun this φ
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    n : Nat
    e : Equiv σ (Fin n)
    φ : MvPolynomial (Option σ) R
    ⊢ Eq ((MvPolynomial.finSuccEquiv R n).toRingEquiv.toRingHom.comp (MvPolynomial …
  -/
  apply ringHom_ext
    /-
      case hC
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      n : Nat
      e : Equiv σ (Fin n)
      φ : MvPolynomial (Option σ) R
      ⊢ ∀ (r : R), Eq (((MvPolynomial.finSuccEquiv R n).toRingEquiv.toRingHom.comp ( …
    -/
  · simp [Polynomial.algebraMap_apply, algebraMap_eq, finSuccEquiv_apply]
    /-
      🎉 no goals
    -/
    /-
      case hX
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      n : Nat
      e : Equiv σ (Fin n)
      φ : MvPolynomial (Option σ) R
      ⊢ ∀ (i : Option σ), Eq (((MvPolynomial.finSuccEquiv R n).toRingEquiv.toRingHom …
    -/
                     /-
                       🎉 no goals
                     -/
  · rintro (i|i) <;> simp [finSuccEquiv_apply]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem rename_polynomial_aeval_X {σ τ : Type*} (f : σ → τ) (i : σ) (p : R[X]) :
    rename f (Polynomial.aeval (X i) p) = Polynomial.aeval (X (f i) : MvPolynomial τ R) p := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    σ : Type u_2
    τ : Type u_3
    f : σ → τ
    i : σ
    p : Polynomial R
    ⊢ Eq ((MvPolynomial.rename f) ((Polynomial.aeval (MvPolynomial.X i)) p)) ((Pol …
  -/
  rw [← aeval_algHom_apply, rename_X]
  /-
    🎉 no goals
  -/


