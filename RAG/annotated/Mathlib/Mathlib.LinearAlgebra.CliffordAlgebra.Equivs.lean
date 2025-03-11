@[simp]
theorem ι_eq_zero : ι (0 : QuadraticForm R Unit) = 0 :=
  Subsingleton.elim _ _


/-- Since the vector space is empty the ring is commutative. -/
instance : CommRing (CliffordAlgebra (0 : QuadraticForm R Unit)) :=
  { CliffordAlgebra.instRing _ with
    mul_comm := fun x y => by
      induction x using CliffordAlgebra.induction with
      | algebraMap r => apply Algebra.commutes
      | ι x => simp
      | add x₁ x₂ hx₁ hx₂ => rw [mul_add, add_mul, hx₁, hx₂]
      | mul x₁ x₂ hx₁ hx₂ => rw [mul_assoc, hx₂, ← mul_assoc, hx₁, ← mul_assoc] }

-- Porting note: Changed `x.reverse` to `reverse (R := R) x`

theorem reverse_apply (x : CliffordAlgebra (0 : QuadraticForm R Unit)) :
    reverse (R := R) x = x := by
  induction x using CliffordAlgebra.induction with
  | algebraMap r => exact reverse.commutes _
  | ι x => rw [ι_eq_zero, LinearMap.zero_apply, reverse.map_zero]
  | mul x₁ x₂ hx₁ hx₂ => rw [reverse.map_mul, mul_comm, hx₁, hx₂]
  | add x₁ x₂ hx₁ hx₂ => rw [reverse.map_add, hx₁, hx₂]


@[simp]
theorem reverse_eq_id :
    (reverse : CliffordAlgebra (0 : QuadraticForm R Unit) →ₗ[R] _) = LinearMap.id :=
  LinearMap.ext reverse_apply


@[simp]
theorem involute_eq_id :
                                                                                          /-
                                                                                            R : Type u_1
                                                                                            inst✝ : CommRing R
                                                                                            ⊢ Eq CliffordAlgebra.involute (AlgHom.id R (CliffordAlgebra 0))
                                                                                          -/
    (involute : CliffordAlgebra (0 : QuadraticForm R Unit) →ₐ[R] _) = AlgHom.id R _ := by ext; simp
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


/-- The clifford algebra over a 0-dimensional vector space is isomorphic to its scalars. -/
protected def equiv : CliffordAlgebra (0 : QuadraticForm R Unit) ≃ₐ[R] R :=
  AlgEquiv.ofAlgHom
    (CliffordAlgebra.lift (0 : QuadraticForm R Unit) <|
      ⟨0, fun _ : Unit => (zero_mul (0 : R)).trans (algebraMap R _).map_zero.symm⟩)
                           /-
                             R : Type u_1
                             inst✝ : CommRing R
                             ⊢ Eq (((CliffordAlgebra.lift 0) ⟨0, ⋯⟩).comp (Algebra.ofId R (CliffordAlgebra  …
                           -/
    (Algebra.ofId R _) (by ext)
                           /-
                             🎉 no goals
                           -/
        /-
          R : Type u_1
          inst✝ : CommRing R
          ⊢ Eq ((Algebra.ofId R (CliffordAlgebra 0)).comp ((CliffordAlgebra.lift 0) ⟨0,  …
        -/
    (by ext : 1; rw [ι_eq_zero, LinearMap.comp_zero, LinearMap.comp_zero])
                 /-
                   🎉 no goals
                 -/


/-- The quadratic form sending elements to the negation of their square. -/
def Q : QuadraticForm ℝ ℝ :=
  -QuadraticMap.sq (R := ℝ) -- Porting note: Added `(R := ℝ)`


@[simp]
theorem Q_apply (r : ℝ) : Q r = -(r * r) :=
  rfl


/-- Intermediate result for `CliffordAlgebraComplex.equiv`: clifford algebras over
`CliffordAlgebraComplex.Q` above can be converted to `ℂ`. -/
def toComplex : CliffordAlgebra Q →ₐ[ℝ] ℂ :=
  CliffordAlgebra.lift Q
    ⟨LinearMap.toSpanSingleton _ _ Complex.I, fun r => by
      /-
        r : Real
        ⊢ Eq (HMul.hMul ((LinearMap.toSpanSingleton Real Complex Complex.I) r) ((Linea …
      -/
      dsimp [LinearMap.toSpanSingleton, LinearMap.id]
      /-
        r : Real
        ⊢ Eq (HMul.hMul (HMul.hMul (↑r) Complex.I) (HMul.hMul (↑r) Complex.I)) ↑(Neg.n …
      -/
      rw [mul_mul_mul_comm]
      /-
        r : Real
        ⊢ Eq (HMul.hMul (HMul.hMul ↑r ↑r) (HMul.hMul Complex.I Complex.I)) ↑(Neg.neg ( …
      -/
      simp⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem toComplex_ι (r : ℝ) : toComplex (ι Q r) = r • Complex.I :=
  CliffordAlgebra.lift_ι_apply _ _ r


/-- `CliffordAlgebra.involute` is analogous to `Complex.conj`. -/
@[simp]
theorem toComplex_involute (c : CliffordAlgebra Q) :
    toComplex (involute c) = conj (toComplex c) := by
  have : toComplex (involute (ι Q 1)) = conj (toComplex (ι Q 1)) := by
    simp only [involute_ι, toComplex_ι, map_neg, one_smul, Complex.conj_I]
  suffices toComplex.comp involute = Complex.conjAe.toAlgHom.comp toComplex by
    exact AlgHom.congr_fun this c
  /-
    c : CliffordAlgebra CliffordAlgebraComplex.Q
    this : Eq (CliffordAlgebraComplex.toComplex (CliffordAlgebra.involute ((Cliffo …
    ⊢ Eq (CliffordAlgebraComplex.toComplex.comp CliffordAlgebra.involute) ((↑Compl …
  -/
  ext : 2
  /-
    case a.h
    c : CliffordAlgebra CliffordAlgebraComplex.Q
    this : Eq (CliffordAlgebraComplex.toComplex (CliffordAlgebra.involute ((Cliffo …
    ⊢ Eq (((CliffordAlgebraComplex.toComplex.comp CliffordAlgebra.involute).toLine …
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- Intermediate result for `CliffordAlgebraComplex.equiv`: `ℂ` can be converted to
`CliffordAlgebraComplex.Q` above can be converted to. -/
def ofComplex : ℂ →ₐ[ℝ] CliffordAlgebra Q :=
  Complex.lift
    ⟨CliffordAlgebra.ι Q 1, by
      /-
        ⊢ Eq (HMul.hMul ((CliffordAlgebra.ι CliffordAlgebraComplex.Q) 1) ((CliffordAlg …
      -/
      rw [CliffordAlgebra.ι_sq_scalar, Q_apply, one_mul, RingHom.map_neg, RingHom.map_one]⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem ofComplex_I : ofComplex Complex.I = ι Q 1 :=
                                /-
                                  ⊢ Eq (HMul.hMul ↑⟨(CliffordAlgebra.ι CliffordAlgebraComplex.Q) 1, CliffordAlge …
                                -/
  Complex.liftAux_apply_I _ (by simp)
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem toComplex_comp_ofComplex : toComplex.comp ofComplex = AlgHom.id ℝ ℂ := by
  /-
    ⊢ Eq (CliffordAlgebraComplex.toComplex.comp CliffordAlgebraComplex.ofComplex)  …
  -/
  ext1
  /-
    case h
    ⊢ Eq ((CliffordAlgebraComplex.toComplex.comp CliffordAlgebraComplex.ofComplex) …
  -/
  dsimp only [AlgHom.comp_apply, Subtype.coe_mk, AlgHom.id_apply]
  /-
    case h
    ⊢ Eq (CliffordAlgebraComplex.toComplex (CliffordAlgebraComplex.ofComplex Compl …
  -/
  rw [ofComplex_I, toComplex_ι, one_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem toComplex_ofComplex (c : ℂ) : toComplex (ofComplex c) = c :=
  AlgHom.congr_fun toComplex_comp_ofComplex c


@[simp]
theorem ofComplex_comp_toComplex : ofComplex.comp toComplex = AlgHom.id ℝ (CliffordAlgebra Q) := by
  /-
    ⊢ Eq (CliffordAlgebraComplex.ofComplex.comp CliffordAlgebraComplex.toComplex)  …
  -/
  ext
  dsimp only [LinearMap.comp_apply, Subtype.coe_mk, AlgHom.id_apply, AlgHom.toLinearMap_apply,
    AlgHom.comp_apply]
  /-
    case a.h
    ⊢ Eq (CliffordAlgebraComplex.ofComplex (CliffordAlgebraComplex.toComplex ((Cli …
  -/
  rw [toComplex_ι, one_smul, ofComplex_I]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofComplex_toComplex (c : CliffordAlgebra Q) : ofComplex (toComplex c) = c :=
  AlgHom.congr_fun ofComplex_comp_toComplex c


/-- The clifford algebras over `CliffordAlgebraComplex.Q` is isomorphic as an `ℝ`-algebra to `ℂ`. -/
@[simps!]
protected def equiv : CliffordAlgebra Q ≃ₐ[ℝ] ℂ :=
  AlgEquiv.ofAlgHom toComplex ofComplex toComplex_comp_ofComplex ofComplex_comp_toComplex


/-- The clifford algebra is commutative since it is isomorphic to the complex numbers.

TODO: prove this is true for all `CliffordAlgebra`s over a 1-dimensional vector space. -/
instance : CommRing (CliffordAlgebra Q) :=
  { CliffordAlgebra.instRing _ with
    mul_comm := fun x y =>
      CliffordAlgebraComplex.equiv.injective <| by
        /-
          x y : CliffordAlgebra CliffordAlgebraComplex.Q
          ⊢ Eq (CliffordAlgebraComplex.equiv (HMul.hMul x y)) (CliffordAlgebraComplex.eq …
        -/
        rw [map_mul, mul_comm, map_mul] }
        /-
          🎉 no goals
        -/

-- Porting note: Changed `x.reverse` to `reverse (R := ℝ) x`

/-- `reverse` is a no-op over `CliffordAlgebraComplex.Q`. -/
theorem reverse_apply (x : CliffordAlgebra Q) : reverse (R := ℝ) x = x := by
  induction x using CliffordAlgebra.induction with
  | algebraMap r => exact reverse.commutes _
  | ι x => rw [reverse_ι]
  | mul x₁ x₂ hx₁ hx₂ => rw [reverse.map_mul, mul_comm, hx₁, hx₂]
  | add x₁ x₂ hx₁ hx₂ => rw [reverse.map_add, hx₁, hx₂]


@[simp]
theorem reverse_eq_id : (reverse : CliffordAlgebra Q →ₗ[ℝ] _) = LinearMap.id :=
  LinearMap.ext reverse_apply


/-- `Complex.conj` is analogous to `CliffordAlgebra.involute`. -/
@[simp]
theorem ofComplex_conj (c : ℂ) : ofComplex (conj c) = involute (ofComplex c) :=
  CliffordAlgebraComplex.equiv.injective <| by
    /-
      c : Complex
      ⊢ Eq (CliffordAlgebraComplex.equiv (CliffordAlgebraComplex.ofComplex ((starRin …
    -/
    rw [equiv_apply, equiv_apply, toComplex_involute, toComplex_ofComplex, toComplex_ofComplex]
    /-
      🎉 no goals
    -/

-- this name is too short for us to want it visible after `open CliffordAlgebraComplex`
--attribute [protected] Q -- Porting note: removed


/-- `Q c₁ c₂` is a quadratic form over `R × R` such that `CliffordAlgebra (Q c₁ c₂)` is isomorphic
as an `R`-algebra to `ℍ[R,c₁,c₂]`. -/
def Q : QuadraticForm R (R × R) :=
  (c₁ • QuadraticMap.sq).prod (c₂ • QuadraticMap.sq)


@[simp]
theorem Q_apply (v : R × R) : Q c₁ c₂ v = c₁ * (v.1 * v.1) + c₂ * (v.2 * v.2) :=
  rfl


/-- The quaternion basis vectors within the algebra. -/
@[simps i j k]
def quaternionBasis : QuaternionAlgebra.Basis (CliffordAlgebra (Q c₁ c₂)) c₁ c₂ where
  i := ι (Q c₁ c₂) (1, 0)
  j := ι (Q c₁ c₂) (0, 1)
  k := ι (Q c₁ c₂) (1, 0) * ι (Q c₁ c₂) (0, 1)
  i_mul_i := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      c₁ c₂ : R
      ⊢ Eq (HMul.hMul ((CliffordAlgebra.ι (CliffordAlgebraQuaternion.Q c₁ c₂)) { fst …
    -/
    rw [ι_sq_scalar, Q_apply, ← Algebra.algebraMap_eq_smul_one]
    /-
      R : Type u_1
      inst✝ : CommRing R
      c₁ c₂ : R
      ⊢ Eq ((algebraMap R (CliffordAlgebra (CliffordAlgebraQuaternion.Q c₁ c₂))) (HA …
    -/
    simp
    /-
      🎉 no goals
    -/
  j_mul_j := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      c₁ c₂ : R
      ⊢ Eq (HMul.hMul ((CliffordAlgebra.ι (CliffordAlgebraQuaternion.Q c₁ c₂)) { fst …
    -/
    rw [ι_sq_scalar, Q_apply, ← Algebra.algebraMap_eq_smul_one]
    /-
      R : Type u_1
      inst✝ : CommRing R
      c₁ c₂ : R
      ⊢ Eq ((algebraMap R (CliffordAlgebra (CliffordAlgebraQuaternion.Q c₁ c₂))) (HA …
    -/
    simp
    /-
      🎉 no goals
    -/
  i_mul_j := rfl
  j_mul_i := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      c₁ c₂ : R
      ⊢ Eq (HMul.hMul ((CliffordAlgebra.ι (CliffordAlgebraQuaternion.Q c₁ c₂)) { fst …
    -/
    rw [eq_neg_iff_add_eq_zero, ι_mul_ι_add_swap, QuadraticMap.polar]
    /-
      R : Type u_1
      inst✝ : CommRing R
      c₁ c₂ : R
      ⊢ Eq ((algebraMap R (CliffordAlgebra (CliffordAlgebraQuaternion.Q c₁ c₂))) (HS …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Intermediate result of `CliffordAlgebraQuaternion.equiv`: clifford algebras over
`CliffordAlgebraQuaternion.Q` can be converted to `ℍ[R,c₁,c₂]`. -/
def toQuaternion : CliffordAlgebra (Q c₁ c₂) →ₐ[R] ℍ[R,c₁,c₂] :=
  CliffordAlgebra.lift (Q c₁ c₂)
    ⟨{  toFun := fun v => (⟨0, v.1, v.2, 0⟩ : ℍ[R,c₁,c₂])
                                    /-
                                      R : Type u_1
                                      inst✝ : CommRing R
                                      c₁ c₂ : R
                                      v₁ v₂ : Prod R R
                                      ⊢ Eq ((fun v => { re := 0, imI := v.1, imJ := v.2, imK := 0 }) (HAdd.hAdd v₁ v …
                                    -/
        map_add' := fun v₁ v₂ => by simp
                                    /-
                                      🎉 no goals
                                    -/
                                   /-
                                     R : Type u_1
                                     inst✝ : CommRing R
                                     c₁ c₂ r : R
                                     v : Prod R R
                                     ⊢ Eq ({ toFun := fun v => { re := 0, imI := v.1, imJ := v.2, imK := 0 }, map_a …
                                   -/
        map_smul' := fun r v => by dsimp; rw [mul_zero] }, fun v => by
                                          /-
                                            🎉 no goals
                                          -/
      /-
        R : Type u_1
        inst✝ : CommRing R
        c₁ c₂ : R
        v : Prod R R
        ⊢ Eq (HMul.hMul ({ toFun := fun v => { re := 0, imI := v.1, imJ := v.2, imK := …
      -/
      dsimp
      /-
        R : Type u_1
        inst✝ : CommRing R
        c₁ c₂ : R
        v : Prod R R
        ⊢ Eq { re := HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul 0 0) (HMul.hMul (HMul. …
      -/
      ext
      /-
        case re
        R : Type u_1
        inst✝ : CommRing R
        c₁ c₂ : R
        v : Prod R R
        ⊢ Eq { re := HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul 0 0) (HMul.hMul (HMul. …
      -/
      all_goals dsimp; ring⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem toQuaternion_ι (v : R × R) :
    toQuaternion (ι (Q c₁ c₂) v) = (⟨0, v.1, v.2, 0⟩ : ℍ[R,c₁,c₂]) :=
  CliffordAlgebra.lift_ι_apply _ _ v


/-- The "clifford conjugate" maps to the quaternion conjugate. -/
theorem toQuaternion_star (c : CliffordAlgebra (Q c₁ c₂)) :
    toQuaternion (star c) = star (toQuaternion c) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    c₁ c₂ : R
    c : CliffordAlgebra (CliffordAlgebraQuaternion.Q c₁ c₂)
    ⊢ Eq (CliffordAlgebraQuaternion.toQuaternion (Star.star c)) (Star.star (Cliffo …
  -/
  simp only [CliffordAlgebra.star_def']
  induction c using CliffordAlgebra.induction with
  | algebraMap r =>
    simp only [reverse.commutes, AlgHom.commutes, QuaternionAlgebra.coe_algebraMap,
      QuaternionAlgebra.star_coe]
  | ι x =>
    rw [reverse_ι, involute_ι, toQuaternion_ι, map_neg, toQuaternion_ι,
      QuaternionAlgebra.neg_mk, star_mk, neg_zero]
  | mul x₁ x₂ hx₁ hx₂ => simp only [reverse.map_mul, map_mul, hx₁, hx₂, star_mul]
  | add x₁ x₂ hx₁ hx₂ => simp only [reverse.map_add, map_add, hx₁, hx₂, star_add]


/-- Map a quaternion into the clifford algebra. -/
def ofQuaternion : ℍ[R,c₁,c₂] →ₐ[R] CliffordAlgebra (Q c₁ c₂) :=
  (quaternionBasis c₁ c₂).liftHom


@[simp]
theorem ofQuaternion_mk (a₁ a₂ a₃ a₄ : R) :
    ofQuaternion (⟨a₁, a₂, a₃, a₄⟩ : ℍ[R,c₁,c₂]) =
      algebraMap R _ a₁ + a₂ • ι (Q c₁ c₂) (1, 0) + a₃ • ι (Q c₁ c₂) (0, 1) +
        a₄ • (ι (Q c₁ c₂) (1, 0) * ι (Q c₁ c₂) (0, 1)) :=
  rfl


@[simp]
theorem ofQuaternion_comp_toQuaternion :
    ofQuaternion.comp toQuaternion = AlgHom.id R (CliffordAlgebra (Q c₁ c₂)) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    c₁ c₂ : R
    ⊢ Eq (CliffordAlgebraQuaternion.ofQuaternion.comp CliffordAlgebraQuaternion.to …
  -/
  ext : 1
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    c₁ c₂ : R
    ⊢ Eq ((CliffordAlgebraQuaternion.ofQuaternion.comp CliffordAlgebraQuaternion.t …
  -/
  dsimp -- before we end up with two goals and have to do this twice
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    c₁ c₂ : R
    ⊢ Eq ((CliffordAlgebraQuaternion.ofQuaternion.toLinearMap.comp CliffordAlgebra …
  -/
  ext
  all_goals
    dsimp
    rw [toQuaternion_ι]
    dsimp
    simp only [toQuaternion_ι, zero_smul, one_smul, zero_add, add_zero, RingHom.map_zero]


@[simp]
theorem ofQuaternion_toQuaternion (c : CliffordAlgebra (Q c₁ c₂)) :
    ofQuaternion (toQuaternion c) = c :=
  AlgHom.congr_fun ofQuaternion_comp_toQuaternion c


@[simp]
theorem toQuaternion_comp_ofQuaternion :
    toQuaternion.comp ofQuaternion = AlgHom.id R ℍ[R,c₁,c₂] := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    c₁ c₂ : R
    ⊢ Eq (CliffordAlgebraQuaternion.toQuaternion.comp CliffordAlgebraQuaternion.of …
  -/
              /-
                🎉 no goals
              -/
  ext : 1 <;> simp
              /-
                🎉 no goals
              -/


@[simp]
theorem toQuaternion_ofQuaternion (q : ℍ[R,c₁,c₂]) : toQuaternion (ofQuaternion q) = q :=
  AlgHom.congr_fun toQuaternion_comp_ofQuaternion q


/-- The clifford algebra over `CliffordAlgebraQuaternion.Q c₁ c₂` is isomorphic as an `R`-algebra
to `ℍ[R,c₁,c₂]`. -/
@[simps!]
protected def equiv : CliffordAlgebra (Q c₁ c₂) ≃ₐ[R] ℍ[R,c₁,c₂] :=
  AlgEquiv.ofAlgHom toQuaternion ofQuaternion toQuaternion_comp_ofQuaternion
    ofQuaternion_comp_toQuaternion


/-- The quaternion conjugate maps to the "clifford conjugate" (aka `star`). -/
@[simp]
theorem ofQuaternion_star (q : ℍ[R,c₁,c₂]) : ofQuaternion (star q) = star (ofQuaternion q) :=
  CliffordAlgebraQuaternion.equiv.injective <| by
    rw [equiv_apply, equiv_apply, toQuaternion_star, toQuaternion_ofQuaternion,
      toQuaternion_ofQuaternion]

-- this name is too short for us to want it visible after `open CliffordAlgebraQuaternion`
--attribute [protected] Q -- Porting note: removed


theorem ι_mul_ι (r₁ r₂) : ι (0 : QuadraticForm R R) r₁ * ι (0 : QuadraticForm R R) r₂ = 0 := by
  rw [← mul_one r₁, ← mul_one r₂, ← smul_eq_mul R, ← smul_eq_mul R, LinearMap.map_smul,
    LinearMap.map_smul, smul_mul_smul_comm, ι_sq_scalar, QuadraticMap.zero_apply, RingHom.map_zero,
    smul_zero]


/-- The clifford algebra over a 1-dimensional vector space with 0 quadratic form is isomorphic to
the dual numbers. -/
protected def equiv : CliffordAlgebra (0 : QuadraticForm R R) ≃ₐ[R] R[ε] :=
  AlgEquiv.ofAlgHom
    (CliffordAlgebra.lift (0 : QuadraticForm R R) ⟨inrHom R _, fun m => inr_mul_inr _ m m⟩)
    (DualNumber.lift ⟨
      (Algebra.ofId _ _, ι (R := R) _ 1),
      ι_mul_ι (1 : R) 1,
      fun _ => (Algebra.commutes _ _).symm⟩)
    (by
      /-
        R : Type u_1
        inst✝ : CommRing R
        ⊢ Eq (((CliffordAlgebra.lift 0) ⟨TrivSqZeroExt.inrHom R R, ⋯⟩).comp (DualNumbe …
      -/
      ext : 1
      -- This used to be a single `simp` before https://github.com/leanprover/lean4/pull/2644
      simp only [QuadraticMap.zero_apply, AlgHom.coe_comp, Function.comp_apply, lift_apply_eps,
        AlgHom.coe_id, id_eq]
      /-
        case hε
        R : Type u_1
        inst✝ : CommRing R
        ⊢ Eq (((CliffordAlgebra.lift 0) ⟨TrivSqZeroExt.inrHom R R, ⋯⟩) ((CliffordAlgeb …
      -/
      erw [lift_ι_apply]
      /-
        case hε
        R : Type u_1
        inst✝ : CommRing R
        ⊢ Eq ((TrivSqZeroExt.inrHom R R) 1) DualNumber.eps
      -/
      simp)
      /-
        🎉 no goals
      -/
    -- This used to be a single `simp` before https://github.com/leanprover/lean4/pull/2644
    (by
      /-
        R : Type u_1
        inst✝ : CommRing R
        ⊢ Eq ((DualNumber.lift ⟨{ fst := Algebra.ofId R (CliffordAlgebra 0), snd := (C …
      -/
      ext : 2
      simp only [QuadraticMap.zero_apply, AlgHom.comp_toLinearMap, LinearMap.coe_comp,
        Function.comp_apply, AlgHom.toLinearMap_apply, AlgHom.toLinearMap_id, LinearMap.id_comp]
      /-
        case a.h
        R : Type u_1
        inst✝ : CommRing R
        ⊢ Eq ((DualNumber.lift ⟨{ fst := Algebra.ofId R (CliffordAlgebra 0), snd := (C …
      -/
      erw [lift_ι_apply]
      /-
        case a.h
        R : Type u_1
        inst✝ : CommRing R
        ⊢ Eq ((DualNumber.lift ⟨{ fst := Algebra.ofId R (CliffordAlgebra 0), snd := (C …
      -/
      simp)
      /-
        🎉 no goals
      -/


@[simp]
theorem equiv_ι (r : R) : CliffordAlgebraDualNumber.equiv (ι (R := R) _ r) = r • ε :=
  (lift_ι_apply _ _ r).trans (inr_eq_smul_eps _)


@[simp]
theorem equiv_symm_eps :
    CliffordAlgebraDualNumber.equiv.symm (eps : R[ε]) = ι (0 : QuadraticForm R R) 1 :=
  DualNumber.lift_apply_eps _


