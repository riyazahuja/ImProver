instance algebra {r : CommSemiring R} [s : ∀ i, Semiring (f i)] [∀ i, Algebra R (f i)] :
    Algebra R (∀ i : I, f i) :=
  { (Pi.ringHom fun i => algebraMap R (f i) : R →+* ∀ i : I, f i) with
                               /-
                                 I : Type u
                                 R : Type u_1
                                 f✝ : I → Type v
                                 x y : (i : I) → f✝ i
                                 i : I
                                 r : CommSemiring R
                                 s : (i : I) → Semiring (f✝ i)
                                 inst✝ : (i : I) → Algebra R (f✝ i)
                                 a : R
                                 f : (i : I) → f✝ i
                                 ⊢ Eq (HMul.hMul (__src✝ a) f) (HMul.hMul f (__src✝ a))
                               -/
    commutes' := fun a f => by ext; simp [Algebra.commutes]
                                    /-
                                      🎉 no goals
                                    -/
                               /-
                                 I : Type u
                                 R : Type u_1
                                 f✝ : I → Type v
                                 x y : (i : I) → f✝ i
                                 i : I
                                 r : CommSemiring R
                                 s : (i : I) → Semiring (f✝ i)
                                 inst✝ : (i : I) → Algebra R (f✝ i)
                                 a : R
                                 f : (i : I) → f✝ i
                                 ⊢ Eq (HSMul.hSMul a f) (HMul.hMul (__src✝ a) f)
                               -/
    smul_def' := fun a f => by ext; simp [Algebra.smul_def] }
                                    /-
                                      🎉 no goals
                                    -/


theorem algebraMap_def {_ : CommSemiring R} [_s : ∀ i, Semiring (f i)] [∀ i, Algebra R (f i)]
    (a : R) : algebraMap R (∀ i, f i) a = fun i => algebraMap R (f i) a :=
  rfl


@[simp]
theorem algebraMap_apply {_ : CommSemiring R} [_s : ∀ i, Semiring (f i)] [∀ i, Algebra R (f i)]
    (a : R) (i : I) : algebraMap R (∀ i, f i) a i = algebraMap R (f i) a :=
  rfl

-- One could also build a `∀ i, R i`-algebra structure on `∀ i, A i`,
-- when each `A i` is an `R i`-algebra, although I'm not sure that it's useful.

/-- A family of algebra homomorphisms `g i : A →ₐ[R] f i` defines a ring homomorphism
`Pi.algHom g : A →ₐ[R] Π i, f i` given by `Pi.algHom g x i = f i x`. -/
@[simps!]
def algHom [CommSemiring R] [s : ∀ i, Semiring (f i)] [∀ i, Algebra R (f i)]
    {A : Type*} [Semiring A] [Algebra R A] (g : ∀ i, A →ₐ[R] f i) :
    A →ₐ[R] ∀ i, f i where
  __ := Pi.ringHom fun i ↦ (g i).toRingHom
                    /-
                      I : Type u
                      R : Type u_1
                      f : I → Type v
                      x y : (i : I) → f i
                      i : I
                      inst✝³ : CommSemiring R
                      s : (i : I) → Semiring (f i)
                      inst✝² : (i : I) → Algebra R (f i)
                      A : Type u_2
                      inst✝¹ : Semiring A
                      inst✝ : Algebra R A
                      g : (i : I) → AlgHom R A (f i)
                      r : R
                      ⊢ Eq ((↑↑__spread✝⁻⁰).toFun ((algebraMap R A) r)) ((algebraMap R ((i : I) → f  …
                    -/
  commutes' r := by ext; simp
                         /-
                           🎉 no goals
                         -/


/-- `Function.eval` as an `AlgHom`. The name matches `Pi.evalRingHom`, `Pi.evalMonoidHom`,
etc. -/
@[simps]
def evalAlgHom {_ : CommSemiring R} [∀ i, Semiring (f i)] [∀ i, Algebra R (f i)] (i : I) :
    (∀ i, f i) →ₐ[R] f i :=
  { Pi.evalRingHom f i with
    toFun := fun f => f i
    commutes' := fun _ => rfl }


/-- `Function.const` as an `AlgHom`. The name matches `Pi.constRingHom`, `Pi.constMonoidHom`,
etc. -/
@[simps]
def constAlgHom : B →ₐ[R] A → B :=
  { Pi.constRingHom A B with
    toFun := Function.const _
    commutes' := fun _ => rfl }


/-- When `R` is commutative and permits an `algebraMap`, `Pi.constRingHom` is equal to that
map. -/
@[simp]
theorem constRingHom_eq_algebraMap : constRingHom A R = algebraMap R (A → R) :=
  rfl


@[simp]
theorem constAlgHom_eq_algebra_ofId : constAlgHom R A R = Algebra.ofId R (A → R) :=
  rfl


/-- A special case of `Pi.algebra` for non-dependent types. Lean struggles to elaborate
definitions elsewhere in the library without this, -/
instance Function.algebra {R : Type*} (I : Type*) (A : Type*) [CommSemiring R] [Semiring A]
    [Algebra R A] : Algebra R (I → A) :=
  Pi.algebra _ _


/-- `R`-algebra homomorphism between the function spaces `I → A` and `I → B`, induced by an
`R`-algebra homomorphism `f` between `A` and `B`. -/
@[simps]
protected def compLeft (f : A →ₐ[R] B) (I : Type*) : (I → A) →ₐ[R] I → B :=
  { f.toRingHom.compLeft I with
    toFun := fun h => f ∘ h
    commutes' := fun c => by
      /-
        R : Type u
        A : Type v
        B : Type w
        I✝ : Type u_1
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        f : AlgHom R A B
        I : Type u_2
        c : R
        ⊢ Eq ((↑↑{ toFun := fun h => Function.comp (⇑f) h, map_one' := ⋯, map_mul' :=  …
      -/
      ext
      /-
        case h
        R : Type u
        A : Type v
        B : Type w
        I✝ : Type u_1
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        f : AlgHom R A B
        I : Type u_2
        c : R
        x✝ : I
        ⊢ Eq ((↑↑{ toFun := fun h => Function.comp (⇑f) h, map_one' := ⋯, map_mul' :=  …
      -/
      exact f.commutes' c }
      /-
        🎉 no goals
      -/


/-- A family of algebra equivalences `∀ i, (A₁ i ≃ₐ A₂ i)` generates a
multiplicative equivalence between `∀ i, A₁ i` and `∀ i, A₂ i`.

This is the `AlgEquiv` version of `Equiv.piCongrRight`, and the dependent version of
`AlgEquiv.arrowCongr`.
-/
@[simps apply]
def piCongrRight {R ι : Type*} {A₁ A₂ : ι → Type*} [CommSemiring R] [∀ i, Semiring (A₁ i)]
    [∀ i, Semiring (A₂ i)] [∀ i, Algebra R (A₁ i)] [∀ i, Algebra R (A₂ i)]
    (e : ∀ i, A₁ i ≃ₐ[R] A₂ i) : (∀ i, A₁ i) ≃ₐ[R] ∀ i, A₂ i :=
  { @RingEquiv.piCongrRight ι A₁ A₂ _ _ fun i => (e i).toRingEquiv with
    toFun := fun x j => e j (x j)
    invFun := fun x j => (e j).symm (x j)
    commutes' := fun r => by
      /-
        R : Type u_1
        ι : Type u_2
        A₁ : ι → Type u_3
        A₂ : ι → Type u_4
        inst✝⁴ : CommSemiring R
        inst✝³ : (i : ι) → Semiring (A₁ i)
        inst✝² : (i : ι) → Semiring (A₂ i)
        inst✝¹ : (i : ι) → Algebra R (A₁ i)
        inst✝ : (i : ι) → Algebra R (A₂ i)
        e : (i : ι) → AlgEquiv R (A₁ i) (A₂ i)
        r : R
        ⊢ Eq ({ toFun := fun x j => (e j) (x j), invFun := fun x j => (e j).symm (x j) …
      -/
      ext i
      /-
        case h
        R : Type u_1
        ι : Type u_2
        A₁ : ι → Type u_3
        A₂ : ι → Type u_4
        inst✝⁴ : CommSemiring R
        inst✝³ : (i : ι) → Semiring (A₁ i)
        inst✝² : (i : ι) → Semiring (A₂ i)
        inst✝¹ : (i : ι) → Algebra R (A₁ i)
        inst✝ : (i : ι) → Algebra R (A₂ i)
        e : (i : ι) → AlgEquiv R (A₁ i) (A₂ i)
        r : R
        i : ι
        ⊢ Eq ({ toFun := fun x j => (e j) (x j), invFun := fun x j => (e j).symm (x j) …
      -/
      simp }
      /-
        🎉 no goals
      -/


@[simp]
theorem piCongrRight_refl {R ι : Type*} {A : ι → Type*} [CommSemiring R] [∀ i, Semiring (A i)]
    [∀ i, Algebra R (A i)] :
    (piCongrRight fun i => (AlgEquiv.refl : A i ≃ₐ[R] A i)) = AlgEquiv.refl :=
  rfl


@[simp]
theorem piCongrRight_symm {R ι : Type*} {A₁ A₂ : ι → Type*} [CommSemiring R]
    [∀ i, Semiring (A₁ i)] [∀ i, Semiring (A₂ i)] [∀ i, Algebra R (A₁ i)] [∀ i, Algebra R (A₂ i)]
    (e : ∀ i, A₁ i ≃ₐ[R] A₂ i) : (piCongrRight e).symm = piCongrRight fun i => (e i).symm :=
  rfl


@[simp]
theorem piCongrRight_trans {R ι : Type*} {A₁ A₂ A₃ : ι → Type*} [CommSemiring R]
    [∀ i, Semiring (A₁ i)] [∀ i, Semiring (A₂ i)] [∀ i, Semiring (A₃ i)] [∀ i, Algebra R (A₁ i)]
    [∀ i, Algebra R (A₂ i)] [∀ i, Algebra R (A₃ i)] (e₁ : ∀ i, A₁ i ≃ₐ[R] A₂ i)
    (e₂ : ∀ i, A₂ i ≃ₐ[R] A₃ i) :
    (piCongrRight e₁).trans (piCongrRight e₂) = piCongrRight fun i => (e₁ i).trans (e₂ i) :=
  rfl


