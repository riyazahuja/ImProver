/--A variant of `DirectSum.induction_on` that uses `DirectSum.lof` instead of `.of`-/
theorem induction_lon {R : Type*} [Semiring R] {ι: Type*} [DecidableEq ι]
    {M : ι → Type*} [(i: ι) → AddCommMonoid <| M i] [(i : ι) → Module R (M i)]
    {C: (⨁ i, M i) → Prop} (x : ⨁ i, M i)
    (H_zero : C 0)
    (H_basic : ∀ i (x : M i), C (lof R ι M i x))
    (H_plus : ∀ (x y : ⨁ i, M i), C x → C y → C (x + y)) : C x := by
  induction x using DirectSum.induction_on with
  | H_zero => exact H_zero
  | H_basic => exact H_basic _ _
  | H_plus x y hx hy => exact H_plus x y hx hy


/--If two `R`-algebras are `R`-equivalent and their quotients by a relation `rel` are defined,
then their quotients are also `R`-equivalent.

(Special case of the third isomorphism theorem.)-/
def algEquivQuotAlgEquiv
    {R : Type u} [CommSemiring R] {A B : Type v} [Semiring A] [Semiring B]
    [Algebra R A] [Algebra R B] (f : A ≃ₐ[R] B) (rel : A → A → Prop) :
    RingQuot rel ≃ₐ[R] RingQuot (rel on f.symm) :=
  AlgEquiv.ofAlgHom
    (RingQuot.liftAlgHom R (s := rel)
      ⟨AlgHom.comp (RingQuot.mkAlgHom R (rel on f.symm)) f,
      fun x y h_rel ↦ by
        /-
          R : Type u
          inst✝⁴ : CommSemiring R
          A B : Type v
          inst✝³ : Semiring A
          inst✝² : Semiring B
          inst✝¹ : Algebra R A
          inst✝ : Algebra R B
          f : AlgEquiv R A B
          rel : A → A → Prop
          x y : A
          h_rel : rel x y
          ⊢ Eq (((RingQuot.mkAlgHom R (Function.onFun rel ⇑f.symm)).comp ↑f) x) (((RingQ …
        -/
        apply RingQuot.mkAlgHom_rel
        /-
          case w
          R : Type u
          inst✝⁴ : CommSemiring R
          A B : Type v
          inst✝³ : Semiring A
          inst✝² : Semiring B
          inst✝¹ : Algebra R A
          inst✝ : Algebra R B
          f : AlgEquiv R A B
          rel : A → A → Prop
          x y : A
          h_rel : rel x y
          ⊢ Function.onFun rel (⇑f.symm) (↑↑f x) (↑↑f y)
        -/
        simpa [Function.onFun]⟩)
        /-
          🎉 no goals
        -/
    ((RingQuot.liftAlgHom R (s := rel on f.symm)
      ⟨AlgHom.comp (RingQuot.mkAlgHom R rel) f.symm,
                     /-
                       R : Type u
                       inst✝⁴ : CommSemiring R
                       A B : Type v
                       inst✝³ : Semiring A
                       inst✝² : Semiring B
                       inst✝¹ : Algebra R A
                       inst✝ : Algebra R B
                       f : AlgEquiv R A B
                       rel : A → A → Prop
                       x y : B
                       h : Function.onFun rel (⇑f.symm) x y
                       ⊢ Eq (((RingQuot.mkAlgHom R rel).comp ↑f.symm) x) (((RingQuot.mkAlgHom R rel). …
                     -/
      fun x y h ↦ by apply RingQuot.mkAlgHom_rel; simpa⟩))
                                                  /-
                                                    🎉 no goals
                                                  -/
        /-
          R : Type u
          inst✝⁴ : CommSemiring R
          A B : Type v
          inst✝³ : Semiring A
          inst✝² : Semiring B
          inst✝¹ : Algebra R A
          inst✝ : Algebra R B
          f : AlgEquiv R A B
          rel : A → A → Prop
          ⊢ Eq (((RingQuot.liftAlgHom R) ⟨(RingQuot.mkAlgHom R (Function.onFun rel ⇑f.sy …
        -/
               /-
                 🎉 no goals
               -/
    (by ext b; simp) (by ext a; simp)
                                /-
                                  🎉 no goals
                                -/


@[deprecated (since := "2024-12-07")] alias algEquiv_quot_algEquiv := algEquivQuotAlgEquiv


/--If two (semi)rings are equivalent and their quotients by a relation `rel` are defined,
then their quotients are also equivalent.

(Special case of `algEquiv_quot_algEquiv` when `R = ℕ`, which in turn is a special
case of the third isomorphism theorem.)-/
def equivQuotEquiv {A B : Type v} [Semiring A] [Semiring B] (f : A ≃+* B) (rel : A → A → Prop) :
    RingQuot rel ≃+* RingQuot (rel on f.symm) :=
  let f_alg : A ≃ₐ[ℕ] B :=
                                              /-
                                                A B : Type v
                                                inst✝¹ : Semiring A
                                                inst✝ : Semiring B
                                                f : RingEquiv A B
                                                rel : A → A → Prop
                                                n : Nat
                                                ⊢ Eq (f ((algebraMap Nat A) n)) ((algebraMap Nat B) n)
                                              -/
    AlgEquiv.ofRingEquiv (f := f) (fun n ↦ by simp)
                                              /-
                                                🎉 no goals
                                              -/
  algEquivQuotAlgEquiv f_alg rel |>.toRingEquiv


@[deprecated (since := "2024-12-07")] alias equiv_quot_equiv := equivQuotEquiv


                                     /-
                                       I : Type u
                                       inst✝⁵ : DecidableEq I
                                       i : I
                                       R : Type v
                                       inst✝⁴ : CommSemiring R
                                       A : I → Type w
                                       inst✝³ : (i : I) → Semiring (A i)
                                       inst✝² : (i : I) → Algebra R (A i)
                                       B : Type w'
                                       inst✝¹ : Semiring B
                                       inst✝ : Algebra R B
                                       maps : {i : I} → AlgHom R (A i) B
                                       ⊢ Module R (DirectSum I fun i => A i)
                                     -/
instance : Module R (⨁ i, A i) := by infer_instance
                                     /-
                                       🎉 no goals
                                     -/


/--The free tensor algebra over a direct sum of `R`-algebras, before
taking the quotient by the free product relation.-/
abbrev FreeTensorAlgebra := TensorAlgebra R (⨁ i, A i)


/--The direct sum of tensor powers of a direct sum of `R`-algebras,
before taking the quotient by the free product relation.-/
abbrev PowerAlgebra := ⨁ (n : ℕ), TensorPower R n (⨁ i, A i)


/--The free tensor algebra and its representation as an infinite direct sum
of tensor powers are (noncomputably) equivalent as `R`-algebras.-/
@[reducible] noncomputable def powerAlgebra_equiv_freeAlgebra :
    PowerAlgebra R A ≃ₐ[R] FreeTensorAlgebra R A :=
  TensorAlgebra.equivDirectSum.symm


/--The generating equivalence relation for elements of the free tensor algebra
that are identified in the free product.-/
inductive rel : FreeTensorAlgebra R A → FreeTensorAlgebra R A → Prop
  | id  : ∀ {i : I}, rel (ι R <| lof R I A i 1) 1
  | prod : ∀ {i : I} {a₁ a₂ : A i},
      rel
        (tprod R (⨁ i, A i) 2 (fun | 0 => lof R I A i a₁ | 1 => lof R I A i a₂))
        (ι R <| lof R I A i (a₁ * a₂))


/--The generating equivalence relation for elements of the power algebra
that are identified in the free product. -/
@[reducible, simp] def rel' := rel R A on ofDirectSum


theorem rel_id (i : I) : rel R A (ι R <| lof R I A i 1) 1 := rel.id



/--The free product of the collection of `R`-algebras `A i`, as a quotient of
`FreeTensorAlgebra R A`.-/
@[reducible] def _root_.LinearAlgebra.FreeProduct := RingQuot <| FreeProduct.rel R A


/--The free product of the collection of `R`-algebras `A i`, as a quotient of `PowerAlgebra R A`-/
@[reducible] def _root_.LinearAlgebra.FreeProductOfPowers := RingQuot <| FreeProduct.rel' R A


@[deprecated (since := "2024-12-07")]
alias _root_.LinearAlgebra.FreeProduct_ofPowers := LinearAlgebra.FreeProductOfPowers


/--The `R`-algebra equivalence relating `FreeProduct` and `FreeProduct_ofPowers`-/
noncomputable def equivPowerAlgebra : FreeProductOfPowers R A ≃ₐ[R] FreeProduct R A :=
  RingQuot.algEquivQuotAlgEquiv
    (FreeProduct.powerAlgebra_equiv_freeAlgebra R A |>.symm) (FreeProduct.rel R A)
  |>.symm


local infixr:60 " ∘ₐ " => AlgHom.comp


                                                         /-
                                                           I : Type u
                                                           inst✝⁵ : DecidableEq I
                                                           i : I
                                                           R : Type v
                                                           inst✝⁴ : CommSemiring R
                                                           A : I → Type w
                                                           inst✝³ : (i : I) → Semiring (A i)
                                                           inst✝² : (i : I) → Algebra R (A i)
                                                           B : Type w'
                                                           inst✝¹ : Semiring B
                                                           inst✝ : Algebra R B
                                                           maps : {i : I} → AlgHom R (A i) B
                                                           ⊢ Semiring (LinearAlgebra.FreeProduct R A)
                                                         -/
instance instSemiring : Semiring (FreeProduct R A) := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           I : Type u
                                                           inst✝⁵ : DecidableEq I
                                                           i : I
                                                           R : Type v
                                                           inst✝⁴ : CommSemiring R
                                                           A : I → Type w
                                                           inst✝³ : (i : I) → Semiring (A i)
                                                           inst✝² : (i : I) → Algebra R (A i)
                                                           B : Type w'
                                                           inst✝¹ : Semiring B
                                                           inst✝ : Algebra R B
                                                           maps : {i : I} → AlgHom R (A i) B
                                                           ⊢ Algebra R (LinearAlgebra.FreeProduct R A)
                                                         -/
instance instAlgebra : Algebra R (FreeProduct R A) := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/


/--The canonical quotient map `FreeTensorAlgebra R A →ₐ[R] FreeProduct R A`,
as an `R`-algebra homomorphism.-/
abbrev mkAlgHom : FreeTensorAlgebra R A →ₐ[R] FreeProduct R A :=
  RingQuot.mkAlgHom R (rel R A)


/--The canonical linear map from the direct sum of the `A i` to the free product.-/
abbrev ι' : (⨁ i, A i) →ₗ[R] FreeProduct R A :=
  (mkAlgHom R A).toLinearMap ∘ₗ TensorAlgebra.ι R (M := ⨁ i, A i)


@[simp] theorem ι_apply (x : ⨁ i, A i) :
  ⟨Quot.mk (Rel <| rel R A) (TensorAlgebra.ι R x)⟩ = ι' R A x := by
    /-
      I : Type u
      inst✝³ : DecidableEq I
      R : Type v
      inst✝² : CommSemiring R
      A : I → Type w
      inst✝¹ : (i : I) → Semiring (A i)
      inst✝ : (i : I) → Algebra R (A i)
      x : DirectSum I fun i => A i
      ⊢ Eq { toQuot := Quot.mk (RingQuot.Rel (LinearAlgebra.FreeProduct.rel R A)) (( …
    -/
    aesop (add simp [ι', mkAlgHom, RingQuot.mkAlgHom, mkRingHom])
    /-
      🎉 no goals
    -/


/--The injection into the free product of any `1 : A i` is the 1 of the free product.-/
theorem identify_one (i : I) : ι' R A (DirectSum.lof R I A i 1) = 1 := by
  /-
    I : Type u
    inst✝³ : DecidableEq I
    R : Type v
    inst✝² : CommSemiring R
    A : I → Type w
    inst✝¹ : (i : I) → Semiring (A i)
    inst✝ : (i : I) → Algebra R (A i)
    i : I
    ⊢ Eq ((LinearAlgebra.FreeProduct.ι' R A) ((DirectSum.lof R I A i) 1)) 1
  -/
  suffices ι' R A (DirectSum.lof R I A i 1) = mkAlgHom R A 1 by simpa
  /-
    I : Type u
    inst✝³ : DecidableEq I
    R : Type v
    inst✝² : CommSemiring R
    A : I → Type w
    inst✝¹ : (i : I) → Semiring (A i)
    inst✝ : (i : I) → Algebra R (A i)
    i : I
    ⊢ Eq ((LinearAlgebra.FreeProduct.ι' R A) ((DirectSum.lof R I A i) 1)) ((Linear …
  -/
  exact RingQuot.mkAlgHom_rel R <| rel_id R A (i := i)
  /-
    🎉 no goals
  -/


/--Multiplication in the free product of the injections of any two `aᵢ aᵢ': A i` for
the same `i` is just the injection of multiplication `aᵢ * aᵢ'` in `A i`.-/
theorem mul_injections (a₁ a₂ : A i) :
    ι' R A (DirectSum.lof R I A i a₁) * ι' R A (DirectSum.lof R I A i a₂)
      = ι' R A (DirectSum.lof R I A i (a₁ * a₂)) := by
  /-
    I : Type u
    inst✝³ : DecidableEq I
    i : I
    R : Type v
    inst✝² : CommSemiring R
    A : I → Type w
    inst✝¹ : (i : I) → Semiring (A i)
    inst✝ : (i : I) → Algebra R (A i)
    a₁ a₂ : A i
    ⊢ Eq (HMul.hMul ((LinearAlgebra.FreeProduct.ι' R A) ((DirectSum.lof R I A i) a …
  -/
  convert RingQuot.mkAlgHom_rel R <| rel.prod
  /-
    case h.e'_2
    I : Type u
    inst✝³ : DecidableEq I
    i : I
    R : Type v
    inst✝² : CommSemiring R
    A : I → Type w
    inst✝¹ : (i : I) → Semiring (A i)
    inst✝ : (i : I) → Algebra R (A i)
    a₁ a₂ : A i
    ⊢ Eq (HMul.hMul ((LinearAlgebra.FreeProduct.ι' R A) ((DirectSum.lof R I A i) a …
  -/
  aesop
  /-
    🎉 no goals
  -/


/--The `i`th canonical injection, from `A i` to the free product, as
a linear map.-/
abbrev lof (i : I) : A i →ₗ[R] FreeProduct R A :=
  ι' R A ∘ₗ DirectSum.lof R I A i


/--`lof R A i 1 = 1` for all `i`.-/
theorem lof_map_one (i : I) : lof R A i 1 = 1 := by
  /-
    I : Type u
    inst✝³ : DecidableEq I
    R : Type v
    inst✝² : CommSemiring R
    A : I → Type w
    inst✝¹ : (i : I) → Semiring (A i)
    inst✝ : (i : I) → Algebra R (A i)
    i : I
    ⊢ Eq ((LinearAlgebra.FreeProduct.lof R A i) 1) 1
  -/
  rw [lof]; dsimp [mkAlgHom]; exact identify_one R A i
                              /-
                                🎉 no goals
                              -/


/--The `i`th canonical injection, from `A i` to the free product.-/
irreducible_def ι (i : I) : A i →ₐ[R] FreeProduct R A :=
  AlgHom.ofLinearMap (ι' R A ∘ₗ DirectSum.lof R I A i)
    (lof_map_one R A i) (mul_injections R A · · |>.symm)


/--The family of canonical injection maps, with `i` left implicit.-/
irreducible_def of {i : I} : A i →ₐ[R] FreeProduct R A := ι R A i



/--Universal property of the free product of algebras:
for every `R`-algebra `B`, every family of maps `maps : (i : I) → (A i →ₐ[R] B)` lifts
to a unique arrow `π` from `FreeProduct R A` such that  `π ∘ ι i = maps i`.-/
@[simps] def lift : ({i : I} → A i →ₐ[R] B) ≃ (FreeProduct R A →ₐ[R] B) where
  toFun maps :=
    RingQuot.liftAlgHom R ⟨
        TensorAlgebra.lift R <|
          DirectSum.toModule R I B <|
            (@maps · |>.toLinearMap),
        fun x y r ↦ by
          cases r with
          | id => simp
          | prod => simp⟩
  invFun π i := π ∘ₐ ι R A i
  left_inv π := by
    /-
      I : Type u
      inst✝⁵ : DecidableEq I
      i : I
      R : Type v
      inst✝⁴ : CommSemiring R
      A : I → Type w
      inst✝³ : (i : I) → Semiring (A i)
      inst✝² : (i : I) → Algebra R (A i)
      B : Type w'
      inst✝¹ : Semiring B
      inst✝ : Algebra R B
      maps π : {i : I} → AlgHom R (A i) B
      ⊢ Eq ((fun π i => π.comp (LinearAlgebra.FreeProduct.ι R A i)) ((fun maps => (R …
    -/
    ext i aᵢ
    /-
      case h.H
      I : Type u
      inst✝⁵ : DecidableEq I
      i✝ : I
      R : Type v
      inst✝⁴ : CommSemiring R
      A : I → Type w
      inst✝³ : (i : I) → Semiring (A i)
      inst✝² : (i : I) → Algebra R (A i)
      B : Type w'
      inst✝¹ : Semiring B
      inst✝ : Algebra R B
      maps π : {i : I} → AlgHom R (A i) B
      i : I
      aᵢ : A i
      ⊢ Eq (((fun π i => π.comp (LinearAlgebra.FreeProduct.ι R A i)) ((fun maps => ( …
    -/
    aesop (add simp [ι, ι'])
    /-
      🎉 no goals
    -/
  right_inv maps := by
    /-
      I : Type u
      inst✝⁵ : DecidableEq I
      i : I
      R : Type v
      inst✝⁴ : CommSemiring R
      A : I → Type w
      inst✝³ : (i : I) → Semiring (A i)
      inst✝² : (i : I) → Algebra R (A i)
      B : Type w'
      inst✝¹ : Semiring B
      inst✝ : Algebra R B
      maps✝ : {i : I} → AlgHom R (A i) B
      maps : AlgHom R (LinearAlgebra.FreeProduct R A) B
      ⊢ Eq ((fun maps => (RingQuot.liftAlgHom R) ⟨(TensorAlgebra.lift R) (DirectSum. …
    -/
    ext i a
    /-
      case w.w.H.h
      I : Type u
      inst✝⁵ : DecidableEq I
      i✝ : I
      R : Type v
      inst✝⁴ : CommSemiring R
      A : I → Type w
      inst✝³ : (i : I) → Semiring (A i)
      inst✝² : (i : I) → Algebra R (A i)
      B : Type w'
      inst✝¹ : Semiring B
      inst✝ : Algebra R B
      maps✝ : {i : I} → AlgHom R (A i) B
      maps : AlgHom R (LinearAlgebra.FreeProduct R A) B
      i : I
      a : A i
      ⊢ Eq ((((((fun maps => (RingQuot.liftAlgHom R) ⟨(TensorAlgebra.lift R) (Direct …
    -/
    aesop (add simp [ι, ι'])
    /-
      🎉 no goals
    -/


/--Universal property of the free product of algebras, property:
for every `R`-algebra `B`, every family of maps `maps : (i : I) → (A i →ₐ[R] B)` lifts
to a unique arrow `π` from `FreeProduct R A` such that  `π ∘ ι i = maps i`.-/
theorem lift_comp_ι : (lift R A maps) ∘ₐ (ι R A i) = maps := by
  /-
    I : Type u
    inst✝⁵ : DecidableEq I
    i : I
    R : Type v
    inst✝⁴ : CommSemiring R
    A : I → Type w
    inst✝³ : (i : I) → Semiring (A i)
    inst✝² : (i : I) → Algebra R (A i)
    B : Type w'
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    maps : {i : I} → AlgHom R (A i) B
    ⊢ Eq (((LinearAlgebra.FreeProduct.lift R A) fun {i} => maps).comp (LinearAlgeb …
  -/
  ext a
  /-
    case H
    I : Type u
    inst✝⁵ : DecidableEq I
    i : I
    R : Type v
    inst✝⁴ : CommSemiring R
    A : I → Type w
    inst✝³ : (i : I) → Semiring (A i)
    inst✝² : (i : I) → Algebra R (A i)
    B : Type w'
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    maps : {i : I} → AlgHom R (A i) B
    a : A i
    ⊢ Eq ((((LinearAlgebra.FreeProduct.lift R A) fun {i} => maps).comp (LinearAlge …
  -/
  simp [lift_apply, ι]
  /-
    🎉 no goals
  -/


@[aesop safe destruct] theorem lift_unique
    (f : FreeProduct R A →ₐ[R] B) (h : ∀ i, f ∘ₐ ι R A i = maps) :
    f = lift R A maps := by
  /-
    I : Type u
    inst✝⁵ : DecidableEq I
    R : Type v
    inst✝⁴ : CommSemiring R
    A : I → Type w
    inst✝³ : (i : I) → Semiring (A i)
    inst✝² : (i : I) → Algebra R (A i)
    B : Type w'
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    maps : {i : I} → AlgHom R (A i) B
    f : AlgHom R (LinearAlgebra.FreeProduct R A) B
    h : ∀ (i : I), Eq (f.comp (LinearAlgebra.FreeProduct.ι R A i)) maps
    ⊢ Eq f ((LinearAlgebra.FreeProduct.lift R A) fun {i} => maps)
  -/
  ext i a; simp_rw [AlgHom.ext_iff] at h; specialize h i a
  /-
    case w.w.H.h
    I : Type u
    inst✝⁵ : DecidableEq I
    R : Type v
    inst✝⁴ : CommSemiring R
    A : I → Type w
    inst✝³ : (i : I) → Semiring (A i)
    inst✝² : (i : I) → Algebra R (A i)
    B : Type w'
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    maps : {i : I} → AlgHom R (A i) B
    f : AlgHom R (LinearAlgebra.FreeProduct R A) B
    i : I
    a : A i
    h : Eq ((f.comp (LinearAlgebra.FreeProduct.ι R A i)) a) (maps a)
    ⊢ Eq ((((f.comp (RingQuot.mkAlgHom R (LinearAlgebra.FreeProduct.rel R A))).toL …
  -/
  simp [h.symm, ι]
  /-
    🎉 no goals
  -/


