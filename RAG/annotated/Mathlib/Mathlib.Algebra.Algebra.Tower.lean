/-- The `R`-algebra morphism `A → End (M)` corresponding to the representation of the algebra `A`
on the `B`-module `M`.

This is a stronger version of `DistribMulAction.toLinearMap`, and could also have been
called `Algebra.toModuleEnd`.

The typeclasses correspond to the situation where the types act on each other as
```
R ----→ B
| ⟍     |
|   ⟍   |
↓     ↘ ↓
A ----→ M
```
where the diagram commutes, the action by `R` commutes with everything, and the action by `A` and
`B` on `M` commute.

Typically this is most useful with `B = R` as `Algebra.lsmul R R A : A →ₐ[R] Module.End R M`.
However this can be used to get the fact that left-multiplication by `A` is right `A`-linear, and
vice versa, as
```lean
example : A →ₐ[R] Module.End Aᵐᵒᵖ A := Algebra.lsmul R Aᵐᵒᵖ A
example : Aᵐᵒᵖ →ₐ[R] Module.End A A := Algebra.lsmul R A A
```
respectively; though `LinearMap.mulLeft` and `LinearMap.mulRight` can also be used here.
-/
def lsmul : A →ₐ[R] Module.End B M where
  toFun := DistribMulAction.toLinearMap B M
  map_one' := LinearMap.ext fun _ => one_smul A _
  map_mul' a b := LinearMap.ext <| smul_assoc a b
  map_zero' := LinearMap.ext fun _ => zero_smul A _
  map_add' _a _b := LinearMap.ext fun _ => add_smul _ _ _
  commutes' r := LinearMap.ext <| algebraMap_smul A r


@[simp]
theorem lsmul_coe (a : A) : (lsmul R B M a : M → M) = (a • ·) := rfl


theorem algebraMap_smul [SMul R M] [IsScalarTower R A M] (r : R) (x : M) :
    algebraMap R A r • x = r • x := by
  /-
    R : Type u
    A : Type w
    M : Type v₁
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    inst✝² : MulAction A M
    inst✝¹ : SMul R M
    inst✝ : IsScalarTower R A M
    r : R
    x : M
    ⊢ Eq (HSMul.hSMul ((algebraMap R A) r) x) (HSMul.hSMul r x)
  -/
  rw [Algebra.algebraMap_eq_smul_one, smul_assoc, one_smul]
  /-
    🎉 no goals
  -/


variable {A} in
theorem of_algebraMap_smul [SMul R M] (h : ∀ (r : R) (x : M), algebraMap R A r • x = r • x) :
    IsScalarTower R A M where
                         /-
                           R : Type u
                           A : Type w
                           M : Type v₁
                           inst✝⁴ : CommSemiring R
                           inst✝³ : Semiring A
                           inst✝² : Algebra R A
                           inst✝¹ : MulAction A M
                           inst✝ : SMul R M
                           h : ∀ (r : R) (x : M), Eq (HSMul.hSMul ((algebraMap R A) r) x) (HSMul.hSMul r x)
                           r : R
                           a : A
                           x : M
                           ⊢ Eq (HSMul.hSMul (HSMul.hSMul r a) x) (HSMul.hSMul r (HSMul.hSMul a x))
                         -/
  smul_assoc r a x := by rw [Algebra.smul_def, mul_smul, h]
                         /-
                           🎉 no goals
                         -/


variable (R M) in
theorem of_compHom : letI := MulAction.compHom M (algebraMap R A : R →* A); IsScalarTower R A M :=
  letI := MulAction.compHom M (algebraMap R A : R →* A); of_algebraMap_smul fun _ _ ↦ rfl


theorem of_algebraMap_eq [Algebra R A]
    (h : ∀ x, algebraMap R A x = algebraMap S A (algebraMap R S x)) : IsScalarTower R S A :=
                   /-
                     R : Type u
                     S : Type v
                     A : Type w
                     inst✝⁵ : CommSemiring R
                     inst✝⁴ : CommSemiring S
                     inst✝³ : Semiring A
                     inst✝² : Algebra R S
                     inst✝¹ : Algebra S A
                     inst✝ : Algebra R A
                     h : ∀ (x : R), Eq ((algebraMap R A) x) ((algebraMap S A) ((algebraMap R S) x))
                     x : R
                     y : S
                     z : A
                     ⊢ Eq (HSMul.hSMul (HSMul.hSMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
                   -/
  ⟨fun x y z => by simp_rw [Algebra.smul_def, RingHom.map_mul, mul_assoc, h]⟩
                   /-
                     🎉 no goals
                   -/


/-- See note [partially-applied ext lemmas]. -/
theorem of_algebraMap_eq' [Algebra R A]
    (h : algebraMap R A = (algebraMap S A).comp (algebraMap R S)) : IsScalarTower R S A :=
  of_algebraMap_eq <| RingHom.ext_iff.1 h


theorem algebraMap_eq : algebraMap R A = (algebraMap S A).comp (algebraMap R S) :=
  RingHom.ext fun x => by
    /-
      R : Type u
      S : Type v
      A : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Semiring A
      inst✝³ : Algebra R S
      inst✝² : Algebra S A
      inst✝¹ : Algebra R A
      inst✝ : IsScalarTower R S A
      x : R
      ⊢ Eq ((algebraMap R A) x) (((algebraMap S A).comp (algebraMap R S)) x)
    -/
    simp_rw [RingHom.comp_apply, Algebra.algebraMap_eq_smul_one, smul_assoc, one_smul]
    /-
      🎉 no goals
    -/


theorem algebraMap_apply (x : R) : algebraMap R A x = algebraMap S A (algebraMap R S x) := by
  /-
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R S
    inst✝² : Algebra S A
    inst✝¹ : Algebra R A
    inst✝ : IsScalarTower R S A
    x : R
    ⊢ Eq ((algebraMap R A) x) ((algebraMap S A) ((algebraMap R S) x))
  -/
  rw [algebraMap_eq R S A, RingHom.comp_apply]
  /-
    🎉 no goals
  -/


@[ext]
theorem Algebra.ext {S : Type u} {A : Type v} [CommSemiring S] [Semiring A] (h1 h2 : Algebra S A)
                                /-
                                  R : Type u
                                  S✝ : Type v
                                  A✝ : Type w
                                  B : Type u₁
                                  M : Type v₁
                                  inst✝¹² : CommSemiring R
                                  inst✝¹¹ : CommSemiring S✝
                                  inst✝¹⁰ : Semiring A✝
                                  inst✝⁹ : Semiring B
                                  inst✝⁸ : Algebra R S✝
                                  inst✝⁷ : Algebra S✝ A✝
                                  inst✝⁶ : Algebra S✝ B
                                  inst✝⁵ : Algebra R A✝
                                  inst✝⁴ : Algebra R B
                                  inst✝³ : IsScalarTower R S✝ A✝
                                  inst✝² : IsScalarTower R S✝ B
                                  S : Type u
                                  A : Type v
                                  inst✝¹ : CommSemiring S
                                  inst✝ : Semiring A
                                  h1 h2 : Algebra S A
                                  r : S
                                  x : A
                                  ⊢ A
                                -/
    (h : ∀ (r : S) (x : A), (by have I := h1; exact r • x) = r • x) : h1 = h2 :=
                                              /-
                                                🎉 no goals
                                              -/
  Algebra.algebra_ext _ _ fun r => by
    /-
      S : Type u
      A : Type v
      inst✝¹ : CommSemiring S
      inst✝ : Semiring A
      h1 h2 : Algebra S A
      h : ∀ (r : S) (x : A), Eq (letFun h1 fun I => HSMul.hSMul r x) (HSMul.hSMul r x)
      r : S
      ⊢ Eq ((algebraMap S A) r) ((algebraMap S A) r)
    -/
    simpa only [@Algebra.smul_def _ _ _ _ h1, @Algebra.smul_def _ _ _ _ h2, mul_one] using h r 1
    /-
      🎉 no goals
    -/


/-- In a tower, the canonical map from the middle element to the top element is an
algebra homomorphism over the bottom element. -/
def toAlgHom : S →ₐ[R] A :=
  { algebraMap S A with commutes' := fun _ => (algebraMap_apply _ _ _ _).symm }


theorem toAlgHom_apply (y : S) : toAlgHom R S A y = algebraMap S A y := rfl


@[simp]
theorem coe_toAlgHom : ↑(toAlgHom R S A) = algebraMap S A :=
  RingHom.ext fun _ => rfl


@[simp]
theorem coe_toAlgHom' : (toAlgHom R S A : S → A) = algebraMap S A := rfl


@[simp]
theorem _root_.AlgHom.map_algebraMap (f : A →ₐ[S] B) (r : R) :
    f (algebraMap R A r) = algebraMap R B r := by
  /-
    R : Type u
    S : Type v
    A : Type w
    B : Type u₁
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Semiring A
    inst✝⁷ : Semiring B
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra S A
    inst✝⁴ : Algebra S B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : IsScalarTower R S A
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    r : R
    ⊢ Eq (f ((algebraMap R A) r)) ((algebraMap R B) r)
  -/
  rw [algebraMap_apply R S A r, f.commutes, ← algebraMap_apply R S B]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.AlgHom.comp_algebraMap_of_tower (f : A →ₐ[S] B) :
    (f : A →+* B).comp (algebraMap R A) = algebraMap R B :=
  RingHom.ext (AlgHom.map_algebraMap f)

-- conflicts with IsScalarTower.Subalgebra

instance (priority := 999) subsemiring (U : Subsemiring S) : IsScalarTower U S A :=
  of_algebraMap_eq fun _x => rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/12096): removed @[nolint instance_priority], linter not ported yet

instance (priority := 999) of_algHom {R A B : Type*} [CommSemiring R] [CommSemiring A]
    [CommSemiring B] [Algebra R A] [Algebra R B] (f : A →ₐ[R] B) :
    @IsScalarTower R A B _ f.toRingHom.toAlgebra.toSMul _ :=
  letI := (f : A →+* B).toAlgebra
  of_algebraMap_eq fun x => (f.commutes x).symm


/-- R ⟶ S induces S-Alg ⥤ R-Alg -/
def restrictScalars (f : A →ₐ[S] B) : A →ₐ[R] B :=
  { (f : A →+* B) with
    commutes' := fun r => by
      /-
        R : Type u
        S : Type v
        A : Type w
        B : Type u₁
        M : Type v₁
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Algebra R S
        inst✝⁵ : Algebra S A
        inst✝⁴ : Algebra S B
        inst✝³ : Algebra R A
        inst✝² : Algebra R B
        inst✝¹ : IsScalarTower R S A
        inst✝ : IsScalarTower R S B
        f : AlgHom S A B
        r : R
        ⊢ Eq ((↑↑__src✝).toFun ((algebraMap R A) r)) ((algebraMap R B) r)
      -/
      rw [algebraMap_apply R S A, algebraMap_apply R S B]
      /-
        R : Type u
        S : Type v
        A : Type w
        B : Type u₁
        M : Type v₁
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Algebra R S
        inst✝⁵ : Algebra S A
        inst✝⁴ : Algebra S B
        inst✝³ : Algebra R A
        inst✝² : Algebra R B
        inst✝¹ : IsScalarTower R S A
        inst✝ : IsScalarTower R S B
        f : AlgHom S A B
        r : R
        ⊢ Eq ((↑↑__src✝).toFun ((algebraMap S A) ((algebraMap R S) r))) ((algebraMap S …
      -/
      exact f.commutes (algebraMap R S r) }
      /-
        🎉 no goals
      -/


theorem restrictScalars_apply (f : A →ₐ[S] B) (x : A) : f.restrictScalars R x = f x := rfl


@[simp]
theorem coe_restrictScalars (f : A →ₐ[S] B) : (f.restrictScalars R : A →+* B) = f := rfl


@[simp]
theorem coe_restrictScalars' (f : A →ₐ[S] B) : (restrictScalars R f : A → B) = f := rfl


theorem restrictScalars_injective :
    Function.Injective (restrictScalars R : (A →ₐ[S] B) → A →ₐ[R] B) := fun _ _ h =>
  AlgHom.ext (AlgHom.congr_fun h : _)


/-- R ⟶ S induces S-Alg ⥤ R-Alg -/
def restrictScalars (f : A ≃ₐ[S] B) : A ≃ₐ[R] B :=
  { (f : A ≃+* B) with
    commutes' := fun r => by
      /-
        R : Type u
        S : Type v
        A : Type w
        B : Type u₁
        M : Type v₁
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Algebra R S
        inst✝⁵ : Algebra S A
        inst✝⁴ : Algebra S B
        inst✝³ : Algebra R A
        inst✝² : Algebra R B
        inst✝¹ : IsScalarTower R S A
        inst✝ : IsScalarTower R S B
        f : AlgEquiv S A B
        r : R
        ⊢ Eq (__src✝.toFun ((algebraMap R A) r)) ((algebraMap R B) r)
      -/
      rw [algebraMap_apply R S A, algebraMap_apply R S B]
      /-
        R : Type u
        S : Type v
        A : Type w
        B : Type u₁
        M : Type v₁
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Algebra R S
        inst✝⁵ : Algebra S A
        inst✝⁴ : Algebra S B
        inst✝³ : Algebra R A
        inst✝² : Algebra R B
        inst✝¹ : IsScalarTower R S A
        inst✝ : IsScalarTower R S B
        f : AlgEquiv S A B
        r : R
        ⊢ Eq (__src✝.toFun ((algebraMap S A) ((algebraMap R S) r))) ((algebraMap S B)  …
      -/
      exact f.commutes (algebraMap R S r) }
      /-
        🎉 no goals
      -/


theorem restrictScalars_apply (f : A ≃ₐ[S] B) (x : A) : f.restrictScalars R x = f x := rfl


@[simp]
theorem coe_restrictScalars (f : A ≃ₐ[S] B) : (f.restrictScalars R : A ≃+* B) = f := rfl


@[simp]
theorem coe_restrictScalars' (f : A ≃ₐ[S] B) : (restrictScalars R f : A → B) = f := rfl


theorem restrictScalars_injective :
    Function.Injective (restrictScalars R : (A ≃ₐ[S] B) → A ≃ₐ[R] B) := fun _ _ h =>
  AlgEquiv.ext (AlgEquiv.congr_fun h : _)


/-- If `A` is an `R`-algebra such that the induced morphism `R →+* A` is surjective, then the
`R`-module generated by a set `X` equals the `A`-module generated by `X`. -/
theorem restrictScalars_span (hsur : Function.Surjective (algebraMap R A)) (X : Set M) :
    restrictScalars R (span A X) = span R X := by
  /-
    R : Type u
    A : Type w
    M : Type v₁
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    hsur : Function.Surjective ⇑(algebraMap R A)
    X : Set M
    ⊢ Eq (Submodule.restrictScalars R (Submodule.span A X)) (Submodule.span R X)
  -/
  refine ((span_le_restrictScalars R A X).antisymm fun m hm => ?_).symm
  /-
    R : Type u
    A : Type w
    M : Type v₁
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    hsur : Function.Surjective ⇑(algebraMap R A)
    X : Set M
    m : M
    hm : Membership.mem (Submodule.restrictScalars R (Submodule.span A X)) m
    ⊢ Membership.mem (Submodule.span R X) m
  -/
  refine span_induction subset_span (zero_mem _) (fun _ _ _ _ => add_mem) (fun a m _ hm => ?_) hm
  /-
    R : Type u
    A : Type w
    M : Type v₁
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    hsur : Function.Surjective ⇑(algebraMap R A)
    X : Set M
    m✝ : M
    hm✝ : Membership.mem (Submodule.restrictScalars R (Submodule.span A X)) m✝
    a : A
    m : M
    x✝ : Membership.mem (Submodule.span A X) m
    hm : Membership.mem (Submodule.span R X) m
    ⊢ Membership.mem (Submodule.span R X) (HSMul.hSMul a m)
  -/
  obtain ⟨r, rfl⟩ := hsur a
  /-
    case intro
    R : Type u
    A : Type w
    M : Type v₁
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    hsur : Function.Surjective ⇑(algebraMap R A)
    X : Set M
    m✝ : M
    hm✝ : Membership.mem (Submodule.restrictScalars R (Submodule.span A X)) m✝
    m : M
    x✝ : Membership.mem (Submodule.span A X) m
    hm : Membership.mem (Submodule.span R X) m
    r : R
    ⊢ Membership.mem (Submodule.span R X) (HSMul.hSMul ((algebraMap R A) r) m)
  -/
  simpa [algebraMap_smul] using smul_mem _ r hm
  /-
    🎉 no goals
  -/


theorem coe_span_eq_span_of_surjective (h : Function.Surjective (algebraMap R A)) (s : Set M) :
    (Submodule.span A s : Set M) = Submodule.span R s :=
  congr_arg ((↑) : Submodule R M → Set M) (Submodule.restrictScalars_span R A h s)


theorem smul_mem_span_smul_of_mem {s : Set S} {t : Set A} {k : S} (hks : k ∈ span R s) {x : A}
    (hx : x ∈ t) : k • x ∈ span R (s • t) :=
  span_induction (fun _ hc => subset_span <| Set.smul_mem_smul hc hx)
        /-
          R : Type u
          S : Type v
          A : Type w
          inst✝⁶ : Semiring R
          inst✝⁵ : Semiring S
          inst✝⁴ : AddCommMonoid A
          inst✝³ : Module R S
          inst✝² : Module S A
          inst✝¹ : Module R A
          inst✝ : IsScalarTower R S A
          s : Set S
          t : Set A
          k : S
          hks : Membership.mem (Submodule.span R s) k
          x : A
          hx : Membership.mem t x
          ⊢ Membership.mem (Submodule.span R (HSMul.hSMul s t)) (HSMul.hSMul 0 x)
        -/
    (by rw [zero_smul]; exact zero_mem _)
                        /-
                          🎉 no goals
                        -/
                                 /-
                                   R : Type u
                                   S : Type v
                                   A : Type w
                                   inst✝⁶ : Semiring R
                                   inst✝⁵ : Semiring S
                                   inst✝⁴ : AddCommMonoid A
                                   inst✝³ : Module R S
                                   inst✝² : Module S A
                                   inst✝¹ : Module R A
                                   inst✝ : IsScalarTower R S A
                                   s : Set S
                                   t : Set A
                                   k : S
                                   hks : Membership.mem (Submodule.span R s) k
                                   x : A
                                   hx : Membership.mem t x
                                   c₁ c₂ : S
                                   x✝¹ : Membership.mem (Submodule.span R s) c₁
                                   x✝ : Membership.mem (Submodule.span R s) c₂
                                   ih₁ : Membership.mem (Submodule.span R (HSMul.hSMul s t)) (HSMul.hSMul c₁ x)
                                   ih₂ : Membership.mem (Submodule.span R (HSMul.hSMul s t)) (HSMul.hSMul c₂ x)
                                   ⊢ Membership.mem (Submodule.span R (HSMul.hSMul s t)) (HSMul.hSMul (HAdd.hAdd  …
                                 -/
    (fun c₁ c₂ _ _ ih₁ ih₂ => by rw [add_smul]; exact add_mem ih₁ ih₂)
                                                /-
                                                  🎉 no goals
                                                -/
                        /-
                          R : Type u
                          S : Type v
                          A : Type w
                          inst✝⁶ : Semiring R
                          inst✝⁵ : Semiring S
                          inst✝⁴ : AddCommMonoid A
                          inst✝³ : Module R S
                          inst✝² : Module S A
                          inst✝¹ : Module R A
                          inst✝ : IsScalarTower R S A
                          s : Set S
                          t : Set A
                          k : S
                          hks : Membership.mem (Submodule.span R s) k
                          x : A
                          hx : Membership.mem t x
                          b : R
                          c : S
                          x✝ : Membership.mem (Submodule.span R s) c
                          hc : Membership.mem (Submodule.span R (HSMul.hSMul s t)) (HSMul.hSMul c x)
                          ⊢ Membership.mem (Submodule.span R (HSMul.hSMul s t)) (HSMul.hSMul (HSMul.hSMu …
                        -/
    (fun b c _ hc => by rw [IsScalarTower.smul_assoc]; exact smul_mem _ _ hc) hks
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem span_smul_of_span_eq_top {s : Set S} (hs : span R s = ⊤) (t : Set A) :
    span R (s • t) = (span S t).restrictScalars R :=
  le_antisymm
    (span_le.2 fun _x ⟨p, _hps, _q, hqt, hpqx⟩ ↦ hpqx ▸ (span S t).smul_mem p (subset_span hqt))
    fun _ hp ↦ closure_induction (hx := hp) (zero_mem _) (fun _ _ _ _ ↦ add_mem) fun s0 y hy ↦ by
      refine span_induction (fun x hx ↦ subset_span <| by exact ⟨x, hx, y, hy, rfl⟩) ?_ ?_ ?_
        (hs ▸ mem_top : s0 ∈ span R s)
        /-
          case refine_1
          R : Type u
          S : Type v
          A : Type w
          inst✝⁶ : Semiring R
          inst✝⁵ : Semiring S
          inst✝⁴ : AddCommMonoid A
          inst✝³ : Module R S
          inst✝² : Module S A
          inst✝¹ : Module R A
          inst✝ : IsScalarTower R S A
          s : Set S
          hs : Eq (Submodule.span R s) Top.top
          t : Set A
          x✝ : A
          hp : Membership.mem (Submodule.restrictScalars R (Submodule.span S t)) x✝
          s0 : S
          y : A
          hy : Membership.mem t y
          ⊢ Membership.mem (Submodule.span R (HSMul.hSMul s t)) (HSMul.hSMul 0 y)
        -/
      · rw [zero_smul]; apply zero_mem
                        /-
                          🎉 no goals
                        -/
        /-
          case refine_2
          R : Type u
          S : Type v
          A : Type w
          inst✝⁶ : Semiring R
          inst✝⁵ : Semiring S
          inst✝⁴ : AddCommMonoid A
          inst✝³ : Module R S
          inst✝² : Module S A
          inst✝¹ : Module R A
          inst✝ : IsScalarTower R S A
          s : Set S
          hs : Eq (Submodule.span R s) Top.top
          t : Set A
          x✝ : A
          hp : Membership.mem (Submodule.restrictScalars R (Submodule.span S t)) x✝
          s0 : S
          y : A
          hy : Membership.mem t y
          ⊢ ∀ (x y_1 : S), Membership.mem (Submodule.span R s) x → Membership.mem (Submo …
        -/
      · intro _ _ _ _; rw [add_smul]; apply add_mem
                                      /-
                                        🎉 no goals
                                      -/
        /-
          case refine_3
          R : Type u
          S : Type v
          A : Type w
          inst✝⁶ : Semiring R
          inst✝⁵ : Semiring S
          inst✝⁴ : AddCommMonoid A
          inst✝³ : Module R S
          inst✝² : Module S A
          inst✝¹ : Module R A
          inst✝ : IsScalarTower R S A
          s : Set S
          hs : Eq (Submodule.span R s) Top.top
          t : Set A
          x✝ : A
          hp : Membership.mem (Submodule.restrictScalars R (Submodule.span S t)) x✝
          s0 : S
          y : A
          hy : Membership.mem t y
          ⊢ ∀ (a : R) (x : S), Membership.mem (Submodule.span R s) x → Membership.mem (S …
        -/
      · intro r s0 _ hy; rw [IsScalarTower.smul_assoc]; exact smul_mem _ r hy
                                                        /-
                                                          🎉 no goals
                                                        -/

-- The following two lemmas were originally used to prove `span_smul_of_span_eq_top`
-- but are now not needed.

theorem smul_mem_span_smul' {s : Set S} (hs : span R s = ⊤) {t : Set A} {k : S} {x : A}
    (hx : x ∈ span R (s • t)) : k • x ∈ span R (s • t) := by
  /-
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : AddCommMonoid A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    s : Set S
    hs : Eq (Submodule.span R s) Top.top
    t : Set A
    k : S
    x : A
    hx : Membership.mem (Submodule.span R (HSMul.hSMul s t)) x
    ⊢ Membership.mem (Submodule.span R (HSMul.hSMul s t)) (HSMul.hSMul k x)
  -/
  rw [span_smul_of_span_eq_top hs] at hx ⊢; exact (span S t).smul_mem k hx
                                            /-
                                              🎉 no goals
                                            -/


theorem smul_mem_span_smul {s : Set S} (hs : span R s = ⊤) {t : Set A} {k : S} {x : A}
    (hx : x ∈ span R t) : k • x ∈ span R (s • t) := by
  /-
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : AddCommMonoid A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    s : Set S
    hs : Eq (Submodule.span R s) Top.top
    t : Set A
    k : S
    x : A
    hx : Membership.mem (Submodule.span R t) x
    ⊢ Membership.mem (Submodule.span R (HSMul.hSMul s t)) (HSMul.hSMul k x)
  -/
  rw [span_smul_of_span_eq_top hs]
  /-
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : AddCommMonoid A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    s : Set S
    hs : Eq (Submodule.span R s) Top.top
    t : Set A
    k : S
    x : A
    hx : Membership.mem (Submodule.span R t) x
    ⊢ Membership.mem (Submodule.restrictScalars R (Submodule.span S t)) (HSMul.hSM …
  -/
  exact (span S t).smul_mem k (span_le_restrictScalars R S t hx)
  /-
    🎉 no goals
  -/


/-- A variant of `Submodule.span_image` for `algebraMap`. -/
theorem span_algebraMap_image (a : Set R) :
    Submodule.span R (algebraMap R S '' a) = (Submodule.span R a).map (Algebra.linearMap R S) :=
  (Submodule.span_image <| Algebra.linearMap R S).trans rfl


theorem span_algebraMap_image_of_tower {S T : Type*} [CommSemiring S] [Semiring T] [Module R S]
    [Algebra R T] [Algebra S T] [IsScalarTower R S T] (a : Set S) :
    Submodule.span R (algebraMap S T '' a) =
      (Submodule.span R a).map ((Algebra.linearMap S T).restrictScalars R) :=
  (Submodule.span_image <| (Algebra.linearMap S T).restrictScalars R).trans rfl


theorem map_mem_span_algebraMap_image {S T : Type*} [CommSemiring S] [Semiring T] [Algebra R S]
    [Algebra R T] [Algebra S T] [IsScalarTower R S T] (x : S) (a : Set S)
    (hx : x ∈ Submodule.span R a) : algebraMap S T x ∈ Submodule.span R (algebraMap S T '' a) := by
  /-
    R : Type u
    inst✝⁶ : CommSemiring R
    S : Type u_1
    T : Type u_2
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Semiring T
    inst✝³ : Algebra R S
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    x : S
    a : Set S
    hx : Membership.mem (Submodule.span R a) x
    ⊢ Membership.mem (Submodule.span R (Set.image (⇑(algebraMap S T)) a)) ((algebr …
  -/
  rw [span_algebraMap_image_of_tower, mem_map]
  /-
    R : Type u
    inst✝⁶ : CommSemiring R
    S : Type u_1
    T : Type u_2
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Semiring T
    inst✝³ : Algebra R S
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    x : S
    a : Set S
    hx : Membership.mem (Submodule.span R a) x
    ⊢ Exists fun y => And (Membership.mem (Submodule.span R a) y) (Eq ((↑R (Algebr …
  -/
  exact ⟨x, hx, rfl⟩
  /-
    🎉 no goals
  -/


theorem lsmul_injective [NoZeroSMulDivisors A M] {x : A} (hx : x ≠ 0) :
    Function.Injective (lsmul R B M x) :=
  smul_right_injective M hx


