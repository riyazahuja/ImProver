/-- `Rel` relates each `ι m * ι m`, for `m : M`, with `Q m`.

The Clifford algebra of `M` is defined as the quotient modulo this relation.
-/
inductive Rel : TensorAlgebra R M → TensorAlgebra R M → Prop
  | of (m : M) : Rel (ι R m * ι R m) (algebraMap R _ (Q m))


/-- The Clifford algebra of an `R`-module `M` equipped with a quadratic_form `Q`.
-/
def CliffordAlgebra :=
  RingQuot (CliffordAlgebra.Rel Q)


instance instInhabited : Inhabited (CliffordAlgebra Q) := RingQuot.instInhabited _

instance instRing : Ring (CliffordAlgebra Q) := RingQuot.instRing _


instance (priority := 900) instAlgebra' {R A M} [CommSemiring R] [AddCommGroup M] [CommRing A]
    [Algebra R A] [Module R M] [Module A M] (Q : QuadraticForm A M)
    [IsScalarTower R A M] :
    Algebra R (CliffordAlgebra Q) :=
  RingQuot.instAlgebra _

-- verify there are no diamonds
-- but doesn't work at `reducible_and_instances` https://github.com/leanprover-community/mathlib4/issues/10906

instance instAlgebra : Algebra R (CliffordAlgebra Q) := instAlgebra' _


instance {R S A M} [CommSemiring R] [CommSemiring S] [AddCommGroup M] [CommRing A]
    [Algebra R A] [Algebra S A] [Module R M] [Module S M] [Module A M] (Q : QuadraticForm A M)
    [IsScalarTower R A M] [IsScalarTower S A M] :
    SMulCommClass R S (CliffordAlgebra Q) :=
  RingQuot.instSMulCommClass _


instance {R S A M} [CommSemiring R] [CommSemiring S] [AddCommGroup M] [CommRing A]
    [SMul R S] [Algebra R A] [Algebra S A] [Module R M] [Module S M] [Module A M]
    [IsScalarTower R A M] [IsScalarTower S A M] [IsScalarTower R S A] (Q : QuadraticForm A M) :
    IsScalarTower R S (CliffordAlgebra Q) :=
  RingQuot.instIsScalarTower _


/-- The canonical linear map `M →ₗ[R] CliffordAlgebra Q`.
-/
def ι : M →ₗ[R] CliffordAlgebra Q :=
  (RingQuot.mkAlgHom R _).toLinearMap.comp (TensorAlgebra.ι R)


/-- As well as being linear, `ι Q` squares to the quadratic form -/
@[simp]
theorem ι_sq_scalar (m : M) : ι Q m * ι Q m = algebraMap R _ (Q m) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    m : M
    ⊢ Eq (HMul.hMul ((CliffordAlgebra.ι Q) m) ((CliffordAlgebra.ι Q) m)) ((algebra …
  -/
  rw [ι]
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    m : M
    ⊢ Eq (HMul.hMul (((RingQuot.mkAlgHom R (CliffordAlgebra.Rel Q)).toLinearMap.co …
  -/
  erw [LinearMap.comp_apply]
  rw [AlgHom.toLinearMap_apply, ← map_mul (RingQuot.mkAlgHom R (Rel Q)),
    RingQuot.mkAlgHom_rel R (Rel.of m), AlgHom.commutes]
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    m : M
    ⊢ Eq ((algebraMap R (RingQuot (CliffordAlgebra.Rel Q))) (Q m)) ((algebraMap R  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_ι_sq_scalar (g : CliffordAlgebra Q →ₐ[R] A) (m : M) :
    g (ι Q m) * g (ι Q m) = algebraMap _ _ (Q m) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    g : AlgHom R (CliffordAlgebra Q) A
    m : M
    ⊢ Eq (HMul.hMul (g ((CliffordAlgebra.ι Q) m)) (g ((CliffordAlgebra.ι Q) m))) ( …
  -/
  rw [← map_mul, ι_sq_scalar, AlgHom.commutes]
  /-
    🎉 no goals
  -/


/-- Given a linear map `f : M →ₗ[R] A` into an `R`-algebra `A`, which satisfies the condition:
`cond : ∀ m : M, f m * f m = Q(m)`, this is the canonical lift of `f` to a morphism of `R`-algebras
from `CliffordAlgebra Q` to `A`.
-/
@[simps symm_apply]
def lift :
    { f : M →ₗ[R] A // ∀ m, f m * f m = algebraMap _ _ (Q m) } ≃ (CliffordAlgebra Q →ₐ[R] A) where
  toFun f :=
    RingQuot.liftAlgHom R
      ⟨TensorAlgebra.lift R (f : M →ₗ[R] A), fun x y (h : Rel Q x y) => by
        /-
          R : Type u_1
          inst✝⁴ : CommRing R
          M : Type u_2
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          Q : QuadraticForm R M
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : Subtype fun f => ∀ (m : M), Eq (HMul.hMul (f m) (f m)) ((algebraMap R A) ( …
          x y : TensorAlgebra R M
          h : CliffordAlgebra.Rel Q x y
          ⊢ Eq (((TensorAlgebra.lift R) ↑f) x) (((TensorAlgebra.lift R) ↑f) y)
        -/
        induction h
        /-
          case of
          R : Type u_1
          inst✝⁴ : CommRing R
          M : Type u_2
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          Q : QuadraticForm R M
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : Subtype fun f => ∀ (m : M), Eq (HMul.hMul (f m) (f m)) ((algebraMap R A) ( …
          x y : TensorAlgebra R M
          m✝ : M
          ⊢ Eq (((TensorAlgebra.lift R) ↑f) (HMul.hMul ((TensorAlgebra.ι R) m✝) ((Tensor …
        -/
        rw [AlgHom.commutes, map_mul, TensorAlgebra.lift_ι_apply, f.prop]⟩
        /-
          🎉 no goals
        -/
  invFun F :=
    ⟨F.toLinearMap.comp (ι Q), fun m => by
      /-
        R : Type u_1
        inst✝⁴ : CommRing R
        M : Type u_2
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        F : AlgHom R (CliffordAlgebra Q) A
        m : M
        ⊢ Eq (HMul.hMul ((F.toLinearMap.comp (CliffordAlgebra.ι Q)) m) ((F.toLinearMap …
      -/
      rw [LinearMap.comp_apply, AlgHom.toLinearMap_apply, comp_ι_sq_scalar]⟩
      /-
        🎉 no goals
      -/
  left_inv f := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      Q : QuadraticForm R M
      A : Type u_3
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : Subtype fun f => ∀ (m : M), Eq (HMul.hMul (f m) (f m)) ((algebraMap R A) ( …
      ⊢ Eq ((fun F => ⟨F.toLinearMap.comp (CliffordAlgebra.ι Q), ⋯⟩) ((fun f => (Rin …
    -/
    ext x
    -- Porting note: removed `simp only` proof which gets stuck simplifying `LinearMap.comp_apply`
    /-
      case a.h
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      Q : QuadraticForm R M
      A : Type u_3
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : Subtype fun f => ∀ (m : M), Eq (HMul.hMul (f m) (f m)) ((algebraMap R A) ( …
      x : M
      ⊢ Eq (↑((fun F => ⟨F.toLinearMap.comp (CliffordAlgebra.ι Q), ⋯⟩) ((fun f => (R …
    -/
    exact (RingQuot.liftAlgHom_mkAlgHom_apply _ _ _ _).trans (TensorAlgebra.lift_ι_apply _ x)
    /-
      🎉 no goals
    -/
  right_inv F :=
    -- Porting note: replaced with proof derived from the one for `TensorAlgebra`
    RingQuot.ringQuot_ext' _ _ _ <|
      TensorAlgebra.hom_ext <|
        LinearMap.ext fun x => by
          exact
            (RingQuot.liftAlgHom_mkAlgHom_apply _ _ _ _).trans (TensorAlgebra.lift_ι_apply _ _)


@[simp]
theorem ι_comp_lift (f : M →ₗ[R] A) (cond : ∀ m, f m * f m = algebraMap _ _ (Q m)) :
    (lift Q ⟨f, cond⟩).toLinearMap.comp (ι Q) = f :=
  Subtype.mk_eq_mk.mp <| (lift Q).symm_apply_apply ⟨f, cond⟩


@[simp]
theorem lift_ι_apply (f : M →ₗ[R] A) (cond : ∀ m, f m * f m = algebraMap _ _ (Q m)) (x) :
    lift Q ⟨f, cond⟩ (ι Q x) = f x :=
  (LinearMap.ext_iff.mp <| ι_comp_lift f cond) x


@[simp]
theorem lift_unique (f : M →ₗ[R] A) (cond : ∀ m : M, f m * f m = algebraMap _ _ (Q m))
    (g : CliffordAlgebra Q →ₐ[R] A) : g.toLinearMap.comp (ι Q) = f ↔ g = lift Q ⟨f, cond⟩ := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M A
    cond : ∀ (m : M), Eq (HMul.hMul (f m) (f m)) ((algebraMap R A) (Q m))
    g : AlgHom R (CliffordAlgebra Q) A
    ⊢ Iff (Eq (g.toLinearMap.comp (CliffordAlgebra.ι Q)) f) (Eq g ((CliffordAlgebr …
  -/
  convert (lift Q : _ ≃ (CliffordAlgebra Q →ₐ[R] A)).symm_apply_eq
  -- Porting note: added `Subtype.mk_eq_mk`
  /-
    case h.e'_1.a
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M A
    cond : ∀ (m : M), Eq (HMul.hMul (f m) (f m)) ((algebraMap R A) (Q m))
    g : AlgHom R (CliffordAlgebra Q) A
    ⊢ Iff (Eq (g.toLinearMap.comp (CliffordAlgebra.ι Q)) f) (Eq ((CliffordAlgebra. …
  -/
  rw [lift_symm_apply, Subtype.mk_eq_mk]
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_comp_ι (g : CliffordAlgebra Q →ₐ[R] A) :
    lift Q ⟨g.toLinearMap.comp (ι Q), comp_ι_sq_scalar _⟩ = g := by
  -- Porting note: removed `rw [lift_symm_apply]; rfl`, changed `convert` to `exact`
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    g : AlgHom R (CliffordAlgebra Q) A
    ⊢ Eq ((CliffordAlgebra.lift Q) ⟨g.toLinearMap.comp (CliffordAlgebra.ι Q), ⋯⟩) g
  -/
  exact (lift Q : _ ≃ (CliffordAlgebra Q →ₐ[R] A)).apply_symm_apply g
  /-
    🎉 no goals
  -/


/-- See note [partially-applied ext lemmas]. -/
@[ext high]
theorem hom_ext {A : Type*} [Semiring A] [Algebra R A] {f g : CliffordAlgebra Q →ₐ[R] A} :
    f.toLinearMap.comp (ι Q) = g.toLinearMap.comp (ι Q) → f = g := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_4
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R (CliffordAlgebra Q) A
    ⊢ Eq (f.toLinearMap.comp (CliffordAlgebra.ι Q)) (g.toLinearMap.comp (CliffordA …
  -/
  intro h
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_4
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R (CliffordAlgebra Q) A
    h : Eq (f.toLinearMap.comp (CliffordAlgebra.ι Q)) (g.toLinearMap.comp (Cliffor …
    ⊢ Eq f g
  -/
  apply (lift Q).symm.injective
  /-
    case a
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_4
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R (CliffordAlgebra Q) A
    h : Eq (f.toLinearMap.comp (CliffordAlgebra.ι Q)) (g.toLinearMap.comp (Cliffor …
    ⊢ Eq ((CliffordAlgebra.lift Q).symm f) ((CliffordAlgebra.lift Q).symm g)
  -/
  rw [lift_symm_apply, lift_symm_apply]
  /-
    case a
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_4
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R (CliffordAlgebra Q) A
    h : Eq (f.toLinearMap.comp (CliffordAlgebra.ι Q)) (g.toLinearMap.comp (Cliffor …
    ⊢ Eq ⟨f.toLinearMap.comp (CliffordAlgebra.ι Q), ⋯⟩ ⟨g.toLinearMap.comp (Cliffo …
  -/
  simp only [h]
  /-
    🎉 no goals
  -/

-- This proof closely follows `TensorAlgebra.induction`

/-- If `C` holds for the `algebraMap` of `r : R` into `CliffordAlgebra Q`, the `ι` of `x : M`,
and is preserved under addition and multiplication, then it holds for all of `CliffordAlgebra Q`.

See also the stronger `CliffordAlgebra.left_induction` and `CliffordAlgebra.right_induction`.
-/
@[elab_as_elim]
theorem induction {C : CliffordAlgebra Q → Prop}
    (algebraMap : ∀ r, C (algebraMap R (CliffordAlgebra Q) r)) (ι : ∀ x, C (ι Q x))
    (mul : ∀ a b, C a → C b → C (a * b)) (add : ∀ a b, C a → C b → C (a + b))
    (a : CliffordAlgebra Q) : C a := by
  -- the arguments are enough to construct a subalgebra, and a mapping into it from M
  let s : Subalgebra R (CliffordAlgebra Q) :=
    { carrier := C
      mul_mem' := @mul
      add_mem' := @add
      algebraMap_mem' := algebraMap }
  -- Porting note: Added `h`. `h` is needed for `of`.
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    C : CliffordAlgebra Q → Prop
    algebraMap : ∀ (r : R), C ((_root_.algebraMap R (CliffordAlgebra Q)) r)
    ι : ∀ (x : M), C ((CliffordAlgebra.ι Q) x)
    mul : ∀ (a b : CliffordAlgebra Q), C a → C b → C (HMul.hMul a b)
    add : ∀ (a b : CliffordAlgebra Q), C a → C b → C (HAdd.hAdd a b)
    a : CliffordAlgebra Q
    s : Subalgebra R (CliffordAlgebra Q) := { carrier := C, mul_mem' := mul, one_m …
    ⊢ C a
  -/
  letI h : AddCommMonoid s := inferInstanceAs (AddCommMonoid (Subalgebra.toSubmodule s))
  let of : { f : M →ₗ[R] s // ∀ m, f m * f m = _root_.algebraMap _ _ (Q m) } :=
    ⟨(CliffordAlgebra.ι Q).codRestrict (Subalgebra.toSubmodule s) ι,
      fun m => Subtype.eq <| ι_sq_scalar Q m⟩
  -- the mapping through the subalgebra is the identity
  have of_id : AlgHom.id R (CliffordAlgebra Q) = s.val.comp (lift Q of) := by
    ext
    simp [of, h]
    -- Porting note: `simp` can't apply this
    erw [LinearMap.codRestrict_apply]
  -- finding a proof is finding an element of the subalgebra
  -- Porting note: was `convert Subtype.prop (lift Q of a); exact AlgHom.congr_fun of_id a`
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    C : CliffordAlgebra Q → Prop
    algebraMap : ∀ (r : R), C ((_root_.algebraMap R (CliffordAlgebra Q)) r)
    ι : ∀ (x : M), C ((CliffordAlgebra.ι Q) x)
    mul : ∀ (a b : CliffordAlgebra Q), C a → C b → C (HMul.hMul a b)
    add : ∀ (a b : CliffordAlgebra Q), C a → C b → C (HAdd.hAdd a b)
    a : CliffordAlgebra Q
    s : Subalgebra R (CliffordAlgebra Q) := { carrier := C, mul_mem' := mul, one_m …
    h : AddCommMonoid (Subtype fun x => Membership.mem s x) := inferInstanceAs (Ad …
    of : Subtype fun f => ∀ (m : M), Eq (HMul.hMul (f m) (f m)) ((_root_.algebraMa …
    of_id : Eq (AlgHom.id R (CliffordAlgebra Q)) (s.val.comp ((CliffordAlgebra.lif …
    ⊢ C a
  -/
  rw [← AlgHom.id_apply (R := R) a, of_id]
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    C : CliffordAlgebra Q → Prop
    algebraMap : ∀ (r : R), C ((_root_.algebraMap R (CliffordAlgebra Q)) r)
    ι : ∀ (x : M), C ((CliffordAlgebra.ι Q) x)
    mul : ∀ (a b : CliffordAlgebra Q), C a → C b → C (HMul.hMul a b)
    add : ∀ (a b : CliffordAlgebra Q), C a → C b → C (HAdd.hAdd a b)
    a : CliffordAlgebra Q
    s : Subalgebra R (CliffordAlgebra Q) := { carrier := C, mul_mem' := mul, one_m …
    h : AddCommMonoid (Subtype fun x => Membership.mem s x) := inferInstanceAs (Ad …
    of : Subtype fun f => ∀ (m : M), Eq (HMul.hMul (f m) (f m)) ((_root_.algebraMa …
    of_id : Eq (AlgHom.id R (CliffordAlgebra Q)) (s.val.comp ((CliffordAlgebra.lif …
    ⊢ C ((s.val.comp ((CliffordAlgebra.lift Q) of)) a)
  -/
  exact Subtype.prop (lift Q of a)
  /-
    🎉 no goals
  -/


theorem mul_add_swap_eq_polar_of_forall_mul_self_eq {A : Type*} [Ring A] [Algebra R A]
    (f : M →ₗ[R] A) (hf : ∀ x, f x * f x = algebraMap _ _ (Q x)) (a b : M) :
    f a * f b + f b * f a = algebraMap R _ (QuadraticMap.polar Q a b) :=
  calc
    f a * f b + f b * f a = f (a + b) * f (a + b) - f a * f a - f b * f b := by
      /-
        R : Type u_1
        inst✝⁴ : CommRing R
        M : Type u_2
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type u_4
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : LinearMap (RingHom.id R) M A
        hf : ∀ (x : M), Eq (HMul.hMul (f x) (f x)) ((algebraMap R A) (Q x))
        a b : M
        ⊢ Eq (HAdd.hAdd (HMul.hMul (f a) (f b)) (HMul.hMul (f b) (f a))) (HSub.hSub (H …
      -/
                                                 /-
                                                   🎉 no goals
                                                 -/
      rw [f.map_add, mul_add, add_mul, add_mul]; abel
                                                 /-
                                                   🎉 no goals
                                                 -/
    _ = algebraMap R _ (Q (a + b)) - algebraMap R _ (Q a) - algebraMap R _ (Q b) := by
      /-
        R : Type u_1
        inst✝⁴ : CommRing R
        M : Type u_2
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type u_4
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : LinearMap (RingHom.id R) M A
        hf : ∀ (x : M), Eq (HMul.hMul (f x) (f x)) ((algebraMap R A) (Q x))
        a b : M
        ⊢ Eq (HSub.hSub (HSub.hSub (HMul.hMul (f (HAdd.hAdd a b)) (f (HAdd.hAdd a b))) …
      -/
      rw [hf, hf, hf]
      /-
        🎉 no goals
      -/
                                                     /-
                                                       R : Type u_1
                                                       inst✝⁴ : CommRing R
                                                       M : Type u_2
                                                       inst✝³ : AddCommGroup M
                                                       inst✝² : Module R M
                                                       Q : QuadraticForm R M
                                                       A : Type u_4
                                                       inst✝¹ : Ring A
                                                       inst✝ : Algebra R A
                                                       f : LinearMap (RingHom.id R) M A
                                                       hf : ∀ (x : M), Eq (HMul.hMul (f x) (f x)) ((algebraMap R A) (Q x))
                                                       a b : M
                                                       ⊢ Eq (HSub.hSub (HSub.hSub ((algebraMap R A) (Q (HAdd.hAdd a b))) ((algebraMap …
                                                     -/
    _ = algebraMap R _ (Q (a + b) - Q a - Q b) := by rw [← RingHom.map_sub, ← RingHom.map_sub]
                                                     /-
                                                       🎉 no goals
                                                     -/
    _ = algebraMap R _ (QuadraticMap.polar Q a b) := rfl


/-- An alternative way to provide the argument to `CliffordAlgebra.lift` when `2` is invertible.

To show a function squares to the quadratic form, it suffices to show that
`f x * f y + f y * f x = algebraMap _ _ (polar Q x y)` -/
theorem forall_mul_self_eq_iff {A : Type*} [Ring A] [Algebra R A] (h2 : IsUnit (2 : A))
    (f : M →ₗ[R] A) :
    (∀ x, f x * f x = algebraMap _ _ (Q x)) ↔
      (LinearMap.mul R A).compl₂ f ∘ₗ f + (LinearMap.mul R A).flip.compl₂ f ∘ₗ f =
        Q.polarBilin.compr₂ (Algebra.linearMap R A) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_4
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    h2 : IsUnit 2
    f : LinearMap (RingHom.id R) M A
    ⊢ Iff (∀ (x : M), Eq (HMul.hMul (f x) (f x)) ((algebraMap R A) (Q x))) (Eq (HA …
  -/
  simp_rw [DFunLike.ext_iff]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_4
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    h2 : IsUnit 2
    f : LinearMap (RingHom.id R) M A
    ⊢ Iff (∀ (x : M), Eq (HMul.hMul (f x) (f x)) ((algebraMap R A) (Q x))) (∀ (x x …
  -/
  refine ⟨mul_add_swap_eq_polar_of_forall_mul_self_eq _, fun h x => ?_⟩
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_4
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    h2 : IsUnit 2
    f : LinearMap (RingHom.id R) M A
    h : ∀ (x x_1 : M), Eq (((HAdd.hAdd (((LinearMap.mul R A).compl₂ f).comp f) ((( …
    x : M
    ⊢ Eq (HMul.hMul (f x) (f x)) ((algebraMap R A) (Q x))
  -/
  change ∀ x y : M, f x * f y + f y * f x = algebraMap R A (QuadraticMap.polar Q x y) at h
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_4
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    h2 : IsUnit 2
    f : LinearMap (RingHom.id R) M A
    x : M
    h : ∀ (x y : M), Eq (HAdd.hAdd (HMul.hMul (f x) (f y)) (HMul.hMul (f y) (f x)) …
    ⊢ Eq (HMul.hMul (f x) (f x)) ((algebraMap R A) (Q x))
  -/
  apply h2.mul_left_cancel
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_4
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    h2 : IsUnit 2
    f : LinearMap (RingHom.id R) M A
    x : M
    h : ∀ (x y : M), Eq (HAdd.hAdd (HMul.hMul (f x) (f y)) (HMul.hMul (f y) (f x)) …
    ⊢ Eq (HMul.hMul 2 (HMul.hMul (f x) (f x))) (HMul.hMul 2 ((algebraMap R A) (Q x …
  -/
  rw [two_mul, two_mul, h x x, QuadraticMap.polar_self, two_smul, map_add]
  /-
    🎉 no goals
  -/


/-- The symmetric product of vectors is a scalar -/
theorem ι_mul_ι_add_swap (a b : M) :
    ι Q a * ι Q b + ι Q b * ι Q a = algebraMap R _ (QuadraticMap.polar Q a b) :=
  mul_add_swap_eq_polar_of_forall_mul_self_eq _ (ι_sq_scalar _) _ _


theorem ι_mul_ι_comm (a b : M) :
    ι Q a * ι Q b = algebraMap R _ (QuadraticMap.polar Q a b) - ι Q b * ι Q a :=
  eq_sub_of_add_eq (ι_mul_ι_add_swap a b)


@[simp] theorem ι_mul_ι_add_swap_of_isOrtho {a b : M} (h : Q.IsOrtho a b) :
    ι Q a * ι Q b + ι Q b * ι Q a = 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    a b : M
    h : QuadraticMap.IsOrtho Q a b
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((CliffordAlgebra.ι Q) a) ((CliffordAlgebra.ι Q) b) …
  -/
  rw [ι_mul_ι_add_swap, h.polar_eq_zero]
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    a b : M
    h : QuadraticMap.IsOrtho Q a b
    ⊢ Eq ((algebraMap R (CliffordAlgebra Q)) 0) 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem ι_mul_ι_comm_of_isOrtho {a b : M} (h : Q.IsOrtho a b) :
    ι Q a * ι Q b = -(ι Q b * ι Q a) :=
  eq_neg_of_add_eq_zero_left <| ι_mul_ι_add_swap_of_isOrtho h


theorem mul_ι_mul_ι_of_isOrtho (x : CliffordAlgebra Q) {a b : M} (h : Q.IsOrtho a b) :
    x * ι Q a * ι Q b = -(x * ι Q b * ι Q a) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    a b : M
    h : QuadraticMap.IsOrtho Q a b
    ⊢ Eq (HMul.hMul (HMul.hMul x ((CliffordAlgebra.ι Q) a)) ((CliffordAlgebra.ι Q) …
  -/
  rw [mul_assoc, ι_mul_ι_comm_of_isOrtho h, mul_neg, mul_assoc]
  /-
    🎉 no goals
  -/


theorem ι_mul_ι_mul_of_isOrtho (x : CliffordAlgebra Q) {a b : M} (h : Q.IsOrtho a b) :
    ι Q a * (ι Q b * x) = -(ι Q b * (ι Q a * x)) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    a b : M
    h : QuadraticMap.IsOrtho Q a b
    ⊢ Eq (HMul.hMul ((CliffordAlgebra.ι Q) a) (HMul.hMul ((CliffordAlgebra.ι Q) b) …
  -/
  rw [← mul_assoc, ι_mul_ι_comm_of_isOrtho h, neg_mul, mul_assoc]
  /-
    🎉 no goals
  -/


/-- $aba$ is a vector. -/
theorem ι_mul_ι_mul_ι (a b : M) :
    ι Q a * ι Q b * ι Q a = ι Q (QuadraticMap.polar Q a b • a - Q a • b) := by
  rw [ι_mul_ι_comm, sub_mul, mul_assoc, ι_sq_scalar, ← Algebra.smul_def, ← Algebra.commutes, ←
    Algebra.smul_def, ← map_smul, ← map_smul, ← map_sub]


@[simp]
theorem ι_range_map_lift (f : M →ₗ[R] A) (cond : ∀ m, f m * f m = algebraMap _ _ (Q m)) :
    (ι Q).range.map (lift Q ⟨f, cond⟩).toLinearMap = LinearMap.range f := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M A
    cond : ∀ (m : M), Eq (HMul.hMul (f m) (f m)) ((algebraMap R A) (Q m))
    ⊢ Eq (Submodule.map ((CliffordAlgebra.lift Q) ⟨f, cond⟩).toLinearMap (LinearMa …
  -/
  rw [← LinearMap.range_comp, ι_comp_lift]
  /-
    🎉 no goals
  -/


/-- Any linear map that preserves the quadratic form lifts to an `AlgHom` between algebras.

See `CliffordAlgebra.equivOfIsometry` for the case when `f` is a `QuadraticForm.IsometryEquiv`. -/
def map (f : Q₁ →qᵢ Q₂) :
    CliffordAlgebra Q₁ →ₐ[R] CliffordAlgebra Q₂ :=
  CliffordAlgebra.lift Q₁
    ⟨ι Q₂ ∘ₗ f.toLinearMap, fun m => (ι_sq_scalar _ _).trans <| RingHom.congr_arg _ <| f.map_app m⟩


@[simp]
theorem map_comp_ι (f : Q₁ →qᵢ Q₂) :
    (map f).toLinearMap ∘ₗ ι Q₁ = ι Q₂ ∘ₗ f.toLinearMap :=
  ι_comp_lift _ _


@[simp]
theorem map_apply_ι (f : Q₁ →qᵢ Q₂) (m : M₁) : map f (ι Q₁ m) = ι Q₂ (f m) :=
  lift_ι_apply _ _ m


variable (Q₁) in
@[simp]
theorem map_id : map (QuadraticMap.Isometry.id Q₁) = AlgHom.id R (CliffordAlgebra Q₁) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M₁ : Type u_4
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    Q₁ : QuadraticForm R M₁
    ⊢ Eq (CliffordAlgebra.map (QuadraticMap.Isometry.id Q₁)) (AlgHom.id R (Cliffor …
  -/
  ext m; exact map_apply_ι _ m
         /-
           🎉 no goals
         -/


@[simp]
theorem map_comp_map (f : Q₂ →qᵢ Q₃) (g : Q₁ →qᵢ Q₂) :
    (map f).comp (map g) = map (f.comp g) := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁵ : AddCommGroup M₁
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    f : QuadraticMap.Isometry Q₂ Q₃
    g : QuadraticMap.Isometry Q₁ Q₂
    ⊢ Eq ((CliffordAlgebra.map f).comp (CliffordAlgebra.map g)) (CliffordAlgebra.m …
  -/
  ext m
  /-
    case a.h
    R : Type u_1
    inst✝⁶ : CommRing R
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁵ : AddCommGroup M₁
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    f : QuadraticMap.Isometry Q₂ Q₃
    g : QuadraticMap.Isometry Q₁ Q₂
    m : M₁
    ⊢ Eq ((((CliffordAlgebra.map f).comp (CliffordAlgebra.map g)).toLinearMap.comp …
  -/
  dsimp only [LinearMap.comp_apply, AlgHom.comp_apply, AlgHom.toLinearMap_apply, AlgHom.id_apply]
  /-
    case a.h
    R : Type u_1
    inst✝⁶ : CommRing R
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁵ : AddCommGroup M₁
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    f : QuadraticMap.Isometry Q₂ Q₃
    g : QuadraticMap.Isometry Q₁ Q₂
    m : M₁
    ⊢ Eq ((CliffordAlgebra.map f) ((CliffordAlgebra.map g) ((CliffordAlgebra.ι Q₁) …
  -/
  rw [map_apply_ι, map_apply_ι, map_apply_ι, QuadraticMap.Isometry.comp_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem ι_range_map_map (f : Q₁ →qᵢ Q₂) :
    (ι Q₁).range.map (map f).toLinearMap = f.range.map (ι Q₂) :=
  (ι_range_map_lift _ _).trans (LinearMap.range_comp _ _)


open Function in
/-- If `f` is a linear map from `M₁` to `M₂` that preserves the quadratic forms, and if it has
a linear retraction `g` that also preserves the quadratic forms, then `CliffordAlgebra.map g`
is a retraction of `CliffordAlgebra.map f`. -/
lemma leftInverse_map_of_leftInverse {Q₁ : QuadraticForm R M₁} {Q₂ : QuadraticForm R M₂}
    (f : Q₁ →qᵢ Q₂) (g : Q₂ →qᵢ Q₁) (h : LeftInverse g f) : LeftInverse (map g) (map f) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M₁ : Type u_4
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    f : QuadraticMap.Isometry Q₁ Q₂
    g : QuadraticMap.Isometry Q₂ Q₁
    h : Function.LeftInverse ⇑g ⇑f
    ⊢ Function.LeftInverse ⇑(CliffordAlgebra.map g) ⇑(CliffordAlgebra.map f)
  -/
  refine fun x => ?_
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M₁ : Type u_4
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    f : QuadraticMap.Isometry Q₁ Q₂
    g : QuadraticMap.Isometry Q₂ Q₁
    h : Function.LeftInverse ⇑g ⇑f
    x : CliffordAlgebra Q₁
    ⊢ Eq ((CliffordAlgebra.map g) ((CliffordAlgebra.map f) x)) x
  -/
  replace h : g.comp f = QuadraticMap.Isometry.id Q₁ := DFunLike.ext _ _ h
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M₁ : Type u_4
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    f : QuadraticMap.Isometry Q₁ Q₂
    g : QuadraticMap.Isometry Q₂ Q₁
    x : CliffordAlgebra Q₁
    h : Eq (g.comp f) (QuadraticMap.Isometry.id Q₁)
    ⊢ Eq ((CliffordAlgebra.map g) ((CliffordAlgebra.map f) x)) x
  -/
  rw [← AlgHom.comp_apply, map_comp_map, h, map_id, AlgHom.coe_id, id_eq]
  /-
    🎉 no goals
  -/


/-- If a linear map preserves the quadratic forms and is surjective, then the algebra
maps it induces between Clifford algebras is also surjective. -/
lemma map_surjective {Q₁ : QuadraticForm R M₁} {Q₂ : QuadraticForm R M₂} (f : Q₁ →qᵢ Q₂)
    (hf : Function.Surjective f) : Function.Surjective (CliffordAlgebra.map f) :=
  CliffordAlgebra.induction
                                                      /-
                                                        R : Type u_1
                                                        inst✝⁴ : CommRing R
                                                        M₁ : Type u_4
                                                        M₂ : Type u_5
                                                        inst✝³ : AddCommGroup M₁
                                                        inst✝² : AddCommGroup M₂
                                                        inst✝¹ : Module R M₁
                                                        inst✝ : Module R M₂
                                                        Q₁ : QuadraticForm R M₁
                                                        Q₂ : QuadraticForm R M₂
                                                        f : QuadraticMap.Isometry Q₁ Q₂
                                                        hf : Function.Surjective ⇑f
                                                        r : R
                                                        ⊢ Eq ((CliffordAlgebra.map f) ((algebraMap R (CliffordAlgebra Q₁)) r)) ((algeb …
                                                      -/
    (fun r ↦ ⟨algebraMap R (CliffordAlgebra Q₁) r, by simp only [AlgHom.commutes]⟩)
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                              /-
                                                                R : Type u_1
                                                                inst✝⁴ : CommRing R
                                                                M₁ : Type u_4
                                                                M₂ : Type u_5
                                                                inst✝³ : AddCommGroup M₁
                                                                inst✝² : AddCommGroup M₂
                                                                inst✝¹ : Module R M₁
                                                                inst✝ : Module R M₂
                                                                Q₁ : QuadraticForm R M₁
                                                                Q₂ : QuadraticForm R M₂
                                                                f : QuadraticMap.Isometry Q₁ Q₂
                                                                hf : Function.Surjective ⇑f
                                                                y : M₂
                                                                x : M₁
                                                                hx : Eq (f x) y
                                                                ⊢ Eq ((CliffordAlgebra.map f) ((CliffordAlgebra.ι Q₁) x)) ((CliffordAlgebra.ι  …
                                                              -/
    (fun y ↦ let ⟨x, hx⟩ := hf y; ⟨CliffordAlgebra.ι Q₁ x, by simp only [map_apply_ι, hx]⟩)
                                                              /-
                                                                🎉 no goals
                                                              -/
                                          /-
                                            R : Type u_1
                                            inst✝⁴ : CommRing R
                                            M₁ : Type u_4
                                            M₂ : Type u_5
                                            inst✝³ : AddCommGroup M₁
                                            inst✝² : AddCommGroup M₂
                                            inst✝¹ : Module R M₁
                                            inst✝ : Module R M₂
                                            Q₁ : QuadraticForm R M₁
                                            Q₂ : QuadraticForm R M₂
                                            f : QuadraticMap.Isometry Q₁ Q₂
                                            hf : Function.Surjective ⇑f
                                            x✝³ x✝² : CliffordAlgebra Q₂
                                            x✝¹ : Exists fun a => Eq ((CliffordAlgebra.map f) a) x✝³
                                            x✝ : Exists fun a => Eq ((CliffordAlgebra.map f) a) x✝²
                                            x : CliffordAlgebra Q₁
                                            hx : Eq ((CliffordAlgebra.map f) x) x✝³
                                            y : CliffordAlgebra Q₁
                                            hy : Eq ((CliffordAlgebra.map f) y) x✝²
                                            ⊢ Eq ((CliffordAlgebra.map f) (HMul.hMul x y)) (HMul.hMul x✝³ x✝²)
                                          -/
    (fun _ _ ⟨x, hx⟩ ⟨y, hy⟩ ↦ ⟨x * y, by simp only [map_mul, hx, hy]⟩)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            R : Type u_1
                                            inst✝⁴ : CommRing R
                                            M₁ : Type u_4
                                            M₂ : Type u_5
                                            inst✝³ : AddCommGroup M₁
                                            inst✝² : AddCommGroup M₂
                                            inst✝¹ : Module R M₁
                                            inst✝ : Module R M₂
                                            Q₁ : QuadraticForm R M₁
                                            Q₂ : QuadraticForm R M₂
                                            f : QuadraticMap.Isometry Q₁ Q₂
                                            hf : Function.Surjective ⇑f
                                            x✝³ x✝² : CliffordAlgebra Q₂
                                            x✝¹ : Exists fun a => Eq ((CliffordAlgebra.map f) a) x✝³
                                            x✝ : Exists fun a => Eq ((CliffordAlgebra.map f) a) x✝²
                                            x : CliffordAlgebra Q₁
                                            hx : Eq ((CliffordAlgebra.map f) x) x✝³
                                            y : CliffordAlgebra Q₁
                                            hy : Eq ((CliffordAlgebra.map f) y) x✝²
                                            ⊢ Eq ((CliffordAlgebra.map f) (HAdd.hAdd x y)) (HAdd.hAdd x✝³ x✝²)
                                          -/
    (fun _ _ ⟨x, hx⟩ ⟨y, hy⟩ ↦ ⟨x + y, by simp only [map_add, hx, hy]⟩)
                                          /-
                                            🎉 no goals
                                          -/


/-- Two `CliffordAlgebra`s are equivalent as algebras if their quadratic forms are
equivalent. -/
@[simps! apply]
def equivOfIsometry (e : Q₁.IsometryEquiv Q₂) : CliffordAlgebra Q₁ ≃ₐ[R] CliffordAlgebra Q₂ :=
  AlgEquiv.ofAlgHom (map e.toIsometry) (map e.symm.toIsometry)
    ((map_comp_map _ _).trans <| by
      /-
        R : Type u_1
        inst✝¹⁰ : CommRing R
        M : Type u_2
        inst✝⁹ : AddCommGroup M
        inst✝⁸ : Module R M
        Q : QuadraticForm R M
        A : Type u_3
        inst✝⁷ : Semiring A
        inst✝⁶ : Algebra R A
        M₁ : Type u_4
        M₂ : Type u_5
        M₃ : Type u_6
        inst✝⁵ : AddCommGroup M₁
        inst✝⁴ : AddCommGroup M₂
        inst✝³ : AddCommGroup M₃
        inst✝² : Module R M₁
        inst✝¹ : Module R M₂
        inst✝ : Module R M₃
        Q₁ : QuadraticForm R M₁
        Q₂ : QuadraticForm R M₂
        Q₃ : QuadraticForm R M₃
        e : QuadraticMap.IsometryEquiv Q₁ Q₂
        ⊢ Eq (CliffordAlgebra.map (e.toIsometry.comp e.symm.toIsometry)) (AlgHom.id R  …
      -/
      convert map_id Q₂ using 2  -- Porting note: replaced `_` with `Q₂`
      /-
        case h.e'_2.h.e'_11
        R : Type u_1
        inst✝¹⁰ : CommRing R
        M : Type u_2
        inst✝⁹ : AddCommGroup M
        inst✝⁸ : Module R M
        Q : QuadraticForm R M
        A : Type u_3
        inst✝⁷ : Semiring A
        inst✝⁶ : Algebra R A
        M₁ : Type u_4
        M₂ : Type u_5
        M₃ : Type u_6
        inst✝⁵ : AddCommGroup M₁
        inst✝⁴ : AddCommGroup M₂
        inst✝³ : AddCommGroup M₃
        inst✝² : Module R M₁
        inst✝¹ : Module R M₂
        inst✝ : Module R M₃
        Q₁ : QuadraticForm R M₁
        Q₂ : QuadraticForm R M₂
        Q₃ : QuadraticForm R M₃
        e : QuadraticMap.IsometryEquiv Q₁ Q₂
        ⊢ Eq (e.toIsometry.comp e.symm.toIsometry) (QuadraticMap.Isometry.id Q₂)
      -/
      ext m
      /-
        case h.e'_2.h.e'_11.h
        R : Type u_1
        inst✝¹⁰ : CommRing R
        M : Type u_2
        inst✝⁹ : AddCommGroup M
        inst✝⁸ : Module R M
        Q : QuadraticForm R M
        A : Type u_3
        inst✝⁷ : Semiring A
        inst✝⁶ : Algebra R A
        M₁ : Type u_4
        M₂ : Type u_5
        M₃ : Type u_6
        inst✝⁵ : AddCommGroup M₁
        inst✝⁴ : AddCommGroup M₂
        inst✝³ : AddCommGroup M₃
        inst✝² : Module R M₁
        inst✝¹ : Module R M₂
        inst✝ : Module R M₃
        Q₁ : QuadraticForm R M₁
        Q₂ : QuadraticForm R M₂
        Q₃ : QuadraticForm R M₃
        e : QuadraticMap.IsometryEquiv Q₁ Q₂
        m : M₂
        ⊢ Eq ((e.toIsometry.comp e.symm.toIsometry) m) ((QuadraticMap.Isometry.id Q₂) m)
      -/
      exact e.toLinearEquiv.apply_symm_apply m)
      /-
        🎉 no goals
      -/
    ((map_comp_map _ _).trans <| by
      /-
        R : Type u_1
        inst✝¹⁰ : CommRing R
        M : Type u_2
        inst✝⁹ : AddCommGroup M
        inst✝⁸ : Module R M
        Q : QuadraticForm R M
        A : Type u_3
        inst✝⁷ : Semiring A
        inst✝⁶ : Algebra R A
        M₁ : Type u_4
        M₂ : Type u_5
        M₃ : Type u_6
        inst✝⁵ : AddCommGroup M₁
        inst✝⁴ : AddCommGroup M₂
        inst✝³ : AddCommGroup M₃
        inst✝² : Module R M₁
        inst✝¹ : Module R M₂
        inst✝ : Module R M₃
        Q₁ : QuadraticForm R M₁
        Q₂ : QuadraticForm R M₂
        Q₃ : QuadraticForm R M₃
        e : QuadraticMap.IsometryEquiv Q₁ Q₂
        ⊢ Eq (CliffordAlgebra.map (e.symm.toIsometry.comp e.toIsometry)) (AlgHom.id R  …
      -/
      convert map_id Q₁ using 2  -- Porting note: replaced `_` with `Q₁`
      /-
        case h.e'_2.h.e'_11
        R : Type u_1
        inst✝¹⁰ : CommRing R
        M : Type u_2
        inst✝⁹ : AddCommGroup M
        inst✝⁸ : Module R M
        Q : QuadraticForm R M
        A : Type u_3
        inst✝⁷ : Semiring A
        inst✝⁶ : Algebra R A
        M₁ : Type u_4
        M₂ : Type u_5
        M₃ : Type u_6
        inst✝⁵ : AddCommGroup M₁
        inst✝⁴ : AddCommGroup M₂
        inst✝³ : AddCommGroup M₃
        inst✝² : Module R M₁
        inst✝¹ : Module R M₂
        inst✝ : Module R M₃
        Q₁ : QuadraticForm R M₁
        Q₂ : QuadraticForm R M₂
        Q₃ : QuadraticForm R M₃
        e : QuadraticMap.IsometryEquiv Q₁ Q₂
        ⊢ Eq (e.symm.toIsometry.comp e.toIsometry) (QuadraticMap.Isometry.id Q₁)
      -/
      ext m
      /-
        case h.e'_2.h.e'_11.h
        R : Type u_1
        inst✝¹⁰ : CommRing R
        M : Type u_2
        inst✝⁹ : AddCommGroup M
        inst✝⁸ : Module R M
        Q : QuadraticForm R M
        A : Type u_3
        inst✝⁷ : Semiring A
        inst✝⁶ : Algebra R A
        M₁ : Type u_4
        M₂ : Type u_5
        M₃ : Type u_6
        inst✝⁵ : AddCommGroup M₁
        inst✝⁴ : AddCommGroup M₂
        inst✝³ : AddCommGroup M₃
        inst✝² : Module R M₁
        inst✝¹ : Module R M₂
        inst✝ : Module R M₃
        Q₁ : QuadraticForm R M₁
        Q₂ : QuadraticForm R M₂
        Q₃ : QuadraticForm R M₃
        e : QuadraticMap.IsometryEquiv Q₁ Q₂
        m : M₁
        ⊢ Eq ((e.symm.toIsometry.comp e.toIsometry) m) ((QuadraticMap.Isometry.id Q₁) m)
      -/
      exact e.toLinearEquiv.symm_apply_apply m)
      /-
        🎉 no goals
      -/


@[simp]
theorem equivOfIsometry_symm (e : Q₁.IsometryEquiv Q₂) :
    (equivOfIsometry e).symm = equivOfIsometry e.symm :=
  rfl


@[simp]
theorem equivOfIsometry_trans (e₁₂ : Q₁.IsometryEquiv Q₂) (e₂₃ : Q₂.IsometryEquiv Q₃) :
    (equivOfIsometry e₁₂).trans (equivOfIsometry e₂₃) = equivOfIsometry (e₁₂.trans e₂₃) := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁵ : AddCommGroup M₁
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    e₁₂ : QuadraticMap.IsometryEquiv Q₁ Q₂
    e₂₃ : QuadraticMap.IsometryEquiv Q₂ Q₃
    ⊢ Eq ((CliffordAlgebra.equivOfIsometry e₁₂).trans (CliffordAlgebra.equivOfIsom …
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝⁶ : CommRing R
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁵ : AddCommGroup M₁
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    e₁₂ : QuadraticMap.IsometryEquiv Q₁ Q₂
    e₂₃ : QuadraticMap.IsometryEquiv Q₂ Q₃
    x : CliffordAlgebra Q₁
    ⊢ Eq (((CliffordAlgebra.equivOfIsometry e₁₂).trans (CliffordAlgebra.equivOfIso …
  -/
  exact AlgHom.congr_fun (map_comp_map _ _) x
  /-
    🎉 no goals
  -/


@[simp]
theorem equivOfIsometry_refl :
    (equivOfIsometry <| QuadraticMap.IsometryEquiv.refl Q₁) = AlgEquiv.refl := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M₁ : Type u_4
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    Q₁ : QuadraticForm R M₁
    ⊢ Eq (CliffordAlgebra.equivOfIsometry (QuadraticMap.IsometryEquiv.refl Q₁)) Al …
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    M₁ : Type u_4
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    Q₁ : QuadraticForm R M₁
    x : CliffordAlgebra Q₁
    ⊢ Eq ((CliffordAlgebra.equivOfIsometry (QuadraticMap.IsometryEquiv.refl Q₁)) x …
  -/
  exact AlgHom.congr_fun (map_id Q₁) x
  /-
    🎉 no goals
  -/


/-- The canonical image of the `TensorAlgebra` in the `CliffordAlgebra`, which maps
`TensorAlgebra.ι R x` to `CliffordAlgebra.ι Q x`. -/
def toClifford : TensorAlgebra R M →ₐ[R] CliffordAlgebra Q :=
  TensorAlgebra.lift R (CliffordAlgebra.ι Q)


@[simp]
theorem toClifford_ι (m : M) : toClifford (TensorAlgebra.ι R m) = CliffordAlgebra.ι Q m := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    m : M
    ⊢ Eq (TensorAlgebra.toClifford ((TensorAlgebra.ι R) m)) ((CliffordAlgebra.ι Q) …
  -/
  simp [toClifford]
  /-
    🎉 no goals
  -/


