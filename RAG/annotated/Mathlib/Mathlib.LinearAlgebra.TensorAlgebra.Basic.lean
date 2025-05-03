/-- An inductively defined relation on `Pre R M` used to force the initial algebra structure on
the associated quotient.
-/
inductive Rel : FreeAlgebra R M → FreeAlgebra R M → Prop
  -- force `ι` to be linear
  | add {a b : M} : Rel (FreeAlgebra.ι R (a + b)) (FreeAlgebra.ι R a + FreeAlgebra.ι R b)
  | smul {r : R} {a : M} :
    Rel (FreeAlgebra.ι R (r • a)) (algebraMap R (FreeAlgebra R M) r * FreeAlgebra.ι R a)


/-- The tensor algebra of the module `M` over the commutative semiring `R`.
-/
def TensorAlgebra :=
  RingQuot (TensorAlgebra.Rel R M)

-- Porting note: Expanded `deriving Inhabited, Semiring, Algebra`

instance : Inhabited (TensorAlgebra R M) := RingQuot.instInhabited _

instance : Semiring (TensorAlgebra R M) := RingQuot.instSemiring _

-- `IsScalarTower` is not needed, but the instance isn't really canonical without it.

@[nolint unusedArguments]
instance instAlgebra {R A M} [CommSemiring R] [AddCommMonoid M] [CommSemiring A]
    [Algebra R A] [Module R M] [Module A M]
    [IsScalarTower R A M] :
    Algebra R (TensorAlgebra A M) :=
  RingQuot.instAlgebra _

-- verify there is no diamond
-- but doesn't work at `reducible_and_instances` https://github.com/leanprover-community/mathlib4/issues/10906

instance {R S A M} [CommSemiring R] [CommSemiring S] [AddCommMonoid M] [CommSemiring A]
    [Algebra R A] [Algebra S A] [Module R M] [Module S M] [Module A M]
    [IsScalarTower R A M] [IsScalarTower S A M] :
    SMulCommClass R S (TensorAlgebra A M) :=
  RingQuot.instSMulCommClass _


instance {R S A M} [CommSemiring R] [CommSemiring S] [AddCommMonoid M] [CommSemiring A]
    [SMul R S] [Algebra R A] [Algebra S A] [Module R M] [Module S M] [Module A M]
    [IsScalarTower R A M] [IsScalarTower S A M] [IsScalarTower R S A] :
    IsScalarTower R S (TensorAlgebra A M) :=
  RingQuot.instIsScalarTower _


instance {S : Type*} [CommRing S] [Module S M] : Ring (TensorAlgebra S M) :=
  RingQuot.instRing (Rel S M)

-- verify there is no diamond
-- but doesn't work at `reducible_and_instances` https://github.com/leanprover-community/mathlib4/issues/10906

/-- The canonical linear map `M →ₗ[R] TensorAlgebra R M`.
-/
irreducible_def ι : M →ₗ[R] TensorAlgebra R M :=
  { toFun := fun m => RingQuot.mkAlgHom R _ (FreeAlgebra.ι R m)
    map_add' := fun x y => by
      /-
        R : Type u_1
        inst✝² : CommSemiring R
        M : Type u_2
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        x y : M
        ⊢ Eq ((fun m => (RingQuot.mkAlgHom R (TensorAlgebra.Rel R M)) (FreeAlgebra.ι R …
      -/
      rw [← map_add (RingQuot.mkAlgHom R (Rel R M))]
      /-
        R : Type u_1
        inst✝² : CommSemiring R
        M : Type u_2
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        x y : M
        ⊢ Eq ((fun m => (RingQuot.mkAlgHom R (TensorAlgebra.Rel R M)) (FreeAlgebra.ι R …
      -/
      exact RingQuot.mkAlgHom_rel R Rel.add
      /-
        🎉 no goals
      -/
    map_smul' := fun r x => by
      /-
        R : Type u_1
        inst✝² : CommSemiring R
        M : Type u_2
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        r : R
        x : M
        ⊢ Eq ({ toFun := fun m => (RingQuot.mkAlgHom R (TensorAlgebra.Rel R M)) (FreeA …
      -/
      rw [← map_smul (RingQuot.mkAlgHom R (Rel R M))]
      /-
        R : Type u_1
        inst✝² : CommSemiring R
        M : Type u_2
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        r : R
        x : M
        ⊢ Eq ({ toFun := fun m => (RingQuot.mkAlgHom R (TensorAlgebra.Rel R M)) (FreeA …
      -/
      exact RingQuot.mkAlgHom_rel R Rel.smul }
      /-
        🎉 no goals
      -/


theorem ringQuot_mkAlgHom_freeAlgebra_ι_eq_ι (m : M) :
    RingQuot.mkAlgHom R (Rel R M) (FreeAlgebra.ι R m) = ι R m := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : M
    ⊢ Eq ((RingQuot.mkAlgHom R (TensorAlgebra.Rel R M)) (FreeAlgebra.ι R m)) ((Ten …
  -/
  rw [ι]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : M
    ⊢ Eq ((RingQuot.mkAlgHom R (TensorAlgebra.Rel R M)) (FreeAlgebra.ι R m)) ({ to …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note: Changed `irreducible_def` to `def` to get `@[simps symm_apply]` to work

/-- Given a linear map `f : M → A` where `A` is an `R`-algebra, `lift R f` is the unique lift
of `f` to a morphism of `R`-algebras `TensorAlgebra R M → A`.
-/
@[simps symm_apply]
def lift {A : Type*} [Semiring A] [Algebra R A] : (M →ₗ[R] A) ≃ (TensorAlgebra R M →ₐ[R] A) :=
  { toFun :=
      RingQuot.liftAlgHom R ∘ fun f =>
        ⟨FreeAlgebra.lift R (⇑f), fun x y (h : Rel R M x y) => by
          /-
            R : Type u_1
            inst✝⁴ : CommSemiring R
            M : Type u_2
            inst✝³ : AddCommMonoid M
            inst✝² : Module R M
            A : Type u_3
            inst✝¹ : Semiring A
            inst✝ : Algebra R A
            f : LinearMap (RingHom.id R) M A
            x y : FreeAlgebra R M
            h : TensorAlgebra.Rel R M x y
            ⊢ Eq (((FreeAlgebra.lift R) ⇑f) x) (((FreeAlgebra.lift R) ⇑f) y)
          -/
          induction h <;>
            simp only [Algebra.smul_def, FreeAlgebra.lift_ι_apply, LinearMap.map_smulₛₗ,
              RingHom.id_apply, map_mul, AlgHom.commutes, map_add]⟩
    invFun := fun F => F.toLinearMap.comp (ι R)
    left_inv := fun f => by
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Type u_2
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : LinearMap (RingHom.id R) M A
        ⊢ Eq ((fun F => F.toLinearMap.comp (TensorAlgebra.ι R)) (Function.comp (⇑(Ring …
      -/
      rw [ι]
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Type u_2
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : LinearMap (RingHom.id R) M A
        ⊢ Eq ((fun F => F.toLinearMap.comp { toFun := fun m => (RingQuot.mkAlgHom R (T …
      -/
      ext1 x
      /-
        case h
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Type u_2
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : LinearMap (RingHom.id R) M A
        x : M
        ⊢ Eq (((fun F => F.toLinearMap.comp { toFun := fun m => (RingQuot.mkAlgHom R ( …
      -/
      exact (RingQuot.liftAlgHom_mkAlgHom_apply _ _ _ _).trans (FreeAlgebra.lift_ι_apply f x)
      /-
        🎉 no goals
      -/
    right_inv := fun F =>
      RingQuot.ringQuot_ext' _ _ _ <|
        FreeAlgebra.hom_ext <|
          funext fun x => by
            /-
              R : Type u_1
              inst✝⁴ : CommSemiring R
              M : Type u_2
              inst✝³ : AddCommMonoid M
              inst✝² : Module R M
              A : Type u_3
              inst✝¹ : Semiring A
              inst✝ : Algebra R A
              F : AlgHom R (TensorAlgebra R M) A
              x : M
              ⊢ Eq (Function.comp (⇑((Function.comp (⇑(RingQuot.liftAlgHom R)) (fun f => ⟨(F …
            -/
            rw [ι]
            exact
              (RingQuot.liftAlgHom_mkAlgHom_apply _ _ _ _).trans (FreeAlgebra.lift_ι_apply _ _) }


@[simp]
theorem ι_comp_lift {A : Type*} [Semiring A] [Algebra R A] (f : M →ₗ[R] A) :
    (lift R f).toLinearMap.comp (ι R) = f := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M A
    ⊢ Eq (((TensorAlgebra.lift R) f).toLinearMap.comp (TensorAlgebra.ι R)) f
  -/
  convert (lift R).symm_apply_apply f
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_ι_apply {A : Type*} [Semiring A] [Algebra R A] (f : M →ₗ[R] A) (x) :
    lift R f (ι R x) = f x := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M A
    x : M
    ⊢ Eq (((TensorAlgebra.lift R) f) ((TensorAlgebra.ι R) x)) (f x)
  -/
  conv_rhs => rw [← ι_comp_lift f]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M A
    x : M
    ⊢ Eq (((TensorAlgebra.lift R) f) ((TensorAlgebra.ι R) x)) ((((TensorAlgebra.li …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_unique {A : Type*} [Semiring A] [Algebra R A] (f : M →ₗ[R] A)
    (g : TensorAlgebra R M →ₐ[R] A) : g.toLinearMap.comp (ι R) = f ↔ g = lift R f := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M A
    g : AlgHom R (TensorAlgebra R M) A
    ⊢ Iff (Eq (g.toLinearMap.comp (TensorAlgebra.ι R)) f) (Eq g ((TensorAlgebra.li …
  -/
  rw [← (lift R).symm_apply_eq]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M A
    g : AlgHom R (TensorAlgebra R M) A
    ⊢ Iff (Eq (g.toLinearMap.comp (TensorAlgebra.ι R)) f) (Eq ((TensorAlgebra.lift …
  -/
  simp only [lift, Equiv.coe_fn_symm_mk]
  /-
    🎉 no goals
  -/

-- Marking `TensorAlgebra` irreducible makes `Ring` instances inaccessible on quotients.
-- https://leanprover.zulipchat.com/#narrow/stream/113488-general/topic/algebra.2Esemiring_to_ring.20breaks.20semimodule.20typeclass.20lookup/near/212580241
-- For now, we avoid this by not marking it irreducible.

@[simp]
theorem lift_comp_ι {A : Type*} [Semiring A] [Algebra R A] (g : TensorAlgebra R M →ₐ[R] A) :
    lift R (g.toLinearMap.comp (ι R)) = g := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    g : AlgHom R (TensorAlgebra R M) A
    ⊢ Eq ((TensorAlgebra.lift R) (g.toLinearMap.comp (TensorAlgebra.ι R))) g
  -/
  rw [← lift_symm_apply]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    g : AlgHom R (TensorAlgebra R M) A
    ⊢ Eq ((TensorAlgebra.lift R) ((TensorAlgebra.lift R).symm g)) g
  -/
  exact (lift R).apply_symm_apply g
  /-
    🎉 no goals
  -/


/-- See note [partially-applied ext lemmas]. -/
@[ext]
theorem hom_ext {A : Type*} [Semiring A] [Algebra R A] {f g : TensorAlgebra R M →ₐ[R] A}
    (w : f.toLinearMap.comp (ι R) = g.toLinearMap.comp (ι R)) : f = g := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R (TensorAlgebra R M) A
    w : Eq (f.toLinearMap.comp (TensorAlgebra.ι R)) (g.toLinearMap.comp (TensorAlg …
    ⊢ Eq f g
  -/
  rw [← lift_symm_apply, ← lift_symm_apply] at w
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R (TensorAlgebra R M) A
    w : Eq ((TensorAlgebra.lift R).symm f) ((TensorAlgebra.lift R).symm g)
    ⊢ Eq f g
  -/
  exact (lift R).symm.injective w
  /-
    🎉 no goals
  -/

-- This proof closely follows `FreeAlgebra.induction`

/-- If `C` holds for the `algebraMap` of `r : R` into `TensorAlgebra R M`, the `ι` of `x : M`,
and is preserved under addition and multiplication, then it holds for all of `TensorAlgebra R M`.
-/
@[elab_as_elim]
theorem induction {C : TensorAlgebra R M → Prop}
    (algebraMap : ∀ r, C (algebraMap R (TensorAlgebra R M) r)) (ι : ∀ x, C (ι R x))
    (mul : ∀ a b, C a → C b → C (a * b)) (add : ∀ a b, C a → C b → C (a + b))
    (a : TensorAlgebra R M) : C a := by
  -- the arguments are enough to construct a subalgebra, and a mapping into it from M
  let s : Subalgebra R (TensorAlgebra R M) :=
    { carrier := C
      mul_mem' := @mul
      add_mem' := @add
      algebraMap_mem' := algebraMap }
  -- Porting note: Added `h`. `h` is needed for `of`.
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    C : TensorAlgebra R M → Prop
    algebraMap : ∀ (r : R), C ((_root_.algebraMap R (TensorAlgebra R M)) r)
    ι : ∀ (x : M), C ((TensorAlgebra.ι R) x)
    mul : ∀ (a b : TensorAlgebra R M), C a → C b → C (HMul.hMul a b)
    add : ∀ (a b : TensorAlgebra R M), C a → C b → C (HAdd.hAdd a b)
    a : TensorAlgebra R M
    s : Subalgebra R (TensorAlgebra R M) := { carrier := C, mul_mem' := mul, one_m …
    ⊢ C a
  -/
  let h : AddCommMonoid s := inferInstanceAs (AddCommMonoid (Subalgebra.toSubmodule s))
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    C : TensorAlgebra R M → Prop
    algebraMap : ∀ (r : R), C ((_root_.algebraMap R (TensorAlgebra R M)) r)
    ι : ∀ (x : M), C ((TensorAlgebra.ι R) x)
    mul : ∀ (a b : TensorAlgebra R M), C a → C b → C (HMul.hMul a b)
    add : ∀ (a b : TensorAlgebra R M), C a → C b → C (HAdd.hAdd a b)
    a : TensorAlgebra R M
    s : Subalgebra R (TensorAlgebra R M) := { carrier := C, mul_mem' := mul, one_m …
    h : AddCommMonoid (Subtype fun x => Membership.mem s x) := inferInstanceAs (Ad …
    ⊢ C a
  -/
  let of : M →ₗ[R] s := (TensorAlgebra.ι R).codRestrict (Subalgebra.toSubmodule s) ι
  -- the mapping through the subalgebra is the identity
  have of_id : AlgHom.id R (TensorAlgebra R M) = s.val.comp (lift R of) := by
    ext
    simp only [AlgHom.toLinearMap_id, LinearMap.id_comp, AlgHom.comp_toLinearMap,
      LinearMap.coe_comp, Function.comp_apply, AlgHom.toLinearMap_apply, lift_ι_apply,
      Subalgebra.coe_val]
    erw [LinearMap.codRestrict_apply]
  -- finding a proof is finding an element of the subalgebra
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    C : TensorAlgebra R M → Prop
    algebraMap : ∀ (r : R), C ((_root_.algebraMap R (TensorAlgebra R M)) r)
    ι : ∀ (x : M), C ((TensorAlgebra.ι R) x)
    mul : ∀ (a b : TensorAlgebra R M), C a → C b → C (HMul.hMul a b)
    add : ∀ (a b : TensorAlgebra R M), C a → C b → C (HAdd.hAdd a b)
    a : TensorAlgebra R M
    s : Subalgebra R (TensorAlgebra R M) := { carrier := C, mul_mem' := mul, one_m …
    h : AddCommMonoid (Subtype fun x => Membership.mem s x) := inferInstanceAs (Ad …
    of : LinearMap (RingHom.id R) M (Subtype fun x => Membership.mem s x) := Linea …
    of_id : Eq (AlgHom.id R (TensorAlgebra R M)) (s.val.comp ((TensorAlgebra.lift  …
    ⊢ C a
  -/
  rw [← AlgHom.id_apply (R := R) a, of_id]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    C : TensorAlgebra R M → Prop
    algebraMap : ∀ (r : R), C ((_root_.algebraMap R (TensorAlgebra R M)) r)
    ι : ∀ (x : M), C ((TensorAlgebra.ι R) x)
    mul : ∀ (a b : TensorAlgebra R M), C a → C b → C (HMul.hMul a b)
    add : ∀ (a b : TensorAlgebra R M), C a → C b → C (HAdd.hAdd a b)
    a : TensorAlgebra R M
    s : Subalgebra R (TensorAlgebra R M) := { carrier := C, mul_mem' := mul, one_m …
    h : AddCommMonoid (Subtype fun x => Membership.mem s x) := inferInstanceAs (Ad …
    of : LinearMap (RingHom.id R) M (Subtype fun x => Membership.mem s x) := Linea …
    of_id : Eq (AlgHom.id R (TensorAlgebra R M)) (s.val.comp ((TensorAlgebra.lift  …
    ⊢ C ((s.val.comp ((TensorAlgebra.lift R) of)) a)
  -/
  exact Subtype.prop (lift R of a)
  /-
    🎉 no goals
  -/


/-- The left-inverse of `algebraMap`. -/
def algebraMapInv : TensorAlgebra R M →ₐ[R] R :=
  lift R (0 : M →ₗ[R] R)


theorem algebraMap_leftInverse :
    Function.LeftInverse algebraMapInv (algebraMap R <| TensorAlgebra R M) := fun x => by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : R
    ⊢ Eq (TensorAlgebra.algebraMapInv ((algebraMap R (TensorAlgebra R M)) x)) x
  -/
  simp [algebraMapInv]
  /-
    🎉 no goals
  -/


@[simp]
theorem algebraMap_inj (x y : R) :
    algebraMap R (TensorAlgebra R M) x = algebraMap R (TensorAlgebra R M) y ↔ x = y :=
  (algebraMap_leftInverse M).injective.eq_iff


@[simp]
theorem algebraMap_eq_zero_iff (x : R) : algebraMap R (TensorAlgebra R M) x = 0 ↔ x = 0 :=
  map_eq_zero_iff (algebraMap _ _) (algebraMap_leftInverse _).injective


@[simp]
theorem algebraMap_eq_one_iff (x : R) : algebraMap R (TensorAlgebra R M) x = 1 ↔ x = 1 :=
  map_eq_one_iff (algebraMap _ _) (algebraMap_leftInverse _).injective


/-- A `TensorAlgebra` over a nontrivial semiring is nontrivial. -/
instance [Nontrivial R] : Nontrivial (TensorAlgebra R M) :=
  (algebraMap_leftInverse M).injective.nontrivial


/-- The canonical map from `TensorAlgebra R M` into `TrivSqZeroExt R M` that sends
`TensorAlgebra.ι` to `TrivSqZeroExt.inr`. -/
def toTrivSqZeroExt [Module Rᵐᵒᵖ M] [IsCentralScalar R M] :
    TensorAlgebra R M →ₐ[R] TrivSqZeroExt R M :=
  lift R (TrivSqZeroExt.inrHom R M)


@[simp]
theorem toTrivSqZeroExt_ι (x : M) [Module Rᵐᵒᵖ M] [IsCentralScalar R M] :
    toTrivSqZeroExt (ι R x) = TrivSqZeroExt.inr x :=
  lift_ι_apply _ _


/-- The left-inverse of `ι`.

As an implementation detail, we implement this using `TrivSqZeroExt` which has a suitable
algebra structure. -/
def ιInv : TensorAlgebra R M →ₗ[R] M := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ LinearMap (RingHom.id R) (TensorAlgebra R M) M
  -/
  letI : Module Rᵐᵒᵖ M := Module.compHom _ ((RingHom.id R).fromOpposite mul_comm)
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    this : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOpposi …
    ⊢ LinearMap (RingHom.id R) (TensorAlgebra R M) M
  -/
  haveI : IsCentralScalar R M := ⟨fun r m => rfl⟩
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    this✝ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppos …
    this : IsCentralScalar R M
    ⊢ LinearMap (RingHom.id R) (TensorAlgebra R M) M
  -/
  exact (TrivSqZeroExt.sndHom R M).comp toTrivSqZeroExt.toLinearMap
  /-
    🎉 no goals
  -/


theorem ι_leftInverse : Function.LeftInverse ιInv (ι R : M → TensorAlgebra R M) := fun x ↦ by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : M
    ⊢ Eq (TensorAlgebra.ιInv ((TensorAlgebra.ι R) x)) x
  -/
  simp [ιInv]
  /-
    🎉 no goals
  -/


@[simp]
theorem ι_inj (x y : M) : ι R x = ι R y ↔ x = y :=
  ι_leftInverse.injective.eq_iff


@[simp]
                                                        /-
                                                          R : Type u_1
                                                          inst✝² : CommSemiring R
                                                          M : Type u_2
                                                          inst✝¹ : AddCommMonoid M
                                                          inst✝ : Module R M
                                                          x : M
                                                          ⊢ Iff (Eq ((TensorAlgebra.ι R) x) 0) (Eq x 0)
                                                        -/
theorem ι_eq_zero_iff (x : M) : ι R x = 0 ↔ x = 0 := by rw [← ι_inj R x 0, LinearMap.map_zero]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem ι_eq_algebraMap_iff (x : M) (r : R) : ι R x = algebraMap R _ r ↔ x = 0 ∧ r = 0 := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : M
    r : R
    ⊢ Iff (Eq ((TensorAlgebra.ι R) x) ((algebraMap R (TensorAlgebra R M)) r)) (And …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((TensorAlgebra.ι R) x) ((algebraMap R (TensorAlgebra R M)) r)
      ⊢ And (Eq x 0) (Eq r 0)
    -/
  · letI : Module Rᵐᵒᵖ M := Module.compHom _ ((RingHom.id R).fromOpposite mul_comm)
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((TensorAlgebra.ι R) x) ((algebraMap R (TensorAlgebra R M)) r)
      this : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOpposi …
      ⊢ And (Eq x 0) (Eq r 0)
    -/
    haveI : IsCentralScalar R M := ⟨fun r m => rfl⟩
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((TensorAlgebra.ι R) x) ((algebraMap R (TensorAlgebra R M)) r)
      this✝ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppos …
      this : IsCentralScalar R M
      ⊢ And (Eq x 0) (Eq r 0)
    -/
    have hf0 : toTrivSqZeroExt (ι R x) = (0, x) := lift_ι_apply _ _
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((TensorAlgebra.ι R) x) ((algebraMap R (TensorAlgebra R M)) r)
      this✝ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppos …
      this : IsCentralScalar R M
      hf0 : Eq (TensorAlgebra.toTrivSqZeroExt ((TensorAlgebra.ι R) x)) { fst := 0, s …
      ⊢ And (Eq x 0) (Eq r 0)
    -/
    rw [h, AlgHom.commutes] at hf0
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((TensorAlgebra.ι R) x) ((algebraMap R (TensorAlgebra R M)) r)
      this✝ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppos …
      this : IsCentralScalar R M
      hf0 : Eq ((algebraMap R (TrivSqZeroExt R M)) r) { fst := 0, snd := x }
      ⊢ And (Eq x 0) (Eq r 0)
    -/
    have : r = 0 ∧ 0 = x := Prod.ext_iff.1 hf0
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((TensorAlgebra.ι R) x) ((algebraMap R (TensorAlgebra R M)) r)
      this✝¹ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppo …
      this✝ : IsCentralScalar R M
      hf0 : Eq ((algebraMap R (TrivSqZeroExt R M)) r) { fst := 0, snd := x }
      this : And (Eq r 0) (Eq 0 x)
      ⊢ And (Eq x 0) (Eq r 0)
    -/
    exact this.symm.imp_left Eq.symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommSemiring R
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x : M
      r : R
      ⊢ And (Eq x 0) (Eq r 0) → Eq ((TensorAlgebra.ι R) x) ((algebraMap R (TensorAlg …
    -/
  · rintro ⟨rfl, rfl⟩
    /-
      case refine_2.intro
      R : Type u_1
      inst✝² : CommSemiring R
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ⊢ Eq ((TensorAlgebra.ι R) 0) ((algebraMap R (TensorAlgebra R M)) 0)
    -/
    rw [LinearMap.map_zero, RingHom.map_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem ι_ne_one [Nontrivial R] (x : M) : ι R x ≠ 1 := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    x : M
    ⊢ Ne ((TensorAlgebra.ι R) x) 1
  -/
  rw [← (algebraMap R (TensorAlgebra R M)).map_one, Ne, ι_eq_algebraMap_iff]
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    x : M
    ⊢ Not (And (Eq x 0) (Eq 1 0))
  -/
  exact one_ne_zero ∘ And.right
  /-
    🎉 no goals
  -/


/-- The generators of the tensor algebra are disjoint from its scalars. -/
theorem ι_range_disjoint_one :
    Disjoint (LinearMap.range (ι R : M →ₗ[R] TensorAlgebra R M))
      (1 : Submodule R (TensorAlgebra R M)) := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Disjoint (LinearMap.range (TensorAlgebra.ι R)) 1
  -/
  rw [Submodule.disjoint_def, Submodule.one_eq_range]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ ∀ (x : TensorAlgebra R M), Membership.mem (LinearMap.range (TensorAlgebra.ι  …
  -/
  rintro _ ⟨x, hx⟩ ⟨r, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : M
    r : R
    hx : Eq ((TensorAlgebra.ι R) x) ((Algebra.linearMap R (TensorAlgebra R M)) r)
    ⊢ Eq ((Algebra.linearMap R (TensorAlgebra R M)) r) 0
  -/
  rw [Algebra.linearMap_apply, ι_eq_algebraMap_iff] at hx
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : M
    r : R
    hx : And (Eq x 0) (Eq r 0)
    ⊢ Eq ((Algebra.linearMap R (TensorAlgebra R M)) r) 0
  -/
  rw [hx.2, map_zero]
  /-
    🎉 no goals
  -/


/-- Construct a product of `n` elements of the module within the tensor algebra.

See also `PiTensorProduct.tprod`. -/
def tprod (n : ℕ) : MultilinearMap R (fun _ : Fin n => M) (TensorAlgebra R M) :=
  (MultilinearMap.mkPiAlgebraFin R n (TensorAlgebra R M)).compLinearMap fun _ => ι R


@[simp]
theorem tprod_apply {n : ℕ} (x : Fin n → M) : tprod R M n x = (List.ofFn fun i => ι R (x i)).prod :=
  rfl


/-- The canonical image of the `FreeAlgebra` in the `TensorAlgebra`, which maps
`FreeAlgebra.ι R x` to `TensorAlgebra.ι R x`. -/
def toTensor : FreeAlgebra R M →ₐ[R] TensorAlgebra R M :=
  FreeAlgebra.lift R (TensorAlgebra.ι R)


@[simp]
theorem toTensor_ι (m : M) : FreeAlgebra.toTensor (FreeAlgebra.ι R m) = TensorAlgebra.ι R m := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : M
    ⊢ Eq (FreeAlgebra.toTensor (FreeAlgebra.ι R m)) ((TensorAlgebra.ι R) m)
  -/
  simp [toTensor]
  /-
    🎉 no goals
  -/


