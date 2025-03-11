/-- The relation on `FreeAddMonoid (M × N)` that generates a congruence whose quotient is
the tensor product. -/
inductive Eqv : FreeAddMonoid (M × N) → FreeAddMonoid (M × N) → Prop
  | of_zero_left : ∀ n : N, Eqv (.of (0, n)) 0
  | of_zero_right : ∀ m : M, Eqv (.of (m, 0)) 0
  | of_add_left : ∀ (m₁ m₂ : M) (n : N), Eqv (.of (m₁, n) + .of (m₂, n)) (.of (m₁ + m₂, n))
  | of_add_right : ∀ (m : M) (n₁ n₂ : N), Eqv (.of (m, n₁) + .of (m, n₂)) (.of (m, n₁ + n₂))
  | of_smul : ∀ (r : R) (m : M) (n : N), Eqv (.of (r • m, n)) (.of (m, r • n))
  | add_comm : ∀ x y, Eqv (x + y) (y + x)


/-- The tensor product of two modules `M` and `N` over the same commutative semiring `R`.
The localized notations are `M ⊗ N` and `M ⊗[R] N`, accessed by `open scoped TensorProduct`. -/
def TensorProduct : Type _ :=
  (addConGen (TensorProduct.Eqv R M N)).Quotient


set_option quotPrecheck false in
@[inherit_doc TensorProduct] scoped[TensorProduct] infixl:100 " ⊗ " => TensorProduct _


@[inherit_doc] scoped[TensorProduct] notation:100 M " ⊗[" R "] " N:100 => TensorProduct R M N


protected instance zero : Zero (M ⊗[R] N) :=
  (addConGen (TensorProduct.Eqv R M N)).zero


protected instance add : Add (M ⊗[R] N) :=
  (addConGen (TensorProduct.Eqv R M N)).hasAdd


instance addZeroClass : AddZeroClass (M ⊗[R] N) :=
  { (addConGen (TensorProduct.Eqv R M N)).addMonoid with
    /- The `toAdd` field is given explicitly as `TensorProduct.add` for performance reasons.
    This avoids any need to unfold `Con.addMonoid` when the type checker is checking
    that instance diagrams commute -/
    toAdd := TensorProduct.add _ _
    toZero := TensorProduct.zero _ _ }


instance addSemigroup : AddSemigroup (M ⊗[R] N) :=
  { (addConGen (TensorProduct.Eqv R M N)).addMonoid with
    toAdd := TensorProduct.add _ _ }


instance addCommSemigroup : AddCommSemigroup (M ⊗[R] N) :=
  { (addConGen (TensorProduct.Eqv R M N)).addMonoid with
    toAddSemigroup := TensorProduct.addSemigroup _ _
    add_comm := fun x y =>
      AddCon.induction_on₂ x y fun _ _ =>
        Quotient.sound' <| AddConGen.Rel.of _ _ <| Eqv.add_comm _ _ }


instance : Inhabited (M ⊗[R] N) :=
  ⟨0⟩


/-- The canonical function `M → N → M ⊗ N`. The localized notations are `m ⊗ₜ n` and `m ⊗ₜ[R] n`,
accessed by `open scoped TensorProduct`. -/
def tmul (m : M) (n : N) : M ⊗[R] N :=
  AddCon.mk' _ <| FreeAddMonoid.of (m, n)


/-- The canonical function `M → N → M ⊗ N`. -/
infixl:100 " ⊗ₜ " => tmul _


/-- The canonical function `M → N → M ⊗ N`. -/
notation:100 x " ⊗ₜ[" R "] " y:100 => tmul R x y

-- Porting note: make the arguments of induction_on explicit

@[elab_as_elim, induction_eliminator]
protected theorem induction_on {motive : M ⊗[R] N → Prop} (z : M ⊗[R] N)
    (zero : motive 0)
    (tmul : ∀ x y, motive <| x ⊗ₜ[R] y)
    (add : ∀ x y, motive x → motive y → motive (x + y)) : motive z :=
  AddCon.induction_on z fun x =>
    FreeAddMonoid.recOn x zero fun ⟨m, n⟩ y ih => by
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Type u_5
        N : Type u_6
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R M
        inst✝ : Module R N
        motive : TensorProduct R M N → Prop
        z : TensorProduct R M N
        zero : motive 0
        tmul : ∀ (x : M) (y : N), motive (TensorProduct.tmul R x y)
        add : ∀ (x y : TensorProduct R M N), motive x → motive y → motive (HAdd.hAdd x …
        x : FreeAddMonoid (Prod M N)
        x✝ : Prod M N
        y : FreeAddMonoid (Prod M N)
        ih : motive ↑y
        m : M
        n : N
        ⊢ motive ↑(HAdd.hAdd (FreeAddMonoid.of { fst := m, snd := n }) y)
      -/
      rw [AddCon.coe_add]
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Type u_5
        N : Type u_6
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R M
        inst✝ : Module R N
        motive : TensorProduct R M N → Prop
        z : TensorProduct R M N
        zero : motive 0
        tmul : ∀ (x : M) (y : N), motive (TensorProduct.tmul R x y)
        add : ∀ (x y : TensorProduct R M N), motive x → motive y → motive (HAdd.hAdd x …
        x : FreeAddMonoid (Prod M N)
        x✝ : Prod M N
        y : FreeAddMonoid (Prod M N)
        ih : motive ↑y
        m : M
        n : N
        ⊢ motive (HAdd.hAdd ↑(FreeAddMonoid.of { fst := m, snd := n }) ↑y)
      -/
      exact add _ _ (tmul ..) ih
      /-
        🎉 no goals
      -/


/-- Lift an `R`-balanced map to the tensor product.

A map `f : M →+ N →+ P` additive in both components is `R`-balanced, or middle linear with respect
to `R`, if scalar multiplication in either argument is equivalent, `f (r • m) n = f m (r • n)`.

Note that strictly the first action should be a right-action by `R`, but for now `R` is commutative
so it doesn't matter. -/
-- TODO: use this to implement `lift` and `SMul.aux`. For now we do not do this as it causes
-- performance issues elsewhere.
def liftAddHom (f : M →+ N →+ P)
    (hf : ∀ (r : R) (m : M) (n : N), f (r • m) n = f m (r • n)) :
    M ⊗[R] N →+ P :=
  (addConGen (TensorProduct.Eqv R M N)).lift (FreeAddMonoid.lift (fun mn : M × N => f mn.1 mn.2)) <|
    AddCon.addConGen_le fun x y hxy =>
      match x, y, hxy with
      | _, _, .of_zero_left n =>
        (AddCon.ker_rel _).2 <| by simp_rw [map_zero, FreeAddMonoid.lift_eval_of, map_zero,
          AddMonoidHom.zero_apply]
      | _, _, .of_zero_right m =>
                                   /-
                                     R : Type u_1
                                     inst✝¹⁵ : CommSemiring R
                                     R' : Type u_2
                                     inst✝¹⁴ : Monoid R'
                                     R'' : Type u_3
                                     inst✝¹³ : Semiring R''
                                     A : Type u_4
                                     M : Type u_5
                                     N : Type u_6
                                     P : Type u_7
                                     Q : Type u_8
                                     S : Type u_9
                                     T : Type u_10
                                     inst✝¹² : AddCommMonoid M
                                     inst✝¹¹ : AddCommMonoid N
                                     inst✝¹⁰ : AddCommMonoid P
                                     inst✝⁹ : AddCommMonoid Q
                                     inst✝⁸ : AddCommMonoid S
                                     inst✝⁷ : AddCommMonoid T
                                     inst✝⁶ : Module R M
                                     inst✝⁵ : Module R N
                                     inst✝⁴ : Module R Q
                                     inst✝³ : Module R S
                                     inst✝² : Module R T
                                     inst✝¹ : DistribMulAction R' M
                                     inst✝ : Module R'' M
                                     f : AddMonoidHom M (AddMonoidHom N P)
                                     hf : ∀ (r : R) (m : M) (n : N), Eq ((f (HSMul.hSMul r m)) n) ((f m) (HSMul.hSM …
                                     x y : FreeAddMonoid (Prod M N)
                                     hxy : TensorProduct.Eqv R M N x y
                                     m : M
                                     ⊢ Eq ((FreeAddMonoid.lift fun mn => (f mn.1) mn.2) (FreeAddMonoid.of { fst :=  …
                                   -/
        (AddCon.ker_rel _).2 <| by simp_rw [map_zero, FreeAddMonoid.lift_eval_of, map_zero]
                                   /-
                                     🎉 no goals
                                   -/
      | _, _, .of_add_left m₁ m₂ n =>
        (AddCon.ker_rel _).2 <| by simp_rw [map_add, FreeAddMonoid.lift_eval_of, map_add,
          AddMonoidHom.add_apply]
      | _, _, .of_add_right m n₁ n₂ =>
                                   /-
                                     R : Type u_1
                                     inst✝¹⁵ : CommSemiring R
                                     R' : Type u_2
                                     inst✝¹⁴ : Monoid R'
                                     R'' : Type u_3
                                     inst✝¹³ : Semiring R''
                                     A : Type u_4
                                     M : Type u_5
                                     N : Type u_6
                                     P : Type u_7
                                     Q : Type u_8
                                     S : Type u_9
                                     T : Type u_10
                                     inst✝¹² : AddCommMonoid M
                                     inst✝¹¹ : AddCommMonoid N
                                     inst✝¹⁰ : AddCommMonoid P
                                     inst✝⁹ : AddCommMonoid Q
                                     inst✝⁸ : AddCommMonoid S
                                     inst✝⁷ : AddCommMonoid T
                                     inst✝⁶ : Module R M
                                     inst✝⁵ : Module R N
                                     inst✝⁴ : Module R Q
                                     inst✝³ : Module R S
                                     inst✝² : Module R T
                                     inst✝¹ : DistribMulAction R' M
                                     inst✝ : Module R'' M
                                     f : AddMonoidHom M (AddMonoidHom N P)
                                     hf : ∀ (r : R) (m : M) (n : N), Eq ((f (HSMul.hSMul r m)) n) ((f m) (HSMul.hSM …
                                     x y : FreeAddMonoid (Prod M N)
                                     hxy : TensorProduct.Eqv R M N x y
                                     m : M
                                     n₁ n₂ : N
                                     ⊢ Eq ((FreeAddMonoid.lift fun mn => (f mn.1) mn.2) (HAdd.hAdd (FreeAddMonoid.o …
                                   -/
        (AddCon.ker_rel _).2 <| by simp_rw [map_add, FreeAddMonoid.lift_eval_of, map_add]
                                   /-
                                     🎉 no goals
                                   -/
      | _, _, .of_smul s m n =>
                                   /-
                                     R : Type u_1
                                     inst✝¹⁵ : CommSemiring R
                                     R' : Type u_2
                                     inst✝¹⁴ : Monoid R'
                                     R'' : Type u_3
                                     inst✝¹³ : Semiring R''
                                     A : Type u_4
                                     M : Type u_5
                                     N : Type u_6
                                     P : Type u_7
                                     Q : Type u_8
                                     S : Type u_9
                                     T : Type u_10
                                     inst✝¹² : AddCommMonoid M
                                     inst✝¹¹ : AddCommMonoid N
                                     inst✝¹⁰ : AddCommMonoid P
                                     inst✝⁹ : AddCommMonoid Q
                                     inst✝⁸ : AddCommMonoid S
                                     inst✝⁷ : AddCommMonoid T
                                     inst✝⁶ : Module R M
                                     inst✝⁵ : Module R N
                                     inst✝⁴ : Module R Q
                                     inst✝³ : Module R S
                                     inst✝² : Module R T
                                     inst✝¹ : DistribMulAction R' M
                                     inst✝ : Module R'' M
                                     f : AddMonoidHom M (AddMonoidHom N P)
                                     hf : ∀ (r : R) (m : M) (n : N), Eq ((f (HSMul.hSMul r m)) n) ((f m) (HSMul.hSM …
                                     x y : FreeAddMonoid (Prod M N)
                                     hxy : TensorProduct.Eqv R M N x y
                                     s : R
                                     m : M
                                     n : N
                                     ⊢ Eq ((FreeAddMonoid.lift fun mn => (f mn.1) mn.2) (FreeAddMonoid.of { fst :=  …
                                   -/
        (AddCon.ker_rel _).2 <| by rw [FreeAddMonoid.lift_eval_of, FreeAddMonoid.lift_eval_of, hf]
                                   /-
                                     🎉 no goals
                                   -/
      | _, _, .add_comm x y =>
                                   /-
                                     R : Type u_1
                                     inst✝¹⁵ : CommSemiring R
                                     R' : Type u_2
                                     inst✝¹⁴ : Monoid R'
                                     R'' : Type u_3
                                     inst✝¹³ : Semiring R''
                                     A : Type u_4
                                     M : Type u_5
                                     N : Type u_6
                                     P : Type u_7
                                     Q : Type u_8
                                     S : Type u_9
                                     T : Type u_10
                                     inst✝¹² : AddCommMonoid M
                                     inst✝¹¹ : AddCommMonoid N
                                     inst✝¹⁰ : AddCommMonoid P
                                     inst✝⁹ : AddCommMonoid Q
                                     inst✝⁸ : AddCommMonoid S
                                     inst✝⁷ : AddCommMonoid T
                                     inst✝⁶ : Module R M
                                     inst✝⁵ : Module R N
                                     inst✝⁴ : Module R Q
                                     inst✝³ : Module R S
                                     inst✝² : Module R T
                                     inst✝¹ : DistribMulAction R' M
                                     inst✝ : Module R'' M
                                     f : AddMonoidHom M (AddMonoidHom N P)
                                     hf : ∀ (r : R) (m : M) (n : N), Eq ((f (HSMul.hSMul r m)) n) ((f m) (HSMul.hSM …
                                     x✝ y✝ : FreeAddMonoid (Prod M N)
                                     hxy : TensorProduct.Eqv R M N x✝ y✝
                                     x y : FreeAddMonoid (Prod M N)
                                     ⊢ Eq ((FreeAddMonoid.lift fun mn => (f mn.1) mn.2) (HAdd.hAdd x y)) ((FreeAddM …
                                   -/
        (AddCon.ker_rel _).2 <| by simp_rw [map_add, add_comm]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem liftAddHom_tmul (f : M →+ N →+ P)
    (hf : ∀ (r : R) (m : M) (n : N), f (r • m) n = f m (r • n)) (m : M) (n : N) :
    liftAddHom f hf (m ⊗ₜ n) = f m n :=
  rfl


@[simp]
theorem zero_tmul (n : N) : (0 : M) ⊗ₜ[R] n = 0 :=
  Quotient.sound' <| AddConGen.Rel.of _ _ <| Eqv.of_zero_left _


theorem add_tmul (m₁ m₂ : M) (n : N) : (m₁ + m₂) ⊗ₜ n = m₁ ⊗ₜ n + m₂ ⊗ₜ[R] n :=
  Eq.symm <| Quotient.sound' <| AddConGen.Rel.of _ _ <| Eqv.of_add_left _ _ _


@[simp]
theorem tmul_zero (m : M) : m ⊗ₜ[R] (0 : N) = 0 :=
  Quotient.sound' <| AddConGen.Rel.of _ _ <| Eqv.of_zero_right _


theorem tmul_add (m : M) (n₁ n₂ : N) : m ⊗ₜ (n₁ + n₂) = m ⊗ₜ n₁ + m ⊗ₜ[R] n₂ :=
  Eq.symm <| Quotient.sound' <| AddConGen.Rel.of _ _ <| Eqv.of_add_right _ _ _


instance uniqueLeft [Subsingleton M] : Unique (M ⊗[R] N) where
  default := 0
                                             /-
                                               R : Type u_1
                                               inst✝¹⁶ : CommSemiring R
                                               R' : Type u_2
                                               inst✝¹⁵ : Monoid R'
                                               R'' : Type u_3
                                               inst✝¹⁴ : Semiring R''
                                               A : Type u_4
                                               M : Type u_5
                                               N : Type u_6
                                               P : Type u_7
                                               Q : Type u_8
                                               S : Type u_9
                                               T : Type u_10
                                               inst✝¹³ : AddCommMonoid M
                                               inst✝¹² : AddCommMonoid N
                                               inst✝¹¹ : AddCommMonoid P
                                               inst✝¹⁰ : AddCommMonoid Q
                                               inst✝⁹ : AddCommMonoid S
                                               inst✝⁸ : AddCommMonoid T
                                               inst✝⁷ : Module R M
                                               inst✝⁶ : Module R N
                                               inst✝⁵ : Module R Q
                                               inst✝⁴ : Module R S
                                               inst✝³ : Module R T
                                               inst✝² : DistribMulAction R' M
                                               inst✝¹ : Module R'' M
                                               inst✝ : Subsingleton M
                                               z : TensorProduct R M N
                                               x : M
                                               y : N
                                               ⊢ Eq (TensorProduct.tmul R x y) Inhabited.default
                                             -/
  uniq z := z.induction_on rfl (fun x y ↦ by rw [Subsingleton.elim x 0, zero_tmul]; rfl) <| by
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    /-
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Subsingleton M
      z : TensorProduct R M N
      ⊢ ∀ (x y : TensorProduct R M N), Eq x Inhabited.default → Eq y Inhabited.defau …
    -/
    rintro _ _ rfl rfl; apply add_zero
                        /-
                          🎉 no goals
                        -/


instance uniqueRight [Subsingleton N] : Unique (M ⊗[R] N) where
  default := 0
                                             /-
                                               R : Type u_1
                                               inst✝¹⁶ : CommSemiring R
                                               R' : Type u_2
                                               inst✝¹⁵ : Monoid R'
                                               R'' : Type u_3
                                               inst✝¹⁴ : Semiring R''
                                               A : Type u_4
                                               M : Type u_5
                                               N : Type u_6
                                               P : Type u_7
                                               Q : Type u_8
                                               S : Type u_9
                                               T : Type u_10
                                               inst✝¹³ : AddCommMonoid M
                                               inst✝¹² : AddCommMonoid N
                                               inst✝¹¹ : AddCommMonoid P
                                               inst✝¹⁰ : AddCommMonoid Q
                                               inst✝⁹ : AddCommMonoid S
                                               inst✝⁸ : AddCommMonoid T
                                               inst✝⁷ : Module R M
                                               inst✝⁶ : Module R N
                                               inst✝⁵ : Module R Q
                                               inst✝⁴ : Module R S
                                               inst✝³ : Module R T
                                               inst✝² : DistribMulAction R' M
                                               inst✝¹ : Module R'' M
                                               inst✝ : Subsingleton N
                                               z : TensorProduct R M N
                                               x : M
                                               y : N
                                               ⊢ Eq (TensorProduct.tmul R x y) Inhabited.default
                                             -/
  uniq z := z.induction_on rfl (fun x y ↦ by rw [Subsingleton.elim y 0, tmul_zero]; rfl) <| by
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    /-
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Subsingleton N
      z : TensorProduct R M N
      ⊢ ∀ (x y : TensorProduct R M N), Eq x Inhabited.default → Eq y Inhabited.defau …
    -/
    rintro _ _ rfl rfl; apply add_zero
                        /-
                          🎉 no goals
                        -/


/-- A typeclass for `SMul` structures which can be moved across a tensor product.

This typeclass is generated automatically from an `IsScalarTower` instance, but exists so that
we can also add an instance for `AddCommGroup.toIntModule`, allowing `z •` to be moved even if
`R` does not support negation.

Note that `Module R' (M ⊗[R] N)` is available even without this typeclass on `R'`; it's only
needed if `TensorProduct.smul_tmul`, `TensorProduct.smul_tmul'`, or `TensorProduct.tmul_smul` is
used.
-/
class CompatibleSMul [DistribMulAction R' N] : Prop where
  smul_tmul : ∀ (r : R') (m : M) (n : N), (r • m) ⊗ₜ n = m ⊗ₜ[R] (r • n)


/-- Note that this provides the default `CompatibleSMul R R M N` instance through
`IsScalarTower.left`. -/
instance (priority := 100) CompatibleSMul.isScalarTower [SMul R' R] [IsScalarTower R' R M]
    [DistribMulAction R' N] [IsScalarTower R' R N] : CompatibleSMul R R' M N :=
  ⟨fun r m n => by
    /-
      R : Type u_1
      inst✝¹⁹ : CommSemiring R
      R' : Type u_2
      inst✝¹⁸ : Monoid R'
      R'' : Type u_3
      inst✝¹⁷ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹⁶ : AddCommMonoid M
      inst✝¹⁵ : AddCommMonoid N
      inst✝¹⁴ : AddCommMonoid P
      inst✝¹³ : AddCommMonoid Q
      inst✝¹² : AddCommMonoid S
      inst✝¹¹ : AddCommMonoid T
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R N
      inst✝⁸ : Module R Q
      inst✝⁷ : Module R S
      inst✝⁶ : Module R T
      inst✝⁵ : DistribMulAction R' M
      inst✝⁴ : Module R'' M
      inst✝³ : SMul R' R
      inst✝² : IsScalarTower R' R M
      inst✝¹ : DistribMulAction R' N
      inst✝ : IsScalarTower R' R N
      r : R'
      m : M
      n : N
      ⊢ Eq (TensorProduct.tmul R (HSMul.hSMul r m) n) (TensorProduct.tmul R m (HSMul …
    -/
    conv_lhs => rw [← one_smul R m]
    /-
      R : Type u_1
      inst✝¹⁹ : CommSemiring R
      R' : Type u_2
      inst✝¹⁸ : Monoid R'
      R'' : Type u_3
      inst✝¹⁷ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹⁶ : AddCommMonoid M
      inst✝¹⁵ : AddCommMonoid N
      inst✝¹⁴ : AddCommMonoid P
      inst✝¹³ : AddCommMonoid Q
      inst✝¹² : AddCommMonoid S
      inst✝¹¹ : AddCommMonoid T
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R N
      inst✝⁸ : Module R Q
      inst✝⁷ : Module R S
      inst✝⁶ : Module R T
      inst✝⁵ : DistribMulAction R' M
      inst✝⁴ : Module R'' M
      inst✝³ : SMul R' R
      inst✝² : IsScalarTower R' R M
      inst✝¹ : DistribMulAction R' N
      inst✝ : IsScalarTower R' R N
      r : R'
      m : M
      n : N
      ⊢ Eq (TensorProduct.tmul R (HSMul.hSMul r (HSMul.hSMul 1 m)) n) (TensorProduct …
    -/
    conv_rhs => rw [← one_smul R n]
    /-
      R : Type u_1
      inst✝¹⁹ : CommSemiring R
      R' : Type u_2
      inst✝¹⁸ : Monoid R'
      R'' : Type u_3
      inst✝¹⁷ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹⁶ : AddCommMonoid M
      inst✝¹⁵ : AddCommMonoid N
      inst✝¹⁴ : AddCommMonoid P
      inst✝¹³ : AddCommMonoid Q
      inst✝¹² : AddCommMonoid S
      inst✝¹¹ : AddCommMonoid T
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R N
      inst✝⁸ : Module R Q
      inst✝⁷ : Module R S
      inst✝⁶ : Module R T
      inst✝⁵ : DistribMulAction R' M
      inst✝⁴ : Module R'' M
      inst✝³ : SMul R' R
      inst✝² : IsScalarTower R' R M
      inst✝¹ : DistribMulAction R' N
      inst✝ : IsScalarTower R' R N
      r : R'
      m : M
      n : N
      ⊢ Eq (TensorProduct.tmul R (HSMul.hSMul r (HSMul.hSMul 1 m)) n) (TensorProduct …
    -/
    rw [← smul_assoc, ← smul_assoc]
    /-
      R : Type u_1
      inst✝¹⁹ : CommSemiring R
      R' : Type u_2
      inst✝¹⁸ : Monoid R'
      R'' : Type u_3
      inst✝¹⁷ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹⁶ : AddCommMonoid M
      inst✝¹⁵ : AddCommMonoid N
      inst✝¹⁴ : AddCommMonoid P
      inst✝¹³ : AddCommMonoid Q
      inst✝¹² : AddCommMonoid S
      inst✝¹¹ : AddCommMonoid T
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R N
      inst✝⁸ : Module R Q
      inst✝⁷ : Module R S
      inst✝⁶ : Module R T
      inst✝⁵ : DistribMulAction R' M
      inst✝⁴ : Module R'' M
      inst✝³ : SMul R' R
      inst✝² : IsScalarTower R' R M
      inst✝¹ : DistribMulAction R' N
      inst✝ : IsScalarTower R' R N
      r : R'
      m : M
      n : N
      ⊢ Eq (TensorProduct.tmul R (HSMul.hSMul (HSMul.hSMul r 1) m) n) (TensorProduct …
    -/
    exact Quotient.sound' <| AddConGen.Rel.of _ _ <| Eqv.of_smul _ _ _⟩
    /-
      🎉 no goals
    -/


/-- `smul` can be moved from one side of the product to the other . -/
theorem smul_tmul [DistribMulAction R' N] [CompatibleSMul R R' M N] (r : R') (m : M) (n : N) :
    (r • m) ⊗ₜ n = m ⊗ₜ[R] (r • n) :=
  CompatibleSMul.smul_tmul _ _ _

-- Porting note: This is added as a local instance for `SMul.aux`.
-- For some reason type-class inference in Lean 3 unfolded this definition.

private def addMonoidWithWrongNSMul : AddMonoid (M ⊗[R] N) :=
  { (addConGen (TensorProduct.Eqv R M N)).addMonoid with }


attribute [local instance] addMonoidWithWrongNSMul in
/-- Auxiliary function to defining scalar multiplication on tensor product. -/
def SMul.aux {R' : Type*} [SMul R' M] (r : R') : FreeAddMonoid (M × N) →+ M ⊗[R] N :=
  FreeAddMonoid.lift fun p : M × N => (r • p.1) ⊗ₜ p.2


theorem SMul.aux_of {R' : Type*} [SMul R' M] (r : R') (m : M) (n : N) :
    SMul.aux r (.of (m, n)) = (r • m) ⊗ₜ[R] n :=
  rfl


/-- Given two modules over a commutative semiring `R`, if one of the factors carries a
(distributive) action of a second type of scalars `R'`, which commutes with the action of `R`, then
the tensor product (over `R`) carries an action of `R'`.

This instance defines this `R'` action in the case that it is the left module which has the `R'`
action. Two natural ways in which this situation arises are:
 * Extension of scalars
 * A tensor product of a group representation with a module not carrying an action

Note that in the special case that `R = R'`, since `R` is commutative, we just get the usual scalar
action on a tensor product of two modules. This special case is important enough that, for
performance reasons, we define it explicitly below. -/
instance leftHasSMul : SMul R' (M ⊗[R] N) :=
  ⟨fun r =>
    (addConGen (TensorProduct.Eqv R M N)).lift (SMul.aux r : _ →+ M ⊗[R] N) <|
      AddCon.addConGen_le fun x y hxy =>
        match x, y, hxy with
        | _, _, .of_zero_left n =>
                                     /-
                                       R : Type u_1
                                       inst✝¹⁷ : CommSemiring R
                                       R' : Type u_2
                                       inst✝¹⁶ : Monoid R'
                                       R'' : Type u_3
                                       inst✝¹⁵ : Semiring R''
                                       A : Type u_4
                                       M : Type u_5
                                       N : Type u_6
                                       P : Type u_7
                                       Q : Type u_8
                                       S : Type u_9
                                       T : Type u_10
                                       inst✝¹⁴ : AddCommMonoid M
                                       inst✝¹³ : AddCommMonoid N
                                       inst✝¹² : AddCommMonoid P
                                       inst✝¹¹ : AddCommMonoid Q
                                       inst✝¹⁰ : AddCommMonoid S
                                       inst✝⁹ : AddCommMonoid T
                                       inst✝⁸ : Module R M
                                       inst✝⁷ : Module R N
                                       inst✝⁶ : Module R Q
                                       inst✝⁵ : Module R S
                                       inst✝⁴ : Module R T
                                       inst✝³ : DistribMulAction R' M
                                       inst✝² : Module R'' M
                                       inst✝¹ : SMulCommClass R R' M
                                       inst✝ : SMulCommClass R R'' M
                                       r : R'
                                       x y : FreeAddMonoid (Prod M N)
                                       hxy : TensorProduct.Eqv R M N x y
                                       n : N
                                       ⊢ Eq ((TensorProduct.SMul.aux r) (FreeAddMonoid.of { fst := 0, snd := n })) (( …
                                     -/
          (AddCon.ker_rel _).2 <| by simp_rw [map_zero, SMul.aux_of, smul_zero, zero_tmul]
                                     /-
                                       🎉 no goals
                                     -/
        | _, _, .of_zero_right m =>
                                     /-
                                       R : Type u_1
                                       inst✝¹⁷ : CommSemiring R
                                       R' : Type u_2
                                       inst✝¹⁶ : Monoid R'
                                       R'' : Type u_3
                                       inst✝¹⁵ : Semiring R''
                                       A : Type u_4
                                       M : Type u_5
                                       N : Type u_6
                                       P : Type u_7
                                       Q : Type u_8
                                       S : Type u_9
                                       T : Type u_10
                                       inst✝¹⁴ : AddCommMonoid M
                                       inst✝¹³ : AddCommMonoid N
                                       inst✝¹² : AddCommMonoid P
                                       inst✝¹¹ : AddCommMonoid Q
                                       inst✝¹⁰ : AddCommMonoid S
                                       inst✝⁹ : AddCommMonoid T
                                       inst✝⁸ : Module R M
                                       inst✝⁷ : Module R N
                                       inst✝⁶ : Module R Q
                                       inst✝⁵ : Module R S
                                       inst✝⁴ : Module R T
                                       inst✝³ : DistribMulAction R' M
                                       inst✝² : Module R'' M
                                       inst✝¹ : SMulCommClass R R' M
                                       inst✝ : SMulCommClass R R'' M
                                       r : R'
                                       x y : FreeAddMonoid (Prod M N)
                                       hxy : TensorProduct.Eqv R M N x y
                                       m : M
                                       ⊢ Eq ((TensorProduct.SMul.aux r) (FreeAddMonoid.of { fst := m, snd := 0 })) (( …
                                     -/
          (AddCon.ker_rel _).2 <| by simp_rw [map_zero, SMul.aux_of, tmul_zero]
                                     /-
                                       🎉 no goals
                                     -/
        | _, _, .of_add_left m₁ m₂ n =>
                                     /-
                                       R : Type u_1
                                       inst✝¹⁷ : CommSemiring R
                                       R' : Type u_2
                                       inst✝¹⁶ : Monoid R'
                                       R'' : Type u_3
                                       inst✝¹⁵ : Semiring R''
                                       A : Type u_4
                                       M : Type u_5
                                       N : Type u_6
                                       P : Type u_7
                                       Q : Type u_8
                                       S : Type u_9
                                       T : Type u_10
                                       inst✝¹⁴ : AddCommMonoid M
                                       inst✝¹³ : AddCommMonoid N
                                       inst✝¹² : AddCommMonoid P
                                       inst✝¹¹ : AddCommMonoid Q
                                       inst✝¹⁰ : AddCommMonoid S
                                       inst✝⁹ : AddCommMonoid T
                                       inst✝⁸ : Module R M
                                       inst✝⁷ : Module R N
                                       inst✝⁶ : Module R Q
                                       inst✝⁵ : Module R S
                                       inst✝⁴ : Module R T
                                       inst✝³ : DistribMulAction R' M
                                       inst✝² : Module R'' M
                                       inst✝¹ : SMulCommClass R R' M
                                       inst✝ : SMulCommClass R R'' M
                                       r : R'
                                       x y : FreeAddMonoid (Prod M N)
                                       hxy : TensorProduct.Eqv R M N x y
                                       m₁ m₂ : M
                                       n : N
                                       ⊢ Eq ((TensorProduct.SMul.aux r) (HAdd.hAdd (FreeAddMonoid.of { fst := m₁, snd …
                                     -/
          (AddCon.ker_rel _).2 <| by simp_rw [map_add, SMul.aux_of, smul_add, add_tmul]
                                     /-
                                       🎉 no goals
                                     -/
        | _, _, .of_add_right m n₁ n₂ =>
                                     /-
                                       R : Type u_1
                                       inst✝¹⁷ : CommSemiring R
                                       R' : Type u_2
                                       inst✝¹⁶ : Monoid R'
                                       R'' : Type u_3
                                       inst✝¹⁵ : Semiring R''
                                       A : Type u_4
                                       M : Type u_5
                                       N : Type u_6
                                       P : Type u_7
                                       Q : Type u_8
                                       S : Type u_9
                                       T : Type u_10
                                       inst✝¹⁴ : AddCommMonoid M
                                       inst✝¹³ : AddCommMonoid N
                                       inst✝¹² : AddCommMonoid P
                                       inst✝¹¹ : AddCommMonoid Q
                                       inst✝¹⁰ : AddCommMonoid S
                                       inst✝⁹ : AddCommMonoid T
                                       inst✝⁸ : Module R M
                                       inst✝⁷ : Module R N
                                       inst✝⁶ : Module R Q
                                       inst✝⁵ : Module R S
                                       inst✝⁴ : Module R T
                                       inst✝³ : DistribMulAction R' M
                                       inst✝² : Module R'' M
                                       inst✝¹ : SMulCommClass R R' M
                                       inst✝ : SMulCommClass R R'' M
                                       r : R'
                                       x y : FreeAddMonoid (Prod M N)
                                       hxy : TensorProduct.Eqv R M N x y
                                       m : M
                                       n₁ n₂ : N
                                       ⊢ Eq ((TensorProduct.SMul.aux r) (HAdd.hAdd (FreeAddMonoid.of { fst := m, snd  …
                                     -/
          (AddCon.ker_rel _).2 <| by simp_rw [map_add, SMul.aux_of, tmul_add]
                                     /-
                                       🎉 no goals
                                     -/
        | _, _, .of_smul s m n =>
                                     /-
                                       R : Type u_1
                                       inst✝¹⁷ : CommSemiring R
                                       R' : Type u_2
                                       inst✝¹⁶ : Monoid R'
                                       R'' : Type u_3
                                       inst✝¹⁵ : Semiring R''
                                       A : Type u_4
                                       M : Type u_5
                                       N : Type u_6
                                       P : Type u_7
                                       Q : Type u_8
                                       S : Type u_9
                                       T : Type u_10
                                       inst✝¹⁴ : AddCommMonoid M
                                       inst✝¹³ : AddCommMonoid N
                                       inst✝¹² : AddCommMonoid P
                                       inst✝¹¹ : AddCommMonoid Q
                                       inst✝¹⁰ : AddCommMonoid S
                                       inst✝⁹ : AddCommMonoid T
                                       inst✝⁸ : Module R M
                                       inst✝⁷ : Module R N
                                       inst✝⁶ : Module R Q
                                       inst✝⁵ : Module R S
                                       inst✝⁴ : Module R T
                                       inst✝³ : DistribMulAction R' M
                                       inst✝² : Module R'' M
                                       inst✝¹ : SMulCommClass R R' M
                                       inst✝ : SMulCommClass R R'' M
                                       r : R'
                                       x y : FreeAddMonoid (Prod M N)
                                       hxy : TensorProduct.Eqv R M N x y
                                       s : R
                                       m : M
                                       n : N
                                       ⊢ Eq ((TensorProduct.SMul.aux r) (FreeAddMonoid.of { fst := HSMul.hSMul s m, s …
                                     -/
          (AddCon.ker_rel _).2 <| by rw [SMul.aux_of, SMul.aux_of, ← smul_comm, smul_tmul]
                                     /-
                                       🎉 no goals
                                     -/
        | _, _, .add_comm x y =>
                                     /-
                                       R : Type u_1
                                       inst✝¹⁷ : CommSemiring R
                                       R' : Type u_2
                                       inst✝¹⁶ : Monoid R'
                                       R'' : Type u_3
                                       inst✝¹⁵ : Semiring R''
                                       A : Type u_4
                                       M : Type u_5
                                       N : Type u_6
                                       P : Type u_7
                                       Q : Type u_8
                                       S : Type u_9
                                       T : Type u_10
                                       inst✝¹⁴ : AddCommMonoid M
                                       inst✝¹³ : AddCommMonoid N
                                       inst✝¹² : AddCommMonoid P
                                       inst✝¹¹ : AddCommMonoid Q
                                       inst✝¹⁰ : AddCommMonoid S
                                       inst✝⁹ : AddCommMonoid T
                                       inst✝⁸ : Module R M
                                       inst✝⁷ : Module R N
                                       inst✝⁶ : Module R Q
                                       inst✝⁵ : Module R S
                                       inst✝⁴ : Module R T
                                       inst✝³ : DistribMulAction R' M
                                       inst✝² : Module R'' M
                                       inst✝¹ : SMulCommClass R R' M
                                       inst✝ : SMulCommClass R R'' M
                                       r : R'
                                       x✝ y✝ : FreeAddMonoid (Prod M N)
                                       hxy : TensorProduct.Eqv R M N x✝ y✝
                                       x y : FreeAddMonoid (Prod M N)
                                       ⊢ Eq ((TensorProduct.SMul.aux r) (HAdd.hAdd x y)) ((TensorProduct.SMul.aux r)  …
                                     -/
          (AddCon.ker_rel _).2 <| by simp_rw [map_add, add_comm]⟩
                                     /-
                                       🎉 no goals
                                     -/


instance : SMul R (M ⊗[R] N) :=
  TensorProduct.leftHasSMul


protected theorem smul_zero (r : R') : r • (0 : M ⊗[R] N) = 0 :=
  AddMonoidHom.map_zero _


protected theorem smul_add (r : R') (x y : M ⊗[R] N) : r • (x + y) = r • x + r • y :=
  AddMonoidHom.map_add _ _ _


protected theorem zero_smul (x : M ⊗[R] N) : (0 : R'') • x = 0 :=
  have : ∀ (r : R'') (m : M) (n : N), r • m ⊗ₜ[R] n = (r • m) ⊗ₜ n := fun _ _ _ => rfl
                     /-
                       R : Type u_1
                       inst✝⁷ : CommSemiring R
                       R'' : Type u_3
                       inst✝⁶ : Semiring R''
                       M : Type u_5
                       N : Type u_6
                       inst✝⁵ : AddCommMonoid M
                       inst✝⁴ : AddCommMonoid N
                       inst✝³ : Module R M
                       inst✝² : Module R N
                       inst✝¹ : Module R'' M
                       inst✝ : SMulCommClass R R'' M
                       x : TensorProduct R M N
                       this : ∀ (r : R'') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m  …
                       ⊢ Eq (HSMul.hSMul 0 0) 0
                     -/
  x.induction_on (by rw [TensorProduct.smul_zero])
                     /-
                       🎉 no goals
                     -/
                   /-
                     R : Type u_1
                     inst✝⁷ : CommSemiring R
                     R'' : Type u_3
                     inst✝⁶ : Semiring R''
                     M : Type u_5
                     N : Type u_6
                     inst✝⁵ : AddCommMonoid M
                     inst✝⁴ : AddCommMonoid N
                     inst✝³ : Module R M
                     inst✝² : Module R N
                     inst✝¹ : Module R'' M
                     inst✝ : SMulCommClass R R'' M
                     x : TensorProduct R M N
                     this : ∀ (r : R'') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m  …
                     m : M
                     n : N
                     ⊢ Eq (HSMul.hSMul 0 (TensorProduct.tmul R m n)) 0
                   -/
    (fun m n => by rw [this, zero_smul, zero_tmul]) fun x y ihx ihy => by
                   /-
                     🎉 no goals
                   -/
    /-
      R : Type u_1
      inst✝⁷ : CommSemiring R
      R'' : Type u_3
      inst✝⁶ : Semiring R''
      M : Type u_5
      N : Type u_6
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module R'' M
      inst✝ : SMulCommClass R R'' M
      x✝ : TensorProduct R M N
      this : ∀ (r : R'') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m  …
      x y : TensorProduct R M N
      ihx : Eq (HSMul.hSMul 0 x) 0
      ihy : Eq (HSMul.hSMul 0 y) 0
      ⊢ Eq (HSMul.hSMul 0 (HAdd.hAdd x y)) 0
    -/
    rw [TensorProduct.smul_add, ihx, ihy, add_zero]
    /-
      🎉 no goals
    -/


protected theorem one_smul (x : M ⊗[R] N) : (1 : R') • x = x :=
  have : ∀ (r : R') (m : M) (n : N), r • m ⊗ₜ[R] n = (r • m) ⊗ₜ n := fun _ _ _ => rfl
                     /-
                       R : Type u_1
                       inst✝⁷ : CommSemiring R
                       R' : Type u_2
                       inst✝⁶ : Monoid R'
                       M : Type u_5
                       N : Type u_6
                       inst✝⁵ : AddCommMonoid M
                       inst✝⁴ : AddCommMonoid N
                       inst✝³ : Module R M
                       inst✝² : Module R N
                       inst✝¹ : DistribMulAction R' M
                       inst✝ : SMulCommClass R R' M
                       x : TensorProduct R M N
                       this : ∀ (r : R') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m n …
                       ⊢ Eq (HSMul.hSMul 1 0) 0
                     -/
  x.induction_on (by rw [TensorProduct.smul_zero])
                     /-
                       🎉 no goals
                     -/
                   /-
                     R : Type u_1
                     inst✝⁷ : CommSemiring R
                     R' : Type u_2
                     inst✝⁶ : Monoid R'
                     M : Type u_5
                     N : Type u_6
                     inst✝⁵ : AddCommMonoid M
                     inst✝⁴ : AddCommMonoid N
                     inst✝³ : Module R M
                     inst✝² : Module R N
                     inst✝¹ : DistribMulAction R' M
                     inst✝ : SMulCommClass R R' M
                     x : TensorProduct R M N
                     this : ∀ (r : R') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m n …
                     m : M
                     n : N
                     ⊢ Eq (HSMul.hSMul 1 (TensorProduct.tmul R m n)) (TensorProduct.tmul R m n)
                   -/
    (fun m n => by rw [this, one_smul])
                   /-
                     🎉 no goals
                   -/
                          /-
                            R : Type u_1
                            inst✝⁷ : CommSemiring R
                            R' : Type u_2
                            inst✝⁶ : Monoid R'
                            M : Type u_5
                            N : Type u_6
                            inst✝⁵ : AddCommMonoid M
                            inst✝⁴ : AddCommMonoid N
                            inst✝³ : Module R M
                            inst✝² : Module R N
                            inst✝¹ : DistribMulAction R' M
                            inst✝ : SMulCommClass R R' M
                            x✝ : TensorProduct R M N
                            this : ∀ (r : R') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m n …
                            x y : TensorProduct R M N
                            ihx : Eq (HSMul.hSMul 1 x) x
                            ihy : Eq (HSMul.hSMul 1 y) y
                            ⊢ Eq (HSMul.hSMul 1 (HAdd.hAdd x y)) (HAdd.hAdd x y)
                          -/
    fun x y ihx ihy => by rw [TensorProduct.smul_add, ihx, ihy]
                          /-
                            🎉 no goals
                          -/


protected theorem add_smul (r s : R'') (x : M ⊗[R] N) : (r + s) • x = r • x + s • x :=
  have : ∀ (r : R'') (m : M) (n : N), r • m ⊗ₜ[R] n = (r • m) ⊗ₜ n := fun _ _ _ => rfl
                     /-
                       R : Type u_1
                       inst✝⁷ : CommSemiring R
                       R'' : Type u_3
                       inst✝⁶ : Semiring R''
                       M : Type u_5
                       N : Type u_6
                       inst✝⁵ : AddCommMonoid M
                       inst✝⁴ : AddCommMonoid N
                       inst✝³ : Module R M
                       inst✝² : Module R N
                       inst✝¹ : Module R'' M
                       inst✝ : SMulCommClass R R'' M
                       r s : R''
                       x : TensorProduct R M N
                       this : ∀ (r : R'') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m  …
                       ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) 0) (HAdd.hAdd (HSMul.hSMul r 0) (HSMul.hSMul …
                     -/
  x.induction_on (by simp_rw [TensorProduct.smul_zero, add_zero])
                     /-
                       🎉 no goals
                     -/
                   /-
                     R : Type u_1
                     inst✝⁷ : CommSemiring R
                     R'' : Type u_3
                     inst✝⁶ : Semiring R''
                     M : Type u_5
                     N : Type u_6
                     inst✝⁵ : AddCommMonoid M
                     inst✝⁴ : AddCommMonoid N
                     inst✝³ : Module R M
                     inst✝² : Module R N
                     inst✝¹ : Module R'' M
                     inst✝ : SMulCommClass R R'' M
                     r s : R''
                     x : TensorProduct R M N
                     this : ∀ (r : R'') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m  …
                     m : M
                     n : N
                     ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) (TensorProduct.tmul R m n)) (HAdd.hAdd (HSMu …
                   -/
    (fun m n => by simp_rw [this, add_smul, add_tmul]) fun x y ihx ihy => by
                   /-
                     🎉 no goals
                   -/
    /-
      R : Type u_1
      inst✝⁷ : CommSemiring R
      R'' : Type u_3
      inst✝⁶ : Semiring R''
      M : Type u_5
      N : Type u_6
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module R'' M
      inst✝ : SMulCommClass R R'' M
      r s : R''
      x✝ : TensorProduct R M N
      this : ∀ (r : R'') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m  …
      x y : TensorProduct R M N
      ihx : Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.h …
      ihy : Eq (HSMul.hSMul (HAdd.hAdd r s) y) (HAdd.hAdd (HSMul.hSMul r y) (HSMul.h …
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul r ( …
    -/
    simp_rw [TensorProduct.smul_add]
    /-
      R : Type u_1
      inst✝⁷ : CommSemiring R
      R'' : Type u_3
      inst✝⁶ : Semiring R''
      M : Type u_5
      N : Type u_6
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module R'' M
      inst✝ : SMulCommClass R R'' M
      r s : R''
      x✝ : TensorProduct R M N
      this : ∀ (r : R'') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m  …
      x y : TensorProduct R M N
      ihx : Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.h …
      ihy : Eq (HSMul.hSMul (HAdd.hAdd r s) y) (HAdd.hAdd (HSMul.hSMul r y) (HSMul.h …
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd r s) x) (HSMul.hSMul (HAdd.hAdd r s) y …
    -/
    rw [ihx, ihy, add_add_add_comm]
    /-
      🎉 no goals
    -/


instance addMonoid : AddMonoid (M ⊗[R] N) :=
  { TensorProduct.addZeroClass _ _ with
    toAddSemigroup := TensorProduct.addSemigroup _ _
    toZero := TensorProduct.zero _ _
    nsmul := fun n v => n • v
                     /-
                       R : Type u_1
                       inst✝¹⁷ : CommSemiring R
                       R' : Type u_2
                       inst✝¹⁶ : Monoid R'
                       R'' : Type u_3
                       inst✝¹⁵ : Semiring R''
                       A : Type u_4
                       M : Type u_5
                       N : Type u_6
                       P : Type u_7
                       Q : Type u_8
                       S : Type u_9
                       T : Type u_10
                       inst✝¹⁴ : AddCommMonoid M
                       inst✝¹³ : AddCommMonoid N
                       inst✝¹² : AddCommMonoid P
                       inst✝¹¹ : AddCommMonoid Q
                       inst✝¹⁰ : AddCommMonoid S
                       inst✝⁹ : AddCommMonoid T
                       inst✝⁸ : Module R M
                       inst✝⁷ : Module R N
                       inst✝⁶ : Module R Q
                       inst✝⁵ : Module R S
                       inst✝⁴ : Module R T
                       inst✝³ : DistribMulAction R' M
                       inst✝² : Module R'' M
                       inst✝¹ : SMulCommClass R R' M
                       inst✝ : SMulCommClass R R'' M
                       ⊢ ∀ (x : TensorProduct R M N), Eq ((fun n v => HSMul.hSMul n v) 0 x) 0
                     -/
    nsmul_zero := by simp [TensorProduct.zero_smul]
                     /-
                       🎉 no goals
                     -/
    nsmul_succ := by simp only [TensorProduct.one_smul, TensorProduct.add_smul, add_comm,
      forall_const] }


instance addCommMonoid : AddCommMonoid (M ⊗[R] N) :=
  { TensorProduct.addCommSemigroup _ _ with
    toAddMonoid := TensorProduct.addMonoid }


instance leftDistribMulAction : DistribMulAction R' (M ⊗[R] N) :=
  have : ∀ (r : R') (m : M) (n : N), r • m ⊗ₜ[R] n = (r • m) ⊗ₜ n := fun _ _ _ => rfl
  { smul_add := fun r x y => TensorProduct.smul_add r x y
    mul_smul := fun r s x =>
                         /-
                           R : Type u_1
                           inst✝¹⁷ : CommSemiring R
                           R' : Type u_2
                           inst✝¹⁶ : Monoid R'
                           R'' : Type u_3
                           inst✝¹⁵ : Semiring R''
                           A : Type u_4
                           M : Type u_5
                           N : Type u_6
                           P : Type u_7
                           Q : Type u_8
                           S : Type u_9
                           T : Type u_10
                           inst✝¹⁴ : AddCommMonoid M
                           inst✝¹³ : AddCommMonoid N
                           inst✝¹² : AddCommMonoid P
                           inst✝¹¹ : AddCommMonoid Q
                           inst✝¹⁰ : AddCommMonoid S
                           inst✝⁹ : AddCommMonoid T
                           inst✝⁸ : Module R M
                           inst✝⁷ : Module R N
                           inst✝⁶ : Module R Q
                           inst✝⁵ : Module R S
                           inst✝⁴ : Module R T
                           inst✝³ : DistribMulAction R' M
                           inst✝² : Module R'' M
                           inst✝¹ : SMulCommClass R R' M
                           inst✝ : SMulCommClass R R'' M
                           this : ∀ (r : R') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m n …
                           r s : R'
                           x : TensorProduct R M N
                           ⊢ Eq (HSMul.hSMul (HMul.hMul r s) 0) (HSMul.hSMul r (HSMul.hSMul s 0))
                         -/
      x.induction_on (by simp_rw [TensorProduct.smul_zero])
                         /-
                           🎉 no goals
                         -/
                       /-
                         R : Type u_1
                         inst✝¹⁷ : CommSemiring R
                         R' : Type u_2
                         inst✝¹⁶ : Monoid R'
                         R'' : Type u_3
                         inst✝¹⁵ : Semiring R''
                         A : Type u_4
                         M : Type u_5
                         N : Type u_6
                         P : Type u_7
                         Q : Type u_8
                         S : Type u_9
                         T : Type u_10
                         inst✝¹⁴ : AddCommMonoid M
                         inst✝¹³ : AddCommMonoid N
                         inst✝¹² : AddCommMonoid P
                         inst✝¹¹ : AddCommMonoid Q
                         inst✝¹⁰ : AddCommMonoid S
                         inst✝⁹ : AddCommMonoid T
                         inst✝⁸ : Module R M
                         inst✝⁷ : Module R N
                         inst✝⁶ : Module R Q
                         inst✝⁵ : Module R S
                         inst✝⁴ : Module R T
                         inst✝³ : DistribMulAction R' M
                         inst✝² : Module R'' M
                         inst✝¹ : SMulCommClass R R' M
                         inst✝ : SMulCommClass R R'' M
                         this : ∀ (r : R') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m n …
                         r s : R'
                         x : TensorProduct R M N
                         m : M
                         n : N
                         ⊢ Eq (HSMul.hSMul (HMul.hMul r s) (TensorProduct.tmul R m n)) (HSMul.hSMul r ( …
                       -/
        (fun m n => by simp_rw [this, mul_smul]) fun x y ihx ihy => by
                       /-
                         🎉 no goals
                       -/
        /-
          R : Type u_1
          inst✝¹⁷ : CommSemiring R
          R' : Type u_2
          inst✝¹⁶ : Monoid R'
          R'' : Type u_3
          inst✝¹⁵ : Semiring R''
          A : Type u_4
          M : Type u_5
          N : Type u_6
          P : Type u_7
          Q : Type u_8
          S : Type u_9
          T : Type u_10
          inst✝¹⁴ : AddCommMonoid M
          inst✝¹³ : AddCommMonoid N
          inst✝¹² : AddCommMonoid P
          inst✝¹¹ : AddCommMonoid Q
          inst✝¹⁰ : AddCommMonoid S
          inst✝⁹ : AddCommMonoid T
          inst✝⁸ : Module R M
          inst✝⁷ : Module R N
          inst✝⁶ : Module R Q
          inst✝⁵ : Module R S
          inst✝⁴ : Module R T
          inst✝³ : DistribMulAction R' M
          inst✝² : Module R'' M
          inst✝¹ : SMulCommClass R R' M
          inst✝ : SMulCommClass R R'' M
          this : ∀ (r : R') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m n …
          r s : R'
          x✝ x y : TensorProduct R M N
          ihx : Eq (HSMul.hSMul (HMul.hMul r s) x) (HSMul.hSMul r (HSMul.hSMul s x))
          ihy : Eq (HSMul.hSMul (HMul.hMul r s) y) (HSMul.hSMul r (HSMul.hSMul s y))
          ⊢ Eq (HSMul.hSMul (HMul.hMul r s) (HAdd.hAdd x y)) (HSMul.hSMul r (HSMul.hSMul …
        -/
        simp_rw [TensorProduct.smul_add]
        /-
          R : Type u_1
          inst✝¹⁷ : CommSemiring R
          R' : Type u_2
          inst✝¹⁶ : Monoid R'
          R'' : Type u_3
          inst✝¹⁵ : Semiring R''
          A : Type u_4
          M : Type u_5
          N : Type u_6
          P : Type u_7
          Q : Type u_8
          S : Type u_9
          T : Type u_10
          inst✝¹⁴ : AddCommMonoid M
          inst✝¹³ : AddCommMonoid N
          inst✝¹² : AddCommMonoid P
          inst✝¹¹ : AddCommMonoid Q
          inst✝¹⁰ : AddCommMonoid S
          inst✝⁹ : AddCommMonoid T
          inst✝⁸ : Module R M
          inst✝⁷ : Module R N
          inst✝⁶ : Module R Q
          inst✝⁵ : Module R S
          inst✝⁴ : Module R T
          inst✝³ : DistribMulAction R' M
          inst✝² : Module R'' M
          inst✝¹ : SMulCommClass R R' M
          inst✝ : SMulCommClass R R'' M
          this : ∀ (r : R') (m : M) (n : N), Eq (HSMul.hSMul r (TensorProduct.tmul R m n …
          r s : R'
          x✝ x y : TensorProduct R M N
          ihx : Eq (HSMul.hSMul (HMul.hMul r s) x) (HSMul.hSMul r (HSMul.hSMul s x))
          ihy : Eq (HSMul.hSMul (HMul.hMul r s) y) (HSMul.hSMul r (HSMul.hSMul s y))
          ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul r s) x) (HSMul.hSMul (HMul.hMul r s) y …
        -/
        rw [ihx, ihy]
        /-
          🎉 no goals
        -/
    one_smul := TensorProduct.one_smul
    smul_zero := TensorProduct.smul_zero }


instance : DistribMulAction R (M ⊗[R] N) :=
  TensorProduct.leftDistribMulAction


theorem smul_tmul' (r : R') (m : M) (n : N) : r • m ⊗ₜ[R] n = (r • m) ⊗ₜ n :=
  rfl


@[simp]
theorem tmul_smul [DistribMulAction R' N] [CompatibleSMul R R' M N] (r : R') (x : M) (y : N) :
    x ⊗ₜ (r • y) = r • x ⊗ₜ[R] y :=
  (smul_tmul _ _ _).symm


theorem smul_tmul_smul (r s : R) (m : M) (n : N) : (r • m) ⊗ₜ[R] (s • n) = (r * s) • m ⊗ₜ[R] n := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    r s : R
    m : M
    n : N
    ⊢ Eq (TensorProduct.tmul R (HSMul.hSMul r m) (HSMul.hSMul s n)) (HSMul.hSMul ( …
  -/
  simp_rw [smul_tmul, tmul_smul, mul_smul]
  /-
    🎉 no goals
  -/


instance leftModule : Module R'' (M ⊗[R] N) :=
  { add_smul := TensorProduct.add_smul
    zero_smul := TensorProduct.zero_smul }


instance : Module R (M ⊗[R] N) :=
  TensorProduct.leftModule


instance [Module R''ᵐᵒᵖ M] [IsCentralScalar R'' M] : IsCentralScalar R'' (M ⊗[R] N) where
  op_smul_eq_smul r x :=
                       /-
                         R : Type u_1
                         inst✝¹⁹ : CommSemiring R
                         R' : Type u_2
                         inst✝¹⁸ : Monoid R'
                         R'' : Type u_3
                         inst✝¹⁷ : Semiring R''
                         A : Type u_4
                         M : Type u_5
                         N : Type u_6
                         P : Type u_7
                         Q : Type u_8
                         S : Type u_9
                         T : Type u_10
                         inst✝¹⁶ : AddCommMonoid M
                         inst✝¹⁵ : AddCommMonoid N
                         inst✝¹⁴ : AddCommMonoid P
                         inst✝¹³ : AddCommMonoid Q
                         inst✝¹² : AddCommMonoid S
                         inst✝¹¹ : AddCommMonoid T
                         inst✝¹⁰ : Module R M
                         inst✝⁹ : Module R N
                         inst✝⁸ : Module R Q
                         inst✝⁷ : Module R S
                         inst✝⁶ : Module R T
                         inst✝⁵ : DistribMulAction R' M
                         inst✝⁴ : Module R'' M
                         inst✝³ : SMulCommClass R R' M
                         inst✝² : SMulCommClass R R'' M
                         inst✝¹ : Module (MulOpposite R'') M
                         inst✝ : IsCentralScalar R'' M
                         r : R''
                         x : TensorProduct R M N
                         ⊢ Eq (HSMul.hSMul (MulOpposite.op r) 0) (HSMul.hSMul r 0)
                       -/
    x.induction_on (by rw [smul_zero, smul_zero])
                       /-
                         🎉 no goals
                       -/
                     /-
                       R : Type u_1
                       inst✝¹⁹ : CommSemiring R
                       R' : Type u_2
                       inst✝¹⁸ : Monoid R'
                       R'' : Type u_3
                       inst✝¹⁷ : Semiring R''
                       A : Type u_4
                       M : Type u_5
                       N : Type u_6
                       P : Type u_7
                       Q : Type u_8
                       S : Type u_9
                       T : Type u_10
                       inst✝¹⁶ : AddCommMonoid M
                       inst✝¹⁵ : AddCommMonoid N
                       inst✝¹⁴ : AddCommMonoid P
                       inst✝¹³ : AddCommMonoid Q
                       inst✝¹² : AddCommMonoid S
                       inst✝¹¹ : AddCommMonoid T
                       inst✝¹⁰ : Module R M
                       inst✝⁹ : Module R N
                       inst✝⁸ : Module R Q
                       inst✝⁷ : Module R S
                       inst✝⁶ : Module R T
                       inst✝⁵ : DistribMulAction R' M
                       inst✝⁴ : Module R'' M
                       inst✝³ : SMulCommClass R R' M
                       inst✝² : SMulCommClass R R'' M
                       inst✝¹ : Module (MulOpposite R'') M
                       inst✝ : IsCentralScalar R'' M
                       r : R''
                       x✝ : TensorProduct R M N
                       x : M
                       y : N
                       ⊢ Eq (HSMul.hSMul (MulOpposite.op r) (TensorProduct.tmul R x y)) (HSMul.hSMul  …
                     -/
      (fun x y => by rw [smul_tmul', smul_tmul', op_smul_eq_smul]) fun x y hx hy => by
                     /-
                       🎉 no goals
                     -/
      /-
        R : Type u_1
        inst✝¹⁹ : CommSemiring R
        R' : Type u_2
        inst✝¹⁸ : Monoid R'
        R'' : Type u_3
        inst✝¹⁷ : Semiring R''
        A : Type u_4
        M : Type u_5
        N : Type u_6
        P : Type u_7
        Q : Type u_8
        S : Type u_9
        T : Type u_10
        inst✝¹⁶ : AddCommMonoid M
        inst✝¹⁵ : AddCommMonoid N
        inst✝¹⁴ : AddCommMonoid P
        inst✝¹³ : AddCommMonoid Q
        inst✝¹² : AddCommMonoid S
        inst✝¹¹ : AddCommMonoid T
        inst✝¹⁰ : Module R M
        inst✝⁹ : Module R N
        inst✝⁸ : Module R Q
        inst✝⁷ : Module R S
        inst✝⁶ : Module R T
        inst✝⁵ : DistribMulAction R' M
        inst✝⁴ : Module R'' M
        inst✝³ : SMulCommClass R R' M
        inst✝² : SMulCommClass R R'' M
        inst✝¹ : Module (MulOpposite R'') M
        inst✝ : IsCentralScalar R'' M
        r : R''
        x✝ x y : TensorProduct R M N
        hx : Eq (HSMul.hSMul (MulOpposite.op r) x) (HSMul.hSMul r x)
        hy : Eq (HSMul.hSMul (MulOpposite.op r) y) (HSMul.hSMul r y)
        ⊢ Eq (HSMul.hSMul (MulOpposite.op r) (HAdd.hAdd x y)) (HSMul.hSMul r (HAdd.hAd …
      -/
      rw [smul_add, smul_add, hx, hy]
      /-
        🎉 no goals
      -/


/-- `SMulCommClass R' R'₂ M` implies `SMulCommClass R' R'₂ (M ⊗[R] N)` -/
instance smulCommClass_left [SMulCommClass R' R'₂ M] : SMulCommClass R' R'₂ (M ⊗[R] N) where
  smul_comm r' r'₂ x :=
                                     /-
                                       R : Type u_1
                                       inst✝²¹ : CommSemiring R
                                       R' : Type u_2
                                       inst✝²⁰ : Monoid R'
                                       R'' : Type u_3
                                       inst✝¹⁹ : Semiring R''
                                       A : Type u_4
                                       M : Type u_5
                                       N : Type u_6
                                       P : Type u_7
                                       Q : Type u_8
                                       S : Type u_9
                                       T : Type u_10
                                       inst✝¹⁸ : AddCommMonoid M
                                       inst✝¹⁷ : AddCommMonoid N
                                       inst✝¹⁶ : AddCommMonoid P
                                       inst✝¹⁵ : AddCommMonoid Q
                                       inst✝¹⁴ : AddCommMonoid S
                                       inst✝¹³ : AddCommMonoid T
                                       inst✝¹² : Module R M
                                       inst✝¹¹ : Module R N
                                       inst✝¹⁰ : Module R Q
                                       inst✝⁹ : Module R S
                                       inst✝⁸ : Module R T
                                       inst✝⁷ : DistribMulAction R' M
                                       inst✝⁶ : Module R'' M
                                       inst✝⁵ : SMulCommClass R R' M
                                       inst✝⁴ : SMulCommClass R R'' M
                                       R'₂ : Type u_11
                                       inst✝³ : Monoid R'₂
                                       inst✝² : DistribMulAction R'₂ M
                                       inst✝¹ : SMulCommClass R R'₂ M
                                       inst✝ : SMulCommClass R' R'₂ M
                                       r' : R'
                                       r'₂ : R'₂
                                       x : TensorProduct R M N
                                       ⊢ Eq (HSMul.hSMul r' (HSMul.hSMul r'₂ 0)) (HSMul.hSMul r'₂ (HSMul.hSMul r' 0))
                                     -/
    TensorProduct.induction_on x (by simp_rw [TensorProduct.smul_zero])
                                     /-
                                       🎉 no goals
                                     -/
                     /-
                       R : Type u_1
                       inst✝²¹ : CommSemiring R
                       R' : Type u_2
                       inst✝²⁰ : Monoid R'
                       R'' : Type u_3
                       inst✝¹⁹ : Semiring R''
                       A : Type u_4
                       M : Type u_5
                       N : Type u_6
                       P : Type u_7
                       Q : Type u_8
                       S : Type u_9
                       T : Type u_10
                       inst✝¹⁸ : AddCommMonoid M
                       inst✝¹⁷ : AddCommMonoid N
                       inst✝¹⁶ : AddCommMonoid P
                       inst✝¹⁵ : AddCommMonoid Q
                       inst✝¹⁴ : AddCommMonoid S
                       inst✝¹³ : AddCommMonoid T
                       inst✝¹² : Module R M
                       inst✝¹¹ : Module R N
                       inst✝¹⁰ : Module R Q
                       inst✝⁹ : Module R S
                       inst✝⁸ : Module R T
                       inst✝⁷ : DistribMulAction R' M
                       inst✝⁶ : Module R'' M
                       inst✝⁵ : SMulCommClass R R' M
                       inst✝⁴ : SMulCommClass R R'' M
                       R'₂ : Type u_11
                       inst✝³ : Monoid R'₂
                       inst✝² : DistribMulAction R'₂ M
                       inst✝¹ : SMulCommClass R R'₂ M
                       inst✝ : SMulCommClass R' R'₂ M
                       r' : R'
                       r'₂ : R'₂
                       x : TensorProduct R M N
                       m : M
                       n : N
                       ⊢ Eq (HSMul.hSMul r' (HSMul.hSMul r'₂ (TensorProduct.tmul R m n))) (HSMul.hSMu …
                     -/
      (fun m n => by simp_rw [smul_tmul', smul_comm]) fun x y ihx ihy => by
                     /-
                       🎉 no goals
                     -/
      /-
        R : Type u_1
        inst✝²¹ : CommSemiring R
        R' : Type u_2
        inst✝²⁰ : Monoid R'
        R'' : Type u_3
        inst✝¹⁹ : Semiring R''
        A : Type u_4
        M : Type u_5
        N : Type u_6
        P : Type u_7
        Q : Type u_8
        S : Type u_9
        T : Type u_10
        inst✝¹⁸ : AddCommMonoid M
        inst✝¹⁷ : AddCommMonoid N
        inst✝¹⁶ : AddCommMonoid P
        inst✝¹⁵ : AddCommMonoid Q
        inst✝¹⁴ : AddCommMonoid S
        inst✝¹³ : AddCommMonoid T
        inst✝¹² : Module R M
        inst✝¹¹ : Module R N
        inst✝¹⁰ : Module R Q
        inst✝⁹ : Module R S
        inst✝⁸ : Module R T
        inst✝⁷ : DistribMulAction R' M
        inst✝⁶ : Module R'' M
        inst✝⁵ : SMulCommClass R R' M
        inst✝⁴ : SMulCommClass R R'' M
        R'₂ : Type u_11
        inst✝³ : Monoid R'₂
        inst✝² : DistribMulAction R'₂ M
        inst✝¹ : SMulCommClass R R'₂ M
        inst✝ : SMulCommClass R' R'₂ M
        r' : R'
        r'₂ : R'₂
        x✝ x y : TensorProduct R M N
        ihx : Eq (HSMul.hSMul r' (HSMul.hSMul r'₂ x)) (HSMul.hSMul r'₂ (HSMul.hSMul r' …
        ihy : Eq (HSMul.hSMul r' (HSMul.hSMul r'₂ y)) (HSMul.hSMul r'₂ (HSMul.hSMul r' …
        ⊢ Eq (HSMul.hSMul r' (HSMul.hSMul r'₂ (HAdd.hAdd x y))) (HSMul.hSMul r'₂ (HSMu …
      -/
      simp_rw [TensorProduct.smul_add]; rw [ihx, ihy]
                                        /-
                                          🎉 no goals
                                        -/


/-- `IsScalarTower R'₂ R' M` implies `IsScalarTower R'₂ R' (M ⊗[R] N)` -/
instance isScalarTower_left [IsScalarTower R'₂ R' M] : IsScalarTower R'₂ R' (M ⊗[R] N) :=
  ⟨fun s r x =>
                       /-
                         R : Type u_1
                         inst✝²² : CommSemiring R
                         R' : Type u_2
                         inst✝²¹ : Monoid R'
                         R'' : Type u_3
                         inst✝²⁰ : Semiring R''
                         A : Type u_4
                         M : Type u_5
                         N : Type u_6
                         P : Type u_7
                         Q : Type u_8
                         S : Type u_9
                         T : Type u_10
                         inst✝¹⁹ : AddCommMonoid M
                         inst✝¹⁸ : AddCommMonoid N
                         inst✝¹⁷ : AddCommMonoid P
                         inst✝¹⁶ : AddCommMonoid Q
                         inst✝¹⁵ : AddCommMonoid S
                         inst✝¹⁴ : AddCommMonoid T
                         inst✝¹³ : Module R M
                         inst✝¹² : Module R N
                         inst✝¹¹ : Module R Q
                         inst✝¹⁰ : Module R S
                         inst✝⁹ : Module R T
                         inst✝⁸ : DistribMulAction R' M
                         inst✝⁷ : Module R'' M
                         inst✝⁶ : SMulCommClass R R' M
                         inst✝⁵ : SMulCommClass R R'' M
                         R'₂ : Type u_11
                         inst✝⁴ : Monoid R'₂
                         inst✝³ : DistribMulAction R'₂ M
                         inst✝² : SMulCommClass R R'₂ M
                         inst✝¹ : SMul R'₂ R'
                         inst✝ : IsScalarTower R'₂ R' M
                         s : R'₂
                         r : R'
                         x : TensorProduct R M N
                         ⊢ Eq (HSMul.hSMul (HSMul.hSMul s r) 0) (HSMul.hSMul s (HSMul.hSMul r 0))
                       -/
    x.induction_on (by simp)
                       /-
                         🎉 no goals
                       -/
                     /-
                       R : Type u_1
                       inst✝²² : CommSemiring R
                       R' : Type u_2
                       inst✝²¹ : Monoid R'
                       R'' : Type u_3
                       inst✝²⁰ : Semiring R''
                       A : Type u_4
                       M : Type u_5
                       N : Type u_6
                       P : Type u_7
                       Q : Type u_8
                       S : Type u_9
                       T : Type u_10
                       inst✝¹⁹ : AddCommMonoid M
                       inst✝¹⁸ : AddCommMonoid N
                       inst✝¹⁷ : AddCommMonoid P
                       inst✝¹⁶ : AddCommMonoid Q
                       inst✝¹⁵ : AddCommMonoid S
                       inst✝¹⁴ : AddCommMonoid T
                       inst✝¹³ : Module R M
                       inst✝¹² : Module R N
                       inst✝¹¹ : Module R Q
                       inst✝¹⁰ : Module R S
                       inst✝⁹ : Module R T
                       inst✝⁸ : DistribMulAction R' M
                       inst✝⁷ : Module R'' M
                       inst✝⁶ : SMulCommClass R R' M
                       inst✝⁵ : SMulCommClass R R'' M
                       R'₂ : Type u_11
                       inst✝⁴ : Monoid R'₂
                       inst✝³ : DistribMulAction R'₂ M
                       inst✝² : SMulCommClass R R'₂ M
                       inst✝¹ : SMul R'₂ R'
                       inst✝ : IsScalarTower R'₂ R' M
                       s : R'₂
                       r : R'
                       x : TensorProduct R M N
                       m : M
                       n : N
                       ⊢ Eq (HSMul.hSMul (HSMul.hSMul s r) (TensorProduct.tmul R m n)) (HSMul.hSMul s …
                     -/
      (fun m n => by rw [smul_tmul', smul_tmul', smul_tmul', smul_assoc]) fun x y ihx ihy => by
                     /-
                       🎉 no goals
                     -/
      /-
        R : Type u_1
        inst✝²² : CommSemiring R
        R' : Type u_2
        inst✝²¹ : Monoid R'
        R'' : Type u_3
        inst✝²⁰ : Semiring R''
        A : Type u_4
        M : Type u_5
        N : Type u_6
        P : Type u_7
        Q : Type u_8
        S : Type u_9
        T : Type u_10
        inst✝¹⁹ : AddCommMonoid M
        inst✝¹⁸ : AddCommMonoid N
        inst✝¹⁷ : AddCommMonoid P
        inst✝¹⁶ : AddCommMonoid Q
        inst✝¹⁵ : AddCommMonoid S
        inst✝¹⁴ : AddCommMonoid T
        inst✝¹³ : Module R M
        inst✝¹² : Module R N
        inst✝¹¹ : Module R Q
        inst✝¹⁰ : Module R S
        inst✝⁹ : Module R T
        inst✝⁸ : DistribMulAction R' M
        inst✝⁷ : Module R'' M
        inst✝⁶ : SMulCommClass R R' M
        inst✝⁵ : SMulCommClass R R'' M
        R'₂ : Type u_11
        inst✝⁴ : Monoid R'₂
        inst✝³ : DistribMulAction R'₂ M
        inst✝² : SMulCommClass R R'₂ M
        inst✝¹ : SMul R'₂ R'
        inst✝ : IsScalarTower R'₂ R' M
        s : R'₂
        r : R'
        x✝ x y : TensorProduct R M N
        ihx : Eq (HSMul.hSMul (HSMul.hSMul s r) x) (HSMul.hSMul s (HSMul.hSMul r x))
        ihy : Eq (HSMul.hSMul (HSMul.hSMul s r) y) (HSMul.hSMul s (HSMul.hSMul r y))
        ⊢ Eq (HSMul.hSMul (HSMul.hSMul s r) (HAdd.hAdd x y)) (HSMul.hSMul s (HSMul.hSM …
      -/
      rw [smul_add, smul_add, smul_add, ihx, ihy]⟩
      /-
        🎉 no goals
      -/


/-- `IsScalarTower R'₂ R' N` implies `IsScalarTower R'₂ R' (M ⊗[R] N)` -/
instance isScalarTower_right [IsScalarTower R'₂ R' N] : IsScalarTower R'₂ R' (M ⊗[R] N) :=
  ⟨fun s r x =>
                       /-
                         R : Type u_1
                         inst✝²⁶ : CommSemiring R
                         R' : Type u_2
                         inst✝²⁵ : Monoid R'
                         R'' : Type u_3
                         inst✝²⁴ : Semiring R''
                         A : Type u_4
                         M : Type u_5
                         N : Type u_6
                         P : Type u_7
                         Q : Type u_8
                         S : Type u_9
                         T : Type u_10
                         inst✝²³ : AddCommMonoid M
                         inst✝²² : AddCommMonoid N
                         inst✝²¹ : AddCommMonoid P
                         inst✝²⁰ : AddCommMonoid Q
                         inst✝¹⁹ : AddCommMonoid S
                         inst✝¹⁸ : AddCommMonoid T
                         inst✝¹⁷ : Module R M
                         inst✝¹⁶ : Module R N
                         inst✝¹⁵ : Module R Q
                         inst✝¹⁴ : Module R S
                         inst✝¹³ : Module R T
                         inst✝¹² : DistribMulAction R' M
                         inst✝¹¹ : Module R'' M
                         inst✝¹⁰ : SMulCommClass R R' M
                         inst✝⁹ : SMulCommClass R R'' M
                         R'₂ : Type u_11
                         inst✝⁸ : Monoid R'₂
                         inst✝⁷ : DistribMulAction R'₂ M
                         inst✝⁶ : SMulCommClass R R'₂ M
                         inst✝⁵ : SMul R'₂ R'
                         inst✝⁴ : DistribMulAction R'₂ N
                         inst✝³ : DistribMulAction R' N
                         inst✝² : TensorProduct.CompatibleSMul R R'₂ M N
                         inst✝¹ : TensorProduct.CompatibleSMul R R' M N
                         inst✝ : IsScalarTower R'₂ R' N
                         s : R'₂
                         r : R'
                         x : TensorProduct R M N
                         ⊢ Eq (HSMul.hSMul (HSMul.hSMul s r) 0) (HSMul.hSMul s (HSMul.hSMul r 0))
                       -/
    x.induction_on (by simp)
                       /-
                         🎉 no goals
                       -/
                     /-
                       R : Type u_1
                       inst✝²⁶ : CommSemiring R
                       R' : Type u_2
                       inst✝²⁵ : Monoid R'
                       R'' : Type u_3
                       inst✝²⁴ : Semiring R''
                       A : Type u_4
                       M : Type u_5
                       N : Type u_6
                       P : Type u_7
                       Q : Type u_8
                       S : Type u_9
                       T : Type u_10
                       inst✝²³ : AddCommMonoid M
                       inst✝²² : AddCommMonoid N
                       inst✝²¹ : AddCommMonoid P
                       inst✝²⁰ : AddCommMonoid Q
                       inst✝¹⁹ : AddCommMonoid S
                       inst✝¹⁸ : AddCommMonoid T
                       inst✝¹⁷ : Module R M
                       inst✝¹⁶ : Module R N
                       inst✝¹⁵ : Module R Q
                       inst✝¹⁴ : Module R S
                       inst✝¹³ : Module R T
                       inst✝¹² : DistribMulAction R' M
                       inst✝¹¹ : Module R'' M
                       inst✝¹⁰ : SMulCommClass R R' M
                       inst✝⁹ : SMulCommClass R R'' M
                       R'₂ : Type u_11
                       inst✝⁸ : Monoid R'₂
                       inst✝⁷ : DistribMulAction R'₂ M
                       inst✝⁶ : SMulCommClass R R'₂ M
                       inst✝⁵ : SMul R'₂ R'
                       inst✝⁴ : DistribMulAction R'₂ N
                       inst✝³ : DistribMulAction R' N
                       inst✝² : TensorProduct.CompatibleSMul R R'₂ M N
                       inst✝¹ : TensorProduct.CompatibleSMul R R' M N
                       inst✝ : IsScalarTower R'₂ R' N
                       s : R'₂
                       r : R'
                       x : TensorProduct R M N
                       m : M
                       n : N
                       ⊢ Eq (HSMul.hSMul (HSMul.hSMul s r) (TensorProduct.tmul R m n)) (HSMul.hSMul s …
                     -/
      (fun m n => by rw [← tmul_smul, ← tmul_smul, ← tmul_smul, smul_assoc]) fun x y ihx ihy => by
                     /-
                       🎉 no goals
                     -/
      /-
        R : Type u_1
        inst✝²⁶ : CommSemiring R
        R' : Type u_2
        inst✝²⁵ : Monoid R'
        R'' : Type u_3
        inst✝²⁴ : Semiring R''
        A : Type u_4
        M : Type u_5
        N : Type u_6
        P : Type u_7
        Q : Type u_8
        S : Type u_9
        T : Type u_10
        inst✝²³ : AddCommMonoid M
        inst✝²² : AddCommMonoid N
        inst✝²¹ : AddCommMonoid P
        inst✝²⁰ : AddCommMonoid Q
        inst✝¹⁹ : AddCommMonoid S
        inst✝¹⁸ : AddCommMonoid T
        inst✝¹⁷ : Module R M
        inst✝¹⁶ : Module R N
        inst✝¹⁵ : Module R Q
        inst✝¹⁴ : Module R S
        inst✝¹³ : Module R T
        inst✝¹² : DistribMulAction R' M
        inst✝¹¹ : Module R'' M
        inst✝¹⁰ : SMulCommClass R R' M
        inst✝⁹ : SMulCommClass R R'' M
        R'₂ : Type u_11
        inst✝⁸ : Monoid R'₂
        inst✝⁷ : DistribMulAction R'₂ M
        inst✝⁶ : SMulCommClass R R'₂ M
        inst✝⁵ : SMul R'₂ R'
        inst✝⁴ : DistribMulAction R'₂ N
        inst✝³ : DistribMulAction R' N
        inst✝² : TensorProduct.CompatibleSMul R R'₂ M N
        inst✝¹ : TensorProduct.CompatibleSMul R R' M N
        inst✝ : IsScalarTower R'₂ R' N
        s : R'₂
        r : R'
        x✝ x y : TensorProduct R M N
        ihx : Eq (HSMul.hSMul (HSMul.hSMul s r) x) (HSMul.hSMul s (HSMul.hSMul r x))
        ihy : Eq (HSMul.hSMul (HSMul.hSMul s r) y) (HSMul.hSMul s (HSMul.hSMul r y))
        ⊢ Eq (HSMul.hSMul (HSMul.hSMul s r) (HAdd.hAdd x y)) (HSMul.hSMul s (HSMul.hSM …
      -/
      rw [smul_add, smul_add, smul_add, ihx, ihy]⟩
      /-
        🎉 no goals
      -/


/-- A short-cut instance for the common case, where the requirements for the `compatible_smul`
instances are sufficient. -/
instance isScalarTower [SMul R' R] [IsScalarTower R' R M] : IsScalarTower R' R (M ⊗[R] N) :=
  TensorProduct.isScalarTower_left

-- or right

/-- The canonical bilinear map `M → N → M ⊗[R] N`. -/
def mk : M →ₗ[R] N →ₗ[R] M ⊗[R] N :=
                                                     /-
                                                       R : Type u_1
                                                       inst✝¹⁷ : CommSemiring R
                                                       R' : Type u_2
                                                       inst✝¹⁶ : Monoid R'
                                                       R'' : Type u_3
                                                       inst✝¹⁵ : Semiring R''
                                                       A : Type u_4
                                                       M : Type u_5
                                                       N : Type u_6
                                                       P : Type u_7
                                                       Q : Type u_8
                                                       S : Type u_9
                                                       T : Type u_10
                                                       inst✝¹⁴ : AddCommMonoid M
                                                       inst✝¹³ : AddCommMonoid N
                                                       inst✝¹² : AddCommMonoid P
                                                       inst✝¹¹ : AddCommMonoid Q
                                                       inst✝¹⁰ : AddCommMonoid S
                                                       inst✝⁹ : AddCommMonoid T
                                                       inst✝⁸ : Module R M
                                                       inst✝⁷ : Module R N
                                                       inst✝⁶ : Module R Q
                                                       inst✝⁵ : Module R S
                                                       inst✝⁴ : Module R T
                                                       inst✝³ : DistribMulAction R' M
                                                       inst✝² : Module R'' M
                                                       inst✝¹ : SMulCommClass R R' M
                                                       inst✝ : SMulCommClass R R'' M
                                                       c : R
                                                       m : M
                                                       n : N
                                                       ⊢ Eq ((fun x1 x2 => TensorProduct.tmul R x1 x2) (HSMul.hSMul c m) n) (HSMul.hS …
                                                     -/
  LinearMap.mk₂ R (· ⊗ₜ ·) add_tmul (fun c m n => by simp_rw [smul_tmul, tmul_smul])
                                                     /-
                                                       🎉 no goals
                                                     -/
    tmul_add tmul_smul


@[simp]
theorem mk_apply (m : M) (n : N) : mk R M N m n = m ⊗ₜ n :=
  rfl


theorem ite_tmul (x₁ : M) (x₂ : N) (P : Prop) [Decidable P] :
                                                                     /-
                                                                       R : Type u_1
                                                                       inst✝⁵ : CommSemiring R
                                                                       M : Type u_5
                                                                       N : Type u_6
                                                                       inst✝⁴ : AddCommMonoid M
                                                                       inst✝³ : AddCommMonoid N
                                                                       inst✝² : Module R M
                                                                       inst✝¹ : Module R N
                                                                       x₁ : M
                                                                       x₂ : N
                                                                       P : Prop
                                                                       inst✝ : Decidable P
                                                                       ⊢ Eq (TensorProduct.tmul R (ite P x₁ 0) x₂) (ite P (TensorProduct.tmul R x₁ x₂ …
                                                                     -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
    (if P then x₁ else 0) ⊗ₜ[R] x₂ = if P then x₁ ⊗ₜ x₂ else 0 := by split_ifs <;> simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem tmul_ite (x₁ : M) (x₂ : N) (P : Prop) [Decidable P] :
                                                                     /-
                                                                       R : Type u_1
                                                                       inst✝⁵ : CommSemiring R
                                                                       M : Type u_5
                                                                       N : Type u_6
                                                                       inst✝⁴ : AddCommMonoid M
                                                                       inst✝³ : AddCommMonoid N
                                                                       inst✝² : Module R M
                                                                       inst✝¹ : Module R N
                                                                       x₁ : M
                                                                       x₂ : N
                                                                       P : Prop
                                                                       inst✝ : Decidable P
                                                                       ⊢ Eq (TensorProduct.tmul R x₁ (ite P x₂ 0)) (ite P (TensorProduct.tmul R x₁ x₂ …
                                                                     -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
    (x₁ ⊗ₜ[R] if P then x₂ else 0) = if P then x₁ ⊗ₜ x₂ else 0 := by split_ifs <;> simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


lemma tmul_single {ι : Type*} [DecidableEq ι] {M : ι → Type*} [∀ i, AddCommMonoid (M i)]
    [∀ i, Module R (M i)] (i : ι) (x : N) (m : M i) (j : ι) :
    x ⊗ₜ[R] Pi.single i m j = (Pi.single i (x ⊗ₜ[R] m) : ∀ i, N ⊗[R] M i) j := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    N : Type u_6
    inst✝⁴ : AddCommMonoid N
    inst✝³ : Module R N
    ι : Type u_11
    inst✝² : DecidableEq ι
    M : ι → Type u_12
    inst✝¹ : (i : ι) → AddCommMonoid (M i)
    inst✝ : (i : ι) → Module R (M i)
    i : ι
    x : N
    m : M i
    j : ι
    ⊢ Eq (TensorProduct.tmul R x (Pi.single i m j)) (Pi.single i (TensorProduct.tm …
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : i = j <;> aesop
                         /-
                           🎉 no goals
                         -/


lemma single_tmul {ι : Type*} [DecidableEq ι] {M : ι → Type*} [∀ i, AddCommMonoid (M i)]
    [∀ i, Module R (M i)] (i : ι) (x : N) (m : M i) (j : ι) :
    Pi.single i m j ⊗ₜ[R] x = (Pi.single i (m ⊗ₜ[R] x) : ∀ i, M i ⊗[R] N) j := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    N : Type u_6
    inst✝⁴ : AddCommMonoid N
    inst✝³ : Module R N
    ι : Type u_11
    inst✝² : DecidableEq ι
    M : ι → Type u_12
    inst✝¹ : (i : ι) → AddCommMonoid (M i)
    inst✝ : (i : ι) → Module R (M i)
    i : ι
    x : N
    m : M i
    j : ι
    ⊢ Eq (TensorProduct.tmul R (Pi.single i m j) x) (Pi.single i (TensorProduct.tm …
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : i = j <;> aesop
                         /-
                           🎉 no goals
                         -/


theorem sum_tmul {α : Type*} (s : Finset α) (m : α → M) (n : N) :
    (∑ a ∈ s, m a) ⊗ₜ[R] n = ∑ a ∈ s, m a ⊗ₜ[R] n := by
  classical
    induction' s using Finset.induction with a s has ih h
    · simp
    · simp [Finset.sum_insert has, add_tmul, ih]


theorem tmul_sum (m : M) {α : Type*} (s : Finset α) (n : α → N) :
    (m ⊗ₜ[R] ∑ a ∈ s, n a) = ∑ a ∈ s, m ⊗ₜ[R] n a := by
  classical
    induction' s using Finset.induction with a s has ih h
    · simp
    · simp [Finset.sum_insert has, tmul_add, ih]


/-- The simple (aka pure) elements span the tensor product. -/
theorem span_tmul_eq_top : Submodule.span R { t : M ⊗[R] N | ∃ m n, m ⊗ₜ n = t } = ⊤ := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    ⊢ Eq (Submodule.span R (setOf fun t => Exists fun m => Exists fun n => Eq (Ten …
  -/
  ext t; simp only [Submodule.mem_top, iff_true]
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    t : TensorProduct R M N
    ⊢ Membership.mem (Submodule.span R (setOf fun t => Exists fun m => Exists fun  …
  -/
  refine t.induction_on ?_ ?_ ?_
    /-
      case h.refine_1
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Type u_5
      N : Type u_6
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R M
      inst✝ : Module R N
      t : TensorProduct R M N
      ⊢ Membership.mem (Submodule.span R (setOf fun t => Exists fun m => Exists fun  …
    -/
  · exact Submodule.zero_mem _
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Type u_5
      N : Type u_6
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R M
      inst✝ : Module R N
      t : TensorProduct R M N
      ⊢ ∀ (x : M) (y : N), Membership.mem (Submodule.span R (setOf fun t => Exists f …
    -/
  · intro m n
    /-
      case h.refine_2
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Type u_5
      N : Type u_6
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R M
      inst✝ : Module R N
      t : TensorProduct R M N
      m : M
      n : N
      ⊢ Membership.mem (Submodule.span R (setOf fun t => Exists fun m => Exists fun  …
    -/
    apply Submodule.subset_span
    /-
      case h.refine_2.a
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Type u_5
      N : Type u_6
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R M
      inst✝ : Module R N
      t : TensorProduct R M N
      m : M
      n : N
      ⊢ Membership.mem (setOf fun t => Exists fun m => Exists fun n => Eq (TensorPro …
    -/
    use m, n
    /-
      🎉 no goals
    -/
    /-
      case h.refine_3
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Type u_5
      N : Type u_6
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R M
      inst✝ : Module R N
      t : TensorProduct R M N
      ⊢ ∀ (x y : TensorProduct R M N), Membership.mem (Submodule.span R (setOf fun t …
    -/
  · intro t₁ t₂ ht₁ ht₂
    /-
      case h.refine_3
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Type u_5
      N : Type u_6
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R M
      inst✝ : Module R N
      t t₁ t₂ : TensorProduct R M N
      ht₁ : Membership.mem (Submodule.span R (setOf fun t => Exists fun m => Exists  …
      ht₂ : Membership.mem (Submodule.span R (setOf fun t => Exists fun m => Exists  …
      ⊢ Membership.mem (Submodule.span R (setOf fun t => Exists fun m => Exists fun  …
    -/
    exact Submodule.add_mem _ ht₁ ht₂
    /-
      🎉 no goals
    -/


@[simp]
theorem map₂_mk_top_top_eq_top : Submodule.map₂ (mk R M N) ⊤ ⊤ = ⊤ := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    ⊢ Eq (Submodule.map₂ (TensorProduct.mk R M N) Top.top Top.top) Top.top
  -/
  rw [← top_le_iff, ← span_tmul_eq_top, Submodule.map₂_eq_span_image2]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    ⊢ LE.le (Submodule.span R (setOf fun t => Exists fun m => Exists fun n => Eq ( …
  -/
  exact Submodule.span_mono fun _ ⟨m, n, h⟩ => ⟨m, trivial, n, trivial, h⟩
  /-
    🎉 no goals
  -/


theorem exists_eq_tmul_of_forall (x : TensorProduct R M N)
    (h : ∀ (m₁ m₂ : M) (n₁ n₂ : N), ∃ m n, m₁ ⊗ₜ n₁ + m₂ ⊗ₜ n₂ = m ⊗ₜ[R] n) :
    ∃ m n, x = m ⊗ₜ n := by
  induction x with
  | zero =>
    use 0, 0
    rw [TensorProduct.zero_tmul]
  | tmul m n => use m, n
  | add x y h₁ h₂ =>
    obtain ⟨m₁, n₁, rfl⟩ := h₁
    obtain ⟨m₂, n₂, rfl⟩ := h₂
    apply h


/-- Auxiliary function to constructing a linear map `M ⊗ N → P` given a bilinear map `M → N → P`
with the property that its composition with the canonical bilinear map `M → N → M ⊗ N` is
the given bilinear map `M → N → P`. -/
def liftAux : M ⊗[R] N →+ P :=
  liftAddHom (LinearMap.toAddMonoidHom'.comp <| f.toAddMonoidHom)
                    /-
                      R : Type u_1
                      inst✝¹⁶ : CommSemiring R
                      R' : Type u_2
                      inst✝¹⁵ : Monoid R'
                      R'' : Type u_3
                      inst✝¹⁴ : Semiring R''
                      A : Type u_4
                      M : Type u_5
                      N : Type u_6
                      P : Type u_7
                      Q : Type u_8
                      S : Type u_9
                      T : Type u_10
                      inst✝¹³ : AddCommMonoid M
                      inst✝¹² : AddCommMonoid N
                      inst✝¹¹ : AddCommMonoid P
                      inst✝¹⁰ : AddCommMonoid Q
                      inst✝⁹ : AddCommMonoid S
                      inst✝⁸ : AddCommMonoid T
                      inst✝⁷ : Module R M
                      inst✝⁶ : Module R N
                      inst✝⁵ : Module R Q
                      inst✝⁴ : Module R S
                      inst✝³ : Module R T
                      inst✝² : DistribMulAction R' M
                      inst✝¹ : Module R'' M
                      inst✝ : Module R P
                      f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
                      r : R
                      m : M
                      n : N
                      ⊢ Eq (((LinearMap.toAddMonoidHom'.comp f.toAddMonoidHom) (HSMul.hSMul r m)) n) …
                    -/
    fun r m n => by dsimp; rw [LinearMap.map_smul₂, map_smul]
                           /-
                             🎉 no goals
                           -/


theorem liftAux_tmul (m n) : liftAux f (m ⊗ₜ n) = f m n :=
  rfl


@[simp]
theorem liftAux.smul (r : R) (x) : liftAux f (r • x) = r • liftAux f x :=
  TensorProduct.induction_on x (smul_zero _).symm
                   /-
                     R : Type u_1
                     inst✝⁶ : CommSemiring R
                     M : Type u_5
                     N : Type u_6
                     P : Type u_7
                     inst✝⁵ : AddCommMonoid M
                     inst✝⁴ : AddCommMonoid N
                     inst✝³ : AddCommMonoid P
                     inst✝² : Module R M
                     inst✝¹ : Module R N
                     inst✝ : Module R P
                     f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
                     r : R
                     x : TensorProduct R M N
                     p : M
                     q : N
                     ⊢ Eq ((TensorProduct.liftAux f) (HSMul.hSMul r (TensorProduct.tmul R p q))) (H …
                   -/
    (fun p q => by simp_rw [← tmul_smul, liftAux_tmul, (f p).map_smul])
                   /-
                     🎉 no goals
                   -/
                          /-
                            R : Type u_1
                            inst✝⁶ : CommSemiring R
                            M : Type u_5
                            N : Type u_6
                            P : Type u_7
                            inst✝⁵ : AddCommMonoid M
                            inst✝⁴ : AddCommMonoid N
                            inst✝³ : AddCommMonoid P
                            inst✝² : Module R M
                            inst✝¹ : Module R N
                            inst✝ : Module R P
                            f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
                            r : R
                            x p q : TensorProduct R M N
                            ih1 : Eq ((TensorProduct.liftAux f) (HSMul.hSMul r p)) (HSMul.hSMul r ((Tensor …
                            ih2 : Eq ((TensorProduct.liftAux f) (HSMul.hSMul r q)) (HSMul.hSMul r ((Tensor …
                            ⊢ Eq ((TensorProduct.liftAux f) (HSMul.hSMul r (HAdd.hAdd p q))) (HSMul.hSMul  …
                          -/
    fun p q ih1 ih2 => by simp_rw [smul_add, (liftAux f).map_add, ih1, ih2, smul_add]
                          /-
                            🎉 no goals
                          -/


/-- Constructing a linear map `M ⊗ N → P` given a bilinear map `M → N → P` with the property that
its composition with the canonical bilinear map `M → N → M ⊗ N` is
the given bilinear map `M → N → P`. -/
def lift : M ⊗[R] N →ₗ[R] P :=
  { liftAux f with map_smul' := liftAux.smul }


@[simp]
theorem lift.tmul (x y) : lift f (x ⊗ₜ y) = f x y :=
  rfl


@[simp]
theorem lift.tmul' (x y) : (lift f).1 (x ⊗ₜ y) = f x y :=
  rfl


theorem ext' {g h : M ⊗[R] N →ₗ[R] P} (H : ∀ x y, g (x ⊗ₜ y) = h (x ⊗ₜ y)) : g = h :=
  LinearMap.ext fun z =>
                                     /-
                                       R : Type u_1
                                       inst✝⁶ : CommSemiring R
                                       M : Type u_5
                                       N : Type u_6
                                       P : Type u_7
                                       inst✝⁵ : AddCommMonoid M
                                       inst✝⁴ : AddCommMonoid N
                                       inst✝³ : AddCommMonoid P
                                       inst✝² : Module R M
                                       inst✝¹ : Module R N
                                       inst✝ : Module R P
                                       g h : LinearMap (RingHom.id R) (TensorProduct R M N) P
                                       H : ∀ (x : M) (y : N), Eq (g (TensorProduct.tmul R x y)) (h (TensorProduct.tmu …
                                       z : TensorProduct R M N
                                       ⊢ Eq (g 0) (h 0)
                                     -/
    TensorProduct.induction_on z (by simp_rw [LinearMap.map_zero]) H fun x y ihx ihy => by
                                     /-
                                       🎉 no goals
                                     -/
      /-
        R : Type u_1
        inst✝⁶ : CommSemiring R
        M : Type u_5
        N : Type u_6
        P : Type u_7
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : AddCommMonoid N
        inst✝³ : AddCommMonoid P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        g h : LinearMap (RingHom.id R) (TensorProduct R M N) P
        H : ∀ (x : M) (y : N), Eq (g (TensorProduct.tmul R x y)) (h (TensorProduct.tmu …
        z x y : TensorProduct R M N
        ihx : Eq (g x) (h x)
        ihy : Eq (g y) (h y)
        ⊢ Eq (g (HAdd.hAdd x y)) (h (HAdd.hAdd x y))
      -/
      rw [g.map_add, h.map_add, ihx, ihy]
      /-
        🎉 no goals
      -/


theorem lift.unique {g : M ⊗[R] N →ₗ[R] P} (H : ∀ x y, g (x ⊗ₜ y) = f x y) : g = lift f :=
                     /-
                       R : Type u_1
                       inst✝⁶ : CommSemiring R
                       M : Type u_5
                       N : Type u_6
                       P : Type u_7
                       inst✝⁵ : AddCommMonoid M
                       inst✝⁴ : AddCommMonoid N
                       inst✝³ : AddCommMonoid P
                       inst✝² : Module R M
                       inst✝¹ : Module R N
                       inst✝ : Module R P
                       f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
                       g : LinearMap (RingHom.id R) (TensorProduct R M N) P
                       H : ∀ (x : M) (y : N), Eq (g (TensorProduct.tmul R x y)) ((f x) y)
                       m : M
                       n : N
                       ⊢ Eq (g (TensorProduct.tmul R m n)) ((TensorProduct.lift f) (TensorProduct.tmu …
                     -/
  ext' fun m n => by rw [H, lift.tmul]
                     /-
                       🎉 no goals
                     -/


theorem lift_mk : lift (mk R M N) = LinearMap.id :=
  Eq.symm <| lift.unique fun _ _ => rfl


theorem lift_compr₂ (g : P →ₗ[R] Q) : lift (f.compr₂ g) = g.comp (lift f) :=
                                       /-
                                         R : Type u_1
                                         inst✝⁸ : CommSemiring R
                                         M : Type u_5
                                         N : Type u_6
                                         P : Type u_7
                                         Q : Type u_8
                                         inst✝⁷ : AddCommMonoid M
                                         inst✝⁶ : AddCommMonoid N
                                         inst✝⁵ : AddCommMonoid P
                                         inst✝⁴ : AddCommMonoid Q
                                         inst✝³ : Module R M
                                         inst✝² : Module R N
                                         inst✝¹ : Module R Q
                                         inst✝ : Module R P
                                         f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
                                         g : LinearMap (RingHom.id R) P Q
                                         x✝¹ : M
                                         x✝ : N
                                         ⊢ Eq ((g.comp (TensorProduct.lift f)) (TensorProduct.tmul R x✝¹ x✝)) (((f.comp …
                                       -/
  Eq.symm <| lift.unique fun _ _ => by simp
                                       /-
                                         🎉 no goals
                                       -/


theorem lift_mk_compr₂ (f : M ⊗ N →ₗ[R] P) : lift ((mk R M N).compr₂ f) = f := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) (TensorProduct R M N) P
    ⊢ Eq (TensorProduct.lift ((TensorProduct.mk R M N).compr₂ f)) f
  -/
  rw [lift_compr₂ f, lift_mk, LinearMap.comp_id]
  /-
    🎉 no goals
  -/


/-- This used to be an `@[ext]` lemma, but it fails very slowly when the `ext` tactic tries to apply
it in some cases, notably when one wants to show equality of two linear maps. The `@[ext]`
attribute is now added locally where it is needed. Using this as the `@[ext]` lemma instead of
`TensorProduct.ext'` allows `ext` to apply lemmas specific to `M →ₗ _` and `N →ₗ _`.

See note [partially-applied ext lemmas]. -/
theorem ext {g h : M ⊗ N →ₗ[R] P} (H : (mk R M N).compr₂ g = (mk R M N).compr₂ h) : g = h := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    g h : LinearMap (RingHom.id R) (TensorProduct R M N) P
    H : Eq ((TensorProduct.mk R M N).compr₂ g) ((TensorProduct.mk R M N).compr₂ h)
    ⊢ Eq g h
  -/
  rw [← lift_mk_compr₂ g, H, lift_mk_compr₂]
  /-
    🎉 no goals
  -/


attribute [local ext high] ext


/-- Linearly constructing a linear map `M ⊗ N → P` given a bilinear map `M → N → P`
with the property that its composition with the canonical bilinear map `M → N → M ⊗ N` is
the given bilinear map `M → N → P`. -/
def uncurry : (M →ₗ[R] N →ₗ[R] P) →ₗ[R] M ⊗[R] N →ₗ[R] P :=
  LinearMap.flip <| lift <| LinearMap.lflip.comp (LinearMap.flip LinearMap.id)


@[simp]
theorem uncurry_apply (f : M →ₗ[R] N →ₗ[R] P) (m : M) (n : N) :
                                             /-
                                               R : Type u_1
                                               inst✝⁶ : CommSemiring R
                                               M : Type u_5
                                               N : Type u_6
                                               P : Type u_7
                                               inst✝⁵ : AddCommMonoid M
                                               inst✝⁴ : AddCommMonoid N
                                               inst✝³ : AddCommMonoid P
                                               inst✝² : Module R M
                                               inst✝¹ : Module R N
                                               inst✝ : Module R P
                                               f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
                                               m : M
                                               n : N
                                               ⊢ Eq (((TensorProduct.uncurry R M N P) f) (TensorProduct.tmul R m n)) ((f m) n)
                                             -/
    uncurry R M N P f (m ⊗ₜ n) = f m n := by rw [uncurry, LinearMap.flip_apply, lift.tmul]; rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


/-- A linear equivalence constructing a linear map `M ⊗ N → P` given a bilinear map `M → N → P`
with the property that its composition with the canonical bilinear map `M → N → M ⊗ N` is
the given bilinear map `M → N → P`. -/
def lift.equiv : (M →ₗ[R] N →ₗ[R] P) ≃ₗ[R] M ⊗[R] N →ₗ[R] P :=
  { uncurry R M N P with
    invFun := fun f => (mk R M N).compr₂ f
    left_inv := fun _ => LinearMap.ext₂ fun _ _ => lift.tmul _ _
    right_inv := fun _ => ext' fun _ _ => lift.tmul _ _ }


@[simp]
theorem lift.equiv_apply (f : M →ₗ[R] N →ₗ[R] P) (m : M) (n : N) :
    lift.equiv R M N P f (m ⊗ₜ n) = f m n :=
  uncurry_apply f m n


@[simp]
theorem lift.equiv_symm_apply (f : M ⊗[R] N →ₗ[R] P) (m : M) (n : N) :
    (lift.equiv R M N P).symm f m n = f (m ⊗ₜ n) :=
  rfl


/-- Given a linear map `M ⊗ N → P`, compose it with the canonical bilinear map `M → N → M ⊗ N` to
form a bilinear map `M → N → P`. -/
def lcurry : (M ⊗[R] N →ₗ[R] P) →ₗ[R] M →ₗ[R] N →ₗ[R] P :=
  (lift.equiv R M N P).symm


@[simp]
theorem lcurry_apply (f : M ⊗[R] N →ₗ[R] P) (m : M) (n : N) : lcurry R M N P f m n = f (m ⊗ₜ n) :=
  rfl


/-- Given a linear map `M ⊗ N → P`, compose it with the canonical bilinear map `M → N → M ⊗ N` to
form a bilinear map `M → N → P`. -/
def curry (f : M ⊗[R] N →ₗ[R] P) : M →ₗ[R] N →ₗ[R] P :=
  lcurry R M N P f


@[simp]
theorem curry_apply (f : M ⊗ N →ₗ[R] P) (m : M) (n : N) : curry f m n = f (m ⊗ₜ n) :=
  rfl


theorem curry_injective : Function.Injective (curry : (M ⊗[R] N →ₗ[R] P) → M →ₗ[R] N →ₗ[R] P) :=
  fun _ _ H => ext H


theorem ext_threefold {g h : (M ⊗[R] N) ⊗[R] P →ₗ[R] Q}
    (H : ∀ x y z, g (x ⊗ₜ y ⊗ₜ z) = h (x ⊗ₜ y ⊗ₜ z)) : g = h := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    g h : LinearMap (RingHom.id R) (TensorProduct R (TensorProduct R M N) P) Q
    H : ∀ (x : M) (y : N) (z : P), Eq (g (TensorProduct.tmul R (TensorProduct.tmul …
    ⊢ Eq g h
  -/
  ext x y z
  /-
    case H.H.h.h.h
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    g h : LinearMap (RingHom.id R) (TensorProduct R (TensorProduct R M N) P) Q
    H : ∀ (x : M) (y : N) (z : P), Eq (g (TensorProduct.tmul R (TensorProduct.tmul …
    x : M
    y : N
    z : P
    ⊢ Eq (((((TensorProduct.mk R M N).compr₂ ((TensorProduct.mk R (TensorProduct R …
  -/
  exact H x y z
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")] alias ext₃ := ext_threefold

-- We'll need this one for checking the pentagon identity!

theorem ext_fourfold {g h : ((M ⊗[R] N) ⊗[R] P) ⊗[R] Q →ₗ[R] S}
    (H : ∀ w x y z, g (w ⊗ₜ x ⊗ₜ y ⊗ₜ z) = h (w ⊗ₜ x ⊗ₜ y ⊗ₜ z)) : g = h := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    S : Type u_9
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : AddCommMonoid P
    inst✝⁶ : AddCommMonoid Q
    inst✝⁵ : AddCommMonoid S
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R Q
    inst✝¹ : Module R S
    inst✝ : Module R P
    g h : LinearMap (RingHom.id R) (TensorProduct R (TensorProduct R (TensorProduc …
    H : ∀ (w : M) (x : N) (y : P) (z : Q), Eq (g (TensorProduct.tmul R (TensorProd …
    ⊢ Eq g h
  -/
  ext w x y z
  /-
    case H.H.H.h.h.h.h
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    S : Type u_9
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : AddCommMonoid P
    inst✝⁶ : AddCommMonoid Q
    inst✝⁵ : AddCommMonoid S
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R Q
    inst✝¹ : Module R S
    inst✝ : Module R P
    g h : LinearMap (RingHom.id R) (TensorProduct R (TensorProduct R (TensorProduc …
    H : ∀ (w : M) (x : N) (y : P) (z : Q), Eq (g (TensorProduct.tmul R (TensorProd …
    w : M
    x : N
    y : P
    z : Q
    ⊢ Eq ((((((TensorProduct.mk R M N).compr₂ ((TensorProduct.mk R (TensorProduct  …
  -/
  exact H w x y z
  /-
    🎉 no goals
  -/


/-- Two linear maps (M ⊗ N) ⊗ (P ⊗ Q) → S which agree on all elements of the
form (m ⊗ₜ n) ⊗ₜ (p ⊗ₜ q) are equal. -/
theorem ext_fourfold' {φ ψ : (M ⊗[R] N) ⊗[R] P ⊗[R] Q →ₗ[R] S}
    (H : ∀ w x y z, φ (w ⊗ₜ x ⊗ₜ (y ⊗ₜ z)) = ψ (w ⊗ₜ x ⊗ₜ (y ⊗ₜ z))) : φ = ψ := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    S : Type u_9
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : AddCommMonoid P
    inst✝⁶ : AddCommMonoid Q
    inst✝⁵ : AddCommMonoid S
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R Q
    inst✝¹ : Module R S
    inst✝ : Module R P
    φ ψ : LinearMap (RingHom.id R) (TensorProduct R (TensorProduct R M N) (TensorP …
    H : ∀ (w : M) (x : N) (y : P) (z : Q), Eq (φ (TensorProduct.tmul R (TensorProd …
    ⊢ Eq φ ψ
  -/
  ext m n p q
  /-
    case H.H.h.h.H.h.h
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    S : Type u_9
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : AddCommMonoid P
    inst✝⁶ : AddCommMonoid Q
    inst✝⁵ : AddCommMonoid S
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R Q
    inst✝¹ : Module R S
    inst✝ : Module R P
    φ ψ : LinearMap (RingHom.id R) (TensorProduct R (TensorProduct R M N) (TensorP …
    H : ∀ (w : M) (x : N) (y : P) (z : Q), Eq (φ (TensorProduct.tmul R (TensorProd …
    m : M
    n : N
    p : P
    q : Q
    ⊢ Eq ((((TensorProduct.mk R P Q).compr₂ ((((TensorProduct.mk R M N).compr₂ ((T …
  -/
  exact H m n p q
  /-
    🎉 no goals
  -/


/-- The base ring is a left identity for the tensor product of modules, up to linear equivalence.
-/
protected def lid : R ⊗[R] M ≃ₗ[R] M :=
                                                                                             /-
                                                                                               R : Type u_1
                                                                                               inst✝¹⁶ : CommSemiring R
                                                                                               R' : Type u_2
                                                                                               inst✝¹⁵ : Monoid R'
                                                                                               R'' : Type u_3
                                                                                               inst✝¹⁴ : Semiring R''
                                                                                               A : Type u_4
                                                                                               M : Type u_5
                                                                                               N : Type u_6
                                                                                               P : Type u_7
                                                                                               Q : Type u_8
                                                                                               S : Type u_9
                                                                                               T : Type u_10
                                                                                               inst✝¹³ : AddCommMonoid M
                                                                                               inst✝¹² : AddCommMonoid N
                                                                                               inst✝¹¹ : AddCommMonoid P
                                                                                               inst✝¹⁰ : AddCommMonoid Q
                                                                                               inst✝⁹ : AddCommMonoid S
                                                                                               inst✝⁸ : AddCommMonoid T
                                                                                               inst✝⁷ : Module R M
                                                                                               inst✝⁶ : Module R N
                                                                                               inst✝⁵ : Module R Q
                                                                                               inst✝⁴ : Module R S
                                                                                               inst✝³ : Module R T
                                                                                               inst✝² : DistribMulAction R' M
                                                                                               inst✝¹ : Module R'' M
                                                                                               inst✝ : Module R P
                                                                                               x✝ : M
                                                                                               ⊢ Eq (((TensorProduct.lift (LinearMap.lsmul R M)).comp ((TensorProduct.mk R R  …
                                                                                             -/
  LinearEquiv.ofLinear (lift <| LinearMap.lsmul R M) (mk R R M 1) (LinearMap.ext fun _ => by simp)
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/
                        /-
                          R : Type u_1
                          inst✝¹⁶ : CommSemiring R
                          R' : Type u_2
                          inst✝¹⁵ : Monoid R'
                          R'' : Type u_3
                          inst✝¹⁴ : Semiring R''
                          A : Type u_4
                          M : Type u_5
                          N : Type u_6
                          P : Type u_7
                          Q : Type u_8
                          S : Type u_9
                          T : Type u_10
                          inst✝¹³ : AddCommMonoid M
                          inst✝¹² : AddCommMonoid N
                          inst✝¹¹ : AddCommMonoid P
                          inst✝¹⁰ : AddCommMonoid Q
                          inst✝⁹ : AddCommMonoid S
                          inst✝⁸ : AddCommMonoid T
                          inst✝⁷ : Module R M
                          inst✝⁶ : Module R N
                          inst✝⁵ : Module R Q
                          inst✝⁴ : Module R S
                          inst✝³ : Module R T
                          inst✝² : DistribMulAction R' M
                          inst✝¹ : Module R'' M
                          inst✝ : Module R P
                          r : R
                          m : M
                          ⊢ Eq ((((TensorProduct.mk R R M) 1).comp (TensorProduct.lift (LinearMap.lsmul  …
                        -/
    (ext' fun r m => by simp; rw [← tmul_smul, ← smul_tmul, smul_eq_mul, mul_one])
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem lid_tmul (m : M) (r : R) : (TensorProduct.lid R M : R ⊗ M → M) (r ⊗ₜ m) = r • m :=
  rfl


@[simp]
theorem lid_symm_apply (m : M) : (TensorProduct.lid R M).symm m = 1 ⊗ₜ m :=
  rfl


/-- The tensor product of modules is commutative, up to linear equivalence.
-/
protected def comm : M ⊗[R] N ≃ₗ[R] N ⊗[R] M :=
  LinearEquiv.ofLinear (lift (mk R N M).flip) (lift (mk R M N).flip) (ext' fun _ _ => rfl)
    (ext' fun _ _ => rfl)


@[simp]
theorem comm_tmul (m : M) (n : N) : (TensorProduct.comm R M N) (m ⊗ₜ n) = n ⊗ₜ m :=
  rfl


@[simp]
theorem comm_symm_tmul (m : M) (n : N) : (TensorProduct.comm R M N).symm (n ⊗ₜ m) = m ⊗ₜ n :=
  rfl


lemma lift_comp_comm_eq (f : M →ₗ[R] N →ₗ[R] P) :
    lift f ∘ₗ TensorProduct.comm R N M = lift f.flip :=
  ext rfl

/-- The base ring is a right identity for the tensor product of modules, up to linear equivalence.
-/
protected def rid : M ⊗[R] R ≃ₗ[R] M :=
  LinearEquiv.trans (TensorProduct.comm R M R) (TensorProduct.lid R M)


@[simp]
theorem rid_tmul (m : M) (r : R) : (TensorProduct.rid R M) (m ⊗ₜ r) = r • m :=
  rfl


@[simp]
theorem rid_symm_apply (m : M) : (TensorProduct.rid R M).symm m = m ⊗ₜ 1 :=
  rfl


variable (R) in
theorem lid_eq_rid : TensorProduct.lid R R = TensorProduct.rid R R :=
  LinearEquiv.toLinearMap_injective <| ext' mul_comm


/-- If M and N are both R- and A-modules and their actions on them commute,
and if the A-action on `M ⊗[R] N` can switch between the two factors, then there is a
canonical A-linear map from `M ⊗[A] N` to `M ⊗[R] N`. -/
def mapOfCompatibleSMul : M ⊗[A] N →ₗ[A] M ⊗[R] N :=
  lift
  { toFun := fun m ↦
    { __ := mk R M N m
      map_smul' := fun _ _ ↦ (smul_tmul _ _ _).symm }
                                              /-
                                                R : Type u_1
                                                inst✝²¹ : CommSemiring R
                                                R' : Type u_2
                                                inst✝²⁰ : Monoid R'
                                                R'' : Type u_3
                                                inst✝¹⁹ : Semiring R''
                                                A : Type u_4
                                                M : Type u_5
                                                N : Type u_6
                                                P : Type u_7
                                                Q : Type u_8
                                                S : Type u_9
                                                T : Type u_10
                                                inst✝¹⁸ : AddCommMonoid M
                                                inst✝¹⁷ : AddCommMonoid N
                                                inst✝¹⁶ : AddCommMonoid P
                                                inst✝¹⁵ : AddCommMonoid Q
                                                inst✝¹⁴ : AddCommMonoid S
                                                inst✝¹³ : AddCommMonoid T
                                                inst✝¹² : Module R M
                                                inst✝¹¹ : Module R N
                                                inst✝¹⁰ : Module R Q
                                                inst✝⁹ : Module R S
                                                inst✝⁸ : Module R T
                                                inst✝⁷ : DistribMulAction R' M
                                                inst✝⁶ : Module R'' M
                                                inst✝⁵ : Module R P
                                                inst✝⁴ : CommSemiring A
                                                inst✝³ : Module A M
                                                inst✝² : Module A N
                                                inst✝¹ : SMulCommClass R A M
                                                inst✝ : TensorProduct.CompatibleSMul R A M N
                                                x✝¹ x✝ : M
                                                ⊢ ∀ (x : N),
                                                    Eq
                                                      (((fun m =>
                                                            let __spread.0 := (TensorProduct.mk R M N) m;
                                                            { toAddHom := __spread.0.toAddHom, map_smul' := ⋯ })
                                                          (HAdd.hAdd x✝¹ x✝))
                                                        x)
                                                      ((HAdd.hAdd
                                                          ((fun m =>
                                                              let __spread.0 := (TensorProduct.mk R M N) m;
                                                              { toAddHom := __spread.0.toAddHom, map_smul' := ⋯ })
                                                            x✝¹)
                                                          ((fun m =>
                                                              let __spread.0 := (TensorProduct.mk R M N) m;
                                                              { toAddHom := __spread.0.toAddHom, map_smul' := ⋯ })
                                                            x✝))
                                                        x)
                                              -/
    map_add' := fun _ _ ↦ LinearMap.ext <| by simp
                                              /-
                                                🎉 no goals
                                              -/
    map_smul' := fun _ _ ↦ rfl }


@[simp] theorem mapOfCompatibleSMul_tmul (m n) : mapOfCompatibleSMul R A M N (m ⊗ₜ n) = m ⊗ₜ n :=
  rfl


theorem mapOfCompatibleSMul_surjective : Function.Surjective (mapOfCompatibleSMul R A M N) :=
  fun x ↦ x.induction_on (⟨0, map_zero _⟩) (fun m n ↦ ⟨_, mapOfCompatibleSMul_tmul ..⟩)
                                         /-
                                           R : Type u_1
                                           inst✝⁹ : CommSemiring R
                                           A : Type u_4
                                           M : Type u_5
                                           N : Type u_6
                                           inst✝⁸ : AddCommMonoid M
                                           inst✝⁷ : AddCommMonoid N
                                           inst✝⁶ : Module R M
                                           inst✝⁵ : Module R N
                                           inst✝⁴ : CommSemiring A
                                           inst✝³ : Module A M
                                           inst✝² : Module A N
                                           inst✝¹ : SMulCommClass R A M
                                           inst✝ : TensorProduct.CompatibleSMul R A M N
                                           x✝⁴ x✝³ x✝² : TensorProduct R M N
                                           x✝¹ : Exists fun a => Eq ((TensorProduct.mapOfCompatibleSMul R A M N) a) x✝³
                                           x✝ : Exists fun a => Eq ((TensorProduct.mapOfCompatibleSMul R A M N) a) x✝²
                                           x : TensorProduct A M N
                                           hx : Eq ((TensorProduct.mapOfCompatibleSMul R A M N) x) x✝³
                                           y : TensorProduct A M N
                                           hy : Eq ((TensorProduct.mapOfCompatibleSMul R A M N) y) x✝²
                                           ⊢ Eq ((TensorProduct.mapOfCompatibleSMul R A M N) (HAdd.hAdd x y)) (HAdd.hAdd  …
                                         -/
    fun _ _ ⟨x, hx⟩ ⟨y, hy⟩ ↦ ⟨x + y, by simpa using congr($hx + $hy)⟩
                                         /-
                                           🎉 no goals
                                         -/


/-- `mapOfCompatibleSMul R A M N` is also R-linear. -/
def mapOfCompatibleSMul' : M ⊗[A] N →ₗ[R] M ⊗[R] N where
  __ := mapOfCompatibleSMul R A M N
                                                             /-
                                                               R : Type u_1
                                                               inst✝²¹ : CommSemiring R
                                                               R' : Type u_2
                                                               inst✝²⁰ : Monoid R'
                                                               R'' : Type u_3
                                                               inst✝¹⁹ : Semiring R''
                                                               A : Type u_4
                                                               M : Type u_5
                                                               N : Type u_6
                                                               P : Type u_7
                                                               Q : Type u_8
                                                               S : Type u_9
                                                               T : Type u_10
                                                               inst✝¹⁸ : AddCommMonoid M
                                                               inst✝¹⁷ : AddCommMonoid N
                                                               inst✝¹⁶ : AddCommMonoid P
                                                               inst✝¹⁵ : AddCommMonoid Q
                                                               inst✝¹⁴ : AddCommMonoid S
                                                               inst✝¹³ : AddCommMonoid T
                                                               inst✝¹² : Module R M
                                                               inst✝¹¹ : Module R N
                                                               inst✝¹⁰ : Module R Q
                                                               inst✝⁹ : Module R S
                                                               inst✝⁸ : Module R T
                                                               inst✝⁷ : DistribMulAction R' M
                                                               inst✝⁶ : Module R'' M
                                                               inst✝⁵ : Module R P
                                                               inst✝⁴ : CommSemiring A
                                                               inst✝³ : Module A M
                                                               inst✝² : Module A N
                                                               inst✝¹ : SMulCommClass R A M
                                                               inst✝ : TensorProduct.CompatibleSMul R A M N
                                                               x✝² : R
                                                               x : TensorProduct A M N
                                                               x✝¹ : M
                                                               x✝ : N
                                                               ⊢ Eq (__spread✝⁻⁰.toFun (HSMul.hSMul x✝² (TensorProduct.tmul A x✝¹ x✝))) (HSMu …
                                                             -/
  map_smul' _ x := x.induction_on (map_zero _) (fun _ _ ↦ by simp [smul_tmul'])
                                                             /-
                                                               🎉 no goals
                                                             -/
                      /-
                        R : Type u_1
                        inst✝²¹ : CommSemiring R
                        R' : Type u_2
                        inst✝²⁰ : Monoid R'
                        R'' : Type u_3
                        inst✝¹⁹ : Semiring R''
                        A : Type u_4
                        M : Type u_5
                        N : Type u_6
                        P : Type u_7
                        Q : Type u_8
                        S : Type u_9
                        T : Type u_10
                        inst✝¹⁸ : AddCommMonoid M
                        inst✝¹⁷ : AddCommMonoid N
                        inst✝¹⁶ : AddCommMonoid P
                        inst✝¹⁵ : AddCommMonoid Q
                        inst✝¹⁴ : AddCommMonoid S
                        inst✝¹³ : AddCommMonoid T
                        inst✝¹² : Module R M
                        inst✝¹¹ : Module R N
                        inst✝¹⁰ : Module R Q
                        inst✝⁹ : Module R S
                        inst✝⁸ : Module R T
                        inst✝⁷ : DistribMulAction R' M
                        inst✝⁶ : Module R'' M
                        inst✝⁵ : Module R P
                        inst✝⁴ : CommSemiring A
                        inst✝³ : Module A M
                        inst✝² : Module A N
                        inst✝¹ : SMulCommClass R A M
                        inst✝ : TensorProduct.CompatibleSMul R A M N
                        x✝² : R
                        x x✝¹ x✝ : TensorProduct A M N
                        h : Eq (__spread✝⁻⁰.toFun (HSMul.hSMul x✝² x✝¹)) (HSMul.hSMul ((RingHom.id R)  …
                        h' : Eq (__spread✝⁻⁰.toFun (HSMul.hSMul x✝² x✝)) (HSMul.hSMul ((RingHom.id R)  …
                        ⊢ Eq (__spread✝⁻⁰.toFun (HSMul.hSMul x✝² (HAdd.hAdd x✝¹ x✝))) (HSMul.hSMul ((R …
                      -/
    fun _ _ h h' ↦ by simpa using congr($h + $h')
                      /-
                        🎉 no goals
                      -/


/-- If the R- and A-actions on M and N satisfy `CompatibleSMul` both ways,
then `M ⊗[A] N` is canonically isomorphic to `M ⊗[R] N`. -/
def equivOfCompatibleSMul [CompatibleSMul A R M N] : M ⊗[A] N ≃ₗ[A] M ⊗[R] N where
  __ := mapOfCompatibleSMul R A M N
  invFun := mapOfCompatibleSMul A R M N
  left_inv x := x.induction_on (map_zero _) (fun _ _ ↦ rfl)
                      /-
                        R : Type u_1
                        inst✝²² : CommSemiring R
                        R' : Type u_2
                        inst✝²¹ : Monoid R'
                        R'' : Type u_3
                        inst✝²⁰ : Semiring R''
                        A : Type u_4
                        M : Type u_5
                        N : Type u_6
                        P : Type u_7
                        Q : Type u_8
                        S : Type u_9
                        T : Type u_10
                        inst✝¹⁹ : AddCommMonoid M
                        inst✝¹⁸ : AddCommMonoid N
                        inst✝¹⁷ : AddCommMonoid P
                        inst✝¹⁶ : AddCommMonoid Q
                        inst✝¹⁵ : AddCommMonoid S
                        inst✝¹⁴ : AddCommMonoid T
                        inst✝¹³ : Module R M
                        inst✝¹² : Module R N
                        inst✝¹¹ : Module R Q
                        inst✝¹⁰ : Module R S
                        inst✝⁹ : Module R T
                        inst✝⁸ : DistribMulAction R' M
                        inst✝⁷ : Module R'' M
                        inst✝⁶ : Module R P
                        inst✝⁵ : CommSemiring A
                        inst✝⁴ : Module A M
                        inst✝³ : Module A N
                        inst✝² : SMulCommClass R A M
                        inst✝¹ : TensorProduct.CompatibleSMul R A M N
                        inst✝ : TensorProduct.CompatibleSMul A R M N
                        x x✝¹ x✝ : TensorProduct A M N
                        h : Eq ((TensorProduct.mapOfCompatibleSMul A R M N) (__spread✝⁻⁰.toFun x✝¹)) x✝¹
                        h' : Eq ((TensorProduct.mapOfCompatibleSMul A R M N) (__spread✝⁻⁰.toFun x✝)) x✝
                        ⊢ Eq ((TensorProduct.mapOfCompatibleSMul A R M N) (__spread✝⁻⁰.toFun (HAdd.hAd …
                      -/
    fun _ _ h h' ↦ by simpa using congr($h + $h')
                      /-
                        🎉 no goals
                      -/
  right_inv x := x.induction_on (map_zero _) (fun _ _ ↦ rfl)
                      /-
                        R : Type u_1
                        inst✝²² : CommSemiring R
                        R' : Type u_2
                        inst✝²¹ : Monoid R'
                        R'' : Type u_3
                        inst✝²⁰ : Semiring R''
                        A : Type u_4
                        M : Type u_5
                        N : Type u_6
                        P : Type u_7
                        Q : Type u_8
                        S : Type u_9
                        T : Type u_10
                        inst✝¹⁹ : AddCommMonoid M
                        inst✝¹⁸ : AddCommMonoid N
                        inst✝¹⁷ : AddCommMonoid P
                        inst✝¹⁶ : AddCommMonoid Q
                        inst✝¹⁵ : AddCommMonoid S
                        inst✝¹⁴ : AddCommMonoid T
                        inst✝¹³ : Module R M
                        inst✝¹² : Module R N
                        inst✝¹¹ : Module R Q
                        inst✝¹⁰ : Module R S
                        inst✝⁹ : Module R T
                        inst✝⁸ : DistribMulAction R' M
                        inst✝⁷ : Module R'' M
                        inst✝⁶ : Module R P
                        inst✝⁵ : CommSemiring A
                        inst✝⁴ : Module A M
                        inst✝³ : Module A N
                        inst✝² : SMulCommClass R A M
                        inst✝¹ : TensorProduct.CompatibleSMul R A M N
                        inst✝ : TensorProduct.CompatibleSMul A R M N
                        x x✝¹ x✝ : TensorProduct R M N
                        h : Eq (__spread✝⁻⁰.toFun ((TensorProduct.mapOfCompatibleSMul A R M N) x✝¹)) x✝¹
                        h' : Eq (__spread✝⁻⁰.toFun ((TensorProduct.mapOfCompatibleSMul A R M N) x✝)) x✝
                        ⊢ Eq (__spread✝⁻⁰.toFun ((TensorProduct.mapOfCompatibleSMul A R M N) (HAdd.hAd …
                      -/
    fun _ _ h h' ↦ by simpa using congr($h + $h')
                      /-
                        🎉 no goals
                      -/


/-- If the R- and A- action on A and M satisfy `CompatibleSMul` both ways,
then `A ⊗[R] M` is canonically isomorphic to `M`. -/
def lidOfCompatibleSMul : A ⊗[R] M ≃ₗ[A] M :=
  (equivOfCompatibleSMul R A A M).symm ≪≫ₗ TensorProduct.lid _ _


theorem lidOfCompatibleSMul_tmul (a m) : lidOfCompatibleSMul R A M (a ⊗ₜ[R] m) = a • m := rfl


/-- The associator for tensor product of R-modules, as a linear equivalence. -/
protected def assoc : (M ⊗[R] N) ⊗[R] P ≃ₗ[R] M ⊗[R] N ⊗[R] P := by
  refine
      LinearEquiv.ofLinear (lift <| lift <| comp (lcurry R _ _ _) <| mk _ _ _)
        (lift <| comp (uncurry R _ _ _) <| curry <| mk _ _ _)
        (ext <| LinearMap.ext fun m => ext' fun n p => ?_)
        (ext <| flip_inj <| LinearMap.ext fun p => ext' fun m n => ?_) <;>
    repeat'
      first
        |rw [lift.tmul]|rw [compr₂_apply]|rw [comp_apply]|rw [mk_apply]|rw [flip_apply]
        |rw [lcurry_apply]|rw [uncurry_apply]|rw [curry_apply]|rw [id_apply]


@[simp]
theorem assoc_tmul (m : M) (n : N) (p : P) :
    (TensorProduct.assoc R M N P) (m ⊗ₜ n ⊗ₜ p) = m ⊗ₜ (n ⊗ₜ p) :=
  rfl


@[simp]
theorem assoc_symm_tmul (m : M) (n : N) (p : P) :
    (TensorProduct.assoc R M N P).symm (m ⊗ₜ (n ⊗ₜ p)) = m ⊗ₜ n ⊗ₜ p :=
  rfl


/-- The tensor product of a pair of linear maps between modules. -/
def map (f : M →ₗ[R] P) (g : N →ₗ[R] Q) : M ⊗[R] N →ₗ[R] P ⊗[R] Q :=
  lift <| comp (compl₂ (mk _ _ _) g) f


@[simp]
theorem map_tmul (f : M →ₗ[R] P) (g : N →ₗ[R] Q) (m : M) (n : N) : map f g (m ⊗ₜ n) = f m ⊗ₜ g n :=
  rfl


/-- Given linear maps `f : M → P`, `g : N → Q`, if we identify `M ⊗ N` with `N ⊗ M` and `P ⊗ Q`
with `Q ⊗ P`, then this lemma states that `f ⊗ g = g ⊗ f`. -/
lemma map_comp_comm_eq (f : M →ₗ[R] P) (g : N →ₗ[R] Q) :
    map f g ∘ₗ TensorProduct.comm R N M = TensorProduct.comm R Q P ∘ₗ map g f :=
  ext rfl


lemma map_comm (f : M →ₗ[R] P) (g : N →ₗ[R] Q) (x : N ⊗[R] M) :
    map f g (TensorProduct.comm R N M x) = TensorProduct.comm R Q P (map g f x) :=
  DFunLike.congr_fun (map_comp_comm_eq _ _) _


/-- Given linear maps `f : M → Q`, `g : N → S`, and `h : P → T`, if we identify `(M ⊗ N) ⊗ P`
with `M ⊗ (N ⊗ P)` and `(Q ⊗ S) ⊗ T` with `Q ⊗ (S ⊗ T)`, then this lemma states that
`f ⊗ (g ⊗ h) = (f ⊗ g) ⊗ h`. -/
lemma map_map_comp_assoc_eq (f : M →ₗ[R] Q) (g : N →ₗ[R] S) (h : P →ₗ[R] T) :
    map f (map g h) ∘ₗ TensorProduct.assoc R M N P =
      TensorProduct.assoc R Q S T ∘ₗ map (map f g) h :=
  ext <| ext <| LinearMap.ext fun _ => LinearMap.ext fun _ => LinearMap.ext fun _ => rfl


lemma map_map_assoc (f : M →ₗ[R] Q) (g : N →ₗ[R] S) (h : P →ₗ[R] T) (x : (M ⊗[R] N) ⊗[R] P) :
    map f (map g h) (TensorProduct.assoc R M N P x) =
      TensorProduct.assoc R Q S T (map (map f g) h x) :=
  DFunLike.congr_fun (map_map_comp_assoc_eq _ _ _) _


/-- Given linear maps `f : M → Q`, `g : N → S`, and `h : P → T`, if we identify `M ⊗ (N ⊗ P)`
with `(M ⊗ N) ⊗ P` and `Q ⊗ (S ⊗ T)` with `(Q ⊗ S) ⊗ T`, then this lemma states that
`(f ⊗ g) ⊗ h = f ⊗ (g ⊗ h)`. -/
lemma map_map_comp_assoc_symm_eq (f : M →ₗ[R] Q) (g : N →ₗ[R] S) (h : P →ₗ[R] T) :
    map (map f g) h ∘ₗ (TensorProduct.assoc R M N P).symm =
      (TensorProduct.assoc R Q S T).symm ∘ₗ map f (map g h) :=
  ext <| LinearMap.ext fun _ => ext <| LinearMap.ext fun _ => LinearMap.ext fun _ => rfl


lemma map_map_assoc_symm (f : M →ₗ[R] Q) (g : N →ₗ[R] S) (h : P →ₗ[R] T) (x : M ⊗[R] (N ⊗[R] P)) :
    map (map f g) h ((TensorProduct.assoc R M N P).symm x) =
      (TensorProduct.assoc R Q S T).symm (map f (map g h) x) :=
  DFunLike.congr_fun (map_map_comp_assoc_symm_eq _ _ _) _


theorem map_range_eq_span_tmul (f : M →ₗ[R] P) (g : N →ₗ[R] Q) :
    range (map f g) = Submodule.span R { t | ∃ m n, f m ⊗ₜ g n = t } := by
  simp only [← Submodule.map_top, ← span_tmul_eq_top, Submodule.map_span, Set.mem_image,
    Set.mem_setOf_eq]
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    ⊢ Eq (Submodule.span R (Set.image (⇑(TensorProduct.map f g)) (setOf fun t => E …
  -/
  congr; ext t
  /-
    case e_s.h
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    t : TensorProduct R P Q
    ⊢ Iff (Membership.mem (Set.image (⇑(TensorProduct.map f g)) (setOf fun t => Ex …
  -/
  constructor
    /-
      case e_s.h.mp
      R : Type u_1
      inst✝⁸ : CommSemiring R
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid N
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : AddCommMonoid Q
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module R Q
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      g : LinearMap (RingHom.id R) N Q
      t : TensorProduct R P Q
      ⊢ Membership.mem (Set.image (⇑(TensorProduct.map f g)) (setOf fun t => Exists  …
    -/
  · rintro ⟨_, ⟨⟨m, n, rfl⟩, rfl⟩⟩
    /-
      case e_s.h.mp.intro.intro.intro.intro
      R : Type u_1
      inst✝⁸ : CommSemiring R
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid N
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : AddCommMonoid Q
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module R Q
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      g : LinearMap (RingHom.id R) N Q
      m : M
      n : N
      ⊢ Membership.mem (setOf fun t => Exists fun m => Exists fun n => Eq (TensorPro …
    -/
    use m, n
    /-
      case h
      R : Type u_1
      inst✝⁸ : CommSemiring R
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid N
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : AddCommMonoid Q
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module R Q
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      g : LinearMap (RingHom.id R) N Q
      m : M
      n : N
      ⊢ Eq (TensorProduct.tmul R (f m) (g n)) ((TensorProduct.map f g) (TensorProduc …
    -/
    simp only [map_tmul]
    /-
      🎉 no goals
    -/
    /-
      case e_s.h.mpr
      R : Type u_1
      inst✝⁸ : CommSemiring R
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid N
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : AddCommMonoid Q
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module R Q
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      g : LinearMap (RingHom.id R) N Q
      t : TensorProduct R P Q
      ⊢ Membership.mem (setOf fun t => Exists fun m => Exists fun n => Eq (TensorPro …
    -/
  · rintro ⟨m, n, rfl⟩
    /-
      case e_s.h.mpr.intro.intro
      R : Type u_1
      inst✝⁸ : CommSemiring R
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid N
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : AddCommMonoid Q
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module R Q
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      g : LinearMap (RingHom.id R) N Q
      m : M
      n : N
      ⊢ Membership.mem (Set.image (⇑(TensorProduct.map f g)) (setOf fun t => Exists  …
    -/
    refine ⟨_, ⟨⟨m, n, rfl⟩, ?_⟩⟩
    /-
      case e_s.h.mpr.intro.intro
      R : Type u_1
      inst✝⁸ : CommSemiring R
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid N
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : AddCommMonoid Q
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module R Q
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      g : LinearMap (RingHom.id R) N Q
      m : M
      n : N
      ⊢ Eq ((TensorProduct.map f g) (TensorProduct.tmul R m n)) (TensorProduct.tmul  …
    -/
    simp only [map_tmul]
    /-
      🎉 no goals
    -/


/-- Given submodules `p ⊆ P` and `q ⊆ Q`, this is the natural map: `p ⊗ q → P ⊗ Q`. -/
@[simp]
def mapIncl (p : Submodule R P) (q : Submodule R Q) : p ⊗[R] q →ₗ[R] P ⊗[R] Q :=
  map p.subtype q.subtype


lemma range_mapIncl (p : Submodule R P) (q : Submodule R Q) :
    LinearMap.range (mapIncl p q) = Submodule.span R (Set.image2 (· ⊗ₜ ·) p q) := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    P : Type u_7
    Q : Type u_8
    inst✝³ : AddCommMonoid P
    inst✝² : AddCommMonoid Q
    inst✝¹ : Module R Q
    inst✝ : Module R P
    p : Submodule R P
    q : Submodule R Q
    ⊢ Eq (LinearMap.range (TensorProduct.mapIncl p q)) (Submodule.span R (Set.imag …
  -/
  rw [mapIncl, map_range_eq_span_tmul]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    P : Type u_7
    Q : Type u_8
    inst✝³ : AddCommMonoid P
    inst✝² : AddCommMonoid Q
    inst✝¹ : Module R Q
    inst✝ : Module R P
    p : Submodule R P
    q : Submodule R Q
    ⊢ Eq (Submodule.span R (setOf fun t => Exists fun m => Exists fun n => Eq (Ten …
  -/
  congr; ext; simp
              /-
                🎉 no goals
              -/


theorem map₂_eq_range_lift_comp_mapIncl (f : P →ₗ[R] Q →ₗ[R] M)
    (p : Submodule R P) (q : Submodule R Q) :
    Submodule.map₂ f p q = LinearMap.range (lift f ∘ₗ mapIncl p q) := by
  simp_rw [LinearMap.range_comp, range_mapIncl, Submodule.map_span,
    Set.image_image2, Submodule.map₂_eq_span_image2, lift.tmul]


theorem map_comp (f₂ : P →ₗ[R] P') (f₁ : M →ₗ[R] P) (g₂ : Q →ₗ[R] Q') (g₁ : N →ₗ[R] Q) :
    map (f₂.comp f₁) (g₂.comp g₁) = (map f₂ g₂).comp (map f₁ g₁) :=
  ext' fun _ _ => rfl


lemma range_mapIncl_mono {p p' : Submodule R P} {q q' : Submodule R Q} (hp : p ≤ p') (hq : q ≤ q') :
    LinearMap.range (mapIncl p q) ≤ LinearMap.range (mapIncl p' q') := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    P : Type u_7
    Q : Type u_8
    inst✝³ : AddCommMonoid P
    inst✝² : AddCommMonoid Q
    inst✝¹ : Module R Q
    inst✝ : Module R P
    p p' : Submodule R P
    q q' : Submodule R Q
    hp : LE.le p p'
    hq : LE.le q q'
    ⊢ LE.le (LinearMap.range (TensorProduct.mapIncl p q)) (LinearMap.range (Tensor …
  -/
  simp_rw [range_mapIncl]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    P : Type u_7
    Q : Type u_8
    inst✝³ : AddCommMonoid P
    inst✝² : AddCommMonoid Q
    inst✝¹ : Module R Q
    inst✝ : Module R P
    p p' : Submodule R P
    q q' : Submodule R Q
    hp : LE.le p p'
    hq : LE.le q q'
    ⊢ LE.le (Submodule.span R (Set.image2 (fun x1 x2 => TensorProduct.tmul R x1 x2 …
  -/
  exact Submodule.span_mono (Set.image2_subset hp hq)
  /-
    🎉 no goals
  -/


theorem lift_comp_map (i : P →ₗ[R] Q →ₗ[R] Q') (f : M →ₗ[R] P) (g : N →ₗ[R] Q) :
    (lift i).comp (map f g) = lift ((i.comp f).compl₂ g) :=
  ext' fun _ _ => rfl


@[simp]
theorem map_id : map (id : M →ₗ[R] M) (id : N →ₗ[R] N) = .id := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    ⊢ Eq (TensorProduct.map LinearMap.id LinearMap.id) LinearMap.id
  -/
  ext
  /-
    case H.h.h
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    x✝¹ : M
    x✝ : N
    ⊢ Eq ((((TensorProduct.mk R M N).compr₂ (TensorProduct.map LinearMap.id Linear …
  -/
  simp only [mk_apply, id_coe, compr₂_apply, _root_.id, map_tmul]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem map_one : map (1 : M →ₗ[R] M) (1 : N →ₗ[R] N) = 1 :=
  map_id


protected theorem map_mul (f₁ f₂ : M →ₗ[R] M) (g₁ g₂ : N →ₗ[R] N) :
    map (f₁ * f₂) (g₁ * g₂) = map f₁ g₁ * map f₂ g₂ :=
  map_comp f₁ f₂ g₁ g₂


@[simp]
protected theorem map_pow (f : M →ₗ[R] M) (g : N →ₗ[R] N) (n : ℕ) :
    map f g ^ n = map (f ^ n) (g ^ n) := by
  induction n with
  | zero => simp only [pow_zero, TensorProduct.map_one]
  | succ n ih => simp only [pow_succ', ih, TensorProduct.map_mul]


theorem map_add_left (f₁ f₂ : M →ₗ[R] P) (g : N →ₗ[R] Q) :
    map (f₁ + f₂) g = map f₁ g + map f₂ g := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    f₁ f₂ : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    ⊢ Eq (TensorProduct.map (HAdd.hAdd f₁ f₂) g) (HAdd.hAdd (TensorProduct.map f₁  …
  -/
  ext
  /-
    case H.h.h
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    f₁ f₂ : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    x✝¹ : M
    x✝ : N
    ⊢ Eq ((((TensorProduct.mk R M N).compr₂ (TensorProduct.map (HAdd.hAdd f₁ f₂) g …
  -/
  simp only [add_tmul, compr₂_apply, mk_apply, map_tmul, add_apply]
  /-
    🎉 no goals
  -/


theorem map_add_right (f : M →ₗ[R] P) (g₁ g₂ : N →ₗ[R] Q) :
    map f (g₁ + g₂) = map f g₁ + map f g₂ := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    g₁ g₂ : LinearMap (RingHom.id R) N Q
    ⊢ Eq (TensorProduct.map f (HAdd.hAdd g₁ g₂)) (HAdd.hAdd (TensorProduct.map f g …
  -/
  ext
  /-
    case H.h.h
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    g₁ g₂ : LinearMap (RingHom.id R) N Q
    x✝¹ : M
    x✝ : N
    ⊢ Eq ((((TensorProduct.mk R M N).compr₂ (TensorProduct.map f (HAdd.hAdd g₁ g₂) …
  -/
  simp only [tmul_add, compr₂_apply, mk_apply, map_tmul, add_apply]
  /-
    🎉 no goals
  -/


theorem map_smul_left (r : R) (f : M →ₗ[R] P) (g : N →ₗ[R] Q) : map (r • f) g = r • map f g := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    r : R
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    ⊢ Eq (TensorProduct.map (HSMul.hSMul r f) g) (HSMul.hSMul r (TensorProduct.map …
  -/
  ext
  /-
    case H.h.h
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    r : R
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    x✝¹ : M
    x✝ : N
    ⊢ Eq ((((TensorProduct.mk R M N).compr₂ (TensorProduct.map (HSMul.hSMul r f) g …
  -/
  simp only [smul_tmul, compr₂_apply, mk_apply, map_tmul, smul_apply, tmul_smul]
  /-
    🎉 no goals
  -/


theorem map_smul_right (r : R) (f : M →ₗ[R] P) (g : N →ₗ[R] Q) : map f (r • g) = r • map f g := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    r : R
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    ⊢ Eq (TensorProduct.map f (HSMul.hSMul r g)) (HSMul.hSMul r (TensorProduct.map …
  -/
  ext
  /-
    case H.h.h
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    r : R
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    x✝¹ : M
    x✝ : N
    ⊢ Eq ((((TensorProduct.mk R M N).compr₂ (TensorProduct.map f (HSMul.hSMul r g) …
  -/
  simp only [smul_tmul, compr₂_apply, mk_apply, map_tmul, smul_apply, tmul_smul]
  /-
    🎉 no goals
  -/


/-- The tensor product of a pair of linear maps between modules, bilinear in both maps. -/
def mapBilinear : (M →ₗ[R] P) →ₗ[R] (N →ₗ[R] Q) →ₗ[R] M ⊗[R] N →ₗ[R] P ⊗[R] Q :=
  LinearMap.mk₂ R map map_add_left map_smul_left map_add_right map_smul_right


/-- The canonical linear map from `P ⊗[R] (M →ₗ[R] Q)` to `(M →ₗ[R] P ⊗[R] Q)` -/
def lTensorHomToHomLTensor : P ⊗[R] (M →ₗ[R] Q) →ₗ[R] M →ₗ[R] P ⊗[R] Q :=
  TensorProduct.lift (llcomp R M Q _ ∘ₗ mk R P Q)


/-- The canonical linear map from `(M →ₗ[R] P) ⊗[R] Q` to `(M →ₗ[R] P ⊗[R] Q)` -/
def rTensorHomToHomRTensor : (M →ₗ[R] P) ⊗[R] Q →ₗ[R] M →ₗ[R] P ⊗[R] Q :=
  TensorProduct.lift (llcomp R M P _ ∘ₗ (mk R P Q).flip).flip


/-- The linear map from `(M →ₗ P) ⊗ (N →ₗ Q)` to `(M ⊗ N →ₗ P ⊗ Q)` sending `f ⊗ₜ g` to
the `TensorProduct.map f g`, the tensor product of the two maps. -/
def homTensorHomMap : (M →ₗ[R] P) ⊗[R] (N →ₗ[R] Q) →ₗ[R] M ⊗[R] N →ₗ[R] P ⊗[R] Q :=
  lift (mapBilinear R M N P Q)


/--
This is a binary version of `TensorProduct.map`: Given a bilinear map `f : M ⟶ P ⟶ Q` and a
bilinear map `g : N ⟶ S ⟶ T`, if we think `f` and `g` as linear maps with two inputs, then
`map₂ f g` is a bilinear map taking two inputs `M ⊗ N → P ⊗ S → Q ⊗ S` defined by
`map₂ f g (m ⊗ n) (p ⊗ s) = f m p ⊗ g n s`.

Mathematically, `TensorProduct.map₂` is defined as the composition
`M ⊗ N -map→ Hom(P, Q) ⊗ Hom(S, T) -homTensorHomMap→ Hom(P ⊗ S, Q ⊗ T)`.
-/
def map₂ (f : M →ₗ[R] P →ₗ[R] Q) (g : N →ₗ[R] S →ₗ[R] T) :
    M ⊗[R] N →ₗ[R] P ⊗[R] S →ₗ[R] Q ⊗[R] T :=
  homTensorHomMap R _ _ _ _ ∘ₗ map f g


@[simp]
theorem mapBilinear_apply (f : M →ₗ[R] P) (g : N →ₗ[R] Q) : mapBilinear R M N P Q f g = map f g :=
  rfl


@[simp]
theorem lTensorHomToHomLTensor_apply (p : P) (f : M →ₗ[R] Q) (m : M) :
    lTensorHomToHomLTensor R M P Q (p ⊗ₜ f) m = p ⊗ₜ f m :=
  rfl


@[simp]
theorem rTensorHomToHomRTensor_apply (f : M →ₗ[R] P) (q : Q) (m : M) :
    rTensorHomToHomRTensor R M P Q (f ⊗ₜ q) m = f m ⊗ₜ q :=
  rfl


@[simp]
theorem homTensorHomMap_apply (f : M →ₗ[R] P) (g : N →ₗ[R] Q) :
    homTensorHomMap R M N P Q (f ⊗ₜ g) = map f g :=
  rfl


@[simp]
theorem map₂_apply_tmul (f : M →ₗ[R] P →ₗ[R] Q) (g : N →ₗ[R] S →ₗ[R] T) (m : M) (n : N) :
    map₂ f g (m ⊗ₜ n) = map (f m) (g n) := rfl


@[simp]
theorem map_zero_left (g : N →ₗ[R] Q) : map (0 : M →ₗ[R] P) g = 0 :=
  (mapBilinear R M N P Q).map_zero₂ _


@[simp]
theorem map_zero_right (f : M →ₗ[R] P) : map f (0 : N →ₗ[R] Q) = 0 :=
  (mapBilinear R M N P Q _).map_zero


/-- If `M` and `P` are linearly equivalent and `N` and `Q` are linearly equivalent
then `M ⊗ N` and `P ⊗ Q` are linearly equivalent. -/
def congr (f : M ≃ₗ[R] P) (g : N ≃ₗ[R] Q) : M ⊗[R] N ≃ₗ[R] P ⊗[R] Q :=
  LinearEquiv.ofLinear (map f g) (map f.symm g.symm)
                        /-
                          R : Type u_1
                          inst✝¹⁶ : CommSemiring R
                          R' : Type u_2
                          inst✝¹⁵ : Monoid R'
                          R'' : Type u_3
                          inst✝¹⁴ : Semiring R''
                          A : Type u_4
                          M : Type u_5
                          N : Type u_6
                          P : Type u_7
                          Q : Type u_8
                          S : Type u_9
                          T : Type u_10
                          inst✝¹³ : AddCommMonoid M
                          inst✝¹² : AddCommMonoid N
                          inst✝¹¹ : AddCommMonoid P
                          inst✝¹⁰ : AddCommMonoid Q
                          inst✝⁹ : AddCommMonoid S
                          inst✝⁸ : AddCommMonoid T
                          inst✝⁷ : Module R M
                          inst✝⁶ : Module R N
                          inst✝⁵ : Module R Q
                          inst✝⁴ : Module R S
                          inst✝³ : Module R T
                          inst✝² : DistribMulAction R' M
                          inst✝¹ : Module R'' M
                          inst✝ : Module R P
                          f : LinearEquiv (RingHom.id R) M P
                          g : LinearEquiv (RingHom.id R) N Q
                          m : P
                          n : Q
                          ⊢ Eq (((TensorProduct.map ↑f ↑g).comp (TensorProduct.map ↑f.symm ↑g.symm)) (Te …
                        -/
    (ext' fun m n => by simp)
                        /-
                          🎉 no goals
                        -/
                        /-
                          R : Type u_1
                          inst✝¹⁶ : CommSemiring R
                          R' : Type u_2
                          inst✝¹⁵ : Monoid R'
                          R'' : Type u_3
                          inst✝¹⁴ : Semiring R''
                          A : Type u_4
                          M : Type u_5
                          N : Type u_6
                          P : Type u_7
                          Q : Type u_8
                          S : Type u_9
                          T : Type u_10
                          inst✝¹³ : AddCommMonoid M
                          inst✝¹² : AddCommMonoid N
                          inst✝¹¹ : AddCommMonoid P
                          inst✝¹⁰ : AddCommMonoid Q
                          inst✝⁹ : AddCommMonoid S
                          inst✝⁸ : AddCommMonoid T
                          inst✝⁷ : Module R M
                          inst✝⁶ : Module R N
                          inst✝⁵ : Module R Q
                          inst✝⁴ : Module R S
                          inst✝³ : Module R T
                          inst✝² : DistribMulAction R' M
                          inst✝¹ : Module R'' M
                          inst✝ : Module R P
                          f : LinearEquiv (RingHom.id R) M P
                          g : LinearEquiv (RingHom.id R) N Q
                          m : M
                          n : N
                          ⊢ Eq (((TensorProduct.map ↑f.symm ↑g.symm).comp (TensorProduct.map ↑f ↑g)) (Te …
                        -/
    (ext' fun m n => by simp)
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem congr_tmul (f : M ≃ₗ[R] P) (g : N ≃ₗ[R] Q) (m : M) (n : N) :
    congr f g (m ⊗ₜ n) = f m ⊗ₜ g n :=
  rfl


@[simp]
theorem congr_symm_tmul (f : M ≃ₗ[R] P) (g : N ≃ₗ[R] Q) (p : P) (q : Q) :
    (congr f g).symm (p ⊗ₜ q) = f.symm p ⊗ₜ g.symm q :=
  rfl


theorem congr_symm (f : M ≃ₗ[R] P) (g : N ≃ₗ[R] Q) : (congr f g).symm = congr f.symm g.symm := rfl


@[simp] theorem congr_refl_refl : congr (.refl R M) (.refl R N) = .refl R _ :=
  LinearEquiv.toLinearMap_injective <| ext' fun _ _ ↦ rfl


theorem congr_trans (f : M ≃ₗ[R] P) (g : N ≃ₗ[R] Q) (f' : P ≃ₗ[R] S) (g' : Q ≃ₗ[R] T) :
    congr (f ≪≫ₗ f') (g ≪≫ₗ g') = congr f g ≪≫ₗ congr f' g' :=
  LinearEquiv.toLinearMap_injective <| map_comp _ _ _ _


theorem congr_mul (f : M ≃ₗ[R] M) (g : N ≃ₗ[R] N) (f' : M ≃ₗ[R] M) (g' : N ≃ₗ[R] N) :
    congr (f * f') (g * g') = congr f g * congr f' g' := congr_trans _ _ _ _


@[simp] theorem congr_pow (f : M ≃ₗ[R] M) (g : N ≃ₗ[R] N) (n : ℕ) :
    congr f g ^ n = congr (f ^ n) (g ^ n) := by
  induction n with
  | zero => exact congr_refl_refl.symm
  | succ n ih => simp_rw [pow_succ, ih, congr_mul]


@[simp] theorem congr_zpow (f : M ≃ₗ[R] M) (g : N ≃ₗ[R] N) (n : ℤ) :
    congr f g ^ n = congr (f ^ n) (g ^ n) := by
  induction n with
  | ofNat n => exact congr_pow _ _ _
  | negSucc n => simp_rw [zpow_negSucc, congr_pow]; exact congr_symm _ _


/-- A tensor product analogue of `mul_left_comm`. -/
def leftComm : M ⊗[R] N ⊗[R] P ≃ₗ[R] N ⊗[R] M ⊗[R] P :=
  let e₁ := (TensorProduct.assoc R M N P).symm
  let e₂ := congr (TensorProduct.comm R M N) (1 : P ≃ₗ[R] P)
  let e₃ := TensorProduct.assoc R N M P
  e₁ ≪≫ₗ (e₂ ≪≫ₗ e₃)


@[simp]
theorem leftComm_tmul (m : M) (n : N) (p : P) : leftComm R M N P (m ⊗ₜ (n ⊗ₜ p)) = n ⊗ₜ (m ⊗ₜ p) :=
  rfl


@[simp]
theorem leftComm_symm_tmul (m : M) (n : N) (p : P) :
    (leftComm R M N P).symm (n ⊗ₜ (m ⊗ₜ p)) = m ⊗ₜ (n ⊗ₜ p) :=
  rfl


/-- This special case is worth defining explicitly since it is useful for defining multiplication
on tensor products of modules carrying multiplications (e.g., associative rings, Lie rings, ...).

E.g., suppose `M = P` and `N = Q` and that `M` and `N` carry bilinear multiplications:
`M ⊗ M → M` and `N ⊗ N → N`. Using `map`, we can define `(M ⊗ M) ⊗ (N ⊗ N) → M ⊗ N` which, when
combined with this definition, yields a bilinear multiplication on `M ⊗ N`:
`(M ⊗ N) ⊗ (M ⊗ N) → M ⊗ N`. In particular we could use this to define the multiplication in
the `TensorProduct.semiring` instance (currently defined "by hand" using `TensorProduct.mul`).

See also `mul_mul_mul_comm`. -/
def tensorTensorTensorComm : (M ⊗[R] N) ⊗[R] P ⊗[R] Q ≃ₗ[R] (M ⊗[R] P) ⊗[R] N ⊗[R] Q :=
  let e₁ := TensorProduct.assoc R M N (P ⊗[R] Q)
  let e₂ := congr (1 : M ≃ₗ[R] M) (leftComm R N P Q)
  let e₃ := (TensorProduct.assoc R M P (N ⊗[R] Q)).symm
  e₁ ≪≫ₗ (e₂ ≪≫ₗ e₃)


@[simp]
theorem tensorTensorTensorComm_tmul (m : M) (n : N) (p : P) (q : Q) :
    tensorTensorTensorComm R M N P Q (m ⊗ₜ n ⊗ₜ (p ⊗ₜ q)) = m ⊗ₜ p ⊗ₜ (n ⊗ₜ q) :=
  rfl

-- Porting note: the proof here was `rfl` but that caused a timeout.

@[simp]
theorem tensorTensorTensorComm_symm :
    (tensorTensorTensorComm R M N P Q).symm = tensorTensorTensorComm R M P N Q := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    ⊢ Eq (TensorProduct.tensorTensorTensorComm R M N P Q).symm (TensorProduct.tens …
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


/-- This special case is useful for describing the interplay between `dualTensorHomEquiv` and
composition of linear maps.

E.g., composition of linear maps gives a map `(M → N) ⊗ (N → P) → (M → P)`, and applying
`dual_tensor_hom_equiv.symm` to the three hom-modules gives a map
`(M.dual ⊗ N) ⊗ (N.dual ⊗ P) → (M.dual ⊗ P)`, which agrees with the application of `contractRight`
on `N ⊗ N.dual` after the suitable rebracketting.
-/
def tensorTensorTensorAssoc : (M ⊗[R] N) ⊗[R] P ⊗[R] Q ≃ₗ[R] (M ⊗[R] N ⊗[R] P) ⊗[R] Q :=
  (TensorProduct.assoc R (M ⊗[R] N) P Q).symm ≪≫ₗ
    congr (TensorProduct.assoc R M N P) (1 : Q ≃ₗ[R] Q)


@[simp]
theorem tensorTensorTensorAssoc_tmul (m : M) (n : N) (p : P) (q : Q) :
    tensorTensorTensorAssoc R M N P Q (m ⊗ₜ n ⊗ₜ (p ⊗ₜ q)) = m ⊗ₜ (n ⊗ₜ p) ⊗ₜ q :=
  rfl


@[simp]
theorem tensorTensorTensorAssoc_symm_tmul (m : M) (n : N) (p : P) (q : Q) :
    (tensorTensorTensorAssoc R M N P Q).symm (m ⊗ₜ (n ⊗ₜ p) ⊗ₜ q) = m ⊗ₜ n ⊗ₜ (p ⊗ₜ q) :=
  rfl


/-- `LinearMap.lTensor M f : M ⊗ N →ₗ M ⊗ P` is the natural linear map
induced by `f : N →ₗ P`. -/
def lTensor (f : N →ₗ[R] P) : M ⊗[R] N →ₗ[R] M ⊗[R] P :=
  TensorProduct.map id f


/-- `LinearMap.rTensor M f : N₁ ⊗ M →ₗ N₂ ⊗ M` is the natural linear map
induced by `f : N₁ →ₗ N₂`. -/
def rTensor (f : N →ₗ[R] P) : N ⊗[R] M →ₗ[R] P ⊗[R] M :=
  TensorProduct.map f id


@[simp]
theorem lTensor_tmul (m : M) (n : N) : f.lTensor M (m ⊗ₜ n) = m ⊗ₜ f n :=
  rfl


@[simp]
theorem rTensor_tmul (m : M) (n : N) : f.rTensor M (n ⊗ₜ m) = f n ⊗ₜ m :=
  rfl


@[simp]
theorem lTensor_comp_mk (m : M) :
    f.lTensor M ∘ₗ TensorProduct.mk R M N m = TensorProduct.mk R M P m ∘ₗ f :=
  rfl


@[simp]
theorem rTensor_comp_flip_mk (m : M) :
    f.rTensor M ∘ₗ (TensorProduct.mk R N M).flip m = (TensorProduct.mk R P M).flip m ∘ₗ f :=
  rfl


lemma comm_comp_rTensor_comp_comm_eq (g : N →ₗ[R] P) :
    TensorProduct.comm R P Q ∘ₗ rTensor Q g ∘ₗ TensorProduct.comm R Q N =
      lTensor Q g :=
  TensorProduct.ext rfl


theorem rTensor_tensor : rTensor (M ⊗[R] N) g =
    TensorProduct.assoc R Q M N ∘ₗ rTensor N (rTensor M g) ∘ₗ (TensorProduct.assoc R P M N).symm :=
  TensorProduct.ext <| LinearMap.ext fun _ ↦ TensorProduct.ext rfl


lemma comm_comp_lTensor_comp_comm_eq (g : N →ₗ[R] P) :
    TensorProduct.comm R Q P ∘ₗ lTensor Q g ∘ₗ TensorProduct.comm R N Q =
      rTensor Q g :=
  TensorProduct.ext rfl


/-- Given a linear map `f : N → P`, `f ⊗ M` is injective if and only if `M ⊗ f` is injective. -/
theorem lTensor_inj_iff_rTensor_inj :
    Function.Injective (lTensor M f) ↔ Function.Injective (rTensor M f) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    ⊢ Iff (Function.Injective ⇑(LinearMap.lTensor M f)) (Function.Injective ⇑(Line …
  -/
  simp [← comm_comp_rTensor_comp_comm_eq]
  /-
    🎉 no goals
  -/


/-- Given a linear map `f : N → P`, `f ⊗ M` is surjective if and only if `M ⊗ f` is surjective. -/
theorem lTensor_surj_iff_rTensor_surj :
    Function.Surjective (lTensor M f) ↔ Function.Surjective (rTensor M f) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    ⊢ Iff (Function.Surjective ⇑(LinearMap.lTensor M f)) (Function.Surjective ⇑(Li …
  -/
  simp [← comm_comp_rTensor_comp_comm_eq]
  /-
    🎉 no goals
  -/


/-- Given a linear map `f : N → P`, `f ⊗ M` is bijective if and only if `M ⊗ f` is bijective. -/
theorem lTensor_bij_iff_rTensor_bij :
    Function.Bijective (lTensor M f) ↔ Function.Bijective (rTensor M f) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    ⊢ Iff (Function.Bijective ⇑(LinearMap.lTensor M f)) (Function.Bijective ⇑(Line …
  -/
  simp [← comm_comp_rTensor_comp_comm_eq]
  /-
    🎉 no goals
  -/


/-- `lTensorHom M` is the natural linear map that sends a linear map `f : N →ₗ P` to `M ⊗ f`. -/
def lTensorHom : (N →ₗ[R] P) →ₗ[R] M ⊗[R] N →ₗ[R] M ⊗[R] P where
  toFun := lTensor M
  map_add' f g := by
    /-
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Module R P
      g✝ : LinearMap (RingHom.id R) P Q
      f✝ f g : LinearMap (RingHom.id R) N P
      ⊢ Eq (LinearMap.lTensor M (HAdd.hAdd f g)) (HAdd.hAdd (LinearMap.lTensor M f)  …
    -/
    ext x y
    /-
      case H.h.h
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Module R P
      g✝ : LinearMap (RingHom.id R) P Q
      f✝ f g : LinearMap (RingHom.id R) N P
      x : M
      y : N
      ⊢ Eq ((((TensorProduct.mk R M N).compr₂ (LinearMap.lTensor M (HAdd.hAdd f g))) …
    -/
    simp only [compr₂_apply, mk_apply, add_apply, lTensor_tmul, tmul_add]
    /-
      🎉 no goals
    -/
  map_smul' r f := by
    /-
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Module R P
      g : LinearMap (RingHom.id R) P Q
      f✝ : LinearMap (RingHom.id R) N P
      r : R
      f : LinearMap (RingHom.id R) N P
      ⊢ Eq ({ toFun := LinearMap.lTensor M, map_add' := ⋯ }.toFun (HSMul.hSMul r f)) …
    -/
    dsimp
    /-
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Module R P
      g : LinearMap (RingHom.id R) P Q
      f✝ : LinearMap (RingHom.id R) N P
      r : R
      f : LinearMap (RingHom.id R) N P
      ⊢ Eq (LinearMap.lTensor M (HSMul.hSMul r f)) (HSMul.hSMul r (LinearMap.lTensor …
    -/
    ext x y
    /-
      case H.h.h
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Module R P
      g : LinearMap (RingHom.id R) P Q
      f✝ : LinearMap (RingHom.id R) N P
      r : R
      f : LinearMap (RingHom.id R) N P
      x : M
      y : N
      ⊢ Eq ((((TensorProduct.mk R M N).compr₂ (LinearMap.lTensor M (HSMul.hSMul r f) …
    -/
    simp only [compr₂_apply, mk_apply, tmul_smul, smul_apply, lTensor_tmul]
    /-
      🎉 no goals
    -/


/-- `rTensorHom M` is the natural linear map that sends a linear map `f : N →ₗ P` to `f ⊗ M`. -/
def rTensorHom : (N →ₗ[R] P) →ₗ[R] N ⊗[R] M →ₗ[R] P ⊗[R] M where
  toFun f := f.rTensor M
  map_add' f g := by
    /-
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Module R P
      g✝ : LinearMap (RingHom.id R) P Q
      f✝ f g : LinearMap (RingHom.id R) N P
      ⊢ Eq ((fun f => LinearMap.rTensor M f) (HAdd.hAdd f g)) (HAdd.hAdd ((fun f =>  …
    -/
    ext x y
    /-
      case H.h.h
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Module R P
      g✝ : LinearMap (RingHom.id R) P Q
      f✝ f g : LinearMap (RingHom.id R) N P
      x : N
      y : M
      ⊢ Eq ((((TensorProduct.mk R N M).compr₂ ((fun f => LinearMap.rTensor M f) (HAd …
    -/
    simp only [compr₂_apply, mk_apply, add_apply, rTensor_tmul, add_tmul]
    /-
      🎉 no goals
    -/
  map_smul' r f := by
    /-
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Module R P
      g : LinearMap (RingHom.id R) P Q
      f✝ : LinearMap (RingHom.id R) N P
      r : R
      f : LinearMap (RingHom.id R) N P
      ⊢ Eq ({ toFun := fun f => LinearMap.rTensor M f, map_add' := ⋯ }.toFun (HSMul. …
    -/
    dsimp
    /-
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Module R P
      g : LinearMap (RingHom.id R) P Q
      f✝ : LinearMap (RingHom.id R) N P
      r : R
      f : LinearMap (RingHom.id R) N P
      ⊢ Eq (LinearMap.rTensor M (HSMul.hSMul r f)) (HSMul.hSMul r (LinearMap.rTensor …
    -/
    ext x y
    /-
      case H.h.h
      R : Type u_1
      inst✝¹⁶ : CommSemiring R
      R' : Type u_2
      inst✝¹⁵ : Monoid R'
      R'' : Type u_3
      inst✝¹⁴ : Semiring R''
      A : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      S : Type u_9
      T : Type u_10
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid N
      inst✝¹¹ : AddCommMonoid P
      inst✝¹⁰ : AddCommMonoid Q
      inst✝⁹ : AddCommMonoid S
      inst✝⁸ : AddCommMonoid T
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R Q
      inst✝⁴ : Module R S
      inst✝³ : Module R T
      inst✝² : DistribMulAction R' M
      inst✝¹ : Module R'' M
      inst✝ : Module R P
      g : LinearMap (RingHom.id R) P Q
      f✝ : LinearMap (RingHom.id R) N P
      r : R
      f : LinearMap (RingHom.id R) N P
      x : N
      y : M
      ⊢ Eq ((((TensorProduct.mk R N M).compr₂ (LinearMap.rTensor M (HSMul.hSMul r f) …
    -/
    simp only [compr₂_apply, mk_apply, smul_tmul, tmul_smul, smul_apply, rTensor_tmul]
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_lTensorHom : (lTensorHom M : (N →ₗ[R] P) → M ⊗[R] N →ₗ[R] M ⊗[R] P) = lTensor M :=
  rfl


@[simp]
theorem coe_rTensorHom : (rTensorHom M : (N →ₗ[R] P) → N ⊗[R] M →ₗ[R] P ⊗[R] M) = rTensor M :=
  rfl


@[simp]
theorem lTensor_add (f g : N →ₗ[R] P) : (f + g).lTensor M = f.lTensor M + g.lTensor M :=
  (lTensorHom M).map_add f g


@[simp]
theorem rTensor_add (f g : N →ₗ[R] P) : (f + g).rTensor M = f.rTensor M + g.rTensor M :=
  (rTensorHom M).map_add f g


@[simp]
theorem lTensor_zero : lTensor M (0 : N →ₗ[R] P) = 0 :=
  (lTensorHom M).map_zero


@[simp]
theorem rTensor_zero : rTensor M (0 : N →ₗ[R] P) = 0 :=
  (rTensorHom M).map_zero


@[simp]
theorem lTensor_smul (r : R) (f : N →ₗ[R] P) : (r • f).lTensor M = r • f.lTensor M :=
  (lTensorHom M).map_smul r f


@[simp]
theorem rTensor_smul (r : R) (f : N →ₗ[R] P) : (r • f).rTensor M = r • f.rTensor M :=
  (rTensorHom M).map_smul r f


theorem lTensor_comp : (g.comp f).lTensor M = (g.lTensor M).comp (f.lTensor M) := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    g : LinearMap (RingHom.id R) P Q
    f : LinearMap (RingHom.id R) N P
    ⊢ Eq (LinearMap.lTensor M (g.comp f)) ((LinearMap.lTensor M g).comp (LinearMap …
  -/
  ext m n
  /-
    case H.h.h
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    g : LinearMap (RingHom.id R) P Q
    f : LinearMap (RingHom.id R) N P
    m : M
    n : N
    ⊢ Eq ((((TensorProduct.mk R M N).compr₂ (LinearMap.lTensor M (g.comp f))) m) n …
  -/
  simp only [compr₂_apply, mk_apply, comp_apply, lTensor_tmul]
  /-
    🎉 no goals
  -/


theorem lTensor_comp_apply (x : M ⊗[R] N) :
                                                                   /-
                                                                     R : Type u_1
                                                                     inst✝⁸ : CommSemiring R
                                                                     M : Type u_5
                                                                     N : Type u_6
                                                                     P : Type u_7
                                                                     Q : Type u_8
                                                                     inst✝⁷ : AddCommMonoid M
                                                                     inst✝⁶ : AddCommMonoid N
                                                                     inst✝⁵ : AddCommMonoid P
                                                                     inst✝⁴ : AddCommMonoid Q
                                                                     inst✝³ : Module R M
                                                                     inst✝² : Module R N
                                                                     inst✝¹ : Module R Q
                                                                     inst✝ : Module R P
                                                                     g : LinearMap (RingHom.id R) P Q
                                                                     f : LinearMap (RingHom.id R) N P
                                                                     x : TensorProduct R M N
                                                                     ⊢ Eq ((LinearMap.lTensor M (g.comp f)) x) ((LinearMap.lTensor M g) ((LinearMap …
                                                                   -/
    (g.comp f).lTensor M x = (g.lTensor M) ((f.lTensor M) x) := by rw [lTensor_comp, coe_comp]; rfl
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


theorem rTensor_comp : (g.comp f).rTensor M = (g.rTensor M).comp (f.rTensor M) := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    g : LinearMap (RingHom.id R) P Q
    f : LinearMap (RingHom.id R) N P
    ⊢ Eq (LinearMap.rTensor M (g.comp f)) ((LinearMap.rTensor M g).comp (LinearMap …
  -/
  ext m n
  /-
    case H.h.h
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    g : LinearMap (RingHom.id R) P Q
    f : LinearMap (RingHom.id R) N P
    m : N
    n : M
    ⊢ Eq ((((TensorProduct.mk R N M).compr₂ (LinearMap.rTensor M (g.comp f))) m) n …
  -/
  simp only [compr₂_apply, mk_apply, comp_apply, rTensor_tmul]
  /-
    🎉 no goals
  -/


theorem rTensor_comp_apply (x : N ⊗[R] M) :
                                                                   /-
                                                                     R : Type u_1
                                                                     inst✝⁸ : CommSemiring R
                                                                     M : Type u_5
                                                                     N : Type u_6
                                                                     P : Type u_7
                                                                     Q : Type u_8
                                                                     inst✝⁷ : AddCommMonoid M
                                                                     inst✝⁶ : AddCommMonoid N
                                                                     inst✝⁵ : AddCommMonoid P
                                                                     inst✝⁴ : AddCommMonoid Q
                                                                     inst✝³ : Module R M
                                                                     inst✝² : Module R N
                                                                     inst✝¹ : Module R Q
                                                                     inst✝ : Module R P
                                                                     g : LinearMap (RingHom.id R) P Q
                                                                     f : LinearMap (RingHom.id R) N P
                                                                     x : TensorProduct R N M
                                                                     ⊢ Eq ((LinearMap.rTensor M (g.comp f)) x) ((LinearMap.rTensor M g) ((LinearMap …
                                                                   -/
    (g.comp f).rTensor M x = (g.rTensor M) ((f.rTensor M) x) := by rw [rTensor_comp, coe_comp]; rfl
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


theorem lTensor_mul (f g : Module.End R N) : (f * g).lTensor M = f.lTensor M * g.lTensor M :=
  lTensor_comp M f g


theorem rTensor_mul (f g : Module.End R N) : (f * g).rTensor M = f.rTensor M * g.rTensor M :=
  rTensor_comp M f g


@[simp]
theorem lTensor_id : (id : N →ₗ[R] N).lTensor M = id :=
  map_id

-- `simp` can prove this.

theorem lTensor_id_apply (x : M ⊗[R] N) : (LinearMap.id : N →ₗ[R] N).lTensor M x = x := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    x : TensorProduct R M N
    ⊢ Eq ((LinearMap.lTensor M LinearMap.id) x) x
  -/
  rw [lTensor_id, id_coe, _root_.id]
  /-
    🎉 no goals
  -/


@[simp]
theorem rTensor_id : (id : N →ₗ[R] N).rTensor M = id :=
  map_id

-- `simp` can prove this.

theorem rTensor_id_apply (x : N ⊗[R] M) : (LinearMap.id : N →ₗ[R] N).rTensor M x = x := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    x : TensorProduct R N M
    ⊢ Eq ((LinearMap.rTensor M LinearMap.id) x) x
  -/
  rw [rTensor_id, id_coe, _root_.id]
  /-
    🎉 no goals
  -/


@[simp]
theorem lTensor_smul_action (r : R) :
    (DistribMulAction.toLinearMap R N r).lTensor M =
      DistribMulAction.toLinearMap R (M ⊗[R] N) r :=
  (lTensor_smul M r LinearMap.id).trans (congrArg _ (lTensor_id M N))


@[simp]
theorem rTensor_smul_action (r : R) :
    (DistribMulAction.toLinearMap R N r).rTensor M =
      DistribMulAction.toLinearMap R (N ⊗[R] M) r :=
  (rTensor_smul M r LinearMap.id).trans (congrArg _ (rTensor_id M N))


theorem lid_comp_rTensor (f : N →ₗ[R] R) :
    (TensorProduct.lid R M).comp (rTensor M f) = lift ((lsmul R M).comp f) := ext' fun _ _ ↦ rfl


@[simp]
theorem lTensor_comp_rTensor (f : M →ₗ[R] P) (g : N →ₗ[R] Q) :
    (g.lTensor P).comp (f.rTensor N) = map f g := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    ⊢ Eq ((LinearMap.lTensor P g).comp (LinearMap.rTensor N f)) (TensorProduct.map …
  -/
  simp only [lTensor, rTensor, ← map_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem rTensor_comp_lTensor (f : M →ₗ[R] P) (g : N →ₗ[R] Q) :
    (f.rTensor Q).comp (g.lTensor M) = map f g := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R Q
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    ⊢ Eq ((LinearMap.rTensor Q f).comp (LinearMap.lTensor M g)) (TensorProduct.map …
  -/
  simp only [lTensor, rTensor, ← map_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_comp_rTensor (f : M →ₗ[R] P) (g : N →ₗ[R] Q) (f' : S →ₗ[R] M) :
    (map f g).comp (f'.rTensor _) = map (f.comp f') g := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    S : Type u_9
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : AddCommMonoid P
    inst✝⁶ : AddCommMonoid Q
    inst✝⁵ : AddCommMonoid S
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R Q
    inst✝¹ : Module R S
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    f' : LinearMap (RingHom.id R) S M
    ⊢ Eq ((TensorProduct.map f g).comp (LinearMap.rTensor N f')) (TensorProduct.ma …
  -/
  simp only [lTensor, rTensor, ← map_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_comp_lTensor (f : M →ₗ[R] P) (g : N →ₗ[R] Q) (g' : S →ₗ[R] N) :
    (map f g).comp (g'.lTensor _) = map f (g.comp g') := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    S : Type u_9
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : AddCommMonoid P
    inst✝⁶ : AddCommMonoid Q
    inst✝⁵ : AddCommMonoid S
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R Q
    inst✝¹ : Module R S
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    g' : LinearMap (RingHom.id R) S N
    ⊢ Eq ((TensorProduct.map f g).comp (LinearMap.lTensor M g')) (TensorProduct.ma …
  -/
  simp only [lTensor, rTensor, ← map_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem rTensor_comp_map (f' : P →ₗ[R] S) (f : M →ₗ[R] P) (g : N →ₗ[R] Q) :
    (f'.rTensor _).comp (map f g) = map (f'.comp f) g := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    S : Type u_9
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : AddCommMonoid P
    inst✝⁶ : AddCommMonoid Q
    inst✝⁵ : AddCommMonoid S
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R Q
    inst✝¹ : Module R S
    inst✝ : Module R P
    f' : LinearMap (RingHom.id R) P S
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    ⊢ Eq ((LinearMap.rTensor Q f').comp (TensorProduct.map f g)) (TensorProduct.ma …
  -/
  simp only [lTensor, rTensor, ← map_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem lTensor_comp_map (g' : Q →ₗ[R] S) (f : M →ₗ[R] P) (g : N →ₗ[R] Q) :
    (g'.lTensor _).comp (map f g) = map f (g'.comp g) := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    M : Type u_5
    N : Type u_6
    P : Type u_7
    Q : Type u_8
    S : Type u_9
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : AddCommMonoid P
    inst✝⁶ : AddCommMonoid Q
    inst✝⁵ : AddCommMonoid S
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R Q
    inst✝¹ : Module R S
    inst✝ : Module R P
    g' : LinearMap (RingHom.id R) Q S
    f : LinearMap (RingHom.id R) M P
    g : LinearMap (RingHom.id R) N Q
    ⊢ Eq ((LinearMap.lTensor P g').comp (TensorProduct.map f g)) (TensorProduct.ma …
  -/
  simp only [lTensor, rTensor, ← map_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem rTensor_pow (f : M →ₗ[R] M) (n : ℕ) : f.rTensor N ^ n = (f ^ n).rTensor N := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M M
    n : Nat
    ⊢ Eq (HPow.hPow (LinearMap.rTensor N f) n) (LinearMap.rTensor N (HPow.hPow f n))
  -/
  have h := TensorProduct.map_pow f (id : N →ₗ[R] N) n
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M M
    n : Nat
    h : Eq (HPow.hPow (TensorProduct.map f LinearMap.id) n) (TensorProduct.map (HP …
    ⊢ Eq (HPow.hPow (LinearMap.rTensor N f) n) (LinearMap.rTensor N (HPow.hPow f n))
  -/
  rwa [id_pow] at h
  /-
    🎉 no goals
  -/


@[simp]
theorem lTensor_pow (f : N →ₗ[R] N) (n : ℕ) : f.lTensor M ^ n = (f ^ n).lTensor M := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) N N
    n : Nat
    ⊢ Eq (HPow.hPow (LinearMap.lTensor M f) n) (LinearMap.lTensor M (HPow.hPow f n))
  -/
  have h := TensorProduct.map_pow (id : M →ₗ[R] M) f n
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) N N
    n : Nat
    h : Eq (HPow.hPow (TensorProduct.map LinearMap.id f) n) (TensorProduct.map (HP …
    ⊢ Eq (HPow.hPow (LinearMap.lTensor M f) n) (LinearMap.lTensor M (HPow.hPow f n))
  -/
  rwa [id_pow] at h
  /-
    🎉 no goals
  -/


/-- `LinearEquiv.lTensor M f : M ⊗ N ≃ₗ M ⊗ P` is the natural linear equivalence
induced by `f : N ≃ₗ P`. -/
def lTensor (f : N ≃ₗ[R] P) : M ⊗[R] N ≃ₗ[R] M ⊗[R] P := TensorProduct.congr (refl R M) f


/-- `LinearEquiv.rTensor M f : N₁ ⊗ M ≃ₗ N₂ ⊗ M` is the natural linear equivalence
induced by `f : N₁ ≃ₗ N₂`. -/
def rTensor (f : N ≃ₗ[R] P) : N ⊗[R] M ≃ₗ[R] P ⊗[R] M := TensorProduct.congr f (refl R M)


@[simp] theorem coe_lTensor : lTensor M f = (f : N →ₗ[R] P).lTensor M := rfl


@[simp] theorem coe_lTensor_symm : (lTensor M f).symm = (f.symm : P →ₗ[R] N).lTensor M := rfl


@[simp] theorem coe_rTensor : rTensor M f = (f : N →ₗ[R] P).rTensor M := rfl


@[simp] theorem coe_rTensor_symm : (rTensor M f).symm = (f.symm : P →ₗ[R] N).rTensor M := rfl


@[simp] theorem lTensor_tmul : f.lTensor M (m ⊗ₜ n) = m ⊗ₜ f n := rfl


@[simp] theorem lTensor_symm_tmul : (f.lTensor M).symm (m ⊗ₜ p) = m ⊗ₜ f.symm p := rfl


@[simp] theorem rTensor_tmul : f.rTensor M (n ⊗ₜ m) = f n ⊗ₜ m := rfl


@[simp] theorem rTensor_symm_tmul : (f.rTensor M).symm (p ⊗ₜ m) = f.symm p ⊗ₜ m := rfl


lemma comm_trans_rTensor_trans_comm_eq (g : N ≃ₗ[R] P) :
    TensorProduct.comm R Q N ≪≫ₗ rTensor Q g ≪≫ₗ TensorProduct.comm R P Q = lTensor Q g :=
  toLinearMap_injective <| TensorProduct.ext rfl


lemma comm_trans_lTensor_trans_comm_eq (g : N ≃ₗ[R] P) :
    TensorProduct.comm R N Q ≪≫ₗ lTensor Q g ≪≫ₗ TensorProduct.comm R Q P = rTensor Q g :=
  toLinearMap_injective <| TensorProduct.ext rfl


theorem lTensor_trans : (f ≪≫ₗ g).lTensor M = f.lTensor M ≪≫ₗ g.lTensor M :=
  toLinearMap_injective <| LinearMap.lTensor_comp M _ _


theorem lTensor_trans_apply : (f ≪≫ₗ g).lTensor M x = g.lTensor M (f.lTensor M x) :=
  LinearMap.lTensor_comp_apply M _ _ x


theorem rTensor_trans : (f ≪≫ₗ g).rTensor M = f.rTensor M ≪≫ₗ g.rTensor M :=
  toLinearMap_injective <| LinearMap.rTensor_comp M _ _


theorem rTensor_trans_apply : (f ≪≫ₗ g).rTensor M y = g.rTensor M (f.rTensor M y) :=
  LinearMap.rTensor_comp_apply M _ _ y


theorem lTensor_mul (f g : N ≃ₗ[R] N) : (f * g).lTensor M = f.lTensor M * g.lTensor M :=
  lTensor_trans M f g


theorem rTensor_mul (f g : N ≃ₗ[R] N) : (f * g).rTensor M = f.rTensor M * g.rTensor M :=
  rTensor_trans M f g


@[simp] theorem lTensor_refl : (refl R N).lTensor M = refl R _ := TensorProduct.congr_refl_refl


                                                              /-
                                                                R : Type u_1
                                                                inst✝⁴ : CommSemiring R
                                                                M : Type u_5
                                                                N : Type u_6
                                                                inst✝³ : AddCommMonoid M
                                                                inst✝² : AddCommMonoid N
                                                                inst✝¹ : Module R M
                                                                inst✝ : Module R N
                                                                x : TensorProduct R M N
                                                                ⊢ Eq ((LinearEquiv.lTensor M (LinearEquiv.refl R N)) x) x
                                                              -/
theorem lTensor_refl_apply : (refl R N).lTensor M x = x := by rw [lTensor_refl, refl_apply]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp] theorem rTensor_refl : (refl R N).rTensor M = refl R _ := TensorProduct.congr_refl_refl


                                                              /-
                                                                R : Type u_1
                                                                inst✝⁴ : CommSemiring R
                                                                M : Type u_5
                                                                N : Type u_6
                                                                inst✝³ : AddCommMonoid M
                                                                inst✝² : AddCommMonoid N
                                                                inst✝¹ : Module R M
                                                                inst✝ : Module R N
                                                                y : TensorProduct R N M
                                                                ⊢ Eq ((LinearEquiv.rTensor M (LinearEquiv.refl R N)) y) y
                                                              -/
theorem rTensor_refl_apply : (refl R N).rTensor M y = y := by rw [rTensor_refl, refl_apply]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp] theorem rTensor_trans_lTensor (f : M ≃ₗ[R] P) (g : N ≃ₗ[R] Q) :
    f.rTensor N ≪≫ₗ g.lTensor P = TensorProduct.congr f g :=
  toLinearMap_injective <| LinearMap.lTensor_comp_rTensor M _ _


@[simp] theorem lTensor_trans_rTensor (f : M ≃ₗ[R] P) (g : N ≃ₗ[R] Q) :
    g.lTensor M ≪≫ₗ f.rTensor Q = TensorProduct.congr f g :=
  toLinearMap_injective <| LinearMap.rTensor_comp_lTensor M _ _


@[simp] theorem rTensor_trans_congr (f : M ≃ₗ[R] P) (g : N ≃ₗ[R] Q) (f' : S ≃ₗ[R] M) :
    f'.rTensor _ ≪≫ₗ TensorProduct.congr f g = TensorProduct.congr (f' ≪≫ₗ f) g :=
  toLinearMap_injective <| LinearMap.map_comp_rTensor M _ _ _


@[simp] theorem lTensor_trans_congr (f : M ≃ₗ[R] P) (g : N ≃ₗ[R] Q) (g' : S ≃ₗ[R] N) :
    g'.lTensor _ ≪≫ₗ TensorProduct.congr f g = TensorProduct.congr f (g' ≪≫ₗ g) :=
  toLinearMap_injective <| LinearMap.map_comp_lTensor M _ _ _


@[simp] theorem congr_trans_rTensor (f' : P ≃ₗ[R] S) (f : M ≃ₗ[R] P) (g : N ≃ₗ[R] Q) :
    TensorProduct.congr f g ≪≫ₗ f'.rTensor _ = TensorProduct.congr (f ≪≫ₗ f') g :=
  toLinearMap_injective <| LinearMap.rTensor_comp_map M _ _ _


@[simp] theorem congr_trans_lTensor (g' : Q ≃ₗ[R] S) (f : M ≃ₗ[R] P) (g : N ≃ₗ[R] Q) :
    TensorProduct.congr f g ≪≫ₗ g'.lTensor _ = TensorProduct.congr f (g ≪≫ₗ g') :=
  toLinearMap_injective <| LinearMap.lTensor_comp_map M _ _ _


@[simp] theorem rTensor_pow (f : M ≃ₗ[R] M) (n : ℕ) : f.rTensor N ^ n = (f ^ n).rTensor N := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearEquiv (RingHom.id R) M M
    n : Nat
    ⊢ Eq (HPow.hPow (LinearEquiv.rTensor N f) n) (LinearEquiv.rTensor N (HPow.hPow …
  -/
  simpa only [one_pow] using TensorProduct.congr_pow f (1 : N ≃ₗ[R] N) n
  /-
    🎉 no goals
  -/


@[simp] theorem rTensor_zpow (f : M ≃ₗ[R] M) (n : ℤ) : f.rTensor N ^ n = (f ^ n).rTensor N := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearEquiv (RingHom.id R) M M
    n : Int
    ⊢ Eq (HPow.hPow (LinearEquiv.rTensor N f) n) (LinearEquiv.rTensor N (HPow.hPow …
  -/
  simpa only [one_zpow] using TensorProduct.congr_zpow f (1 : N ≃ₗ[R] N) n
  /-
    🎉 no goals
  -/


@[simp] theorem lTensor_pow (f : N ≃ₗ[R] N) (n : ℕ) : f.lTensor M ^ n = (f ^ n).lTensor M := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearEquiv (RingHom.id R) N N
    n : Nat
    ⊢ Eq (HPow.hPow (LinearEquiv.lTensor M f) n) (LinearEquiv.lTensor M (HPow.hPow …
  -/
  simpa only [one_pow] using TensorProduct.congr_pow (1 : M ≃ₗ[R] M) f n
  /-
    🎉 no goals
  -/


@[simp] theorem lTensor_zpow (f : N ≃ₗ[R] N) (n : ℤ) : f.lTensor M ^ n = (f ^ n).lTensor M := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_5
    N : Type u_6
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearEquiv (RingHom.id R) N N
    n : Int
    ⊢ Eq (HPow.hPow (LinearEquiv.lTensor M f) n) (LinearEquiv.lTensor M (HPow.hPow …
  -/
  simpa only [one_zpow] using TensorProduct.congr_zpow (1 : M ≃ₗ[R] M) f n
  /-
    🎉 no goals
  -/


/-- Auxiliary function to defining negation multiplication on tensor product. -/
def Neg.aux : M ⊗[R] N →ₗ[R] M ⊗[R] N :=
  lift <| (mk R M N).comp (-LinearMap.id)


instance neg : Neg (M ⊗[R] N) where
  neg := Neg.aux R


protected theorem neg_add_cancel (x : M ⊗[R] N) : -x + x = 0 :=
  x.induction_on
        /-
          R : Type u_1
          inst✝⁴ : CommSemiring R
          M : Type u_2
          N : Type u_3
          inst✝³ : AddCommGroup M
          inst✝² : AddCommGroup N
          inst✝¹ : Module R M
          inst✝ : Module R N
          x : TensorProduct R M N
          ⊢ Eq (HAdd.hAdd (-0) 0) 0
        -/
    (by rw [add_zero]; apply (Neg.aux R).map_zero)
                       /-
                         🎉 no goals
                       -/
                   /-
                     R : Type u_1
                     inst✝⁴ : CommSemiring R
                     M : Type u_2
                     N : Type u_3
                     inst✝³ : AddCommGroup M
                     inst✝² : AddCommGroup N
                     inst✝¹ : Module R M
                     inst✝ : Module R N
                     x✝ : TensorProduct R M N
                     x : M
                     y : N
                     ⊢ Eq (HAdd.hAdd (Neg.neg (TensorProduct.tmul R x y)) (TensorProduct.tmul R x y …
                   -/
    (fun x y => by convert (add_tmul (R := R) (-x) x y).symm; rw [neg_add_cancel, zero_tmul])
                                                              /-
                                                                🎉 no goals
                                                              -/
    fun x y hx hy => by
    suffices -x + x + (-y + y) = 0 by
      rw [← this]
      unfold Neg.neg neg
      simp only
      rw [map_add]
      abel
    /-
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Type u_2
      N : Type u_3
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      x✝ x y : TensorProduct R M N
      hx : Eq (HAdd.hAdd (Neg.neg x) x) 0
      hy : Eq (HAdd.hAdd (Neg.neg y) y) 0
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Neg.neg x) x) (HAdd.hAdd (Neg.neg y) y)) 0
    -/
    rw [hx, hy, add_zero]
    /-
      🎉 no goals
    -/


instance addCommGroup : AddCommGroup (M ⊗[R] N) :=
  { TensorProduct.addCommMonoid with
    neg := Neg.neg
    sub := _
    sub_eq_add_neg := fun _ _ => rfl
    neg_add_cancel := fun x => TensorProduct.neg_add_cancel x
    zsmul := fun n v => n • v
                      /-
                        R : Type u_1
                        inst✝¹⁰ : CommSemiring R
                        M : Type u_2
                        N : Type u_3
                        P : Type u_4
                        Q : Type u_5
                        S : Type u_6
                        inst✝⁹ : AddCommGroup M
                        inst✝⁸ : AddCommGroup N
                        inst✝⁷ : AddCommGroup P
                        inst✝⁶ : AddCommGroup Q
                        inst✝⁵ : AddCommGroup S
                        inst✝⁴ : Module R M
                        inst✝³ : Module R N
                        inst✝² : Module R P
                        inst✝¹ : Module R Q
                        inst✝ : Module R S
                        ⊢ ∀ (a : TensorProduct R M N), Eq ((fun n v => HSMul.hSMul n v) 0 a) 0
                      -/
    zsmul_zero' := by simp [TensorProduct.zero_smul]
                      /-
                        🎉 no goals
                      -/
                      /-
                        R : Type u_1
                        inst✝¹⁰ : CommSemiring R
                        M : Type u_2
                        N : Type u_3
                        P : Type u_4
                        Q : Type u_5
                        S : Type u_6
                        inst✝⁹ : AddCommGroup M
                        inst✝⁸ : AddCommGroup N
                        inst✝⁷ : AddCommGroup P
                        inst✝⁶ : AddCommGroup Q
                        inst✝⁵ : AddCommGroup S
                        inst✝⁴ : Module R M
                        inst✝³ : Module R N
                        inst✝² : Module R P
                        inst✝¹ : Module R Q
                        inst✝ : Module R S
                        ⊢ ∀ (n : Nat) (a : TensorProduct R M N), Eq ((fun n v => HSMul.hSMul n v) (↑n. …
                      -/
    zsmul_succ' := by simp [add_comm, TensorProduct.one_smul, TensorProduct.add_smul]
                      /-
                        🎉 no goals
                      -/
    zsmul_neg' := fun n x => by
      /-
        R : Type u_1
        inst✝¹⁰ : CommSemiring R
        M : Type u_2
        N : Type u_3
        P : Type u_4
        Q : Type u_5
        S : Type u_6
        inst✝⁹ : AddCommGroup M
        inst✝⁸ : AddCommGroup N
        inst✝⁷ : AddCommGroup P
        inst✝⁶ : AddCommGroup Q
        inst✝⁵ : AddCommGroup S
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        inst✝¹ : Module R Q
        inst✝ : Module R S
        n : Nat
        x : TensorProduct R M N
        ⊢ Eq ((fun n v => HSMul.hSMul n v) (Int.negSucc n) x) (Neg.neg ((fun n v => HS …
      -/
      change (-n.succ : ℤ) • x = -(((n : ℤ) + 1) • x)
      rw [← zero_add (_ • x), ← TensorProduct.neg_add_cancel ((n.succ : ℤ) • x), add_assoc,
        ← add_smul, ← sub_eq_add_neg, sub_self, zero_smul, add_zero]
      /-
        R : Type u_1
        inst✝¹⁰ : CommSemiring R
        M : Type u_2
        N : Type u_3
        P : Type u_4
        Q : Type u_5
        S : Type u_6
        inst✝⁹ : AddCommGroup M
        inst✝⁸ : AddCommGroup N
        inst✝⁷ : AddCommGroup P
        inst✝⁶ : AddCommGroup Q
        inst✝⁵ : AddCommGroup S
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        inst✝¹ : Module R Q
        inst✝ : Module R S
        n : Nat
        x : TensorProduct R M N
        ⊢ Eq (Neg.neg (HSMul.hSMul (↑n.succ) x)) (Neg.neg (HSMul.hSMul (HAdd.hAdd (↑n) …
      -/
      rfl }
      /-
        🎉 no goals
      -/


theorem neg_tmul (m : M) (n : N) : (-m) ⊗ₜ n = -m ⊗ₜ[R] n :=
  rfl


theorem tmul_neg (m : M) (n : N) : m ⊗ₜ (-n) = -m ⊗ₜ[R] n :=
  (mk R M N _).map_neg _


theorem tmul_sub (m : M) (n₁ n₂ : N) : m ⊗ₜ (n₁ - n₂) = m ⊗ₜ[R] n₁ - m ⊗ₜ[R] n₂ :=
  (mk R M N _).map_sub _ _


theorem sub_tmul (m₁ m₂ : M) (n : N) : (m₁ - m₂) ⊗ₜ n = m₁ ⊗ₜ[R] n - m₂ ⊗ₜ[R] n :=
  (mk R M N).map_sub₂ _ _ _


/-- While the tensor product will automatically inherit a ℤ-module structure from
`AddCommGroup.toIntModule`, that structure won't be compatible with lemmas like `tmul_smul` unless
we use a `ℤ-Module` instance provided by `TensorProduct.left_module`.

When `R` is a `Ring` we get the required `TensorProduct.compatible_smul` instance through
`IsScalarTower`, but when it is only a `Semiring` we need to build it from scratch.
The instance diamond in `compatible_smul` doesn't matter because it's in `Prop`.
-/
instance CompatibleSMul.int : CompatibleSMul R ℤ M N :=
  ⟨fun r m n =>
                           /-
                             R : Type u_1
                             inst✝¹⁰ : CommSemiring R
                             M : Type u_2
                             N : Type u_3
                             P : Type u_4
                             Q : Type u_5
                             S : Type u_6
                             inst✝⁹ : AddCommGroup M
                             inst✝⁸ : AddCommGroup N
                             inst✝⁷ : AddCommGroup P
                             inst✝⁶ : AddCommGroup Q
                             inst✝⁵ : AddCommGroup S
                             inst✝⁴ : Module R M
                             inst✝³ : Module R N
                             inst✝² : Module R P
                             inst✝¹ : Module R Q
                             inst✝ : Module R S
                             r : Int
                             m : M
                             n : N
                             ⊢ Eq (TensorProduct.tmul R (HSMul.hSMul 0 m) n) (TensorProduct.tmul R m (HSMul …
                           -/
                           /-
                             🎉 no goals
                           -/
    Int.induction_on r (by simp) (fun r ih => by simpa [add_smul, tmul_add, add_tmul] using ih)
                                                 /-
                                                   🎉 no goals
                                                 -/
                     /-
                       R : Type u_1
                       inst✝¹⁰ : CommSemiring R
                       M : Type u_2
                       N : Type u_3
                       P : Type u_4
                       Q : Type u_5
                       S : Type u_6
                       inst✝⁹ : AddCommGroup M
                       inst✝⁸ : AddCommGroup N
                       inst✝⁷ : AddCommGroup P
                       inst✝⁶ : AddCommGroup Q
                       inst✝⁵ : AddCommGroup S
                       inst✝⁴ : Module R M
                       inst✝³ : Module R N
                       inst✝² : Module R P
                       inst✝¹ : Module R Q
                       inst✝ : Module R S
                       r✝ : Int
                       m : M
                       n : N
                       r : Nat
                       ih : Eq (TensorProduct.tmul R (HSMul.hSMul (Neg.neg ↑r) m) n) (TensorProduct.t …
                       ⊢ Eq (TensorProduct.tmul R (HSMul.hSMul (HSub.hSub (Neg.neg ↑r) 1) m) n) (Tens …
                     -/
      fun r ih => by simpa [sub_smul, tmul_sub, sub_tmul] using ih⟩
                     /-
                       🎉 no goals
                     -/


instance CompatibleSMul.unit {S} [Monoid S] [DistribMulAction S M] [DistribMulAction S N]
    [CompatibleSMul R S M N] : CompatibleSMul R Sˣ M N :=
  ⟨fun s m n => (CompatibleSMul.smul_tmul (s : S) m n : _)⟩


@[simp]
theorem lTensor_sub (f g : N →ₗ[R] P) : (f - g).lTensor M = f.lTensor M - g.lTensor M := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f g : LinearMap (RingHom.id R) N P
    ⊢ Eq (LinearMap.lTensor M (HSub.hSub f g)) (HSub.hSub (LinearMap.lTensor M f)  …
  -/
  simp_rw [← coe_lTensorHom]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f g : LinearMap (RingHom.id R) N P
    ⊢ Eq ((LinearMap.lTensorHom M) (HSub.hSub f g)) (HSub.hSub ((LinearMap.lTensor …
  -/
  exact (lTensorHom (R := R) (N := N) (P := P) M).map_sub f g
  /-
    🎉 no goals
  -/


@[simp]
theorem rTensor_sub (f g : N →ₗ[R] P) : (f - g).rTensor M = f.rTensor M - g.rTensor M := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f g : LinearMap (RingHom.id R) N P
    ⊢ Eq (LinearMap.rTensor M (HSub.hSub f g)) (HSub.hSub (LinearMap.rTensor M f)  …
  -/
  simp only [← coe_rTensorHom]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f g : LinearMap (RingHom.id R) N P
    ⊢ Eq ((LinearMap.rTensorHom M) (HSub.hSub f g)) (HSub.hSub ((LinearMap.rTensor …
  -/
  exact (rTensorHom (R := R) (N := N) (P := P) M).map_sub f g
  /-
    🎉 no goals
  -/


@[simp]
theorem lTensor_neg (f : N →ₗ[R] P) : (-f).lTensor M = -f.lTensor M := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    ⊢ Eq (LinearMap.lTensor M (Neg.neg f)) (Neg.neg (LinearMap.lTensor M f))
  -/
  simp only [← coe_lTensorHom]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    ⊢ Eq ((LinearMap.lTensorHom M) (Neg.neg f)) (Neg.neg ((LinearMap.lTensorHom M) …
  -/
  exact (lTensorHom (R := R) (N := N) (P := P) M).map_neg f
  /-
    🎉 no goals
  -/


@[simp]
theorem rTensor_neg (f : N →ₗ[R] P) : (-f).rTensor M = -f.rTensor M := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    ⊢ Eq (LinearMap.rTensor M (Neg.neg f)) (Neg.neg (LinearMap.rTensor M f))
  -/
  simp only [← coe_rTensorHom]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    ⊢ Eq ((LinearMap.rTensorHom M) (Neg.neg f)) (Neg.neg ((LinearMap.rTensorHom M) …
  -/
  exact (rTensorHom (R := R) (N := N) (P := P) M).map_neg f
  /-
    🎉 no goals
  -/


