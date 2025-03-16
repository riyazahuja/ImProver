@[to_additive]
instance involutiveInv [InvolutiveInv α] : InvolutiveInv (WithOne α) :=
  { WithOne.inv with
    inv_inv := fun a =>
                                         /-
                                           α : Type u
                                           β : Type v
                                           γ : Type w
                                           inst✝ : InvolutiveInv α
                                           a : WithOne α
                                           ⊢ Eq (Option.map (Function.comp Inv.inv Inv.inv) a) a
                                         -/
      (Option.map_map _ _ _).trans <| by simp_rw [inv_comp_inv, Option.map_id, id] }
                                         /-
                                           🎉 no goals
                                         -/


/-- `WithOne.coe` as a bundled morphism -/
@[to_additive (attr := simps apply) "`WithZero.coe` as a bundled morphism"]
def coeMulHom [Mul α] : α →ₙ* WithOne α where
  toFun := coe
  map_mul' _ _ := rfl


/-- Lift a semigroup homomorphism `f` to a bundled monoid homomorphism. -/
@[to_additive "Lift an add semigroup homomorphism `f` to a bundled add monoid homomorphism."]
def lift : (α →ₙ* β) ≃ (WithOne α →* β) where
  toFun f :=
    { toFun := fun x => Option.casesOn x 1 f, map_one' := rfl,
                                                    /-
                                                      α : Type u
                                                      β : Type v
                                                      γ : Type w
                                                      inst✝¹ : Mul α
                                                      inst✝ : MulOneClass β
                                                      f : MulHom α β
                                                      x y : WithOne α
                                                      ⊢ Eq ({ toFun := fun x => Option.casesOn x 1 ⇑f, map_one' := ⋯ }.toFun (HMul.h …
                                                    -/
      map_mul' := fun x y => WithOne.cases_on x (by rw [one_mul]; exact (one_mul _).symm)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                                         /-
                                           α : Type u
                                           β : Type v
                                           γ : Type w
                                           inst✝¹ : Mul α
                                           inst✝ : MulOneClass β
                                           f : MulHom α β
                                           x✝ y : WithOne α
                                           x : α
                                           ⊢ Eq ({ toFun := fun x => Option.casesOn x 1 ⇑f, map_one' := ⋯ }.toFun (HMul.h …
                                         -/
        (fun x => WithOne.cases_on y (by rw [mul_one]; exact (mul_one _).symm)
                                                       /-
                                                         🎉 no goals
                                                       -/
          (fun y => f.map_mul x y)) }
  invFun F := F.toMulHom.comp coeMulHom
  left_inv _ := MulHom.ext fun _ => rfl
  right_inv F := MonoidHom.ext fun x => WithOne.cases_on x F.map_one.symm (fun _ => rfl)
-- Porting note: the above proofs were broken because they were parenthesized wrong by mathport?


@[to_additive (attr := simp)]
theorem lift_coe (x : α) : lift f x = f x :=
  rfl


@[to_additive (attr := simp)]
theorem lift_one : lift f 1 = 1 :=
  rfl


@[to_additive]
theorem lift_unique (f : WithOne α →* β) : f = lift (f.toMulHom.comp coeMulHom) :=
  (lift.apply_symm_apply f).symm


/-- Given a multiplicative map from `α → β` returns a monoid homomorphism
  from `WithOne α` to `WithOne β` -/
@[to_additive "Given an additive map from `α → β` returns an add monoid homomorphism from
`WithZero α` to `WithZero β`"]
def map (f : α →ₙ* β) : WithOne α →* WithOne β :=
  lift (coeMulHom.comp f)


@[to_additive (attr := simp)]
theorem map_coe (f : α →ₙ* β) (a : α) : map f (a : WithOne α) = f a :=
  rfl


@[to_additive (attr := simp)]
theorem map_id : map (MulHom.id α) = MonoidHom.id (WithOne α) := by
  /-
    α : Type u
    inst✝ : Mul α
    ⊢ Eq (WithOne.map (MulHom.id α)) (MonoidHom.id (WithOne α))
  -/
  ext x
  /-
    case h
    α : Type u
    inst✝ : Mul α
    x : WithOne α
    ⊢ Eq ((WithOne.map (MulHom.id α)) x) ((MonoidHom.id (WithOne α)) x)
  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> rfl
                  /-
                    🎉 no goals
                  -/


@[to_additive]
theorem map_map (f : α →ₙ* β) (g : β →ₙ* γ) (x) : map g (map f x) = map (g.comp f) x := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Mul α
    inst✝¹ : Mul β
    inst✝ : Mul γ
    f : MulHom α β
    g : MulHom β γ
    x : WithOne α
    ⊢ Eq ((WithOne.map g) ((WithOne.map f) x)) ((WithOne.map (g.comp f)) x)
  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> rfl
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem map_comp (f : α →ₙ* β) (g : β →ₙ* γ) : map (g.comp f) = (map g).comp (map f) :=
  MonoidHom.ext fun x => (map_map f g x).symm

-- Porting note: this used to have `@[simps apply]` but it was generating lemmas which
-- weren't in simp normal form.

/-- A version of `Equiv.optionCongr` for `WithOne`. -/
@[to_additive (attr := simps apply) "A version of `Equiv.optionCongr` for `WithZero`."]
def _root_.MulEquiv.withOneCongr (e : α ≃* β) : WithOne α ≃* WithOne β :=
  { map e.toMulHom with
    toFun := map e.toMulHom, invFun := map e.symm.toMulHom,
    left_inv := (by induction · <;> simp)
    right_inv := (by induction · <;> simp) }

-- Porting note: for this declaration and the two below I added the `to_additive` attribute because
-- it seemed to be missing from mathlib3

@[to_additive (attr := simp)]
theorem _root_.MulEquiv.withOneCongr_refl : (MulEquiv.refl α).withOneCongr = MulEquiv.refl _ :=
  MulEquiv.toMonoidHom_injective map_id


@[to_additive (attr := simp)]
theorem _root_.MulEquiv.withOneCongr_symm (e : α ≃* β) :
    e.withOneCongr.symm = e.symm.withOneCongr :=
  rfl


@[to_additive (attr := simp)]
theorem _root_.MulEquiv.withOneCongr_trans (e₁ : α ≃* β) (e₂ : β ≃* γ) :
    e₁.withOneCongr.trans e₂.withOneCongr = (e₁.trans e₂).withOneCongr :=
  MulEquiv.toMonoidHom_injective (map_comp _ _).symm


