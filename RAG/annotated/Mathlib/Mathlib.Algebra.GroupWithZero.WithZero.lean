instance one : One (WithZero α) where
  __ := ‹One α›


@[simp, norm_cast] lemma coe_one : ((1 : α) : WithZero α) = 1 := rfl


instance mulZeroClass : MulZeroClass (WithZero α) where
  mul := Option.map₂ (· * ·)
  zero_mul := Option.map₂_none_left (· * ·)
  mul_zero := Option.map₂_none_right (· * ·)


@[simp, norm_cast] lemma coe_mul (a b : α) : (↑(a * b) : WithZero α) = a * b := rfl


lemma unzero_mul {x y : WithZero α} (hxy : x * y ≠ 0) :
    unzero hxy = unzero (left_ne_zero_of_mul hxy) * unzero (right_ne_zero_of_mul hxy) := by
  /-
    α : Type u_1
    inst✝ : Mul α
    x y : WithZero α
    hxy : Ne (HMul.hMul x y) 0
    ⊢ Eq (WithZero.unzero hxy) (HMul.hMul (WithZero.unzero ⋯) (WithZero.unzero ⋯))
  -/
  simp only [← coe_inj, coe_mul, coe_unzero]
  /-
    🎉 no goals
  -/


instance noZeroDivisors : NoZeroDivisors (WithZero α) := ⟨Option.map₂_eq_none_iff.1⟩


instance semigroupWithZero [Semigroup α] : SemigroupWithZero (WithZero α) where
  __ := mulZeroClass
  mul_assoc _ _ _ := Option.map₂_assoc mul_assoc


instance commSemigroup [CommSemigroup α] : CommSemigroup (WithZero α) where
  __ := semigroupWithZero
  mul_comm _ _ := Option.map₂_comm mul_comm


instance mulZeroOneClass [MulOneClass α] : MulZeroOneClass (WithZero α) where
  __ := mulZeroClass
  one_mul := Option.map₂_left_identity one_mul
  mul_one := Option.map₂_right_identity mul_one


/-- Coercion as a monoid hom. -/
@[simps apply]
def coeMonoidHom : α →* WithZero α where
  toFun        := (↑)
  map_one'     := rfl
  map_mul' _ _ := rfl


@[ext high]
theorem monoidWithZeroHom_ext ⦃f g : WithZero α →*₀ β⦄
    (h : f.toMonoidHom.comp coeMonoidHom = g.toMonoidHom.comp coeMonoidHom) :
    f = g :=
  DFunLike.ext _ _ fun
    | 0 => (map_zero f).trans (map_zero g).symm
    | (g : α) => DFunLike.congr_fun h g


/-- The (multiplicative) universal property of `WithZero`. -/
@[simps! symm_apply_apply]
noncomputable nonrec def lift' : (α →* β) ≃ (WithZero α →*₀ β) where
  toFun f :=
    { toFun := fun
        | 0 => 0
        | (a : α) => f a
      map_zero' := rfl
      map_one' := map_one f
      map_mul' := fun
        | 0, _ => (zero_mul _).symm
        | (_ : α), 0 => (mul_zero _).symm
        | (_ : α), (_ : α) => map_mul f _ _ }
  invFun F := F.toMonoidHom.comp coeMonoidHom
  left_inv _ := rfl
  right_inv _ := monoidWithZeroHom_ext rfl


lemma lift'_zero (f : α →* β) : lift' f (0 : WithZero α) = 0 := rfl


@[simp] lemma lift'_coe (f : α →* β) (x : α) : lift' f (x : WithZero α) = f x := rfl


lemma lift'_unique (f : WithZero α →*₀ β) : f = lift' (f.toMonoidHom.comp coeMonoidHom) :=
  (lift'.apply_symm_apply f).symm


/-- The `MonoidWithZero` homomorphism `WithZero α →* WithZero β` induced by a monoid homomorphism
  `f : α →* β`. -/
noncomputable def map' (f : α →* β) : WithZero α →*₀ WithZero β := lift' (coeMonoidHom.comp f)


lemma map'_zero (f : α →* β) : map' f 0 = 0 := rfl


@[simp] lemma map'_coe (f : α →* β) (x : α) : map' f (x : WithZero α) = f x := rfl


@[simp]
lemma map'_id : map' (MonoidHom.id β) = MonoidHom.id (WithZero β) := by
  /-
    β : Type u_2
    inst✝ : MulOneClass β
    ⊢ Eq (↑(WithZero.map' (MonoidHom.id β))) (MonoidHom.id (WithZero β))
  -/
                         /-
                           🎉 no goals
                         -/
  ext x; induction x <;> rfl
                         /-
                           🎉 no goals
                         -/


lemma map'_map'  (f : α →* β) (g : β →* γ) (x) : map' g (map' f x) = map' (g.comp f) x := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MulOneClass α
    inst✝¹ : MulOneClass β
    inst✝ : MulOneClass γ
    f : MonoidHom α β
    g : MonoidHom β γ
    x : WithZero α
    ⊢ Eq ((WithZero.map' g) ((WithZero.map' f) x)) ((WithZero.map' (g.comp f)) x)
  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> rfl
                  /-
                    🎉 no goals
                  -/


@[simp]
lemma map'_comp (f : α →* β) (g : β →* γ) : map' (g.comp f) = (map' g).comp (map' f) :=
  MonoidWithZeroHom.ext fun x => (map'_map' f g x).symm


instance pow : Pow (WithZero α) ℕ where
  pow x n := match x, n with
    | none, 0 => 1
    | none, _ + 1 => 0
    | some x, n => ↑(x ^ n)


@[simp, norm_cast] lemma coe_pow (a : α) (n : ℕ) : (↑(a ^ n) : WithZero α) = a ^ n := rfl


instance monoidWithZero [Monoid α] : MonoidWithZero (WithZero α) where
  __ := mulZeroOneClass
  __ := semigroupWithZero
  npow n a := a ^ n
  npow_zero a := match a with
    | none => rfl
    | some _ => congr_arg some (pow_zero _)
  npow_succ n a := match a with
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   inst✝ : Monoid α
                   n : Nat
                   a : WithZero α
                   ⊢ Eq ((fun n a => HPow.hPow a n) (HAdd.hAdd n 1) Option.none) (HMul.hMul ((fun …
                 -/
    | none => by change 0 ^ (n + 1) = 0 ^ n * 0; simp only [mul_zero]; rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
    | some _ => congr_arg some <| pow_succ _ _


instance commMonoidWithZero [CommMonoid α] : CommMonoidWithZero (WithZero α) :=
  { WithZero.monoidWithZero, WithZero.commSemigroup with }


/-- Extend the inverse operation on `α` to `WithZero α` by sending `0` to `0`. -/
instance inv : Inv (WithZero α) where inv a := Option.map (·⁻¹) a


@[simp, norm_cast] lemma coe_inv (a : α) : ((a⁻¹ : α) : WithZero α) = (↑a)⁻¹ := rfl


@[simp] protected lemma inv_zero : (0 : WithZero α)⁻¹ = 0 := rfl


instance invOneClass [InvOneClass α] : InvOneClass (WithZero α) where
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    γ : Type u_3
                                                    inst✝ : InvOneClass α
                                                    ⊢ Eq (↑(Inv.inv 1)) 1
                                                  -/
  inv_one := show ((1⁻¹ : α) : WithZero α) = 1 by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


instance div : Div (WithZero α) where div := Option.map₂ (· / ·)


@[norm_cast] lemma coe_div (a b : α) : ↑(a / b : α) = (a / b : WithZero α) := rfl


instance : Pow (WithZero α) ℤ where
  pow a n := match a, n with
    | none, Int.ofNat 0 => 1
    | none, Int.ofNat (Nat.succ _) => 0
    | none, Int.negSucc _ => 0
    | some x, n => ↑(x ^ n)


@[simp, norm_cast] lemma coe_zpow (a : α) (n : ℤ) : ↑(a ^ n) = (↑a : WithZero α) ^ n := rfl


instance divInvMonoid [DivInvMonoid α] : DivInvMonoid (WithZero α) where
  __ := monoidWithZero
  div_eq_mul_inv a b := match a, b with
    | none, _ => rfl
    | some _, none => rfl
    | some a, some b => congr_arg some (div_eq_mul_inv a b)
  zpow n a := a ^ n
  zpow_zero' a := match a with
    | none => rfl
    | some _ => congr_arg some (zpow_zero _)
  zpow_succ' n a := match a with
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   inst✝ : DivInvMonoid α
                   n : Nat
                   a : WithZero α
                   ⊢ Eq ((fun n a => HPow.hPow a n) (↑n.succ) Option.none) (HMul.hMul ((fun n a = …
                 -/
    | none => by change 0 ^ _ = 0 ^ _ * 0; simp only [mul_zero]; rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    | some _ => congr_arg some (DivInvMonoid.zpow_succ' _ _)
  zpow_neg' _ a := match a with
    | none => rfl
    | some _ => congr_arg some (DivInvMonoid.zpow_neg' _ _)


instance divInvOneMonoid [DivInvOneMonoid α] : DivInvOneMonoid (WithZero α) where
  __ := divInvMonoid
  __ := invOneClass


instance involutiveInv [InvolutiveInv α] : InvolutiveInv (WithZero α) where
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    γ : Type u_3
                                                    inst✝ : InvolutiveInv α
                                                    a : WithZero α
                                                    ⊢ Eq (Option.map (Function.comp (fun x => Inv.inv x) fun x => Inv.inv x) a) a
                                                  -/
  inv_inv a := (Option.map_map _ _ _).trans <| by simp [Function.comp]
                                                  /-
                                                    🎉 no goals
                                                  -/


instance divisionMonoid [DivisionMonoid α] : DivisionMonoid (WithZero α) where
  __ := divInvMonoid
  __ := involutiveInv
  mul_inv_rev a b := match a, b with
    | none, none => rfl
    | none, some _ => rfl
    | some _, none => rfl
    | some _, some _ => congr_arg some (mul_inv_rev _ _)
  inv_eq_of_mul a b := match a, b with
    | none, none => fun _ ↦ rfl
                                 /-
                                   α : Type u_1
                                   β : Type u_2
                                   γ : Type u_3
                                   inst✝ : DivisionMonoid α
                                   a b✝ : WithZero α
                                   b : α
                                   x✝ : Eq (HMul.hMul Option.none (Option.some b)) 1
                                   ⊢ Eq (Inv.inv Option.none) (Option.some b)
                                 -/
    | none, some b => fun _ ↦ by contradiction
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   α : Type u_1
                                   β : Type u_2
                                   γ : Type u_3
                                   inst✝ : DivisionMonoid α
                                   a✝ b : WithZero α
                                   a : α
                                   x✝ : Eq (HMul.hMul (Option.some a) Option.none) 1
                                   ⊢ Eq (Inv.inv (Option.some a)) Option.none
                                 -/
    | some a, none => fun _ ↦ by contradiction
                                 /-
                                   🎉 no goals
                                 -/
    | some _, some _ => fun h ↦
      congr_arg some <| inv_eq_of_mul_eq_one_right <| Option.some_injective _ h


instance divisionCommMonoid [DivisionCommMonoid α] : DivisionCommMonoid (WithZero α) where
  __ := divisionMonoid
  __ := commSemigroup


/-- If `α` is a group then `WithZero α` is a group with zero. -/
instance groupWithZero : GroupWithZero (WithZero α) where
  __ := monoidWithZero
  __ := divInvMonoid
  __ := nontrivial
  inv_zero := WithZero.inv_zero
  mul_inv_cancel a ha := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : Group α
      a : WithZero α
      ha : Ne a 0
      ⊢ Eq (HMul.hMul a (Inv.inv a)) 1
    -/
    lift a to α using ha
    /-
      case intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : Group α
      a : α
      ⊢ Eq (HMul.hMul (↑a) (Inv.inv ↑a)) 1
    -/
    norm_cast
    /-
      case intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : Group α
      a : α
      ⊢ Eq (HMul.hMul a (Inv.inv a)) 1
    -/
    apply mul_inv_cancel
    /-
      🎉 no goals
    -/



/-- Any group is isomorphic to the units of itself adjoined with `0`. -/
def unitsWithZeroEquiv : (WithZero α)ˣ ≃* α where
  toFun a := unzero a.ne_zero
  invFun a := Units.mk0 a coe_ne_zero
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  γ : Type u_3
                                  inst✝ : Group α
                                  x✝ : Units (WithZero α)
                                  ⊢ Eq ↑((fun a => Units.mk0 ↑a ⋯) ((fun a => WithZero.unzero ⋯) x✝)) ↑x✝
                                -/
  left_inv _ := Units.ext <| by simp only [coe_unzero, Units.mk0_val]
                                /-
                                  🎉 no goals
                                -/
  right_inv _ := rfl
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     γ : Type u_3
                                     inst✝ : Group α
                                     x✝¹ x✝ : Units (WithZero α)
                                     ⊢ Eq ↑({ toFun := fun a => WithZero.unzero ⋯, invFun := fun a => Units.mk0 ↑a  …
                                   -/
  map_mul' _ _ := coe_inj.mp <| by simp only [Units.val_mul, coe_unzero, coe_mul]
                                   /-
                                     🎉 no goals
                                   -/


/-- Any group with zero is isomorphic to adjoining `0` to the units of itself. -/
def withZeroUnitsEquiv {G : Type*} [GroupWithZero G]
    [DecidablePred (fun a : G ↦ a = 0)] :
    WithZero Gˣ ≃* G where
  toFun := WithZero.recZeroCoe 0 Units.val
  invFun a := if h : a = 0 then 0 else (Units.mk0 a h : Gˣ)
  left_inv := (by induction · <;> simp)
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      inst✝² : Group α
                      G : Type u_4
                      inst✝¹ : GroupWithZero G
                      inst✝ : DecidablePred fun a => Eq a 0
                      x✝ : G
                      ⊢ Eq ((fun n => WithZero.recZeroCoe 0 Units.val n) ((fun a => dite (Eq a 0) (f …
                    -/
                                         /-
                                           🎉 no goals
                                         -/
  right_inv _ := by simp only; split <;> simp_all
                                         /-
                                           🎉 no goals
                                         -/
  map_mul' x y := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝² : Group α
      G : Type u_4
      inst✝¹ : GroupWithZero G
      inst✝ : DecidablePred fun a => Eq a 0
      x y : WithZero (Units G)
      ⊢ Eq ({ toFun := fun n => WithZero.recZeroCoe 0 Units.val n, invFun := fun a = …
    -/
    induction x <;> induction y <;>
    /-
      case h₁.h₁
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝² : Group α
      G : Type u_4
      inst✝¹ : GroupWithZero G
      inst✝ : DecidablePred fun a => Eq a 0
      ⊢ Eq ({ toFun := fun n => WithZero.recZeroCoe 0 Units.val n, invFun := fun a = …
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
    simp [← WithZero.coe_mul, ← Units.val_mul]
    /-
      🎉 no goals
    -/


/-- A version of `Equiv.optionCongr` for `WithZero`. -/
noncomputable def _root_.MulEquiv.withZero [Group β] (e : α ≃* β) :
    WithZero α ≃* WithZero β where
  toFun := map' e.toMonoidHom
  invFun := map' e.symm.toMonoidHom
  left_inv := (by induction · <;> simp)
  right_inv := (by induction · <;> simp)
  map_mul' x y := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝¹ : Group α
      inst✝ : Group β
      e : MulEquiv α β
      x y : WithZero α
      ⊢ Eq ({ toFun := ⇑(WithZero.map' e.toMonoidHom), invFun := ⇑(WithZero.map' e.s …
    -/
    induction x <;> induction y <;>
    /-
      case h₁.h₁
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝¹ : Group α
      inst✝ : Group β
      e : MulEquiv α β
      ⊢ Eq ({ toFun := ⇑(WithZero.map' e.toMonoidHom), invFun := ⇑(WithZero.map' e.s …
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
    simp
    /-
      🎉 no goals
    -/


/-- The inverse of `MulEquiv.withZero`. -/
protected noncomputable def _root_.MulEquiv.unzero [Group β] (e : WithZero α ≃* WithZero β) :
    α ≃* β where
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     γ : Type u_3
                                     inst✝¹ : Group α
                                     inst✝ : Group β
                                     e : MulEquiv (WithZero α) (WithZero β)
                                     x : α
                                     ⊢ Ne (e ↑x) 0
                                   -/
  toFun x := unzero (x := e x) (by simp [ne_eq, ← e.eq_symm_apply])
                                   /-
                                     🎉 no goals
                                   -/
                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           γ : Type u_3
                                           inst✝¹ : Group α
                                           inst✝ : Group β
                                           e : MulEquiv (WithZero α) (WithZero β)
                                           x : β
                                           ⊢ Ne (e.symm ↑x) 0
                                         -/
  invFun x := unzero (x := e.symm x) (by simp [e.symm_apply_eq])
                                         /-
                                           🎉 no goals
                                         -/
                   /-
                     α : Type u_1
                     β : Type u_2
                     γ : Type u_3
                     inst✝¹ : Group α
                     inst✝ : Group β
                     e : MulEquiv (WithZero α) (WithZero β)
                     x✝ : α
                     ⊢ Eq ((fun x => WithZero.unzero ⋯) ((fun x => WithZero.unzero ⋯) x✝)) x✝
                   -/
  left_inv _ := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      inst✝¹ : Group α
                      inst✝ : Group β
                      e : MulEquiv (WithZero α) (WithZero β)
                      x✝ : β
                      ⊢ Eq ((fun x => WithZero.unzero ⋯) ((fun x => WithZero.unzero ⋯) x✝)) x✝
                    -/
  right_inv _ := by simp
                    /-
                      🎉 no goals
                    -/
  map_mul' _ _ := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝¹ : Group α
      inst✝ : Group β
      e : MulEquiv (WithZero α) (WithZero β)
      x✝¹ x✝ : α
      ⊢ Eq ({ toFun := fun x => WithZero.unzero ⋯, invFun := fun x => WithZero.unzer …
    -/
    simp only [coe_mul, map_mul]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝¹ : Group α
      inst✝ : Group β
      e : MulEquiv (WithZero α) (WithZero β)
      x✝¹ x✝ : α
      ⊢ Eq (WithZero.unzero ⋯) (HMul.hMul (WithZero.unzero ⋯) (WithZero.unzero ⋯))
    -/
    generalize_proofs A B C
    suffices ((unzero A : β) : WithZero β) = (unzero B) * (unzero C) by
      rwa [← WithZero.coe_mul, WithZero.coe_inj] at this
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝¹ : Group α
      inst✝ : Group β
      e : MulEquiv (WithZero α) (WithZero β)
      x✝¹ x✝ : α
      A : Ne (HMul.hMul (e ↑x✝¹) (e ↑x✝)) 0
      B : Ne (e ↑x✝¹) 0
      C : Ne (e ↑x✝) 0
      ⊢ Eq (↑(WithZero.unzero A)) (HMul.hMul ↑(WithZero.unzero B) ↑(WithZero.unzero  …
    -/
    simp
    /-
      🎉 no goals
    -/


instance commGroupWithZero [CommGroup α] : CommGroupWithZero (WithZero α) :=
  { WithZero.groupWithZero, WithZero.commMonoidWithZero with }


instance addMonoidWithOne [AddMonoidWithOne α] : AddMonoidWithOne (WithZero α) where
  natCast n := if n = 0 then 0 else (n : α)
  natCast_zero := rfl
  natCast_succ n := by
    cases n with
    | zero => show (((1 : ℕ) : α) : WithZero α) = 0 + 1; · rw [Nat.cast_one, coe_one, zero_add]
    | succ n =>
        show (((n + 2 : ℕ) : α) : WithZero α) = ((n + 1 : ℕ) : α) + 1
        rw [Nat.cast_succ, coe_add, coe_one]


