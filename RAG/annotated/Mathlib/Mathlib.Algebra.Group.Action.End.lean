/-- Push forward the action of `R` on `M` along a compatible surjective map `f : R →* S`.

See also `Function.Surjective.distribMulActionLeft` and `Function.Surjective.moduleLeft`.
-/
@[to_additive
"Push forward the action of `R` on `M` along a compatible surjective map `f : R →+ S`."]
abbrev Function.Surjective.mulActionLeft {R S M : Type*} [Monoid R] [MulAction R M] [Monoid S]
    [SMul S M] (f : R →* S) (hf : Surjective f) (hsmul : ∀ (c) (x : M), f c • x = c • x) :
    MulAction S M where
  smul := (· • ·)
                   /-
                     M✝ : Type u_1
                     N : Type u_2
                     α : Type u_3
                     inst✝⁵ : Monoid M✝
                     inst✝⁴ : MulAction M✝ α
                     R : Type u_4
                     S : Type u_5
                     M : Type u_6
                     inst✝³ : Monoid R
                     inst✝² : MulAction R M
                     inst✝¹ : Monoid S
                     inst✝ : SMul S M
                     f : MonoidHom R S
                     hf : Function.Surjective ⇑f
                     hsmul : ∀ (c : R) (x : M), Eq (HSMul.hSMul (f c) x) (HSMul.hSMul c x)
                     b : M
                     ⊢ Eq (HSMul.hSMul 1 b) b
                   -/
  one_smul b := by rw [← f.map_one, hsmul, one_smul]
                   /-
                     🎉 no goals
                   -/
                                            /-
                                              M✝ : Type u_1
                                              N : Type u_2
                                              α : Type u_3
                                              inst✝⁵ : Monoid M✝
                                              inst✝⁴ : MulAction M✝ α
                                              R : Type u_4
                                              S : Type u_5
                                              M : Type u_6
                                              inst✝³ : Monoid R
                                              inst✝² : MulAction R M
                                              inst✝¹ : Monoid S
                                              inst✝ : SMul S M
                                              f : MonoidHom R S
                                              hf : Function.Surjective ⇑f
                                              hsmul : ∀ (c : R) (x : M), Eq (HSMul.hSMul (f c) x) (HSMul.hSMul c x)
                                              a b : R
                                              x : M
                                              ⊢ Eq (HSMul.hSMul (HMul.hMul (f a) (f b)) x) (HSMul.hSMul (f a) (HSMul.hSMul ( …
                                            -/
  mul_smul := hf.forall₂.mpr fun a b x ↦ by simp only [← f.map_mul, hsmul, mul_smul]
                                            /-
                                              🎉 no goals
                                            -/


/-- A multiplicative action of `M` on `α` and a monoid homomorphism `N → M` induce
a multiplicative action of `N` on `α`.

See note [reducible non-instances]. -/
@[to_additive]
abbrev compHom [Monoid N] (g : N →* M) : MulAction N α where
  smul := SMul.comp.smul g
  -- Porting note: was `by simp [g.map_one, MulAction.one_smul]`
                   /-
                     M : Type u_1
                     N : Type u_2
                     α : Type u_3
                     inst✝² : Monoid M
                     inst✝¹ : MulAction M α
                     inst✝ : Monoid N
                     g : MonoidHom N M
                     x✝ : α
                     ⊢ Eq (HSMul.hSMul 1 x✝) x✝
                   -/
  one_smul _ := by simpa [(· • ·)] using MulAction.one_smul ..
                   /-
                     🎉 no goals
                   -/
  -- Porting note: was `by simp [g.map_mul, MulAction.mul_smul]`
                       /-
                         M : Type u_1
                         N : Type u_2
                         α : Type u_3
                         inst✝² : Monoid M
                         inst✝¹ : MulAction M α
                         inst✝ : Monoid N
                         g : MonoidHom N M
                         x✝² x✝¹ : N
                         x✝ : α
                         ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) x✝) (HSMul.hSMul x✝² (HSMul.hSMul x✝¹ x✝))
                       -/
  mul_smul _ _ _ := by simpa [(· • ·)] using MulAction.mul_smul ..
                       /-
                         🎉 no goals
                       -/


@[to_additive]
lemma compHom_smul_def
    {E F G : Type*} [Monoid E] [Monoid F] [MulAction F G] (f : E →* F) (a : E) (x : G) :
    letI : MulAction E G := MulAction.compHom _ f
    a • x = (f a) • x := rfl


/-- If the multiplicative action of `M` on `N` is compatible with multiplication on `N`, then
`fun x ↦ x • 1` is a monoid homomorphism from `M` to `N`. -/
@[to_additive (attr := simps)
"If the additive action of `M` on `N` is compatible with addition on `N`, then
`fun x ↦ x +ᵥ 0` is an additive monoid homomorphism from `M` to `N`."]
def MonoidHom.smulOneHom {M N} [Monoid M] [MulOneClass N] [MulAction M N] [IsScalarTower M N N] :
    M →* N where
  toFun x := x • (1 : N)
  map_one' := one_smul _ _
                     /-
                       M✝ : Type u_1
                       N✝ : Type u_2
                       α : Type u_3
                       M : Type ?u.2733
                       N : Type ?u.2736
                       inst✝³ : Monoid M
                       inst✝² : MulOneClass N
                       inst✝¹ : MulAction M N
                       inst✝ : IsScalarTower M N N
                       x y : M
                       ⊢ Eq ({ toFun := fun x => HSMul.hSMul x 1, map_one' := ⋯ }.toFun (HMul.hMul x  …
                     -/
  map_mul' x y := by rw [smul_one_mul, smul_smul]
                     /-
                       🎉 no goals
                     -/


/-- A monoid homomorphism between two monoids M and N can be equivalently specified by a
multiplicative action of M on N that is compatible with the multiplication on N. -/
@[to_additive
"A monoid homomorphism between two additive monoids M and N can be equivalently
specified by an additive action of M on N that is compatible with the addition on N."]
def monoidHomEquivMulActionIsScalarTower (M N) [Monoid M] [Monoid N] :
    (M →* N) ≃ {_inst : MulAction M N // IsScalarTower M N N} where
  toFun f := ⟨MulAction.compHom N f, SMul.comp.isScalarTower _⟩
  invFun := fun ⟨_, _⟩ ↦ MonoidHom.smulOneHom
  left_inv f := MonoidHom.ext fun m ↦ mul_one (f m)
  right_inv := fun ⟨_, _⟩ ↦ Subtype.ext <| MulAction.ext <| funext₂ <| smul_one_smul N


/-- The monoid of endomorphisms.

Note that this is generalized by `CategoryTheory.End` to categories other than `Type u`. -/
protected def Function.End := α → α


instance : Monoid (Function.End α) where
  one := id
  mul := (· ∘ ·)
  mul_assoc _ _ _ := rfl
  mul_one _ := rfl
  one_mul _ := rfl
  npow n f := f^[n]
  npow_succ _ _ := Function.iterate_succ _ _


instance : Inhabited (Function.End α) := ⟨1⟩


/-- The tautological action by `Function.End α` on `α`.

This is generalized to bundled endomorphisms by:
* `Equiv.Perm.applyMulAction`
* `AddMonoid.End.applyDistribMulAction`
* `AddMonoid.End.applyModule`
* `AddAut.applyDistribMulAction`
* `MulAut.applyMulDistribMulAction`
* `LinearEquiv.applyDistribMulAction`
* `LinearMap.applyModule`
* `RingHom.applyMulSemiringAction`
* `RingAut.applyMulSemiringAction`
* `AlgEquiv.applyMulSemiringAction`
-/
instance Function.End.applyMulAction : MulAction (Function.End α) α where
  smul := (· <| ·)
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


@[simp] lemma Function.End.smul_def (f : Function.End α) (a : α) : f • a = f a := rfl

--TODO - This statement should be somethting like `toFun (f * g) = toFun f ∘ toFun g`

lemma Function.End.mul_def (f g : Function.End α) : (f * g) = f ∘ g := rfl

--TODO - This statement should be somethting like `toFun 1 = id`

lemma Function.End.one_def : (1 : Function.End α) = id := rfl


/-- The monoid hom representing a monoid action.

When `M` is a group, see `MulAction.toPermHom`. -/
def MulAction.toEndHom [Monoid M] [MulAction M α] : M →* Function.End α where
  toFun := (· • ·)
  map_one' := funext (one_smul M)
  map_mul' x y := funext (mul_smul x y)


/-- The monoid action induced by a monoid hom to `Function.End α`

See note [reducible non-instances]. -/
abbrev MulAction.ofEndHom [Monoid M] (f : M →* Function.End α) : MulAction M α :=
  MulAction.compHom α f

