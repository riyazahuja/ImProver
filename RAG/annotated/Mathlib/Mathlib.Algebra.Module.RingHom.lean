/-- Push forward the action of `R` on `M` along a compatible surjective map `f : R →+* S`.

See also `Function.Surjective.mulActionLeft` and `Function.Surjective.distribMulActionLeft`.
-/
abbrev Function.Surjective.moduleLeft {R S M : Type*} [Semiring R] [AddCommMonoid M] [Module R M]
    [Semiring S] [SMul S M] (f : R →+* S) (hf : Function.Surjective f)
    (hsmul : ∀ (c) (x : M), f c • x = c • x) : Module S M :=
  { hf.distribMulActionLeft f.toMonoidHom hsmul with
                             /-
                               R✝ : Type u_1
                               S✝ : Type u_2
                               M✝ : Type u_3
                               M₂ : Type u_4
                               inst✝⁷ : Semiring R✝
                               inst✝⁶ : AddCommMonoid M✝
                               inst✝⁵ : Module R✝ M✝
                               r s : R✝
                               x✝ : M✝
                               R : Type u_5
                               S : Type u_6
                               M : Type u_7
                               inst✝⁴ : Semiring R
                               inst✝³ : AddCommMonoid M
                               inst✝² : Module R M
                               inst✝¹ : Semiring S
                               inst✝ : SMul S M
                               f : RingHom R S
                               hf : Function.Surjective ⇑f
                               hsmul : ∀ (c : R) (x : M), Eq (HSMul.hSMul (f c) x) (HSMul.hSMul c x)
                               x : M
                               ⊢ Eq (HSMul.hSMul 0 x) 0
                             -/
                                               /-
                                                 R✝ : Type u_1
                                                 S✝ : Type u_2
                                                 M✝ : Type u_3
                                                 M₂ : Type u_4
                                                 inst✝⁷ : Semiring R✝
                                                 inst✝⁶ : AddCommMonoid M✝
                                                 inst✝⁵ : Module R✝ M✝
                                                 r s : R✝
                                                 x✝ : M✝
                                                 R : Type u_5
                                                 S : Type u_6
                                                 M : Type u_7
                                                 inst✝⁴ : Semiring R
                                                 inst✝³ : AddCommMonoid M
                                                 inst✝² : Module R M
                                                 inst✝¹ : Semiring S
                                                 inst✝ : SMul S M
                                                 f : RingHom R S
                                                 hf : Function.Surjective ⇑f
                                                 hsmul : ∀ (c : R) (x : M), Eq (HSMul.hSMul (f c) x) (HSMul.hSMul c x)
                                                 a b : R
                                                 x : M
                                                 ⊢ Eq (HSMul.hSMul (HAdd.hAdd (f a) (f b)) x) (HAdd.hAdd (HSMul.hSMul (f a) x)  …
                                               -/
    zero_smul := fun x => by rw [← f.map_zero, hsmul, zero_smul]
                                               /-
                                                 🎉 no goals
                                               -/
                             /-
                               🎉 no goals
                             -/
    add_smul := hf.forall₂.mpr fun a b x => by simp only [← f.map_add, hsmul, add_smul] }


/-- Compose a `Module` with a `RingHom`, with action `f s • m`.

See note [reducible non-instances]. -/
abbrev Module.compHom [Semiring S] (f : S →+* R) : Module S M :=
  { MulActionWithZero.compHom M f.toMonoidWithZeroHom, DistribMulAction.compHom M (f : S →* R) with
    -- Porting note: the `show f (r + s) • x = f r • x + f s • x` wasn't needed in mathlib3.
    -- Somehow, now that `SMul` is heterogeneous, it can't unfold earlier fields of a definition for
    -- use in later fields.  See
    -- https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/Heterogeneous.20scalar.20multiplication
                                                                       /-
                                                                         R : Type u_1
                                                                         S : Type u_2
                                                                         M : Type u_3
                                                                         M₂ : Type u_4
                                                                         inst✝³ : Semiring R
                                                                         inst✝² : AddCommMonoid M
                                                                         inst✝¹ : Module R M
                                                                         r✝ s✝ : R
                                                                         x✝ : M
                                                                         inst✝ : Semiring S
                                                                         f : RingHom S R
                                                                         r s : S
                                                                         x : M
                                                                         ⊢ Eq (HSMul.hSMul (f (HAdd.hAdd r s)) x) (HAdd.hAdd (HSMul.hSMul (f r) x) (HSM …
                                                                       -/
    add_smul := fun r s x => show f (r + s) • x = f r • x + f s • x by simp [add_smul] }
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- A ring homomorphism `f : R →+* M` defines a module structure by `r • x = f r * x`. -/
def RingHom.toModule [Semiring R] [Semiring S] (f : R →+* S) : Module R S :=
  Module.compHom S f


/-- If the module action of `R` on `S` is compatible with multiplication on `S`, then
`fun x ↦ x • 1` is a ring homomorphism from `R` to `S`.

This is the `RingHom` version of `MonoidHom.smulOneHom`.

When `R` is commutative, usually `algebraMap` should be preferred. -/
@[simps!] def RingHom.smulOneHom
    [Semiring R] [NonAssocSemiring S] [Module R S] [IsScalarTower R S S] : R →+* S where
  __ := MonoidHom.smulOneHom
  map_zero' := zero_smul R 1
  map_add' := (add_smul · · 1)


/-- A homomorphism between semirings R and S can be equivalently specified by a R-module
structure on S such that S/S/R is a scalar tower. -/
def ringHomEquivModuleIsScalarTower [Semiring R] [Semiring S] :
    (R →+* S) ≃ {_inst : Module R S // IsScalarTower R S S} where
  toFun f := ⟨Module.compHom S f, SMul.comp.isScalarTower _⟩
  invFun := fun ⟨_, _⟩ ↦ RingHom.smulOneHom
  left_inv f := RingHom.ext fun r ↦ mul_one (f r)
  right_inv := fun ⟨_, _⟩ ↦ Subtype.ext <| Module.ext <| funext₂ <| smul_one_smul S

