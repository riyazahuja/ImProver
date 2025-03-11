/-- A ring `R` satisfies the Orzech property, if for any finitely generated `R`-module `M`,
any surjective homomorphism `f : N → M` from a submodule `N` of `M` to `M` is injective.

NOTE: In the definition we need to assume that `M` has the same universe level as `R`, but it
in fact implies the universe polymorphic versions
`OrzechProperty.injective_of_surjective_of_injective`
and `OrzechProperty.injective_of_surjective_of_submodule`. -/
@[mk_iff]
class OrzechProperty : Prop where
  injective_of_surjective_of_submodule' : ∀ {M : Type u} [AddCommMonoid M] [Module R M]
    [Module.Finite R M] {N : Submodule R M} (f : N →ₗ[R] M), Surjective f → Injective f


theorem injective_of_surjective_of_injective
    {N : Type w} [AddCommMonoid N] [Module R N]
    (i f : N →ₗ[R] M) (hi : Injective i) (hf : Surjective f) : Injective f := by
  /-
    R : Type u
    inst✝⁶ : Semiring R
    inst✝⁵ : OrzechProperty R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    N : Type w
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    i f : LinearMap (RingHom.id R) N M
    hi : Function.Injective ⇑i
    hf : Function.Surjective ⇑f
    ⊢ Function.Injective ⇑f
  -/
  obtain ⟨n, g, hg⟩ := Module.Finite.exists_fin' R M
  /-
    case intro.intro
    R : Type u
    inst✝⁶ : Semiring R
    inst✝⁵ : OrzechProperty R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    N : Type w
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    i f : LinearMap (RingHom.id R) N M
    hi : Function.Injective ⇑i
    hf : Function.Surjective ⇑f
    n : Nat
    g : LinearMap (RingHom.id R) (Fin n → R) M
    hg : Function.Surjective ⇑g
    ⊢ Function.Injective ⇑f
  -/
  haveI := small_of_surjective hg
  /-
    case intro.intro
    R : Type u
    inst✝⁶ : Semiring R
    inst✝⁵ : OrzechProperty R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    N : Type w
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    i f : LinearMap (RingHom.id R) N M
    hi : Function.Injective ⇑i
    hf : Function.Surjective ⇑f
    n : Nat
    g : LinearMap (RingHom.id R) (Fin n → R) M
    hg : Function.Surjective ⇑g
    this : Small.{max u ?u.1860, v} M
    ⊢ Function.Injective ⇑f
  -/
  letI := Equiv.addCommMonoid (equivShrink M).symm
  /-
    case intro.intro
    R : Type u
    inst✝⁶ : Semiring R
    inst✝⁵ : OrzechProperty R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    N : Type w
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    i f : LinearMap (RingHom.id R) N M
    hi : Function.Injective ⇑i
    hf : Function.Surjective ⇑f
    n : Nat
    g : LinearMap (RingHom.id R) (Fin n → R) M
    hg : Function.Surjective ⇑g
    this✝ : Small.{max u ?u.1860, v} M
    this : AddCommMonoid (Shrink.{max u ?u.1860, v} M) := (equivShrink M).symm.add …
    ⊢ Function.Injective ⇑f
  -/
  letI := Equiv.module R (equivShrink M).symm
  /-
    case intro.intro
    R : Type u
    inst✝⁶ : Semiring R
    inst✝⁵ : OrzechProperty R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    N : Type w
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    i f : LinearMap (RingHom.id R) N M
    hi : Function.Injective ⇑i
    hf : Function.Surjective ⇑f
    n : Nat
    g : LinearMap (RingHom.id R) (Fin n → R) M
    hg : Function.Surjective ⇑g
    this✝¹ : Small.{max u ?u.1860, v} M
    this✝ : AddCommMonoid (Shrink.{max u ?u.1860, v} M) := (equivShrink M).symm.ad …
    this : Module R (Shrink.{max u ?u.1860, v} M) := Equiv.module R (equivShrink M …
    ⊢ Function.Injective ⇑f
  -/
  let j : Shrink.{u} M ≃ₗ[R] M := Equiv.linearEquiv R (equivShrink M).symm
  /-
    case intro.intro
    R : Type u
    inst✝⁶ : Semiring R
    inst✝⁵ : OrzechProperty R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    N : Type w
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    i f : LinearMap (RingHom.id R) N M
    hi : Function.Injective ⇑i
    hf : Function.Surjective ⇑f
    n : Nat
    g : LinearMap (RingHom.id R) (Fin n → R) M
    hg : Function.Surjective ⇑g
    this✝¹ : Small.{u, v} M
    this✝ : AddCommMonoid (Shrink.{u, v} M) := (equivShrink M).symm.addCommMonoid
    this : Module R (Shrink.{u, v} M) := Equiv.module R (equivShrink M).symm
    j : LinearEquiv (RingHom.id R) (Shrink.{u, v} M) M := Equiv.linearEquiv R (equ …
    ⊢ Function.Injective ⇑f
  -/
  haveI := Module.Finite.equiv j.symm
  /-
    case intro.intro
    R : Type u
    inst✝⁶ : Semiring R
    inst✝⁵ : OrzechProperty R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    N : Type w
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    i f : LinearMap (RingHom.id R) N M
    hi : Function.Injective ⇑i
    hf : Function.Surjective ⇑f
    n : Nat
    g : LinearMap (RingHom.id R) (Fin n → R) M
    hg : Function.Surjective ⇑g
    this✝² : Small.{u, v} M
    this✝¹ : AddCommMonoid (Shrink.{u, v} M) := (equivShrink M).symm.addCommMonoid
    this✝ : Module R (Shrink.{u, v} M) := Equiv.module R (equivShrink M).symm
    j : LinearEquiv (RingHom.id R) (Shrink.{u, v} M) M := Equiv.linearEquiv R (equ …
    this : Module.Finite R (Shrink.{u, v} M)
    ⊢ Function.Injective ⇑f
  -/
  let i' := j.symm.toLinearMap ∘ₗ i
  /-
    case intro.intro
    R : Type u
    inst✝⁶ : Semiring R
    inst✝⁵ : OrzechProperty R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    N : Type w
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    i f : LinearMap (RingHom.id R) N M
    hi : Function.Injective ⇑i
    hf : Function.Surjective ⇑f
    n : Nat
    g : LinearMap (RingHom.id R) (Fin n → R) M
    hg : Function.Surjective ⇑g
    this✝² : Small.{u, v} M
    this✝¹ : AddCommMonoid (Shrink.{u, v} M) := (equivShrink M).symm.addCommMonoid
    this✝ : Module R (Shrink.{u, v} M) := Equiv.module R (equivShrink M).symm
    j : LinearEquiv (RingHom.id R) (Shrink.{u, v} M) M := Equiv.linearEquiv R (equ …
    this : Module.Finite R (Shrink.{u, v} M)
    i' : LinearMap (RingHom.id R) N (Shrink.{u, v} M) := (↑j.symm).comp i
    ⊢ Function.Injective ⇑f
  -/
  replace hi : Injective i' := by simpa [i'] using hi
  /-
    case intro.intro
    R : Type u
    inst✝⁶ : Semiring R
    inst✝⁵ : OrzechProperty R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    N : Type w
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    i f : LinearMap (RingHom.id R) N M
    hf : Function.Surjective ⇑f
    n : Nat
    g : LinearMap (RingHom.id R) (Fin n → R) M
    hg : Function.Surjective ⇑g
    this✝² : Small.{u, v} M
    this✝¹ : AddCommMonoid (Shrink.{u, v} M) := (equivShrink M).symm.addCommMonoid
    this✝ : Module R (Shrink.{u, v} M) := Equiv.module R (equivShrink M).symm
    j : LinearEquiv (RingHom.id R) (Shrink.{u, v} M) M := Equiv.linearEquiv R (equ …
    this : Module.Finite R (Shrink.{u, v} M)
    i' : LinearMap (RingHom.id R) N (Shrink.{u, v} M) := (↑j.symm).comp i
    hi : Function.Injective ⇑i'
    ⊢ Function.Injective ⇑f
  -/
  let f' := j.symm.toLinearMap ∘ₗ f ∘ₗ (LinearEquiv.ofInjective i' hi).symm.toLinearMap
  /-
    case intro.intro
    R : Type u
    inst✝⁶ : Semiring R
    inst✝⁵ : OrzechProperty R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    N : Type w
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    i f : LinearMap (RingHom.id R) N M
    hf : Function.Surjective ⇑f
    n : Nat
    g : LinearMap (RingHom.id R) (Fin n → R) M
    hg : Function.Surjective ⇑g
    this✝² : Small.{u, v} M
    this✝¹ : AddCommMonoid (Shrink.{u, v} M) := (equivShrink M).symm.addCommMonoid
    this✝ : Module R (Shrink.{u, v} M) := Equiv.module R (equivShrink M).symm
    j : LinearEquiv (RingHom.id R) (Shrink.{u, v} M) M := Equiv.linearEquiv R (equ …
    this : Module.Finite R (Shrink.{u, v} M)
    i' : LinearMap (RingHom.id R) N (Shrink.{u, v} M) := (↑j.symm).comp i
    hi : Function.Injective ⇑i'
    f' : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (LinearMap.rang …
    ⊢ Function.Injective ⇑f
  -/
  replace hf : Surjective f' := by simpa [f'] using hf
  /-
    case intro.intro
    R : Type u
    inst✝⁶ : Semiring R
    inst✝⁵ : OrzechProperty R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    N : Type w
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    i f : LinearMap (RingHom.id R) N M
    n : Nat
    g : LinearMap (RingHom.id R) (Fin n → R) M
    hg : Function.Surjective ⇑g
    this✝² : Small.{u, v} M
    this✝¹ : AddCommMonoid (Shrink.{u, v} M) := (equivShrink M).symm.addCommMonoid
    this✝ : Module R (Shrink.{u, v} M) := Equiv.module R (equivShrink M).symm
    j : LinearEquiv (RingHom.id R) (Shrink.{u, v} M) M := Equiv.linearEquiv R (equ …
    this : Module.Finite R (Shrink.{u, v} M)
    i' : LinearMap (RingHom.id R) N (Shrink.{u, v} M) := (↑j.symm).comp i
    hi : Function.Injective ⇑i'
    f' : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (LinearMap.rang …
    hf : Function.Surjective ⇑f'
    ⊢ Function.Injective ⇑f
  -/
  simpa [f'] using injective_of_surjective_of_submodule' f' hf
  /-
    🎉 no goals
  -/


theorem injective_of_surjective_of_submodule
    {N : Submodule R M} (f : N →ₗ[R] M) (hf : Surjective f) : Injective f :=
  injective_of_surjective_of_injective N.subtype f N.injective_subtype hf


theorem injective_of_surjective_endomorphism
    (f : M →ₗ[R] M) (hf : Surjective f) : Injective f :=
  injective_of_surjective_of_injective _ f (LinearEquiv.refl _ _).injective hf


theorem bijective_of_surjective_endomorphism
    (f : M →ₗ[R] M) (hf : Surjective f) : Bijective f :=
  ⟨injective_of_surjective_endomorphism f hf, hf⟩


