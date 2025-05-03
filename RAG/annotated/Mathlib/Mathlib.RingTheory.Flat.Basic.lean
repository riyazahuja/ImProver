/-- An `R`-module `M` is flat if for all finitely generated ideals `I` of `R`,
the canonical map `I ⊗ M →ₗ M` is injective. -/
@[mk_iff] class Flat : Prop where
  out : ∀ ⦃I : Ideal R⦄ (_ : I.FG),
    Function.Injective (TensorProduct.lift ((lsmul R M).comp I.subtype))


variable {R} in
instance instSubalgebraToSubmodule {S : Type v} [Ring S] [Algebra R S]
    (A : Subalgebra R S) [Flat R A] : Flat R (Subalgebra.toSubmodule A) := ‹Flat R A›


instance self (R : Type u) [CommRing R] : Flat R R :=
  ⟨by
    /-
      R✝ : Type u
      M : Type v
      inst✝³ : CommRing R✝
      inst✝² : AddCommGroup M
      inst✝¹ : Module R✝ M
      R : Type u
      inst✝ : CommRing R
      ⊢ ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(TensorProduct.lift ((LinearMap. …
    -/
    intro I _
    /-
      R✝ : Type u
      M : Type v
      inst✝³ : CommRing R✝
      inst✝² : AddCommGroup M
      inst✝¹ : Module R✝ M
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      x✝ : I.FG
      ⊢ Function.Injective ⇑(TensorProduct.lift ((LinearMap.lsmul R R).comp (Submodu …
    -/
    rw [← Equiv.injective_comp (TensorProduct.rid R I).symm.toEquiv]
    /-
      R✝ : Type u
      M : Type v
      inst✝³ : CommRing R✝
      inst✝² : AddCommGroup M
      inst✝¹ : Module R✝ M
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      x✝ : I.FG
      ⊢ Function.Injective (Function.comp ⇑(TensorProduct.lift ((LinearMap.lsmul R R …
    -/
    convert Subtype.coe_injective using 1
    /-
      case h.e'_3
      R✝ : Type u
      M : Type v
      inst✝³ : CommRing R✝
      inst✝² : AddCommGroup M
      inst✝¹ : Module R✝ M
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      x✝ : I.FG
      ⊢ Eq (Function.comp ⇑(TensorProduct.lift ((LinearMap.lsmul R R).comp (Submodul …
    -/
    ext x
    simp only [Function.comp_apply, LinearEquiv.coe_toEquiv, rid_symm_apply, comp_apply, mul_one,
      lift.tmul, Submodule.subtype_apply, Algebra.id.smul_eq_mul, lsmul_apply]⟩


/-- An `R`-module `M` is flat iff for all finitely generated ideals `I` of `R`, the
tensor product of the inclusion `I → R` and the identity `M → M` is injective. See
`iff_rTensor_injective'` to extend to all ideals `I`. --/
lemma iff_rTensor_injective :
    Flat R M ↔ ∀ ⦃I : Ideal R⦄ (_ : I.FG), Function.Injective (rTensor M I.subtype) := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Module.Flat R M) (∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMa …
  -/
  simp [flat_iff, ← lid_comp_rTensor]
  /-
    🎉 no goals
  -/


/-- An `R`-module `M` is flat iff for all ideals `I` of `R`, the tensor product of the
inclusion `I → R` and the identity `M → M` is injective. See `iff_rTensor_injective` to
restrict to finitely generated ideals `I`. --/
theorem iff_rTensor_injective' :
    Flat R M ↔ ∀ I : Ideal R, Function.Injective (rTensor M I.subtype) := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Module.Flat R M) (∀ (I : Ideal R), Function.Injective ⇑(LinearMap.rTens …
  -/
  rewrite [Flat.iff_rTensor_injective]
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submo …
  -/
  refine ⟨fun h I => ?_, fun h I _ => h I⟩
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    I : Ideal R
    ⊢ Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype I))
  -/
  rewrite [injective_iff_map_eq_zero]
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    I : Ideal R
    ⊢ ∀ (a : TensorProduct R (Subtype fun x => Membership.mem I x) M), Eq ((Linear …
  -/
  intro x hx₀
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    I : Ideal R
    x : TensorProduct R (Subtype fun x => Membership.mem I x) M
    hx₀ : Eq ((LinearMap.rTensor M (Submodule.subtype I)) x) 0
    ⊢ Eq x 0
  -/
  obtain ⟨J, hfg, hle, y, rfl⟩ := Submodule.exists_fg_le_eq_rTensor_inclusion x
  /-
    case intro.intro.intro.intro
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    I : Ideal R
    J : Submodule R R
    hfg : J.FG
    hle : LE.le J I
    y : TensorProduct R (Subtype fun x => Membership.mem J x) M
    hx₀ : Eq ((LinearMap.rTensor M (Submodule.subtype I)) ((LinearMap.rTensor M (S …
    ⊢ Eq ((LinearMap.rTensor M (Submodule.inclusion hle)) y) 0
  -/
  rewrite [← rTensor_comp_apply] at hx₀
  /-
    case intro.intro.intro.intro
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    I : Ideal R
    J : Submodule R R
    hfg : J.FG
    hle : LE.le J I
    y : TensorProduct R (Subtype fun x => Membership.mem J x) M
    hx₀ : Eq ((LinearMap.rTensor M ((Submodule.subtype I).comp (Submodule.inclusio …
    ⊢ Eq ((LinearMap.rTensor M (Submodule.inclusion hle)) y) 0
  -/
  rw [(injective_iff_map_eq_zero _).mp (h hfg) y hx₀, LinearMap.map_zero]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-29")]
alias lTensor_inj_iff_rTensor_inj := LinearMap.lTensor_inj_iff_rTensor_inj


/-- The `lTensor`-variant of `iff_rTensor_injective`. . -/
theorem iff_lTensor_injective :
    Module.Flat R M ↔ ∀ ⦃I : Ideal R⦄ (_ : I.FG), Function.Injective (lTensor M I.subtype) := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Module.Flat R M) (∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMa …
  -/
  simpa [← comm_comp_rTensor_comp_comm_eq] using Module.Flat.iff_rTensor_injective R M
  /-
    🎉 no goals
  -/


/-- The `lTensor`-variant of `iff_rTensor_injective'`. . -/
theorem iff_lTensor_injective' :
    Module.Flat R M ↔ ∀ (I : Ideal R), Function.Injective (lTensor M I.subtype) := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Module.Flat R M) (∀ (I : Ideal R), Function.Injective ⇑(LinearMap.lTens …
  -/
  simpa [← comm_comp_rTensor_comp_comm_eq] using Module.Flat.iff_rTensor_injective' R M
  /-
    🎉 no goals
  -/


/-- A retract of a flat `R`-module is flat. -/
lemma of_retract [f : Flat R M] (i : N →ₗ[R] M) (r : M →ₗ[R] N) (h : r.comp i = LinearMap.id) :
    Flat R N := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : Module.Flat R M
    i : LinearMap (RingHom.id R) N M
    r : LinearMap (RingHom.id R) M N
    h : Eq (r.comp i) LinearMap.id
    ⊢ Module.Flat R N
  -/
  rw [iff_rTensor_injective] at *
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    i : LinearMap (RingHom.id R) N M
    r : LinearMap (RingHom.id R) M N
    h : Eq (r.comp i) LinearMap.id
    ⊢ ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor N (Submodule. …
  -/
  intro I hI
  have h₁ : Function.Injective (lTensor R i) := by
    apply Function.RightInverse.injective (g := (lTensor R r))
    intro x
    rw [← LinearMap.comp_apply, ← lTensor_comp, h]
    simp
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    i : LinearMap (RingHom.id R) N M
    r : LinearMap (RingHom.id R) M N
    h : Eq (r.comp i) LinearMap.id
    I : Ideal R
    hI : I.FG
    h₁ : Function.Injective ⇑(LinearMap.lTensor R i)
    ⊢ Function.Injective ⇑(LinearMap.rTensor N (Submodule.subtype I))
  -/
  rw [← Function.Injective.of_comp_iff h₁ (rTensor N I.subtype), ← LinearMap.coe_comp]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    i : LinearMap (RingHom.id R) N M
    r : LinearMap (RingHom.id R) M N
    h : Eq (r.comp i) LinearMap.id
    I : Ideal R
    hI : I.FG
    h₁ : Function.Injective ⇑(LinearMap.lTensor R i)
    ⊢ Function.Injective ⇑((LinearMap.lTensor R i).comp (LinearMap.rTensor N (Subm …
  -/
  rw [LinearMap.lTensor_comp_rTensor, ← LinearMap.rTensor_comp_lTensor]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    i : LinearMap (RingHom.id R) N M
    r : LinearMap (RingHom.id R) M N
    h : Eq (r.comp i) LinearMap.id
    I : Ideal R
    hI : I.FG
    h₁ : Function.Injective ⇑(LinearMap.lTensor R i)
    ⊢ Function.Injective ⇑((LinearMap.rTensor M (Submodule.subtype I)).comp (Linea …
  -/
  rw [LinearMap.coe_comp, Function.Injective.of_comp_iff (f hI)]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    i : LinearMap (RingHom.id R) N M
    r : LinearMap (RingHom.id R) M N
    h : Eq (r.comp i) LinearMap.id
    I : Ideal R
    hI : I.FG
    h₁ : Function.Injective ⇑(LinearMap.lTensor R i)
    ⊢ Function.Injective ⇑(LinearMap.lTensor (Subtype fun x => Membership.mem I x) …
  -/
  apply Function.RightInverse.injective (g := lTensor _ r)
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    i : LinearMap (RingHom.id R) N M
    r : LinearMap (RingHom.id R) M N
    h : Eq (r.comp i) LinearMap.id
    I : Ideal R
    hI : I.FG
    h₁ : Function.Injective ⇑(LinearMap.lTensor R i)
    ⊢ Function.RightInverse ⇑(LinearMap.lTensor (Subtype fun x => Membership.mem I …
  -/
  intro x
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    i : LinearMap (RingHom.id R) N M
    r : LinearMap (RingHom.id R) M N
    h : Eq (r.comp i) LinearMap.id
    I : Ideal R
    hI : I.FG
    h₁ : Function.Injective ⇑(LinearMap.lTensor R i)
    x : TensorProduct R (Subtype fun x => Membership.mem I x) N
    ⊢ Eq ((LinearMap.lTensor (Subtype fun x => Membership.mem I x) r) ((LinearMap. …
  -/
  rw [← LinearMap.comp_apply, ← lTensor_comp, h]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor M (Submodul …
    i : LinearMap (RingHom.id R) N M
    r : LinearMap (RingHom.id R) M N
    h : Eq (r.comp i) LinearMap.id
    I : Ideal R
    hI : I.FG
    h₁ : Function.Injective ⇑(LinearMap.lTensor R i)
    x : TensorProduct R (Subtype fun x => Membership.mem I x) N
    ⊢ Eq ((LinearMap.lTensor (Subtype fun x => Membership.mem I x) LinearMap.id) x …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A `R`-module linearly equivalent to a flat `R`-module is flat. -/
lemma of_linearEquiv [f : Flat R M] (e : N ≃ₗ[R] M) : Flat R N := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : Module.Flat R M
    e : LinearEquiv (RingHom.id R) N M
    ⊢ Module.Flat R N
  -/
  have h : e.symm.toLinearMap.comp e.toLinearMap = LinearMap.id := by simp
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : Module.Flat R M
    e : LinearEquiv (RingHom.id R) N M
    h : Eq ((↑e.symm).comp ↑e) LinearMap.id
    ⊢ Module.Flat R N
  -/
  exact of_retract _ _ _ e.toLinearMap e.symm.toLinearMap h
  /-
    🎉 no goals
  -/


/-- If an `R`-module `M` is linearly equivalent to another `R`-module `N`, then `M` is flat
  if and only if `N` is flat. -/
lemma equiv_iff (e : M ≃ₗ[R] N) : Flat R M ↔ Flat R N :=
  ⟨fun _ => of_linearEquiv R M N e.symm, fun _ => of_linearEquiv R N M e⟩


instance ulift [Module.Flat R M] : Module.Flat R (ULift.{v'} M) :=
  of_linearEquiv R M (ULift.{v'} M) ULift.moduleEquiv

-- Making this an instance causes an infinite sequence `M → ULift M → ULift (ULift M) → ...`.

lemma of_ulift [Module.Flat R (ULift.{v'} M)] : Module.Flat R M :=
  of_linearEquiv R (ULift.{v'} M) M ULift.moduleEquiv.symm


instance shrink [Small.{v'} M] [Module.Flat R M] : Module.Flat R (Shrink.{v'} M) :=
  of_linearEquiv R M (Shrink.{v'} M) (Shrink.linearEquiv M R)

-- Making this an instance causes an infinite sequence `M → Shrink M → Shrink (Shrink M) → ...`.

lemma of_shrink [Small.{v'} M] [Module.Flat R (Shrink.{v'} M)] :
    Module.Flat R M :=
  of_linearEquiv R (Shrink.{v'} M) M (Shrink.linearEquiv M R).symm


/-- A direct sum of flat `R`-modules is flat. -/
instance directSum (ι : Type v) (M : ι → Type w) [(i : ι) → AddCommGroup (M i)]
    [(i : ι) → Module R (M i)] [F : (i : ι) → (Flat R (M i))] : Flat R (⨁ i, M i) := by
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    ⊢ Module.Flat R (DirectSum ι fun i => M i)
  -/
  haveI := Classical.decEq ι
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this : DecidableEq ι
    ⊢ Module.Flat R (DirectSum ι fun i => M i)
  -/
  rw [iff_rTensor_injective]
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this : DecidableEq ι
    ⊢ ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor (DirectSum ι  …
  -/
  intro I hI
  -- This instance was added during PR https://github.com/leanprover-community/mathlib4/pull/10828,
  -- see https://leanprover.zulipchat.com/#narrow/stream/144837-PR-reviews/topic/.2310828.20-.20generalizing.20CommRing.20to.20CommSemiring.20etc.2E/near/422684923
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this : DecidableEq ι
    I : Ideal R
    hI : I.FG
    ⊢ Function.Injective ⇑(LinearMap.rTensor (DirectSum ι fun i => M i) (Submodule …
  -/
  letI : ∀ i, AddCommGroup (I ⊗[R] M i) := inferInstance
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    ⊢ Function.Injective ⇑(LinearMap.rTensor (DirectSum ι fun i => M i) (Submodule …
  -/
  rw [← Equiv.comp_injective _ (TensorProduct.lid R (⨁ i, M i)).toEquiv]
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    ⊢ Function.Injective (Function.comp ⇑(TensorProduct.lid R (DirectSum ι fun i = …
  -/
  set η₁ := TensorProduct.lid R (⨁ i, M i)
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    ⊢ Function.Injective (Function.comp ⇑η₁.toEquiv ⇑(LinearMap.rTensor (DirectSum …
  -/
  set η := fun i ↦ TensorProduct.lid R (M i)
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    ⊢ Function.Injective (Function.comp ⇑η₁.toEquiv ⇑(LinearMap.rTensor (DirectSum …
  -/
  set φ := fun i ↦ rTensor (M i) I.subtype
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    ⊢ Function.Injective (Function.comp ⇑η₁.toEquiv ⇑(LinearMap.rTensor (DirectSum …
  -/
  set π := fun i ↦ component R ι (fun j ↦ M j) i
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ⊢ Function.Injective (Function.comp ⇑η₁.toEquiv ⇑(LinearMap.rTensor (DirectSum …
  -/
  set ψ := (TensorProduct.directSumRight R {x // x ∈ I} (fun i ↦ M i)).symm.toLinearMap with psi_def
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ⊢ Function.Injective (Function.comp ⇑η₁.toEquiv ⇑(LinearMap.rTensor (DirectSum …
  -/
  set ρ := rTensor (⨁ i, M i) I.subtype
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    ⊢ Function.Injective (Function.comp ⇑η₁.toEquiv ⇑ρ)
  -/
  set τ := (fun i ↦ component R ι (fun j ↦ ({x // x ∈ I} ⊗[R] (M j))) i)
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    ⊢ Function.Injective (Function.comp ⇑η₁.toEquiv ⇑ρ)
  -/
  rw [← Equiv.injective_comp (TensorProduct.directSumRight _ _ _).symm.toEquiv]
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    ⊢ Function.Injective (Function.comp (Function.comp ⇑η₁.toEquiv ⇑ρ) ⇑(TensorPro …
  -/
  rw [LinearEquiv.coe_toEquiv, ← LinearEquiv.coe_coe, ← LinearMap.coe_comp]
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    ⊢ Function.Injective (Function.comp ⇑((↑η₁).comp ρ) ⇑(TensorProduct.directSumR …
  -/
  rw [LinearEquiv.coe_toEquiv, ← LinearEquiv.coe_coe, ← LinearMap.coe_comp]
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    ⊢ Function.Injective ⇑(((↑η₁).comp ρ).comp ↑(TensorProduct.directSumRight R (S …
  -/
  rw [← psi_def, injective_iff_map_eq_zero ((η₁.comp ρ).comp ψ)]
  have h₁ : ∀ (i : ι), (π i).comp ((η₁.comp ρ).comp ψ) = (η i).comp ((φ i).comp (τ i)) := by
    intro i
    apply DirectSum.linearMap_ext
    intro j
    apply TensorProduct.ext'
    intro a m
    simp only [ρ, ψ, φ, η, η₁, coe_comp, LinearEquiv.coe_coe, Function.comp_apply,
      directSumRight_symm_lof_tmul, rTensor_tmul, Submodule.coe_subtype, lid_tmul, map_smul]
    rw [DirectSum.component.of, DirectSum.component.of]
    by_cases h₂ : j = i
    · subst j; simp
    · simp [h₂]
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    h₁ : ∀ (i : ι), Eq ((π i).comp (((↑η₁).comp ρ).comp ψ)) ((↑(η i)).comp ((φ i). …
    ⊢ ∀ (a : DirectSum ι fun i => TensorProduct R (Subtype fun x => Membership.mem …
  -/
  intro a ha; rw [DirectSum.ext_iff R]; intro i
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    h₁ : ∀ (i : ι), Eq ((π i).comp (((↑η₁).comp ρ).comp ψ)) ((↑(η i)).comp ((φ i). …
    a : DirectSum ι fun i => TensorProduct R (Subtype fun x => Membership.mem I x) …
    ha : Eq ((((↑η₁).comp ρ).comp ψ) a) 0
    i : ι
    ⊢ Eq ((DirectSum.component R ι (fun i => TensorProduct R (Subtype fun x => Mem …
  -/
  have f := LinearMap.congr_arg (f := (π i)) ha
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    h₁ : ∀ (i : ι), Eq ((π i).comp (((↑η₁).comp ρ).comp ψ)) ((↑(η i)).comp ((φ i). …
    a : DirectSum ι fun i => TensorProduct R (Subtype fun x => Membership.mem I x) …
    ha : Eq ((((↑η₁).comp ρ).comp ψ) a) 0
    i : ι
    f : Eq ((π i) ((((↑η₁).comp ρ).comp ψ) a)) ((π i) 0)
    ⊢ Eq ((DirectSum.component R ι (fun i => TensorProduct R (Subtype fun x => Mem …
  -/
  erw [LinearMap.congr_fun (h₁ i) a] at f
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    h₁ : ∀ (i : ι), Eq ((π i).comp (((↑η₁).comp ρ).comp ψ)) ((↑(η i)).comp ((φ i). …
    a : DirectSum ι fun i => TensorProduct R (Subtype fun x => Membership.mem I x) …
    ha : Eq ((((↑η₁).comp ρ).comp ψ) a) 0
    i : ι
    f : Eq (((↑(η i)).comp ((φ i).comp (τ i))) a) ((π i) 0)
    ⊢ Eq ((DirectSum.component R ι (fun i => TensorProduct R (Subtype fun x => Mem …
  -/
  rw [(map_zero (π i) : (π i) 0 = (0 : M i))] at f
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    h₁ : ∀ (i : ι), Eq ((π i).comp (((↑η₁).comp ρ).comp ψ)) ((↑(η i)).comp ((φ i). …
    a : DirectSum ι fun i => TensorProduct R (Subtype fun x => Membership.mem I x) …
    ha : Eq ((((↑η₁).comp ρ).comp ψ) a) 0
    i : ι
    f : Eq (((↑(η i)).comp ((φ i).comp (τ i))) a) 0
    ⊢ Eq ((DirectSum.component R ι (fun i => TensorProduct R (Subtype fun x => Mem …
  -/
  have h₂ := F i
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    h₁ : ∀ (i : ι), Eq ((π i).comp (((↑η₁).comp ρ).comp ψ)) ((↑(η i)).comp ((φ i). …
    a : DirectSum ι fun i => TensorProduct R (Subtype fun x => Membership.mem I x) …
    ha : Eq ((((↑η₁).comp ρ).comp ψ) a) 0
    i : ι
    f : Eq (((↑(η i)).comp ((φ i).comp (τ i))) a) 0
    h₂ : Module.Flat R (M i)
    ⊢ Eq ((DirectSum.component R ι (fun i => TensorProduct R (Subtype fun x => Mem …
  -/
  rw [iff_rTensor_injective] at h₂
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    h₁ : ∀ (i : ι), Eq ((π i).comp (((↑η₁).comp ρ).comp ψ)) ((↑(η i)).comp ((φ i). …
    a : DirectSum ι fun i => TensorProduct R (Subtype fun x => Membership.mem I x) …
    ha : Eq ((((↑η₁).comp ρ).comp ψ) a) 0
    i : ι
    f : Eq (((↑(η i)).comp ((φ i).comp (τ i))) a) 0
    h₂ : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor (M i) (Sub …
    ⊢ Eq ((DirectSum.component R ι (fun i => TensorProduct R (Subtype fun x => Mem …
  -/
  have h₃ := h₂ hI
  simp only [φ, τ, coe_comp, LinearEquiv.coe_coe, Function.comp_apply,
    EmbeddingLike.map_eq_zero_iff, h₃, LinearMap.map_eq_zero_iff] at f
  /-
    R : Type u
    M✝ : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M✝
    inst✝⁴ : Module R M✝
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    ι : Type v
    M : ι → Type w
    inst✝¹ : (i : ι) → AddCommGroup (M i)
    inst✝ : (i : ι) → Module R (M i)
    F : ∀ (i : ι), Module.Flat R (M i)
    this✝ : DecidableEq ι
    I : Ideal R
    hI : I.FG
    this : (i : ι) → AddCommGroup (TensorProduct R (Subtype fun x => Membership.me …
    η₁ : LinearEquiv (RingHom.id R) (TensorProduct R R (DirectSum ι fun i => M i)) …
    η : (i : ι) → LinearEquiv (RingHom.id R) (TensorProduct R R (M i)) (M i) := fu …
    φ : (i : ι) → LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Memb …
    π : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => M i) (M i) := fun …
    ψ : LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R (Subtype fu …
    psi_def : Eq ψ ↑(TensorProduct.directSumRight R (Subtype fun x => Membership.m …
    ρ : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    τ : (i : ι) → LinearMap (RingHom.id R) (DirectSum ι fun i => TensorProduct R ( …
    h₁ : ∀ (i : ι), Eq ((π i).comp (((↑η₁).comp ρ).comp ψ)) ((↑(η i)).comp ((φ i). …
    a : DirectSum ι fun i => TensorProduct R (Subtype fun x => Membership.mem I x) …
    ha : Eq ((((↑η₁).comp ρ).comp ψ) a) 0
    i : ι
    h₂ : ∀ ⦃I : Ideal R⦄, I.FG → Function.Injective ⇑(LinearMap.rTensor (M i) (Sub …
    h₃ : Function.Injective ⇑(LinearMap.rTensor (M i) (Submodule.subtype I))
    f : Eq ((DirectSum.component R ι (fun j => TensorProduct R (Subtype fun x => M …
    ⊢ Eq ((DirectSum.component R ι (fun i => TensorProduct R (Subtype fun x => Mem …
  -/
  simp [f]
  /-
    🎉 no goals
  -/


open scoped Classical in
/-- Free `R`-modules over discrete types are flat. -/
instance finsupp (ι : Type v) : Flat R (ι →₀ R) :=
  of_linearEquiv R _ _ (finsuppLEquivDirectSum R R ι)


instance of_free [Free R M] : Flat R M := of_linearEquiv R _ _ (Free.repr R M)


/-- A projective module with a discrete type of generator is flat -/
lemma of_projective_surjective (ι : Type w) [Projective R M] (p : (ι →₀ R) →ₗ[R] M)
    (hp : Surjective p) : Flat R M := by
  /-
    R : Type u
    M : Type v
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type w
    inst✝ : Module.Projective R M
    p : LinearMap (RingHom.id R) (Finsupp ι R) M
    hp : Function.Surjective ⇑p
    ⊢ Module.Flat R M
  -/
  have h := Module.projective_lifting_property p (LinearMap.id) hp
  cases h with
    | _ e he => exact of_retract R _ _ _ _ he


instance of_projective [h : Projective R M] : Flat R M := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    h : Module.Projective R M
    ⊢ Module.Flat R M
  -/
  rw [Module.projective_def'] at h
  cases h with
    | _ e he => exact of_retract R _ _ _ _ he


/--
Define the character module of `M` to be `M →+ ℚ ⧸ ℤ`.
The character module of `M` is an injective module if and only if
 `L ⊗ 𝟙 M` is injective for any linear map `L` in the same universe as `M`.
-/
lemma injective_characterModule_iff_rTensor_preserves_injective_linearMap :
    Module.Injective R (CharacterModule M) ↔
    ∀ ⦃N N' : Type v⦄ [AddCommGroup N] [AddCommGroup N'] [Module R N] [Module R N']
      (L : N →ₗ[R] N'), Function.Injective L → Function.Injective (L.rTensor M) := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Module.Injective R (CharacterModule M)) (∀ ⦃N N' : Type v⦄ [inst : AddC …
  -/
  simp_rw [injective_iff, rTensor_injective_iff_lcomp_surjective, Surjective, DFunLike.ext_iff]; rfl
                                                                                                 /-
                                                                                                   🎉 no goals
                                                                                                 -/


/-- `CharacterModule M` is Baer iff `M` is flat. -/
theorem iff_characterModule_baer : Flat R M ↔ Module.Baer R (CharacterModule M) := by
  simp_rw [iff_rTensor_injective', Baer, rTensor_injective_iff_lcomp_surjective,
                                                   /-
                                                     R : Type u
                                                     M : Type v
                                                     inst✝² : CommRing R
                                                     inst✝¹ : AddCommGroup M
                                                     inst✝ : Module R M
                                                     ⊢ Iff (∀ (I : Ideal R) (b : LinearMap (RingHom.id R) (Subtype fun x => Members …
                                                   -/
    Surjective, DFunLike.ext_iff, Subtype.forall]; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- `CharacterModule M` is an injective module iff `M` is flat. -/
theorem iff_characterModule_injective [Small.{v} R] :
    Flat R M ↔ Module.Injective R (CharacterModule M) :=
  iff_characterModule_baer.trans Module.Baer.iff_injective


/--
If `M` is a flat module, then `f ⊗ 𝟙 M` is injective for all injective linear maps `f`.
-/
theorem rTensor_preserves_injective_linearMap {N' : Type*} [AddCommGroup N'] [Module R N']
    [h : Flat R M] (L : N →ₗ[R] N') (hL : Function.Injective L) :
    Function.Injective (L.rTensor M) :=
  rTensor_injective_iff_lcomp_surjective.2 ((iff_characterModule_baer.1 h).extension_property _ hL)


@[deprecated (since := "2024-03-29")]
alias preserves_injective_linearMap := rTensor_preserves_injective_linearMap


instance {S} [CommRing S] [Algebra R S] [Module S M] [IsScalarTower R S M] [Flat S M] [Flat R N] :
    Flat S (M ⊗[R] N) :=
  (iff_rTensor_injective' _ _).mpr fun I ↦ by
    simpa [AlgebraTensorModule.rTensor_tensor] using
      rTensor_preserves_injective_linearMap (.restrictScalars R <| I.subtype.rTensor M)
      (rTensor_preserves_injective_linearMap _ I.injective_subtype)


/--
If `M` is a flat module, then `𝟙 M ⊗ f` is injective for all injective linear maps `f`.
-/
theorem lTensor_preserves_injective_linearMap {N' : Type*} [AddCommGroup N'] [Module R N']
    [Flat R M] (L : N →ₗ[R] N') (hL : Function.Injective L) :
    Function.Injective (L.lTensor M) :=
  (L.lTensor_inj_iff_rTensor_inj M).2 (rTensor_preserves_injective_linearMap L hL)


variable (R M) in
/-- `M` is flat if and only if `f ⊗ 𝟙 M` is injective whenever `f` is an injective linear map.
  See `Module.Flat.iff_rTensor_preserves_injective_linearMap` to specialize the universe of
  `N, N', N''` to `Type (max u v)`. -/
lemma iff_rTensor_preserves_injective_linearMap' [Small.{v'} R] [Small.{v'} M] : Flat R M ↔
    ∀ ⦃N N' : Type v'⦄ [AddCommGroup N] [AddCommGroup N'] [Module R N] [Module R N']
      (f : N →ₗ[R] N') (_ : Function.Injective f), Function.Injective (f.rTensor M) :=
  (Module.Flat.equiv_iff R M (Shrink.{v'} M) (Shrink.linearEquiv M R).symm).trans <|
    iff_characterModule_injective.trans <|
      (injective_characterModule_iff_rTensor_preserves_injective_linearMap R (Shrink.{v'} M)).trans
        <| forall₅_congr <| fun N N' _ _ _ => forall₃_congr <| fun _ f _ =>
  let frmu := f.rTensor (Shrink.{v'} M)
  let frm := f.rTensor M
  let emn := TensorProduct.congr (LinearEquiv.refl R N) (Shrink.linearEquiv M R)
  let emn' := TensorProduct.congr (LinearEquiv.refl R N') (Shrink.linearEquiv M R)
  have h : emn'.toLinearMap.comp frmu = frm.comp emn.toLinearMap := TensorProduct.ext rfl
  (EquivLike.comp_injective frmu emn').symm.trans <|
    (congrArg Function.Injective (congrArg DFunLike.coe h)).to_iff.trans <|
      EquivLike.injective_comp emn frm


variable (R M) in
/-- `M` is flat if and only if `f ⊗ 𝟙 M` is injective whenever `f` is an injective linear map.
  See `Module.Flat.iff_rTensor_preserves_injective_linearMap'` to generalize the universe of
  `N, N', N''` to any universe that is higher than `R` and `M`. -/
lemma iff_rTensor_preserves_injective_linearMap : Flat R M ↔
    ∀ ⦃N N' : Type (max u v)⦄ [AddCommGroup N] [AddCommGroup N'] [Module R N] [Module R N']
      (f : N →ₗ[R] N') (_ : Function.Injective f), Function.Injective (f.rTensor M) :=
  iff_rTensor_preserves_injective_linearMap'.{max u v} R M


variable (R M) in
/-- `M` is flat if and only if `𝟙 M ⊗ f` is injective whenever `f` is an injective linear map.
  See `Module.Flat.iff_lTensor_preserves_injective_linearMap` to specialize the universe of
  `N, N', N''` to `Type (max u v)`. -/
lemma iff_lTensor_preserves_injective_linearMap' [Small.{v'} R] [Small.{v'} M] : Flat R M ↔
    ∀ ⦃N N' : Type v'⦄ [AddCommGroup N] [AddCommGroup N'] [Module R N] [Module R N']
      (L : N →ₗ[R] N'), Function.Injective L → Function.Injective (L.lTensor M) := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Small.{v', u} R
    inst✝ : Small.{v', v} M
    ⊢ Iff (Module.Flat R M) (∀ ⦃N N' : Type v'⦄ [inst : AddCommGroup N] [inst_1 :  …
  -/
  simp_rw [iff_rTensor_preserves_injective_linearMap', LinearMap.lTensor_inj_iff_rTensor_inj]
  /-
    🎉 no goals
  -/


variable (R M) in
/-- `M` is flat if and only if `𝟙 M ⊗ f` is injective whenever `f` is an injective linear map.
  See `Module.Flat.iff_lTensor_preserves_injective_linearMap'` to generalize the universe of
  `N, N', N''` to any universe that is higher than `R` and `M`. -/
lemma iff_lTensor_preserves_injective_linearMap : Flat R M ↔
    ∀ ⦃N N' : Type (max u v)⦄ [AddCommGroup N] [AddCommGroup N'] [Module R N] [Module R N']
      (f : N →ₗ[R] N') (_ : Function.Injective f), Function.Injective (f.lTensor M) :=
  iff_lTensor_preserves_injective_linearMap'.{max u v} R M


variable (M) in
/-- If `M` is flat then `M ⊗ -` is an exact functor. -/
lemma lTensor_exact [Flat R M] ⦃N N' N'' : Type*⦄
    [AddCommGroup N] [AddCommGroup N'] [AddCommGroup N''] [Module R N] [Module R N'] [Module R N'']
    ⦃f : N →ₗ[R] N'⦄ ⦃g : N' →ₗ[R] N''⦄ (exact : Function.Exact f g) :
    Function.Exact (f.lTensor M) (g.lTensor M) := by
  /-
    R : Type u
    M : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module.Flat R M
    N : Type u_1
    N' : Type u_2
    N'' : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : AddCommGroup N'
    inst✝³ : AddCommGroup N''
    inst✝² : Module R N
    inst✝¹ : Module R N'
    inst✝ : Module R N''
    f : LinearMap (RingHom.id R) N N'
    g : LinearMap (RingHom.id R) N' N''
    exact : Function.Exact ⇑f ⇑g
    ⊢ Function.Exact ⇑(LinearMap.lTensor M f) ⇑(LinearMap.lTensor M g)
  -/
  let π : N' →ₗ[R] N' ⧸ LinearMap.range f := Submodule.mkQ _
  let ι : N' ⧸ LinearMap.range f →ₗ[R] N'' :=
    Submodule.subtype _ ∘ₗ (LinearMap.quotKerEquivRange g).toLinearMap ∘ₗ
      Submodule.quotEquivOfEq (LinearMap.range f) (LinearMap.ker g)
        (LinearMap.exact_iff.mp exact).symm
  suffices exact1 : Function.Exact (f.lTensor M) (π.lTensor M) by
    rw [show g = ι.comp π from rfl, lTensor_comp]
    exact exact1.comp_injective _ (lTensor_preserves_injective_linearMap ι <| by
      simpa [ι, - Subtype.val_injective] using Subtype.val_injective) (map_zero _)
  /-
    R : Type u
    M : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module.Flat R M
    N : Type u_1
    N' : Type u_2
    N'' : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : AddCommGroup N'
    inst✝³ : AddCommGroup N''
    inst✝² : Module R N
    inst✝¹ : Module R N'
    inst✝ : Module R N''
    f : LinearMap (RingHom.id R) N N'
    g : LinearMap (RingHom.id R) N' N''
    exact : Function.Exact ⇑f ⇑g
    π : LinearMap (RingHom.id R) N' (HasQuotient.Quotient N' (LinearMap.range f))  …
    ι : LinearMap (RingHom.id R) (HasQuotient.Quotient N' (LinearMap.range f)) N'' …
    ⊢ Function.Exact ⇑(LinearMap.lTensor M f) ⇑(LinearMap.lTensor M π)
  -/
  exact _root_.lTensor_exact _ (fun x => by simp [π]) Quotient.mk''_surjective
  /-
    🎉 no goals
  -/


variable (M) in
/-- If `M` is flat then `- ⊗ M` is an exact functor. -/
lemma rTensor_exact [Flat R M] ⦃N N' N'' : Type*⦄
    [AddCommGroup N] [AddCommGroup N'] [AddCommGroup N''] [Module R N] [Module R N'] [Module R N'']
    ⦃f : N →ₗ[R] N'⦄ ⦃g : N' →ₗ[R] N''⦄ (exact : Function.Exact f g) :
    Function.Exact (f.rTensor M) (g.rTensor M) := by
  /-
    R : Type u
    M : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module.Flat R M
    N : Type u_1
    N' : Type u_2
    N'' : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : AddCommGroup N'
    inst✝³ : AddCommGroup N''
    inst✝² : Module R N
    inst✝¹ : Module R N'
    inst✝ : Module R N''
    f : LinearMap (RingHom.id R) N N'
    g : LinearMap (RingHom.id R) N' N''
    exact : Function.Exact ⇑f ⇑g
    ⊢ Function.Exact ⇑(LinearMap.rTensor M f) ⇑(LinearMap.rTensor M g)
  -/
  let π : N' →ₗ[R] N' ⧸ LinearMap.range f := Submodule.mkQ _
  let ι : N' ⧸ LinearMap.range f →ₗ[R] N'' :=
    Submodule.subtype _ ∘ₗ (LinearMap.quotKerEquivRange g).toLinearMap ∘ₗ
      Submodule.quotEquivOfEq (LinearMap.range f) (LinearMap.ker g)
        (LinearMap.exact_iff.mp exact).symm
  suffices exact1 : Function.Exact (f.rTensor M) (π.rTensor M) by
    rw [show g = ι.comp π from rfl, rTensor_comp]
    exact exact1.comp_injective _ (rTensor_preserves_injective_linearMap ι <| by
      simpa [ι, - Subtype.val_injective] using Subtype.val_injective) (map_zero _)
  /-
    R : Type u
    M : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module.Flat R M
    N : Type u_1
    N' : Type u_2
    N'' : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : AddCommGroup N'
    inst✝³ : AddCommGroup N''
    inst✝² : Module R N
    inst✝¹ : Module R N'
    inst✝ : Module R N''
    f : LinearMap (RingHom.id R) N N'
    g : LinearMap (RingHom.id R) N' N''
    exact : Function.Exact ⇑f ⇑g
    π : LinearMap (RingHom.id R) N' (HasQuotient.Quotient N' (LinearMap.range f))  …
    ι : LinearMap (RingHom.id R) (HasQuotient.Quotient N' (LinearMap.range f)) N'' …
    ⊢ Function.Exact ⇑(LinearMap.rTensor M f) ⇑(LinearMap.rTensor M π)
  -/
  exact _root_.rTensor_exact M (fun x => by simp [π]) Quotient.mk''_surjective
  /-
    🎉 no goals
  -/


/-- `M` is flat if and only if `M ⊗ -` is an exact functor. See
  `Module.Flat.iff_lTensor_exact` to specialize the universe of `N, N', N''` to `Type (max u v)`. -/
theorem iff_lTensor_exact' [Small.{v'} R] [Small.{v'} M] : Flat R M ↔
    ∀ ⦃N N' N'' : Type v'⦄ [AddCommGroup N] [AddCommGroup N'] [AddCommGroup N'']
      [Module R N] [Module R N'] [Module R N''] ⦃f : N →ₗ[R] N'⦄ ⦃g : N' →ₗ[R] N''⦄,
        Function.Exact f g → Function.Exact (f.lTensor M) (g.lTensor M) := by
  refine ⟨fun _ => lTensor_exact M, fun H => iff_lTensor_preserves_injective_linearMap' R M |>.mpr
    fun N' N'' _ _ _ _ L hL => LinearMap.ker_eq_bot |>.mp <| eq_bot_iff |>.mpr
      fun x (hx : _ = 0) => ?_⟩
  simpa [Eq.comm] using @H PUnit N' N'' _ _ _ _ _ _ 0 L (fun x => by
    simp_rw [Set.mem_range, LinearMap.zero_apply, exists_const]
    exact (L.map_eq_zero_iff hL).trans eq_comm) x |>.mp  hx


/-- `M` is flat if and only if `M ⊗ -` is an exact functor.
  See `Module.Flat.iff_lTensor_exact'` to generalize the universe of
  `N, N', N''` to any universe that is higher than `R` and `M`. -/
theorem iff_lTensor_exact : Flat R M ↔
    ∀ ⦃N N' N'' : Type (max u v)⦄ [AddCommGroup N] [AddCommGroup N'] [AddCommGroup N'']
      [Module R N] [Module R N'] [Module R N''] ⦃f : N →ₗ[R] N'⦄ ⦃g : N' →ₗ[R] N''⦄,
        Function.Exact f g → Function.Exact (f.lTensor M) (g.lTensor M) :=
  iff_lTensor_exact'.{max u v}


/-- `M` is flat if and only if `- ⊗ M` is an exact functor. See
  `Module.Flat.iff_rTensor_exact` to specialize the universe of `N, N', N''` to `Type (max u v)`. -/
theorem iff_rTensor_exact' [Small.{v'} R] [Small.{v'} M] : Flat R M ↔
    ∀ ⦃N N' N'' : Type v'⦄ [AddCommGroup N] [AddCommGroup N'] [AddCommGroup N'']
      [Module R N] [Module R N'] [Module R N''] ⦃f : N →ₗ[R] N'⦄ ⦃g : N' →ₗ[R] N''⦄,
        Function.Exact f g → Function.Exact (f.rTensor M) (g.rTensor M) := by
  refine ⟨fun _ => rTensor_exact M, fun H => iff_rTensor_preserves_injective_linearMap' R M |>.mpr
    fun N' N'' _ _ _ _ L hL => LinearMap.ker_eq_bot |>.mp <| eq_bot_iff |>.mpr
      fun x (hx : _ = 0) => ?_⟩
  simpa [Eq.comm] using @H PUnit N' N'' _ _ _ _ _ _ 0 L (fun x => by
    simp_rw [Set.mem_range, LinearMap.zero_apply, exists_const]
    exact (L.map_eq_zero_iff hL).trans eq_comm) x |>.mp hx


/-- `M` is flat if and only if `- ⊗ M` is an exact functor.
  See `Module.Flat.iff_rTensor_exact'` to generalize the universe of
  `N, N', N''` to any universe that is higher than `R` and `M`. -/
theorem iff_rTensor_exact : Flat R M ↔
    ∀ ⦃N N' N'' : Type (max u v)⦄ [AddCommGroup N] [AddCommGroup N'] [AddCommGroup N'']
      [Module R N] [Module R N'] [Module R N''] ⦃f : N →ₗ[R] N'⦄ ⦃g : N' →ₗ[R] N''⦄,
        Function.Exact f g → Function.Exact (f.rTensor M) (g.rTensor M) :=
  iff_rTensor_exact'.{max u v}


/-- If p and q are submodules of M and N respectively, and M and q are flat,
then `p ⊗ q → M ⊗ N` is injective. -/
theorem tensorProduct_mapIncl_injective_of_right
    [Flat R M] [Flat R q] : Function.Injective (mapIncl p q) := by
  /-
    R : Type u
    M : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    p : Submodule R M
    q : Submodule R N
    inst✝¹ : Module.Flat R M
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem q x)
    ⊢ Function.Injective ⇑(TensorProduct.mapIncl p q)
  -/
  rw [mapIncl, ← lTensor_comp_rTensor]
  exact (lTensor_preserves_injective_linearMap _ q.injective_subtype).comp
    (rTensor_preserves_injective_linearMap _ p.injective_subtype)


/-- If p and q are submodules of M and N respectively, and N and p are flat,
then `p ⊗ q → M ⊗ N` is injective. -/
theorem tensorProduct_mapIncl_injective_of_left
    [Flat R p] [Flat R N] : Function.Injective (mapIncl p q) := by
  /-
    R : Type u
    M : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type w
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    p : Submodule R M
    q : Submodule R N
    inst✝¹ : Module.Flat R (Subtype fun x => Membership.mem p x)
    inst✝ : Module.Flat R N
    ⊢ Function.Injective ⇑(TensorProduct.mapIncl p q)
  -/
  rw [mapIncl, ← rTensor_comp_lTensor]
  exact (rTensor_preserves_injective_linearMap _ p.injective_subtype).comp
    (lTensor_preserves_injective_linearMap _ q.injective_subtype)


theorem includeLeft_injective [Module.Flat R A] (hb : Function.Injective (algebraMap R B)) :
    Function.Injective (includeLeft : A →ₐ[S] A ⊗[R] B) := by
  convert Module.Flat.lTensor_preserves_injective_linearMap (M := A) (Algebra.linearMap R B) hb
    |>.comp (_root_.TensorProduct.rid R A).symm.injective
  /-
    case h.e'_3
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R B
    inst✝³ : CommSemiring S
    inst✝² : Algebra S A
    inst✝¹ : SMulCommClass R S A
    inst✝ : Module.Flat R A
    hb : Function.Injective ⇑(algebraMap R B)
    ⊢ Eq (⇑Algebra.TensorProduct.includeLeft) (Function.comp ⇑(LinearMap.lTensor A …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem includeRight_injective [Module.Flat R B] (ha : Function.Injective (algebraMap R A)) :
    Function.Injective (includeRight : B →ₐ[R] A ⊗[R] B) := by
  convert Module.Flat.rTensor_preserves_injective_linearMap (M := B) (Algebra.linearMap R A) ha
    |>.comp (_root_.TensorProduct.lid R B).symm.injective
  /-
    case h.e'_3
    R : Type u_1
    A : Type u_3
    B : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    inst✝² : Ring B
    inst✝¹ : Algebra R B
    inst✝ : Module.Flat R B
    ha : Function.Injective ⇑(algebraMap R A)
    ⊢ Eq (⇑Algebra.TensorProduct.includeRight) (Function.comp ⇑(LinearMap.rTensor  …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- If `M`, `N` are `R`-modules, there exists an injective `R`-linear map from `R` to `N`,
and `M` is a nontrivial flat `R`-module, then `M ⊗[R] N` is nontrivial. -/
theorem nontrivial_of_linearMap_injective_of_flat_left (f : R →ₗ[R] N) (h : Function.Injective f)
    [Module.Flat R M] [Nontrivial M] : Nontrivial (M ⊗[R] N) :=
  Module.Flat.lTensor_preserves_injective_linearMap (M := M) f h |>.comp
    (TensorProduct.rid R M).symm.injective |>.nontrivial


/-- If `M`, `N` are `R`-modules, there exists an injective `R`-linear map from `R` to `M`,
and `N` is a nontrivial flat `R`-module, then `M ⊗[R] N` is nontrivial. -/
theorem nontrivial_of_linearMap_injective_of_flat_right (f : R →ₗ[R] M) (h : Function.Injective f)
    [Module.Flat R N] [Nontrivial N] : Nontrivial (M ⊗[R] N) :=
  Module.Flat.rTensor_preserves_injective_linearMap (M := N) f h |>.comp
    (TensorProduct.lid R N).symm.injective |>.nontrivial


/-- If `A`, `B` are `R`-algebras, `R` injects into `B`,
and `A` is a nontrivial flat `R`-algebra, then `A ⊗[R] B` is nontrivial. -/
theorem nontrivial_of_algebraMap_injective_of_flat_left (h : Function.Injective (algebraMap R B))
    [Module.Flat R A] [Nontrivial A] : Nontrivial (A ⊗[R] B) :=
  TensorProduct.nontrivial_of_linearMap_injective_of_flat_left R A B (Algebra.linearMap R B) h


/-- If `A`, `B` are `R`-algebras, `R` injects into `A`,
and `B` is a nontrivial flat `R`-algebra, then `A ⊗[R] B` is nontrivial. -/
theorem nontrivial_of_algebraMap_injective_of_flat_right (h : Function.Injective (algebraMap R A))
    [Module.Flat R B] [Nontrivial B] : Nontrivial (A ⊗[R] B) :=
  TensorProduct.nontrivial_of_linearMap_injective_of_flat_right R A B (Algebra.linearMap R A) h


